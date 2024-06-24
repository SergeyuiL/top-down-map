import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener
import numpy as np
import tf_transformations
from tf2_ros import LookupException

import os
import json
import time
from base import Base

class TargetPublisher(Node):

    def __init__(self, base_frame, world_frame):
        super().__init__('nav_target_publisher')
        
        self.base_frame = base_frame
        self.world_frame = world_frame
        
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_pose = PoseStamped()

    def get_transform(self):
        now = self.get_clock().now()
        past = now - rclpy.duration.Duration(seconds=0.1)  # 小于实际的最小延迟
        future = now + rclpy.duration.Duration(seconds=0.1)  # 未来的短暂时间窗
        try:
            # 尝试获取最接近当前时间的转换，允许一定的时间差
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, self.base_frame, now.to_msg(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            return transform
        except LookupException as e:
            self.get_logger().error(f"Failed to find transform: {e}")
            return None
        
    def print_all_transforms(self):
        all_transforms = self.tf_buffer.all_frames_as_string()
        self.get_logger().info("Current available transforms:\n" + all_transforms)


    @property
    def position(self) -> np.ndarray:
        transform = self.get_transform()
        if transform:
            return np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z], dtype=np.float64)
        else:
            return np.array([0.0, 0.0, 0.0])

    @property
    def rotation(self) -> np.ndarray:
        transform = self.get_transform()
        if transform:
            return np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
        else:
            return np.array([0.0, 0.0, 0.0, 1.0])

    def move_to(self, position: np.ndarray, rotation: np.ndarray, frame_id: str):
        # Sends a `NavToPose` action request
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")
        
        if frame_id == 'world':
            self.goal_pose.header.frame_id = self.world_frame
        elif frame_id == 'robot':
            self.goal_pose.header.frame_id = self.base_frame
        else:
            self.get_logger().error("Wrong frame id input!")
            return
        self.goal_pose.header.stamp = self.get_clock().now().to_msg()
        self.goal_pose.pose.position.x = position[0]
        self.goal_pose.pose.position.y = position[1]
        self.goal_pose.pose.position.z = position[2]
        
        if len(rotation) == 3:
            rotation = tf_transformations.quaternion_from_euler(rotation[0], rotation[1], rotation[2])
        
        self.goal_pose.pose.orientation.x = rotation[0]
        self.goal_pose.pose.orientation.y = rotation[1]
        self.goal_pose.pose.orientation.z = rotation[2]
        self.goal_pose.pose.orientation.w = rotation[3]
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.goal_pose

        self.info('Navigating to goal: ' + str(self.goal_pose.pose.position.x) + ' ' +
                      str(self.goal_pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(self.goal_pose.pose.position.x) + ' ' +
                           str(self.goal_pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return
    
    def debug(self, msg):
        self.get_logger().debug(msg)
        return
    
    def info(self, msg):
        self.get_logger().info(msg)
        return
    
def point_to_line_distance(point, line_start, line_end):
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    nearest = line_start + t * line_vec
    distance = np.linalg.norm(point - nearest)
    return distance, nearest

def main():
    data_dir = "/home/sg/workspace/top-down-map/map06132"
    base_frame = "base_link"
    world_frame = "map"
    
    # Start the ROS 2 Python Client Library
    rclpy.init()
    
    tp = TargetPublisher(base_frame, world_frame)
    base = Base()
    with open(os.path.join(data_dir, "object_info.json"), 'r') as f:
            object_info = json.load(f)
    # contour = np.array(object_info["bed"][0])
    contour = np.array(object_info["couch"][0])
    center = np.mean(contour, axis=0)
    
    cur_position = tp.position[0:2]
    print(f"Current Pose: {tp.position, tp.rotation}")
    # cur_position = np.array([0., 0.])
    robot_radius = 0.3
    safe_distance = 0.2
    
    vector_to_current = cur_position - center
    unit_vector = vector_to_current / np.linalg.norm(vector_to_current)

    min_distance = float('inf')
    nearest_point_on_contour = None

    for point in contour:
        distance, nearest = point_to_line_distance(point, center, cur_position)
        if distance < min_distance:
            min_distance = distance
            nearest_point_on_contour = point

    navigation_point = nearest_point_on_contour + unit_vector * (robot_radius + safe_distance)

    angle = np.arctan2(unit_vector[1], unit_vector[0])
    quaternion = tf_transformations.quaternion_from_euler(0, 0, angle)

    navigation_position = np.array([navigation_point[0], navigation_point[1], 0.0])  
    navigation_orientation = np.array(quaternion)
    print(f"Target Pose: {navigation_position}, {navigation_orientation}")
    
    test_pos = np.array([0.5, 0, 0])
    test_rot = np.array([0, 0, 0, 1.])
    
    base.move_to(test_pos, test_rot, "robot")
    
    tp.move_to(navigation_position, navigation_orientation, "world")

    rclpy.spin(tp)
    tp.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    # main()
    tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
    tf_listener = TransformListener(tf_buffer)
    transform = tf_buffer.lookup_transform(
                "map", "base_link", now.to_msg(),
                timeout=rclpy.duration.Duration(seconds=0.1))
    
