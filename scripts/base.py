from geometry_msgs.msg import PoseStamped
# from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
# from nav2_simple_commander.robot_navigator import 
import rclpy
from rclpy.duration import Duration
import rclpy.duration
import rclpy.logging
from rclpy.node import Node
import numpy as np
from typing import Tuple
import rclpy.task
# from tf_transformations import quaternion_from_euler
from scipy.spatial.transform import Rotation
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose, Spin, DriveOnHeading
import rclpy.timer
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from rclpy.action.client import ClientGoalHandle
import time
class Base(Node):
    def __init__(self,
                 goal_action_name = "/navigate_to_pose",
                 spin_action_name = "/spin",
                 forward_action_name = "/drive_on_heading",
                 base_frame="base_link", 
                 world_frame="map"):
        super().__init__("skillset_base")
        self.goal_base_client = ActionClient(self, NavigateToPose, action_name=goal_action_name)
        self.spin_base_client = ActionClient(self, Spin, spin_action_name)
        self.movefoward_base_client = ActionClient(self, DriveOnHeading, forward_action_name)
        self.target_pose = NavigateToPose.Goal()
        self.controller_handler = None
        self.base_frame = base_frame
        self.world_frame = world_frame

        self.global_costmap = None # self.base_controller.getGlobalCostmap()
        self.local_costmap = None # self.base_controller.getLocalCostmap()

        self.cancel_flag = False

        self.rate = self.create_rate(30)
        self.task_success = False
        self.task_info = ""

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._spin_done = False


    
    @property
    def position(self) -> np.ndarray:
        time.sleep(1)
        base_tf: TransformStamped = self.tf_buffer.lookup_transform(self.world_frame, self.base_frame, rclpy.time.Time())
        return np.array([base_tf.transform.translation.x, base_tf.transform.translation.y, base_tf.transform.translation.z], dtype=np.float64)

    @property
    def rotation(self) -> np.ndarray:
        time.sleep(1)
        base_tf: TransformStamped = self.tf_buffer.lookup_transform(self.world_frame, self.base_frame, rclpy.time.Time())
        return np.array([base_tf.transform.rotation.x, base_tf.transform.rotation.y, base_tf.transform.rotation.z, base_tf.transform.rotation.w])

    def move_to(self, position: np.ndarray,
                rotation: np.ndarray,
                frame_id: str
                ) -> Tuple[bool, str]:
        self.cancel()
        success = True
        info = ""
        self.target_pose.pose.pose.position.x = position[0]
        self.target_pose.pose.pose.position.y = position[1]
        self.target_pose.pose.pose.position.z = position[2]
        if len(rotation) == 3:
            r: Rotation = Rotation.from_euler('xyz', rotation, degrees=False)
            rotation = r.as_quat()
            # rotation = quaternion_from_euler(rotation[0], rotation[1], rotation[2]) # roll, pitch, yaw
        elif len(rotation) == 1:
            r: Rotation = Rotation.from_euler('z', rotation[0], degrees=False)
            rotation = r.as_quat()
            # rotation = quaternion_from_euler(0.0, 0.0, rotation[0])
        self.target_pose.pose.pose.orientation.x = rotation[0]
        self.target_pose.pose.pose.orientation.y = rotation[1]
        self.target_pose.pose.pose.orientation.z = rotation[2]
        self.target_pose.pose.pose.orientation.w = rotation[3]
        if frame_id == "world":
            self.target_pose.pose.header.frame_id = self.world_frame
        elif frame_id == "robot":
            self.target_pose.pose.header.frame_id = self.base_frame
        else:
            self.get_logger().error("ERROR frame_id: {}".format(frame_id))
        
        self.target_pose.pose.header.stamp = self.get_clock().now().to_msg()

        
        # =================== action ========================================
        if self.goal_base_client.wait_for_server(timeout_sec=20):
            self.controller_handler = self.goal_base_client.send_goal_async(self.target_pose, feedback_callback=self._goal_feedback)
            self.controller_handler.add_done_callback(self._goal_response_callback)
            for i in range(20):
            # while not self.controller_handler.done():
                self.get_logger().info("Navigating ....")
                # self.rate.sleep()
                time.sleep(2.0)
            return True, " "
            # result: NavigateToPose.Result = self.controller_handler.result()
            # if result.result_code == 0:
            #     self.get_logger().info("Navigation success!")
            #     success = True
            # else:
            #     self.get_logger().info("Navigation failed!")
            #     success = False
            #     info = result.result_message

        else:
            success = False
            info = "Cannot connected to navigation action server."
        
        self.controller_handler = None
        return success, info
    
    def _goal_feedback(self, fb):
        # fb = fb_msg.feedback
        nav_time = fb.navigation_time
        if fb.navigation_state == NavigateToPose.Feedback.NAV_VOID:
            self.get_logger().info("Navigation action is idle!")
        elif fb.navigation_state == NavigateToPose.Feedback.NAV_RECEIVED_COMMAND:
            self.get_logger().info("Navigation action received goal!")
        elif fb.navigation_state == NavigateToPose.Feedback.NAV_GOAL_PREEMPTED:
            self.get_logger().info("Navigation action is canceled!")
        elif fb.navigation_state == NavigateToPose.Feedback.NAV_PLANNING:
            self.get_logger().info("Navtigation action is planning")
        elif fb.navigation_state == NavigateToPose.Feedback.NAV_CONTROLLING:
            self.get_logger().info("Navigation action is moving!")
        elif fb.navigation_state == NavigateToPose.Feedback.NAV_NEAR_GOAL:
            self.get_logger().info("Navigation action is near the goal!")
        elif fb.navigation_state == NavigateToPose.Feedback.NAV_RECOVERY_SPIN:
            self.get_logger().info("Navigation action is recovery spin!")
        elif fb.navigation_state == NavigateToPose.Feedback.NAV_RECOVERY_WAIT:
            self.get_logger().info("Navigation action is recovery waiting!")
        elif fb.navigation_state == NavigateToPose.Feedback.NAV_GOAL_INVALID:
            self.get_logger().info("Navigation action received invalid goal!")
        elif fb.navigation_state == NavigateToPose.Feedback.NAV_LOCATION_LOST:
            self.get_logger().info("Navigation action lost localization!")
        elif fb.navigation_state == NavigateToPose.Feedback.NAV_IN_VIRTUAL_WALL:
            self.get_logger().info("Navigation action: robot is in the virtual wall!")
        else:
            self.get_logger().error("ERROR navigation state: {}".format(fb.navigation_state))
        # self.get_logger().info("feedback: {}".format(nav_time))

    def _goal_response_callback(self, future):
        goal_handle: ClientGoalHandle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._get_result_callback)
    
    def _get_result_callback(self, future):
        result: NavigateToPose.Result = future.result().result
        if result.result_code == NavigateToPose.Result.PLANNING_FAILED:
            self.get_logger().info("Navigation action planning failed!")
        elif result.result_code == NavigateToPose.Result.EMERGENCY_STOP:
            self.get_logger().info("Navigation action emergency stop!")
        elif result.result_code == NavigateToPose.Result.GOAL_PREEMPTED:
            self.get_logger().info("Navigation action goal preempted!")
        elif result.result_code == NavigateToPose.Result.ROBOT_IN_VIRTUAL_WALL:
            self.get_logger().info("Navigation action: robot in the wall!")
        else:
            self.get_logger().info("invalid result_code: {}".format(result.result_code))
        self.get_logger().info(result.result_message)
    
    def move_forward(self, distance, speed=0.5):
        self.cancel()
        success = True
        info = ""
        target = DriveOnHeading.Goal()
        target.target.x = distance
        target.speed = speed
        target.time_allowance = Duration(seconds=5.0).to_msg()
        if self.movefoward_base_client.wait_for_server(timeout_sec=10.0):
            # result:DriveOnHeading.Result = self.movefoward_base_client.send_goal(target)
            self.controller_handler = self.movefoward_base_client.send_goal_async(target)
            while not self.controller_handler.done():
                self.get_logger().info("moving forward ...")
                # self.rate.sleep()
                time.sleep(1.0)
            
            success = True
            info = ""
        else:
            success = False
            info = "Cannot connected to navigation action server."
        self.controller_handler = None
        return success, info
    
    def turn_around(self, yaw):
        self.cancel()
        self._spin_done = False
        success = True
        info = ""
        target = Spin.Goal()
        target.target_yaw = yaw
        target.time_allowance = Duration(seconds=5.0).to_msg()
        if self.spin_base_client.wait_for_server(timeout_sec=10.0):
            self.controller_handler = self.spin_base_client.send_goal_async(target, feedback_callback=self._spin_feeback_cb)
            self.controller_handler.add_done_callback(self._spin_response_callback)
            while not self._spin_done:
                print(self.controller_handler.done())
                self.get_logger().info("spinning ...")
                time.sleep(1.0)
                
            
            # while not self.controller_handler.done():
            #     self.get_logger().info("spinning ...")
            #     # self.rate.sleep()
            #     time.sleep(1.0)
        else:
            success = False
            info = "Cannot connected to navigation action server."
        self.controller_handler = None
        return success, info
    
    def _spin_feeback_cb(self, future):
        feedback: Spin.Feedback = future.feedback
        self.get_logger().info("feedback: {}".format(feedback.angular_distance_traveled))
    
    def _spin_response_callback(self, future):
        spin_handle = future.result()
        if not spin_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = spin_handle.get_result_async()
        self._get_result_future.add_done_callback(self._get_result_callback)
    
    def _spin_result_callback(self, future):
        result: Spin.Result = future.result().result
        self.get_logger().info(result.total_elapsed_time)
        self._spin_done = True
        
    def cancel(self):
        if self.controller_handler is not None:
            self.controller_handler.cancel()
            if self.controller_handler.cancelled():
                self.get_logger().info("Navigation task is cancelled!")
            else:
                self.get_logger().info("Navigation task is not cancelled!")
        else:
            self.get_logger().info("Nothing to be cancelled!")
    

            


        