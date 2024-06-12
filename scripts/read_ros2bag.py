import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
import message_filters
import tf2_ros
import transformations as tf
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import os
import shutil
import numpy as np

def clear_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

class BagProcessor(Node):
    def __init__(self, data_path):
        super().__init__('bag_processor')
        
        clear_directory(data_path)
        
        self.pose_path = os.path.join(data_path, "node_pose.txt")
        self.rgb_dir = os.path.join(data_path, "rgb")
        self.depth_dir = os.path.join(data_path, "depth")
        
        ensure_directory_exists(self.rgb_dir)
        ensure_directory_exists(self.depth_dir)
        
        self.rgb_sub = message_filters.Subscriber(self, Image, '/realsense1/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/realsense1/aligned_depth_to_color/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 30, 0.1)
        self.ts.registerCallback(self.callback)
        
        self.bridge = CvBridge()
        self.index = 0
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.log_file = open(self.pose_path, 'w')

        self.last_rgb_image = None
        self.last_depth_image = None
        self.last_pose = None
        self.previous_pose = None  # 用于保存上一次保存时的位置

    def callback(self, rgb_msg, depth_msg):
        self.last_rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        self.last_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            position = trans.transform.translation
            orientation = trans.transform.rotation
            euler = tf.euler_from_quaternion([
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w])
            self.last_pose = (position.x, position.y, euler[2])

            if self.previous_pose is None or self.should_save_data(self.previous_pose, self.last_pose):
                self.save_data()
                self.previous_pose = self.last_pose

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            self.get_logger().warn('Transform not available or extrapolation error')

    def should_save_data(self, prev_pose, curr_pose):
        distance_threshold = 0.05  # 5 cm
        angle_threshold = np.radians(5)  # 5 degrees

        prev_x, prev_y, prev_yaw = prev_pose
        curr_x, curr_y, curr_yaw = curr_pose

        distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
        angle_diff = np.abs(curr_yaw - prev_yaw)

        return distance >= distance_threshold or angle_diff >= angle_threshold

    def save_data(self):
        cv2.imwrite(os.path.join(self.rgb_dir, f'rgb_{self.index}.png'), self.last_rgb_image)
        cv2.imwrite(os.path.join(self.depth_dir, f'depth_{self.index}.png'), self.last_depth_image)
        
        x, y, yaw = self.last_pose
        self.log_file.write(f"{self.index} {x} {y} {yaw}\n")
        
        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    processor = BagProcessor("/home/sg/workspace/top-down-map/data_rosbag")
    
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.log_file.close()
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
