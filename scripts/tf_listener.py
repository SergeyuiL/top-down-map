#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException
import time

class TfListener(Node):
    def __init__(self):
        super().__init__('tf_listener')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.on_timer)

    def on_timer(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('base_footprint', 'realsense1_color_optical_frame', now)
            self.get_logger().info(f"Transform: {trans.transform.translation.x}, {trans.transform.translation.y}, {trans.transform.translation.z}")
            self.get_logger().info(f"Rotation: {trans.transform.rotation.x}, {trans.transform.rotation.y}, {trans.transform.rotation.z}, {trans.transform.rotation.w}")
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.get_logger().info('Transform not available')

def main(args=None):
    rclpy.init(args=args)
    node = TfListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
