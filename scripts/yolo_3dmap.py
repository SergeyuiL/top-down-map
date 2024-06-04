import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import cv2 

from ultralytics import YOLO

class mapbuilder:
    def __init__(self, depth_dir, rgb_dir, pose_path, camera_info_path, trans_camera2base, quat_camera2base):
        self.depth_dir = depth_dir
        self.rgb_dir = rgb_dir
        self.pose_path = pose_path
        self.camera_info_path = camera_info_path
        
        r = R.from_quat(quat_camera2base)
        rot_matrix_camera2base = r.as_matrix()
        self.T_base_camera = np.eye(4)
        self.T_base_camera[:3, :3] = rot_matrix_camera2base
        self.T_base_camera[:3, 3] = trans_camera2base

        self.camera_intrinsics = np.zeros((1, 4))
        self.camera_matrix = np.eye(3)
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        
        self.model = YOLO("yolov8x-seg.pt")
        
    def data_load(self):
        # get pose
        base_poses = np.genfromtxt(self.pose_path, delimiter=' ')[:, 1:]
        indices = np.arange(base_poses.shape[0])
        for index in indices:  
            x_base, y_base, yaw_base = base_poses[index]
            T_map_base = np.array([
                                    [np.cos(yaw_base), -np.sin(yaw_base), 0, x_base],
                                    [np.sin(yaw_base), np.cos(yaw_base),  0, y_base],
                                    [0,                0,                 1, 0],
                                    [0,                0,                 0, 1]
                                ])
            T_map_camera = np.dot(T_map_base, self.T_base_camera)
            self.pose_list.append(T_map_camera)
        # get rgb path
        self.rgb_list = [os.path.join(self.rgb_dir, f"rgb_{index}.png") for index in indices]
        # get depth path
        self.depth_list = [os.path.join(self.depth_dir, f"rdepth_{index}.png") for index in indices]
        # get camera info
        with open(self.camera_info_path, 'r') as file:
            camera_info = json.load(file)
        K = camera_info['K']
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        self.camera_intrinsics = np.array([fx, fy, cx, cy])
        self.camera_matrix = np.array([
                                [fx, 0,  cx],
                                [0,  fy, cy],
                                [0,  0,  1]
                            ])
        
    def seg_rgb(self, image):
        results = self.model.predict(source=image, save=True, save_txt=True)
        
if __name__ == "__main__":
    data_dir = "/home/sg/workspace/top-down-map/mapsave"
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    pose_path = os.path.join(data_dir, "node_pose.txt")
    ssmap = mapbuilder()
    
        
        
        
            
            
            
        