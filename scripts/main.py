import numpy as np
import os 
from scipy.spatial.transform import Rotation as R
import json
from tqdm import tqdm




class mapbuilder:
    def __init__(self, data_dir, trans_camera_base, quat_camera_base):
        self.depth_dir = os.path.join(data_dir, "depth")
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.pose_path = os.path.join(data_dir, "node_pose.txt")
        self.camera_info_path = os.path.join(data_dir, "camera_info.json")
        self.seg_dir = os.path.join(data_dir, "segment")
        
        self.rgb_pointcloud_save_path = os.path.join(data_dir, "rgb_pointcloud.ply")
        self.seg_pointcloud_save_path = os.path.join(data_dir, "seg_pointcloud.ply")
        self.object_info_path = os.path.join(data_dir, "object_info.json")
        
        r = R.from_quat(quat_camera_base)
        rot_matrix_camera2base = r.as_matrix()
        self.T_base_camera = np.eye(4)
        self.T_base_camera[:3, :3] = rot_matrix_camera2base
        self.T_base_camera[:3, 3] = trans_camera_base

        self.camera_intrinsics = np.zeros((1, 4))
        self.camera_matrix = np.eye(3)
        self.rgb_list = []
        self.depth_list = []
        self.seg_list = []
        self.pose_list = []
        
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
        self.depth_list = [os.path.join(self.depth_dir, f"depth_{index}.png") for index in indices]
        # get segment path
        self.seg_list = [os.path.join(self.seg_dir, f"rgb_{index}") for index in indices]
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
        
        
        
if __name__ == "__main__":   
    data_dir = "/home/sg/workspace/top-down-map/map06132"
    
    trans_camera_base = [0.08278859935292791, -0.03032243564439939, 1.0932014910932797]
    quat_camera_base = [-0.48836894018639176, 0.48413701319615116, -0.5135400532533373, 0.5132092598729002]
    
    ssmap = mapbuilder(data_dir=data_dir, trans_camera_base=trans_camera_base, quat_camera_base=quat_camera_base)
    ssmap.data_load()
    