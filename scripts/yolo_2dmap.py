import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
from tqdm import tqdm
import shutil
import cv2 
import open3d as o3d

from ultralytics import YOLO
from ultralytics import settings

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

class mapbuilder:
    def __init__(self, data_dir, trans_camera_base, quat_camera_base):
        self.depth_dir = os.path.join(data_dir, "depth")
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.pose_path = os.path.join(data_dir, "node_pose.txt")
        self.camera_info_path = os.path.join(data_dir, "camera_info.json")
        self.seg_dir = os.path.join(data_dir, "segment")
        
        self.rgb_map_save_path = os.path.join(data_dir, "rgb_map.npy")
        self.seg_map_save_path = os.path.join(data_dir, "seg_map.npy")
        
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
        
        settings.update({"runs_dir": data_dir})
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
        
    def seg_rgb(self, confidence):
        clear_directory(self.seg_dir)
        default_jpg_path = os.path.join(self.seg_dir, "predict/image0.jpg")
        default_txt_path = os.path.join(self.seg_dir, "predict/labels/image0.txt")
        pbar = tqdm(total=len(self.rgb_list))
        for rgb_path in self.rgb_list:
            rgb_img = cv2.imread(rgb_path)
            self.model.predict(source=rgb_img, save=True, save_txt=True, conf=confidence)
            
            base_name = os.path.splitext(os.path.basename(rgb_path))[0]
            out_dir = os.path.join(self.seg_dir, base_name)
            os.makedirs(out_dir, exist_ok=True)
            if os.path.exists(default_jpg_path):
                shutil.move(default_jpg_path, os.path.join(out_dir, "rgb.jpg"))
            else:
                print(f"Warning: {default_jpg_path} does not exist.")
            if os.path.exists(default_txt_path):
                shutil.move(default_txt_path, os.path.join(out_dir, "label.txt"))
            else:
                print(f"Warning: {default_txt_path} does not exist.")
            
            pbar.update(1)
                
    def depth_to_pointcloud(self, depth_image, max_depth=6000):
        h, w = depth_image.shape
        fx, fy, cx, cy = self.camera_intrinsics
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        z = depth_image / 1000.0
        mask = z <= max_depth / 1000.0  
        z[~mask] = 0  
        x = (i - cx) * z / fx
        y = (j - cy) * z / fy
        pointcloud = np.stack((x, y, z), axis=-1)
        return pointcloud.reshape(-1, 3), mask.flatten()
    
    def transform_pointcloud(self, pointcloud, pose):
        R_mat = pose[:3, :3]
        t = pose[:3, 3]
        return (R_mat @ pointcloud.T).T + t
    
    def filter_pointcloud(self, pointcloud, z_min, z_max):
        mask = (pointcloud[:, 2] > z_min) & (pointcloud[:, 2] < z_max)
        return pointcloud[mask]
    
    def build_color_map(self, grid_len=2000, grid_size=0.05, z_min=0.1, z_max=2.0, max_depth=6000):
        
        for data_sample in data_iter:
            rgb_path, depth_path, tf, seg_path = data_sample
            try:   
                bgr = cv2.imread(rgb_path)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Failed to load image at {rgb_path}, {e}")
                continue  # Skip this iteration
            # read pose
            try:
                pose = np.array(tf)
            except Exception as e:
                print(f"Failed to load file at {pose_path}, {e}")
                print("Error:", e)
                continue  
            if np.linalg.det(pose) == 0.:
                print(f"Singular matrix {pose_path}")
                continue
            tf_list.append(pose)
            if len(tf_list) == 1:
                init_tf_inv = np.linalg.inv(tf_list[0])
            tf = init_tf_inv @ pose
            # read depth
            try:   
                depth = load_depth(depth_path)
            except Exception as e:
                print(f"Failed to load depth at {depth_path}, {e}")
                continue  # Skip this iteration
            # read seg 
            try:   
                seg_bgr = cv2.imread(seg_path)
                seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Failed to load segmented image at {seg_path}, {e}")
                continue  # Skip this iteration
        
        # project all point cloud onto the ground
        pc, mask = depth2pc(depth, camera_intrinsics, depth_thresh)
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        pc_global = transform_pc(pc, tf)

        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            x, y = pos2grid_id(gs, cs, p[0], p[2])

            rgb_px, rgb_py, rgb_pz = project_point(cam_mat, p_local)
            rgb_v = rgb[rgb_py, rgb_px, :]
            seg_v = seg_rgb[rgb_py, rgb_px, :]

            if p_local[1] < color_top_down_height[y, x]:
                color_top_down[y, x] = rgb_v
                color_top_down_height[y, x] = p_local[1]
            
            if np.any(seg_v > 0) and p_local[1] < seg_top_down_height[y, x]:
                seg_top_down[y, x] = seg_v
                seg_top_down_height[y, x] = p_local[1]

            if p_local[1] > camera_height:
                continue

        pbar.update(1)
        

        

    def save_map(self, color_top_down):
        np.save(self.rgb_map_save_path, color_top_down)
        cv2.imwrite(self.rgb_map_save_path.replace('.npy', '.png'), color_top_down)

# Example usage
if __name__ == "__main__":
    data_dir = "/home/sg/workspace/top-down-map/mapsave"
    
    trans_camera_base = [0.08278859935292791, -0.03032243564439939, 1.0932014910932797]
    quat_camera_base = [-0.48836894018639176, 0.48413701319615116, -0.5135400532533373, 0.5132092598729002]
    
    ssmap = mapbuilder(data_dir, trans_camera_base, quat_camera_base)
    ssmap.data_load()
    # ssmap.seg_rgb(confidence=0.6)
    color_top_down = ssmap.build_color_map(z_min=0.1, z_max=2.0, max_depth=6000)
    ssmap.save_map(color_top_down)
    cv2.imshow('Color Top Down', color_top_down)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
