import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
from tqdm import tqdm
import shutil
import cv2 
import open3d as o3d
import time
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

from ultralytics import YOLO
from ultralytics import settings
            
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

        # self.depth_denoise_kernel_size = depth_denoise_kernel_size
        self.camera_intrinsics = np.zeros((1, 4))
        self.camera_matrix = np.eye(3)
        self.rgb_list = []
        self.depth_list = []
        self.seg_list = []
        self.pose_list = []
        
        settings.update({"runs_dir": data_dir})
        self.model = YOLO("yolov8x-seg.pt")
        self.class_names = self.model.names
        
    def clear_seg_dir(self):
        if not os.path.exists(self.seg_dir):
            print(f"The directory {self.seg_dir} does not exist.")
            return
        
        for filename in os.listdir(self.seg_dir):
            file_path = os.path.join(self.seg_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        
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
        
    def depth_denoise(self, depth_path, kernel_size=7):
        orin_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened_image = cv2.morphologyEx(orin_image, cv2.MORPH_OPEN, kernel)
        denoised_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
        return denoised_image

    def seg_rgb(self, confidence):
        self.clear_seg_dir()
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
                
    def depth_to_pointcloud(self, depth_image):
        h, w = depth_image.shape
        fx, fy, cx, cy = self.camera_intrinsics
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        z = depth_image / 1000.0  
        x = (i - cx) * z / fx
        y = (j - cy) * z / fy
        return np.stack((x, y, z), axis=-1).reshape(-1, 3)
    
    def transform_pointcloud(self, pointcloud, pose):
        R_mat = pose[:3, :3]
        t = pose[:3, 3]
        return (R_mat @ pointcloud.T).T + t
        
    def build_2dmap(self, max_depth=6000, DBSCAN_eps=0.05, DBSCAN_min_samples=50, voxel_size=0.1, kernel_size=7):
        class_point_clouds = {}
        all_class_ids = []
        
        print("Start to merge 2D contours")
        pbar = tqdm(total=len(self.rgb_list))
        for depth_path, seg_dir, pose in zip(self.depth_list, self.seg_list, self.pose_list):
            label_path = os.path.join(seg_dir, "label.txt")
            if not os.path.exists(label_path):
                print(f"Label file not found for {label_path}, skipping.")
                pbar.update(1)
                continue
            try:
                # depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth_image = self.depth_denoise(depth_path, kernel_size)
                seg_image = cv2.imread(os.path.join(seg_dir, "rgb.jpg"))
            except Exception as e:
                print(f"Failed to load source data: {e}")
                pbar.update(1)
                continue  
            with open(label_path, 'r') as f:
                labels = f.readlines()

            point_cloud = self.depth_to_pointcloud(depth_image)
            transformed_point_cloud = self.transform_pointcloud(point_cloud, pose)
            depth_mask = (depth_image > 0) & (depth_image < max_depth)

            for label in labels:
                label_data = list(map(float, label.split()))
                class_id = int(label_data[0])
                points = np.array(label_data[1:]).reshape(-1, 2)

                class_name = self.class_names[class_id]
                if class_name not in class_point_clouds:
                    class_point_clouds[class_name] = o3d.geometry.PointCloud()
                points[:, 0] *= seg_image.shape[1]
                points[:, 1] *= seg_image.shape[0]
                points = points.astype(int)
                seg_mask = np.zeros(seg_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(seg_mask, [points], 1)
                seg_mask = seg_mask.flatten()
                valid_indices = depth_mask.flatten() & (seg_mask > 0)
                obj_points = transformed_point_cloud[valid_indices]
                obj_colors = seg_image.reshape(-1, 3)[valid_indices] / 255.0
                single_point_cloud = o3d.geometry.PointCloud()
                single_point_cloud.points = o3d.utility.Vector3dVector(obj_points)
                single_point_cloud.colors = o3d.utility.Vector3dVector(obj_colors)
                # Apply voxel downsampling
                single_point_cloud = single_point_cloud.voxel_down_sample(voxel_size)
                
                class_point_clouds[class_name] += single_point_cloud
                all_class_ids.extend([class_id] * len(obj_points))
            pbar.update(1)
        pbar.close()
        print("Start to cluster 2D points")
        cluster_2d_start_time = time.time()
        final_object_info = {}
        for class_name, point_cloud in class_point_clouds.items():
            points = np.asarray(point_cloud.points)
            if len(points) == 0:
                continue
            # Project points to 2D plane
            points_2d = points[:, [0, 1]]
            # Perform DBSCAN clustering
            db = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples).fit(points_2d)
            labels = db.labels_
            unique_labels = np.unique(labels)
            contours = []
            for label in unique_labels:
                if label == -1:
                    continue
                mask = (labels == label)
                cluster_points = points_2d[mask]
                hull = cv2.convexHull(cluster_points.astype(np.float32))
                contours.append(hull[:, 0, :].tolist())
            final_object_info[class_name] = contours
        cluster_2d_end_time = time.time()
        with open(self.object_info_path, 'w') as f:
            json.dump(final_object_info, f, indent=4)
        print(f"object contours info saved to {self.object_info_path}, time consumption: {cluster_2d_end_time - cluster_2d_start_time}")
        
    def merge_pointcloud(self, max_depth=6000, DBSCAN_eps=0.05, DBSCAN_min_samples=50, kernel_size=7, batch_size=100000):
        combined_point_cloud = o3d.geometry.PointCloud()

        pbar = tqdm(total=len(self.rgb_list))
        all_class_names = []
        all_transformed_points = []
        all_colors = []

        for depth_path, rgb_path, seg_dir, pose in zip(self.depth_list, self.rgb_list, self.seg_list, self.pose_list):
            label_path = os.path.join(seg_dir, "label.txt")
            if not os.path.exists(label_path):
                print(f"Label file not found for {label_path}, skipping.")
                pbar.update(1)
                continue
            try:
                # depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth_image = self.depth_denoise(depth_path, kernel_size)
                rgb_image = cv2.imread(rgb_path)
                seg_image = cv2.imread(os.path.join(seg_dir, "rgb.jpg"))
            except Exception as e:
                print(f"Failed to load source data: {e}")
                pbar.update(1)
                continue  

            point_cloud = self.depth_to_pointcloud(depth_image)
            transformed_point_cloud = self.transform_pointcloud(point_cloud, pose)
            depth_mask = (depth_image > 0) & (depth_image < max_depth)
            
            
            pbar.update(1)
        pbar.close()

        
        end_time = time.time()
        print(f"Time consumption for clustering and labeling: {end_time - start_time}")

        o3d.io.write_point_cloud(self.seg_pointcloud_save_path, seg_point_cloud)
        o3d.io.write_point_cloud(self.rgb_pointcloud_save_path, rgb_point_cloud)
        
    def show_contours(self):
        with open(self.object_info_path, 'r') as f:
            object_info = json.load(f)
            fig, ax = plt.subplots()
            for class_name, contours in object_info.items():
                for contour in contours:
                    contour = np.array(contour)
                    ax.plot(contour[:, 0], contour[:, 1])
                    
                    center_x = np.mean(contour[:, 0])
                    center_y = np.mean(contour[:, 1])
                    
                    ax.text(center_x, center_y, class_name, fontsize=12, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        ax.set_aspect('equal', adjustable='box')
        plt.title('2D Contours')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
    def show_pointcloud(self, seg=True):
        if seg:
            point_cloud = o3d.io.read_point_cloud(self.seg_pointcloud_save_path)
        else:
            point_cloud = o3d.io.read_point_cloud(self.rgb_pointcloud_save_path)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(point_cloud)
        vis.run()
        vis.destroy_window()
            
if __name__ == "__main__":
    data_dir = "/home/sg/workspace/top-down-map/map06132"
    
    trans_camera_base = [0.08278859935292791, -0.03032243564439939, 1.0932014910932797]
    quat_camera_base = [-0.48836894018639176, 0.48413701319615116, -0.5135400532533373, 0.5132092598729002]
    # trans_camera_footprint = [0.08278859935292791, -0.03032243564439939, 1.3482014910932798]
    # quat_camera_footprint = [-0.48836894018639176, 0.48413701319615116, -0.5135400532533373, 0.5132092598729002]
    
    ssmap = mapbuilder(data_dir, trans_camera_base, quat_camera_base)
    # ssmap = mapbuilder(data_dir, trans_camera_footprint, quat_camera_footprint)
    ssmap.data_load()
    # ssmap.seg_rgb(confidence=0.6)
    ssmap.build_2dmap(max_depth=5000, DBSCAN_eps=0.1, DBSCAN_min_samples=200, voxel_size=0.05)
    ssmap.show_contours()