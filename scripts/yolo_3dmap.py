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
from scipy.spatial import ConvexHull

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
            
def depth_denoise(depth_path, kernel_size=11):
    orin_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_image = cv2.morphologyEx(orin_image, cv2.MORPH_OPEN, kernel)
    denoised_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
    return denoised_image

def fill_contours(contours, shape):
    img = np.zeros(shape, dtype=np.uint8)
    for contour in contours:
        contour_int = np.array(contour, dtype=np.int32)
        cv2.fillPoly(img, [contour_int], 1)
    return np.argwhere(img > 0)

def compute_center(contour):
    contour_np = np.array(contour)
    center_x = np.mean(contour_np[:, 0])
    center_y = np.mean(contour_np[:, 1])
    return [center_x, center_y]

def merge_contours(contours):
    all_points = np.vstack(contours)
    hull = ConvexHull(all_points)
    merged_contour = all_points[hull.vertices].tolist()
    return merged_contour

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
        self.merged_object_info_path = os.path.join(data_dir, "merged_object_info.json")
        
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
        self.class_names = self.model.names
        
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
        
    def build_pointcloud(self, voxel_size=0.05):
        combined_point_cloud = o3d.geometry.PointCloud()
        pbar = tqdm(total=len(self.rgb_list))
        for rgb_path, depth_path, pose in zip(self.rgb_list, self.depth_list, self.pose_list):
            try:
                rgb_image = cv2.imread(rgb_path)
                depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            except Exception as e:
                print("Failed to load source data")
                print("Error:", e)
                continue  
            try:
                point_cloud = self.depth_to_pointcloud(depth_image)
                transformed_point_cloud = self.transform_pointcloud(point_cloud, pose)
                colors = rgb_image.reshape(-1, 3) / 255.0  
                
                single_point_cloud = o3d.geometry.PointCloud()
                single_point_cloud.points = o3d.utility.Vector3dVector(transformed_point_cloud)
                single_point_cloud.colors = o3d.utility.Vector3dVector(colors)
                
                single_point_cloud = single_point_cloud.voxel_down_sample(voxel_size=voxel_size)
                combined_point_cloud += single_point_cloud
            except Exception as e:
                print("Failed to transform and append pointcloud")
                print("Error:", e)
                continue  
            pbar.update(1)
        pbar.close()
        o3d.io.write_point_cloud(self.rgb_pointcloud_save_path, combined_point_cloud)
    
    # TODO:can't show in wsl2 for openGL config, waiting for checking
    def visualize_ply(self, before_seg):
        if before_seg:
            pcd = o3d.io.read_point_cloud(self.rgb_pointcloud_save_path)
        else:
            pcd = o3d.io.read_point_cloud(self.seg_pointcloud_save_path)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        
    def build_ss_pointcloud(self, max_depth=6000, DBSCAN_eps=0.05, DBSCAN_min_samples=50, batch_size=100000):
        combined_point_cloud = o3d.geometry.PointCloud()

        pbar = tqdm(total=len(self.rgb_list))
        all_class_names = []
        all_transformed_points = []
        all_colors = []

        for depth_path, seg_dir, pose in zip(self.depth_list, self.seg_list, self.pose_list):
            label_path = os.path.join(seg_dir, "label.txt")
            if not os.path.exists(label_path):
                print(f"Label file not found for {label_path}, skipping.")
                pbar.update(1)
                continue

            try:
                # depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth_image = depth_denoise(depth_path)
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

                combined_point_cloud += single_point_cloud
                all_class_names.extend([class_name] * len(obj_points))
                all_transformed_points.append(obj_points)
                all_colors.append(obj_colors)

            pbar.update(1)
        pbar.close()

        all_transformed_points = np.vstack(all_transformed_points)
        all_colors = np.vstack(all_colors)
        num_points = all_transformed_points.shape[0]

        print("Start to cluster and label pointcloud")
        print(f"Total points: {num_points}")

        final_object_info = []
        filtered_point_cloud = o3d.geometry.PointCloud()
        start_time = time.time()

        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            batch_points = all_transformed_points[start:end]
            batch_colors = all_colors[start:end]
            batch_class_names = all_class_names[start:end]
            
            db = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples).fit(batch_points)
            labels = db.labels_

            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label == -1:
                    continue
                mask = (labels == label)
                cluster_points = batch_points[mask]
                cluster_colors = batch_colors[mask]
                class_name = max(set([batch_class_names[i] for i in range(len(batch_class_names)) if mask[i]]), key=[batch_class_names[i] for i in range(len(batch_class_names)) if mask[i]].count)

                cluster_2d_points = cluster_points[:, [0, 1]]  # 投影到 xy 平面
                hull = cv2.convexHull(cluster_2d_points.astype(np.float32))

                obj_info = {
                    "class_name": class_name,
                    "contour": hull[:, 0, :].tolist()  # 提取轮廓点
                }
                final_object_info.append(obj_info)

                single_point_cloud = o3d.geometry.PointCloud()
                single_point_cloud.points = o3d.utility.Vector3dVector(cluster_points)
                single_point_cloud.colors = o3d.utility.Vector3dVector(cluster_colors)
                filtered_point_cloud += single_point_cloud

        end_time = time.time()
        print(f"Time consumption for clustering and labeling: {end_time - start_time}")

        o3d.io.write_point_cloud(self.seg_pointcloud_save_path, filtered_point_cloud)
        with open(self.object_info_path, 'w') as f:
            json.dump(final_object_info, f, indent=4)
            
    def merge_object_info(self, eps=0.5, min_samples=2):
        try:
            with open(self.object_info_path, 'r') as f:
                object_info =  json.load(f)
        except Exception as e:
            print(f"Failed to build 3D semantic segment map: {e}")
            
        print("Start 2D clustering")
        start_time = time.time()
        class_contours = {}
        for obj in object_info:
            class_name = obj['class_name']
            contour = obj['contour']
            if class_name not in class_contours:
                class_contours[class_name] = []
            class_contours[class_name].append(contour)
        
        merged_objects = []
        for class_name, contours in class_contours.items():
            points = np.vstack(contours)
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            labels = clustering.labels_
            
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:
                    continue  # 忽略噪声点
                cluster_contours = [contours[i] for i in range(len(contours)) if labels[i] == label]
                if cluster_contours:  # 确保有轮廓可以合并
                    merged_contour = merge_contours(cluster_contours)
                    if merged_contour:  # 检查合并后的轮廓是否有效
                        center = compute_center(merged_contour)
                        merged_objects.append({
                            'class_name': class_name,
                            'center': center,
                            'contour': merged_contour
                        })
        end_time = time.time()
        with open(self.merged_object_info_path, 'w') as f:
            json.dump(merged_objects, f, indent=4)
        print(f"Merged object info saved to {self.merged_object_info_path}, time consumption: {end_time - start_time}")
            
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
    # ssmap.build_pointcloud(voxel_size=0.8, z_min=0.1, z_max=2)
    # ssmap.visualize_ply(before_seg=True)
    ssmap.build_ss_pointcloud(max_depth=5000, DBSCAN_eps=0.1, DBSCAN_min_samples=200, batch_size=100000)
    # ssmap.merge_object_info(eps=0.005, min_samples=2)