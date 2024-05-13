import os
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/sg/Workspace/vlmaps')

from utils.clip_mapping_utils import *

def my_load_depth(depth_filepath, thresh=2.0):
    depth = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Unable to load depth image from {depth_filepath}")
    depth = depth.astype(np.float32) / 1000.0  
    depth[depth > thresh] = 0.0
    return depth

def my_depth2pc(depth, fx, fy, cx, cy, thresh=2.0):
    """
    Convert a depth map into a 3D point cloud

    Parameters:
    - depth (2D numpy array): The depth map where each value is the Z-coordinate in camera space.
    - fx, fy (float): Focal lengths of the camera in pixels.
    - cx, cy (float): Optical center of the camera in pixels.

    Returns:
    - pc (3xN numpy array): The resulting point cloud where each column represents (X, Y, Z) coordinates.
    - mask (1D numpy array): A mask indicating which points are valid based on a depth threshold.
    """
    h, w = depth.shape
    i, j = np.indices((h, w))
    # Normalize pixel coordinates
    x = (j - cx) / fx
    y = (i - cy) / fy
    # Create point cloud
    valid = (depth > 0.1) & (depth <= thresh)  # Only consider depth values within 0.1 to 3.0 meters
    x = x[valid] * depth[valid]  # X = x * Z
    y = y[valid] * depth[valid]  # Y = y * Z
    z = depth[valid]
    # Stack results to a 3xN matrix
    pc = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    # Mask for valid depth entries
    mask = valid.flatten()  # Update this line to use the 'valid' array directly

    return pc, mask

def my_get_camera_matrix(fx, fy, cx, cy):
    """
    Constructs a camera matrix from intrinsic parameters.
    """
    cam_mat = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    return cam_mat

def my_project_point(cam_mat, p):
    """
    Projects a 3D point onto the 2D camera plane using the camera matrix.

    Parameters:
    - cam_mat (numpy array): The 3x3 camera intrinsic matrix.
    - p (numpy array): The 3D point in camera coordinates.

    Returns:
    - x, y (int): The pixel coordinates in the image.
    - z (float): The depth of the point.
    """
    new_p = cam_mat @ p.reshape((3, 1))  # Project the point
    z = new_p[2, 0]                      # Depth of the point
    if z != 0:
        new_p = new_p / z                # Normalize by depth to get image coordinates
    x = int(new_p[0, 0] + 0.5)           # Rounding to nearest integer
    y = int(new_p[1, 0] + 0.5)
    return x, y, z


if __name__ == "__main__":
    data_save_dir = "/home/sg/Workspace/catkin_ws/src/rosbag_play/data"

    mask_version = 1
    depth_sample_rate = 100

    fx = 606.9916381835938
    fy = 605.5125732421875
    cx = 317.2601623535156
    cy = 241.97413635253906
    
    cs = 0.01
    gs = 1200
    
    depth_thresh = 3.0
    camera_height = 0.52725
    
    print(f"loading scene {data_save_dir}")
    rgb_dir = os.path.join(data_save_dir, "color")
    depth_dir = os.path.join(data_save_dir, "depth")
    pose_dir = os.path.join(data_save_dir, "pose")
    
    start_index = 1
    end_index = 2363
    list_index = np.arange(start_index, end_index)
    
    rgb_list = [os.path.join(rgb_dir, f"{index}.png") for index in list_index]
    depth_list = [os.path.join(depth_dir, f"{index}.png") for index in list_index]
    pose_list = [os.path.join(pose_dir, f"{index}.txt") for index in list_index]

    map_save_dir = os.path.join(data_save_dir, "map")
    os.makedirs(map_save_dir, exist_ok=True)
    color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_{mask_version}.npy")
    gt_save_path = os.path.join(map_save_dir, f"grid_{mask_version}_gt.npy")
    weight_save_path = os.path.join(map_save_dir, f"weight_lseg_{mask_version}.npy")
    obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")
    
    # initialize a grid with zero position at the center
    color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)
    color_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)
    gt = np.zeros((gs, gs), dtype=np.int32)
    obstacles = np.ones((gs, gs), dtype=np.uint8)
    weight = np.zeros((gs, gs), dtype=float)

    save_map(color_top_down_save_path, color_top_down)
    save_map(gt_save_path, gt)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)

    tf_list = []
    data_iter = zip(rgb_list, depth_list, pose_list)
    pbar = tqdm(total=len(rgb_list))
    
    rgb_cam_mat = my_get_camera_matrix(fx, fy, cx, cy)
    
    # load all images and depths and poses
    for data_sample in data_iter:
        rgb_path, depth_path, pose_path = data_sample
        st = time.time()

        try:   
            bgr = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Failed to load image at {rgb_path}")
            continue  # Skip this iteration

        # read pose
        try:
            pose = np.loadtxt(pose_path)
        except Exception as e:
            print(f"Failed to load file at {pose_path}")
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
            depth = my_load_depth(depth_path, depth_thresh)
        except Exception as e:
            print(f"Failed to load depth at {depth_path}")
            continue  # Skip this iteration
        
        # project all point cloud onto the ground
        pc, mask = my_depth2pc(depth, fx, fy, cx, cy, depth_thresh)
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        # pc, mask = my_depth2pc(depth, fx, fy, cx, cy)
        # valid_indices = np.where(mask)[0]  
        # shuffle_mask = np.random.permutation(valid_indices)  
        # shuffle_mask = shuffle_mask[::depth_sample_rate]  
        # pc = pc[:, shuffle_mask]
        pc_global = transform_pc(pc, tf)

        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            x, y = pos2grid_id(gs, cs, p[0], p[2])
            if x >= obstacles.shape[0] or y >= obstacles.shape[1] or x < 0 or y < 0 or p_local[1] < -0.5:
                continue

            rgb_px, rgb_py, rgb_pz = my_project_point(rgb_cam_mat, p_local)
            rgb_v = rgb[rgb_py, rgb_px, :]

            if p_local[1] < color_top_down_height[y, x]:
                color_top_down[y, x] = rgb_v
                color_top_down_height[y, x] = p_local[1]

            if p_local[1] > camera_height:
                continue
            obstacles[y, x] = 0

        et = time.time()
        pbar.update(1)

    save_map(color_top_down_save_path, color_top_down)
    save_map(gt_save_path, gt)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)
