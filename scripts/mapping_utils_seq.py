import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json
import argparse
from scipy.spatial.transform import Rotation as R

def get_indices(data_dir):
    files = os.listdir(data_dir)
    indices = {int(file.split('.')[0]) for file in files if file.split('.')[0].isdigit()}
    return indices

def load_camera_intrinsics(data_dir):
    filename = os.path.join(data_dir, 'camera_info.json')
    with open(filename, 'r') as file:
        camera_info = json.load(file)
    K = camera_info['K']
    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]
    camera_intrinsics = np.array([fx, fy, cx, cy])
    return camera_intrinsics

def get_camera_matrix(camera_intrinsics):
    fx = camera_intrinsics[0]
    fy = camera_intrinsics[1]
    cx = camera_intrinsics[2]
    cy = camera_intrinsics[3]
    cam_mat = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    return cam_mat

def load_depth(depth_filepath):
    depth = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Unable to load depth image from {depth_filepath}")
    depth = depth.astype(np.float32) / 1000.0  
    return depth

def depth2pc(depth, camera_intrinsics, thresh=2.0):
    fx = camera_intrinsics[0]
    fy = camera_intrinsics[1]
    cx = camera_intrinsics[2]
    cy = camera_intrinsics[3]
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

def project_point(cam_mat, p):
    new_p = cam_mat @ p.reshape((3, 1))  # Project the point
    z = new_p[2, 0]                      # Depth of the point
    if z != 0:
        new_p = new_p / z                # Normalize by depth to get image coordinates
    x = int(new_p[0, 0] + 0.5)           # Rounding to nearest integer
    y = int(new_p[1, 0] + 0.5)
    return x, y, z

def save_map(save_path, map):
    with open(save_path, "wb") as f:
        np.save(f, map)
        print(f"{save_path} is saved.")
        
def transform_pc(pc, pose):
    pc_homo = np.vstack([pc, np.ones((1, pc.shape[1]))])
    pc_global_homo = pose @ pc_homo
    return pc_global_homo[:3, :]

def pos2grid_id(gs, cs, xx, yy):
    x = int(gs / 2 + int(xx / cs))
    y = int(gs / 2 - int(yy / cs))
    return [x, y]

def load_map(load_path):
    with open(load_path, "rb") as f:
        map = np.load(f)
    return map


if __name__=="__main__":
    depth_sample_rate = 100
    camera_height = 0.52725
    cs = 0.01
    gs = 3000
    depth_thresh = 10.0
    
    trans_camera_base = [0.08278859935292791, -0.03032243564439939, 1.0932014910932797]
    quat_camera_base = [-0.48836894018639176, 0.48413701319615116, -0.5135400532533373, 0.5132092598729002]

    r = R.from_quat(quat_camera_base)
    rot_matrix_camera_base = r.as_matrix()

    T_base_camera = np.eye(4)
    T_base_camera[:3, :3] = rot_matrix_camera_base
    T_base_camera[:3, 3] = trans_camera_base
    
    parser = argparse.ArgumentParser(description="Process data and save top down color map")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory to the raw data.')
    parser.add_argument('--map_save_dir', type=str, required=True, help='Directory to save the map.')
    args = parser.parse_args()
    data_dir = args.data_dir
    map_save_dir = args.map_save_dir
    
    # data_dir = "/home/sg/workspace/top-down-map/data"
    # map_save_dir = "/home/sg/workspace/top-down-map/maps"
    
    camera_intrinsics = load_camera_intrinsics(data_dir)
    cam_mat = get_camera_matrix(camera_intrinsics)

    print(f"loading scene {data_dir}")
    rgb_dir = os.path.join(data_dir, "color")
    depth_dir = os.path.join(data_dir, "depth")
    seg_dir = os.path.join(data_dir, "semantic_proc")
    
    pose_path = os.path.join(data_dir, "node_pose.txt")
    base_poses = data = np.genfromtxt(pose_path, delimiter=' ')[:, 1:]
    
    indices = np.arange(base_poses.shape[0])
    pose_list = []
    for index in indices:
        x_base, y_base, yaw_base = base_poses[index]
        T_map_base = np.array([
                                [np.cos(yaw_base), -np.sin(yaw_base), 0, x_base],
                                [np.sin(yaw_base), np.cos(yaw_base),  0, y_base],
                                [0,                0,                 1, 0],
                                [0,                0,                 0, 1]
                            ])
        T_map_camera = np.dot(T_map_base, T_base_camera)
        pose_list.append(T_map_camera)

    rgb_list = [os.path.join(rgb_dir, f"rgb_{index}.png") for index in indices]
    depth_list = [os.path.join(depth_dir, f"depth_{index}.png") for index in indices]
    seg_list = [os.path.join(seg_dir, f"{index}.png") for index in indices]

    os.makedirs(map_save_dir, exist_ok=True)
    color_top_down_save_path = os.path.join(map_save_dir, "color_top_down.npy")
    seg_top_down_save_path = os.path.join(map_save_dir, "seg_top_down.npy")
    
    # initialize a grid with zero position at the center
    color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)
    seg_top_down_height = color_top_down_height
    color_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)
    seg_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)

    save_map(color_top_down_save_path, color_top_down)
    save_map(seg_top_down_save_path, seg_top_down)

    data_iter = zip(rgb_list, depth_list, pose_list, seg_list)
    pbar = tqdm(total=len(rgb_list))
    
    tf_list = []
    # load all images and depths and poses
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

    save_map(color_top_down_save_path, color_top_down)
    save_map(seg_top_down_save_path, seg_top_down)