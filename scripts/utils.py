import numpy as np
import cv2
import open3d as o3d


            
def depth_denoise(depth_path, kernel_size=11):
    orin_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_image = cv2.morphologyEx(orin_image, cv2.MORPH_OPEN, kernel)
    denoised_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
    return denoised_image

def get_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# pose: 4*4 T
def show_pc(points, colors, poses=None):
    pcd = get_pcd(points, colors)
    if poses is not None:
        cameras_list = []
        for pose in poses:
            camera_cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            cameras_list.append(camera_cf.transform(np.linalg.inv(pose)))
        o3d.visualization.draw_geometries([pcd, *cameras_list])
    else:
        o3d.visualization.draw_geometries([pcd])
        
def save_pc(points, colors, save_path):
    pcd = get_pcd(points, colors)
    o3d.io.write_point_cloud(save_path, pcd)
    
