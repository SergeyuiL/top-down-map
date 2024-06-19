import json
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

def visualize_3d_pointcloud_with_contours(pointcloud_path, json_path):
    # 读取点云文件
    point_cloud = o3d.io.read_point_cloud(pointcloud_path)
    
    # 创建一个Open3D可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    
    # 读取并绘制2D轮廓
    with open(json_path, 'r') as f:
        object_info = json.load(f)

    fig, ax = plt.subplots()
    
    for class_name, contours in object_info.items():
        for contour in contours:
            contour = np.array(contour)
            ax.plot(contour[:, 0], contour[:, 1])
            
            # 计算轮廓的中心点
            center_x = np.mean(contour[:, 0])
            center_y = np.mean(contour[:, 1])
            
            # 将类别名称放置在中心点
            ax.text(center_x, center_y, class_name, fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    ax.set_aspect('equal', adjustable='box')
    plt.title('2D Contours')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # 可视化3D点云
    vis.run()
    vis.destroy_window()

# 调用函数，传入点云文件路径和JSON文件路径
visualize_3d_pointcloud_with_contours('/home/sg/workspace/top-down-map/map06132/seg_pointcloud.ply', '/home/sg/workspace/top-down-map/map06132/object_info.json')
