import matplotlib.pyplot as plt
import json
import numpy as np

# 读取 object_info.json 文件
object_info_path = "/home/sg/workspace/top-down-map/mapsave/object_info.json"
with open(object_info_path, 'r') as f:
    object_info = json.load(f)

# 创建一个新的绘图窗口
plt.figure(figsize=(10, 10))

# 绘制每个对象的轮廓并标注类别
for obj in object_info:
    class_name = obj['class_name']
    contour = obj['contour']
    contour = np.array(contour)
    
    # 绘制轮廓
    plt.plot(contour[:, 0], contour[:, 1], label=class_name)
    
    # 标注类别
    centroid = np.mean(contour, axis=0)
    plt.text(centroid[0], centroid[1], class_name, fontsize=12, ha='center')

# 设置绘图参数
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Contours with Class Labels')
plt.legend()
plt.grid(True)
plt.axis('equal')  # 保持x和y轴比例一致

# 显示绘图
plt.show()