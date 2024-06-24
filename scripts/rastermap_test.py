import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

pgm_path = "/home/sg/workspace/top-down-map/map06132/map06132.pgm"
yaml_path = "/home/sg/workspace/top-down-map/map06132/map06132.yaml"

# 读取yaml文件
with open(yaml_path, 'r') as file:
    map_metadata = yaml.safe_load(file)

# 打印元数据
print(map_metadata)

# 加载图像
img = Image.open(pgm_path)

# 读取元数据中的信息
resolution = map_metadata['resolution']
origin = map_metadata['origin']
negate = map_metadata['negate']
occupied_thresh = map_metadata['occupied_thresh']
free_thresh = map_metadata['free_thresh']
angle = map_metadata['angle']
width = map_metadata['width']
height = map_metadata['height']
x = map_metadata['x']
y = map_metadata['y']

# 显示图像
plt.imshow(img, cmap='gray')
plt.title('Map Image')
plt.axis('off')
plt.show()

# 示例：根据元数据计算一些信息
print(f"Resolution: {resolution}")
print(f"Origin: {origin}")
print(f"Width: {width}, Height: {height}")
