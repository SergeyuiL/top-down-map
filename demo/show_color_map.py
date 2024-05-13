import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.insert(0, '/home/sg/Workspace/vlmaps')

from utils.clip_mapping_utils import load_map, get_new_pallete, get_new_mask_pallete
from utils.clip_utils import get_text_feats
from utils.mp3dcat import mp3dcat

data_dir = "/home/sg/Workspace/catkin_ws/src/rosbag_play/data"

use_self_built_map = True # @param {type: "boolean"} 
map_save_dir = os.path.join(data_dir, "map_correct")
if use_self_built_map:
    map_save_dir = os.path.join(data_dir, "map")
os.makedirs(map_save_dir, exist_ok=True)

color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_1.npy")
# obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

# obstacles = load_map(obstacles_save_path)
# x_indices, y_indices = np.where(obstacles == 0)

# xmin = np.min(x_indices)
# xmax = np.max(x_indices)
# ymin = np.min(y_indices)
# ymax = np.max(y_indices)

# print(np.unique(obstacles))
# obstacles_pil = Image.fromarray(obstacles[xmin:xmax+1, ymin:ymax+1])
# plt.figure(figsize=(8, 6), dpi=120)
# plt.imshow(obstacles_pil, cmap='gray')
# plt.show()

color_top_down = load_map(color_top_down_save_path)
# color_top_down = color_top_down[xmin:xmax+1, ymin:ymax+1]
color_top_down_pil = Image.fromarray(color_top_down)
plt.figure(figsize=(8, 6), dpi=120)
plt.imshow(color_top_down_pil)
plt.show()
