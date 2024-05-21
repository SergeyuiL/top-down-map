import os
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from mapping_utils import load_map

parser = argparse.ArgumentParser(description="Show top down color map")
parser.add_argument('--map_save_dir', type=str, required=True, help='Directory to save the map.')
args = parser.parse_args()
map_save_dir = args.map_save_dir

color_top_down_save_path = os.path.join(map_save_dir, "color_top_down.npy")
seg_top_down_save_path = os.path.join(map_save_dir, "seg_top_down.npy")

color_top_down = load_map(color_top_down_save_path)
seg_top_down = load_map(seg_top_down_save_path)
color_top_down_pil = Image.fromarray(color_top_down)
seg_top_down_pil = Image.fromarray(seg_top_down)
plt.figure(figsize=(8, 6), dpi=120)
plt.imshow(color_top_down_pil)

plt.figure(figsize=(8, 6), dpi=120)
plt.imshow(seg_top_down_pil)
plt.show()
