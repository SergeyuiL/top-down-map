import json
import numpy as np
import os
import random
import cv2
from tqdm import tqdm
import argparse
import re
from pycocotools.mask import decode as coco_decode

def numerical_sort(file):
    numbers = re.findall(r'\d+', file)
    return int(numbers[0]) if numbers else 0

if __name__=="__main__":
    # expected_classes = ["wall", "floor", "box", "curtain", "pillow", "bed", "cabinet", "bowl", "cup", "shelf", "box"]
    expected_classes = ["wall", "floor", "shelf", "bench", "chair", "door", "table", "computer", "cabinet", "sofa"]
              
    parser = argparse.ArgumentParser(description="Filter chosen classes and build top down seg map")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory to the raw data.')
    args = parser.parse_args()
    data_dir = args.data_dir
    
    json_dir = os.path.join(data_dir, "semantic")
    rgb_dir = os.path.join(data_dir, "color")
    seg_proc_dir = os.path.join(data_dir, "semantic_proc")
    os.makedirs(seg_proc_dir, exist_ok=True)
    print(f"loading scene {data_dir}")

    class_colors = {}
    files = [file for file in os.listdir(json_dir) if file.endswith('.json')]
    files_sorted = sorted(files, key=numerical_sort)
    for filename in tqdm(files_sorted, desc="Processing images"):
        if filename.endswith(".json"):
            json_path = os.path.join(json_dir, filename)
            idx = filename.split('_')[1]
            rgb_path = os.path.join(rgb_dir, f"rgb_{idx}.png")
            # read seg json
            with open(json_path, 'r') as file:
                seg_data = json.load(file)
            # read color image
            try:   
                bgr = cv2.imread(rgb_path)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Failed to load image at {rgb_path}, {e}")
                continue  # Skip this iteration

            updated_rgb = np.array(rgb)
            overall_mask = np.zeros(updated_rgb.shape[:2], dtype=bool)
            for item in seg_data['annotations']:
                class_name = item['class_name']
                if class_name in expected_classes:
                    if class_name not in class_colors:
                        class_colors[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    color = class_colors[class_name] + (128,) 
                    mask = coco_decode(item['segmentation'])
                    updated_rgb[mask == 1] = color[:3]  # Apply color where mask is true
                    overall_mask[mask == 1] = True
            updated_rgb[~overall_mask] = 0
            seg_proc_path = os.path.join(seg_proc_dir, f"{idx}.png")
            cv2.imwrite(seg_proc_path, cv2.cvtColor(updated_rgb, cv2.COLOR_RGB2BGR))

