import cv2 

from ultralytics import YOLO

model = YOLO("yolov8x-seg.pt")

# rgb_path = "/home/sg/workspace/top-down-map/mapsave/rgb/rgb_356.png"
# image = cv2.imread(rgb_path)
# results = model.predict(source=image, save=True, save_txt=True)
# results = model.predict(source=image, save=True)
class_names = model.names
print(class_names)