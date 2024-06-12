import cv2
import numpy as np

depth_image = cv2.imread('/home/sg/workspace/top-down-map/data_rosbag/depth/depth_189.png', cv2.IMREAD_UNCHANGED)

kernel = np.ones((9, 9), np.uint8)

opened_image = cv2.morphologyEx(depth_image, cv2.MORPH_OPEN, kernel)

denoised_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original Depth Image', depth_image)
cv2.imshow('Denoised Depth Image', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
