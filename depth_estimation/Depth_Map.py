import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap_right = '/home/ellentuane/Documents/IC/yolov4_obj_detection/helpers/output/horizontal_01_right/horizontal_01_right_510_.jpg'
cap_left = '/home/ellentuane/Documents/IC/yolov4_obj_detection/helpers/output/horizontal_01_left/horizontal_01_left_730_.jpg'

imgL = cv.imread(cap_left,0)
imgR = cv.imread(cap_right,0)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)

print(disparity)
#plt.imshow(disparity)
#plt.show()