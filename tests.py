import cv2
import numpy as np
from bounding_boxes import BoundingBoxes
from yolo_predictions import YoloPredictions
import pandas as pd


save_path = '/home/ellentuane/Documents/IC/output_confusion_matriz/'
video_path = '/home/ellentuane/Documents/IC/videos/Aerial_City.mp4'
labels_path = '/home/ellentuane/Documents/IC/coco.names'
cfg_path = '/home/ellentuane/Documents/IC/yolov4.cfg'
weight_path = '/home/ellentuane/Documents/IC/yolov4.weights'

# .names files with the object's names
labels = open(labels_path).read().strip().split('\n')

# yolo weights and cfg configuration files
net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)

cap = cv2.VideoCapture(video_path)

use_gpu = 0
if use_gpu == 1:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Obter o nome das categorias
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

df = pd.DataFrame(columns=['boxes', 'confidences', 'classID'])
df.to_csv('/home/ellentuane/Documents/IC/output_confusion_matriz/predictions.csv', index=False)

predict = []
stop = 0
i = 0
while True:
    if stop == 0:
        ret, frame = cap.read()
        if ret:
            boxes, confidences, classIDs, idxs = YoloPredictions.make_prediction(net, layer_names, frame, 0.01, 0.03,
                                                                                 960, 960)
            for box in boxes:
               for conf in confidences:
                   for id in classIDs:
                       with open('/home/ellentuane/Documents/IC/output_confusion_matriz/predictions.csv', 'a') as fd:
                           fd.write(box, conf, id, )

        i += 1



