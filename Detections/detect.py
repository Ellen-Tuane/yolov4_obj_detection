import cv2
import numpy as np
from model_performance.bounding_boxes import BoundingBoxes
from Detections.yolo_predictions import YoloPredictions


save_path = '/home/ellentuane/Documents/IC/videos/distance_estimation/'
video_path = '/home/ellentuane/Documents/IC/videos/distance_estimation/test_distancia_horizontal.mp4'
labels_path = '/home/ellentuane/Documents/IC/coco.names'
cfg_path = '/home/ellentuane/Documents/IC/yolov4-tiny.cfg'
weight_path = '/home/ellentuane/Documents/IC/yolov4-tiny.weights'

# .names files with the object's names
labels = open(labels_path).read().strip().split('\n')

# Random colors for each object category
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# yolo weights and cfg configuration files
net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)

use_gpu = 0
if use_gpu == 1:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Obter o nome das categorias
layer_names = YoloPredictions.layer_name(net)

cap = cv2.VideoCapture(video_path)

stop = 0
i = 0
while True:
    if stop == 0:
        ret, frame = cap.read()
        if ret:
            #net, layer_names, image, confidence, threshold, net_height, net_width
            boxes, confidences, classIDs, idxs = YoloPredictions.make_prediction(net, layer_names, frame,
                                                                                 0.01, 0.03, 960, 960)
            for class_id, score, bbox in zip(classIDs, confidences, boxes):
                x, y, w, h = bbox
                name = labels[class_id]
                if name == 'person':
                    frame = BoundingBoxes.draw_bounding_boxes(frame, name, boxes, confidences, classIDs, idxs, colors)
                cv2.imshow('frame', frame)
        else:
            print('Video has ended, failed or wrong path was given, try a different video format!')
            break
        i += 1

    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        stop = not stop
    if key == ord('q'):
        break


cv2.destroyAllWindows()