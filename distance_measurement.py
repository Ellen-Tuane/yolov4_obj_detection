import cv2
import numpy as np
from model_performance.bounding_boxes import BoundingBoxes
from Detections.yolo_predictions import YoloPredictions

save_path = '/home/ellentuane/Documents/IC/videos/distance_estimation/distance_estimation_reference/'
video_path = '/home/ellentuane/Documents/IC/videos/distance_estimation/distance_estimation_reference/distance_estimation_reference_0_.jpg'
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

                    ## Distance Meaasurement for each bounding box
                    #x, y, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
                    ## item() is used to retrieve the value from the tensor
                    distance = (2 * 3.14 * 180) / (w + h * 360) * 1000 + 3  ### Distance measuring in Inch
                    feedback = ("{}".format(labels["Current Object"]) + " " + "is" + " at {} ".format(round(distance)) + "Inches")


                    cv2.putText(img, str("{:.2f} Inches".format(distance)), (text_w + x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
                    cv2.rectangle(img, (bboxes[0], bboxes[1]), (bboxes[2] + text_w - 30, bboxes[3]), color, 2)
                    cv2.putText(img, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)


                    #frame = BoundingBoxes.draw_bounding_boxes(frame, name, boxes, confidences, classIDs, idxs, colors)

            cv2.imshow('frame', frame)
        i += 1

    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        stop = not stop
    if key == ord('q'):
        break

cv2.destroyAllWindows()
