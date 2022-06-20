import cv2
import numpy as np
import os
from model_performance.bounding_boxes import BoundingBoxes
from Detections.pre_processing import PreProcess

class YoloPredictions:
    @staticmethod
    def make_prediction(net, layer_names, image, confidence, threshold, net_height, net_width):
        height, width = image.shape[:2]
        # image pre-processing
        outputs = PreProcess.blob_net(image, layer_names, net, net_height, net_width)

        # extract bbox, confidence e classIDs
        boxes, confidences, classIDs = BoundingBoxes.extract_boxes_confidences_class_ids(outputs, confidence,
                                                                                         width, height)
        # Non-Max Suppression - It is a class of algorithms to select one entity out of many overlapping entities
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

        return boxes, confidences, classIDs, idxs

    @staticmethod
    def layer_name(net):
        layer_names = net.getLayerNames()
        layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return layer_names
