import cv2
from model_performance.bounding_boxes import BoundingBoxes
from Detections.pre_processing import PreProcess


class YoloPredictions:
    def __init__(self, class_id, score, bbox):
        self.class_id = class_id
        self.score = score
        self.bbox = bbox

    def set_class_id(self, class_id):
        self.class_id = class_id

    def get_class_id(self):
        return self.class_id

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return self.score

    def set_bbox(self, bbox):
        self.bbox = bbox

    def get_bbox(self):
        return self.bbox

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
