import cv2
from bounding_boxes import BoundingBoxes


class YoloPredictions:
    def __init__(self, boxes, confidences, classIDs, idxs):
        self.boxes = boxes
        self.confidences = confidences
        self.classIDs = classIDs
        self.idxs = idxs

    @staticmethod
    def make_prediction(net, layer_names, image, confidence, threshold, net_height, net_width):
        height, width = image.shape[:2]
        # image pre-processing
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (net_height, net_width), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(layer_names)

        # extract bbox, confidence e classIDs
        boxes, confidences, classIDs = BoundingBoxes.extract_boxes_confidences_class_ids(outputs, confidence,
                                                                                         width, height)
        # Non-Max Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

        return boxes, confidences, classIDs, idxs
