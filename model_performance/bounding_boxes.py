'''

Object detection bounding boxes
Top left rectangle edge: left_x1, top_y1 and bottom right rectangle edge: left_x2, top_y2

Features:
1. left_x1, top_y1, left_x2, top_y2, width, height

'''
import cv2
import numpy as np


class BoundingBoxes:
    # constructor method
    def __init__(self, x, y, width, height):
        self.left_x = int(x)
        self.top_y = int(y)
        self.width = int(width)
        self.height = int(height)

    def get_x(self):
        return self.left_x

    def get_y(self):
        return self.top_y

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def set_x(self, x):
        self.left_x = int(x)

    def set_y(self, y):
        self.top_y = int(y)

    def set_width(self, width):
        self.width = int(width)

    def set_height(self, height):
        self.height = int(height)

    @staticmethod
    def bbox_to_rectangle(x, y, width, height):
        # this method consists in convert coordinates format like (left_x1, top_y1, width, height)
        # into coordinate necessary to draw a bbox on an image using opencv.
        # left_x1 + rectangle_width
        right_x2 = x + width
        # top_y1 + rectangle height
        bottom_y2 = y + height
        return x, y, right_x2, bottom_y2

    @staticmethod
    def yolo_annotation_to_bbox(x, y, w, h, image_height, image_width):
        # convert yolo bounding boxes annotations into bbox annotation for opencv
        left_x1 = int((x - w / 2) * image_width)
        top_y1 = int((y - h / 2) * image_height)
        right_x2 = int((x + w / 2) * image_width)
        bottom_y2 = int((y + h / 2) * image_height)
        width = right_x2 - left_x1
        height = bottom_y2 - top_y1
        return left_x1, top_y1, width, height

    @staticmethod
    def extract_boxes_confidences_class_ids(outputs, confidence, width, height):
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            for detection in output:
                # Extract the scores, class_id, and the confidence of the prediction
                scores = detection[5:]
                classID = np.argmax(scores)
                conf = scores[classID]

                # Consider only the predictions that are above the confidence threshold
                if conf > confidence:
                    # Scale the bounding box back to the size of the image
                    box = detection[0:4] * np.array([width, height, width, height])
                    centerX, centerY, w, h = box.astype('int')

                    # Use the center coordinates, width and height to get the coordinates of the top left corner
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(conf))
                    classIDs.append(classID)

        return [boxes, confidences, classIDs]

    @staticmethod
    def draw_bounding_boxes(image, labels, boxes, confidences, classids, idxs, colors):
        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                # draw the bounding box and labels on the image
                color = [int(c) for c in colors[classids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        return image

    @staticmethod
    def draw_bounding_boxes_confusion_matriz(image, boxes, color):
        if len(boxes) > 0:
            for i in boxes:
                # extract bounding box coordinates
                x, y = i[0], i[1]
                w, h = i[2], i[3]
                # draw the bounding box and labels on the image

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        return image




