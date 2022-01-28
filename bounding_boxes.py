'''

Object detection bounding boxes
Top left rectangle edge: left_x1, top_y1 and bottom right rectangle edge: left_x2, top_y2

Features:
1. left_x1, top_y1, left_x2, top_y2, width, height

'''


class BoundingBoxes:
    # constructor method
    def __init__(self, x1, y1, x2, y2):
        self.left_x1 = int(x1)
        self.top_y1 = int(y1)
        self.left_x2 = int(x2)
        self.top_y2 = int(y2)

    @classmethod
    def bbox_coordinates_to_rectangle(cls, left_x1, top_y1, width, height):
        # this method consists in convert coordinates format like (left_x1, top_y1, width, height)
        # into coordinate necessary to draw a bbox on an image using opencv.
        # left_x1 + rectangle_width
        left_x2 = left_x1 + width
        # top_y1 + rectangle height
        top_y2 = top_y1 + height
        return BoundingBoxes(left_x1, top_y1, left_x2, top_y2)

    @classmethod
    def yolo_annotation_to_rectangle(cls, x, y, w, h, image_height, image_width):
        # convert yolo bounding boxes annotations into bbox annotation for opencv
        left_x1 = int((x - w / 2) * image_width)
        top_y1 = int((y - h / 2) * image_height)
        left_x2 = int((x + w / 2) * image_width)
        top_y2 = int((y + h / 2) * image_height)
        return BoundingBoxes(left_x1, top_y1, left_x2, top_y2)


