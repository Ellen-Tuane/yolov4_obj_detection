import cv2 as cv

def yolo_annotation_to_rectangle( x, y, w, h, image_height, image_width):
    # convert yolo bounding boxes annotations into bbox annotation for opencv
    left_x1 = round((x - w / 2) * image_width)
    top_y1 = round((y - h / 2) * image_height)
    left_x2 = round((x + w / 2) * image_width)
    top_y2 = round((y + h / 2) * image_height)

    width = left_x2 - left_x1
    height = top_y2 - top_y1
    return left_x1, top_y1, width, height


def bb_labeled(file_label, im_height, im_width):

    with open(file_label, "r") as files_labeled:
        for line_labeled in files_labeled:
            # Split string to float
            _, x, y, w, h = map(float, line_labeled.split(' '))
            rectangle_labeled = yolo_annotation_to_rectangle(x, y, w, h, im_height, im_width)

    return rectangle_labeled

def rescaleFrame(frame, scale=0.75):
    # images, video and live videos
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


labeled = '/home/ellentuane/Documents/IC/output_confusion_matriz/city/city_labels/City_0_.txt'

dir_img = '/home/ellentuane/Documents/IC/output_confusion_matriz/city/city_frame/Aerial_City_0_.jpg'

img = cv.imread(dir_img)
im_h, im_w, _ = img.shape

#img = rescaleFrame(img)

gt = bb_labeled(labeled, im_h, im_w)

img = cv.rectangle(img, (gt[0], gt[1]), (gt[0] + gt[2], gt[1] + gt[3]), (0,0,255), 2)
cv.imshow('Frame', img)
cv.waitKey(0)


print(gt)
print(im_w)
print(im_h)