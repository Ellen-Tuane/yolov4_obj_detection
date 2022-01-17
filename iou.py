import os
import cv2
import matplotlib.pyplot as plt

'''
# DETECTED FILE NAME
txt_name = os.listdir(dir_detected)
for k in txt_name:
    img_name = k.split("_")
'''
# IMAGE SIZE
img = cv2.imread('/home/ellentuane/Documents/IC/image/Aerial_City_0.jpg')
im_h, im_w, _ = img.shape

#Predicted BBOX
dir_predicted = '/home/ellentuane/Documents/IC/detected/320/1'
lines = []

for filename in os.listdir(dir_predicted):
    with open(os.path.join(dir_predicted, filename), "r") as files:
        for line in files:
            lines.append(line)

bb_predicted = []
box = []
for i in lines:
    if i.startswith("person"):
        j = i.split()
        j = [e.replace('%', '') for e in j]
        j = [e.replace(':', '') for e in j]
        j = [e.replace('(', '') for e in j]
        j = [e.replace(')', '') for e in j]
        bb_predicted.append([int(j[3]), int(j[5]), int(j[7]), int(j[9])])
        # bb_predicted = [%, left_x, top_y, width, height]
    else:
        pass

#print(bb_predicted)

for i in bb_predicted:
    x, y, w, h = i[0], i[1], i[2], i[3]
    r = x + w
    b = y + h
    cv2.rectangle(img, (x, y), (r, b), (255, 0, 0), 2)

#plt.imshow(img)
#plt.show()

# OPEN LABELED FILE AND SAVING LINES
dir_labeled = '/home/ellentuane/Documents/IC/labeled/1'
lines_labeled = []

for filename in os.listdir(dir_labeled):
    with open(os.path.join(dir_labeled, filename), "r") as files_labeled:
        for line_labeled in files_labeled:
            lines_labeled.append(line_labeled)

# LABELED BOUNDING BOXES
bb_labeled = []

for l in lines_labeled:
    # Split string to float
    _, x, y, w, h = map(float, l.split(' '))

    x1 = int((x - w / 2) * im_w)
    y1 = int((y - h / 2) * im_h)

    x2 = int((x + w / 2) * im_w)  # r
    y2 = int((y + h / 2) * im_h)  # b

    if x1 < 0:
        x1 = 0
    if x2 > im_w - 1:
        x2 = im_w - 1
    if y1 < 0:
        y1 = 0
    if y2 > im_h - 1:
        y2 = im_h - 1

    bb_labeled.append([x1, y1, x2, y2])

for i in bb_labeled:
    x, y, w, h = i[0], i[1], i[2], i[3]
    cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 1)


print('teste')
plt.imshow(img)
plt.show()