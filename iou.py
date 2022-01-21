import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

#PREDICTED BBOX

def predicted_files(dir_predicted, file_predicted):
    bb_predicted = []

    img_name = file_predicted.split("_")

    with open(os.path.join(dir_predicted, file_predicted), "r") as files:
        for line in files:
            if line.startswith("person"):
                j = line.split()
                j = [e.replace('%', '') for e in j]
                j = [e.replace(':', '') for e in j]
                j = [e.replace('(', '') for e in j]
                j = [e.replace(')', '') for e in j]
                r = int(j[3]) + int(j[7])
                b = int(j[5]) + int(j[9])
                bb_predicted.append([int(j[3]), int(j[5]), r, b, int(j[1]), img_name[1]])
                # bb_predicted = [left_x, top_y, width, height, %]
            else:
                pass
    return bb_predicted

# OPEN LABELED FILE AND SAVING LINES
dir_labeled = '/home/ellentuane/Documents/IC/labeled/1'
lines_labeled = []

for filename in os.listdir(dir_labeled):
    with open(os.path.join(dir_labeled, filename), "r") as files_labeled:
        for line_labeled in files_labeled:
            lines_labeled.append(line_labeled)

# LABELED BOUNDING BOXES

# IMAGE SIZE
def labeled_files(dir_labeled, file_labeled, im_h, im_w):
    bb_labeled = []

    with open(os.path.join(dir_labeled, file_labeled), "r") as files_labeled:
        for line_labeled in files_labeled:
            # Split string to float
            _, x, y, w, h = map(float, line_labeled.split(' '))
            x1 = int((x - w / 2) * im_w)
            y1 = int((y - h / 2) * im_h)
            x2 = int((x + w / 2) * im_w)  # r
            y2 = int((y + h / 2) * im_h)  # b
            bb_labeled.append([x1, y1, x2, y2, 0])

    return bb_labeled

###     IOU

predicted_TP = []
predicted_FP = []
Labeled_FN = []
inter = []

a = 0
while a < len(bb_predicted):
    for boxB in bb_predicted:
        for boxA in bb_labeled:
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            if interArea > 0:
                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
                boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = round((interArea / float(boxAArea + boxBArea - interArea)) * 100, 2)
            else:
                # return the intersection over union value
                iou = 0

            if iou > 50:
                boxA.append(iou)
                boxB.append(iou)
                predicted_TP.append(boxB)
                inter.append([boxA, boxB])
            else:
                pass
        a += 1

### BBOXES LABELED AND PREDICTED
for bbl in bb_labeled:
    x_labeled, y_labeled, w_labeled, h_labeled = bbl[0], bbl[1], bbl[2], bbl[3]
    cv2.rectangle(img, (x_labeled, y_labeled), (w_labeled, h_labeled), (0, 0, 255), 2)

for bbp in bb_predicted:
    x_predicted, y_predicted, w_predicted, h_predicted = bbp[0], bbp[1], bbp[2], bbp[3]
    cv2.rectangle(img, (x_predicted, y_predicted), (w_predicted, h_predicted), (255, 0, 0), 2)

'''plt.imshow(img)
plt.title('LABELED AND PREDICTED')
plt.show()'''


### BBOXES INTERCEPTED
img2 = cv2.imread('/home/ellentuane/Documents/IC/image/Aerial_City_0.jpg')

for i in inter:
    #bb_labeled
    xx, yy, ww, hh = i[0][0], i[0][1], i[0][2], i[0][3]
    cv2.rectangle(img2, (xx, yy), (ww, hh), (0, 0, 255), 2)

    # bb_predicted
    xxx, yyy, www, hhh = i[1][0], i[1][1], i[1][2], i[1][3]
    cv2.rectangle(img2, (xxx, yyy), (www, hhh), (255, 0, 0), 2)

plt.imshow(img2)
plt.title('True Positive')
plt.show()

### BBOXES FALSE POSITIVES ###

for fp1 in bb_predicted:
     if len(fp1) < 6:
         predicted_FP.append(fp1)
     else:
         pass

img3 = cv2.imread('/home/ellentuane/Documents/IC/image/Aerial_City_0.jpg')

for fp in previsao_FP:
    x_fp, y_fp, w_fp, h_fp = fp[0], fp[1], fp[2], fp[3]
    cv2.rectangle(img3, (x_fp, y_fp), (w_fp, h_fp), (0, 0, 255), 2)

plt.imshow(img3)
plt.title('FALSE POSITIVE')
plt.show()

### BBOXES FALSE NEGATIVES ###

for fn1 in bb_labeled:
     if len(fn1) < 6:
         Labeled_FN.append(fn1)
     else:
         pass

img4 = cv2.imread('/home/ellentuane/Documents/IC/image/Aerial_City_0.jpg')

for fn in Labeled_FN:
    x_fn, y_fn, w_fn, h_fn = fn[0], fn[1], fn[2], fn[3]
    cv2.rectangle(img4, (x_fn, y_fn), (w_fn, h_fn), (0, 0, 255), 2)

plt.imshow(img4)
plt.title('FALSE NEGATIVE')
plt.show()

# CONFUSION MATRIZ

TP = len(previsao_TP)
FP = len(previsao_FP)
FN = len(Labeled_FN)

precision = round(((TP / (TP + FP)) * 100), 0)
recall = round((TP / (TP + FN) * 100), 0)
accuracy = round((TP / len(bb_labeled) * 100), 0)
f1_score = round((2 * (precision * recall / (precision + recall))), 0)

confusion_matriz = []
confusion_matriz.append([img_name[0], img_name[1],img_name[2], TP, FP, FN, precision, recall, accuracy, f1_score])

# DATA FRAME

'''df = pd.DataFrame(confusion_matriz, columns=['frame', 'net_size', 'TP', 'FP', 'FN', 'precision', 'recall', 'accuracy', 'f1_score'])

df.to_csv('YOLO_confusion_matriz_.csv')'''

#print(previsao_TP)
#print(previsao_FP)

'''print(TP)
print(FP)
print(FN)
print(precision)
print(recall)
print(Accuracy)
print(f1_score)'''

#print(confusion_matriz)
#print(inter)
#print(len(inter))
#print(len(FN))
#print(bb_predicted)
#print(bb_labeled)