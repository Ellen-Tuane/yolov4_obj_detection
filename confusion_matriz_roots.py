import os
import cv2
import pandas as pd
from bounding_boxes import BoundingBoxes



def bb_predicted(dir_predicted, file_pred):
    bbox_pred = []

    with open(os.path.join(dir_predicted, file_pred), "r") as files:
        for line in files:
            if line.startswith("person"):
                j = line.split()
                j = [e.replace('%', '') for e in j]
                j = [e.replace(':', '') for e in j]
                j = [e.replace('(', '') for e in j]
                j = [e.replace(')', '') for e in j]
                a = BoundingBoxes.rectangle_to_bbox_coordinates(int(j[3]), int(j[5]), int(j[7]), int(j[9]))
                bbox_pred.append([a.left_x1, a.top_y1, a.left_x2, a.top_y2, j[1]])
                #[left_x, top_y, width, height]
            else:
                pass
    return bbox_pred


def bb_labeled(dir_labeled, file_label, im_height, im_width):
    bbox_label = []

    with open(os.path.join(dir_labeled, file_label), "r") as files_labeled:
        for line_labeled in files_labeled:
            # Split string to float
            _, x, y, w, h = map(float, line_labeled.split(' '))
            x1 = int((x - w / 2) * im_width)
            y1 = int((y - h / 2) * im_height)
            x2 = int((x + w / 2) * im_width)  # r
            y2 = int((y + h / 2) * im_height)  # b
            bbox_label.append([x1, y1, x2, y2, 0])

    return bbox_label


def intersection_over_union(boxA, boxB):
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
    return iou


def False_Positive(bb_pred):
    predicted_FP = []

    for fp in bb_pred:
        if len(fp) < 6:
            predicted_FP.append(fp)
        else:
            pass
    return predicted_FP


def False_Negative(bbox_labeled):
    labeled_FN = []

    for fn in bbox_labeled:
        if len(fn) < 6:
            labeled_FN.append(fn)
        else:
            pass
    return labeled_FN


def confusion_matriz(predicted_TP, predicted_FP, labeled_FN, labeled_gt):
    TP = len(predicted_TP)
    FP = len(predicted_FP)
    FN = len(labeled_FN)

    precision = round(((TP / (TP + FP)) * 100), 0) if (TP + FP) != 0 else 0
    recall = round((TP / (TP + FN) * 100), 0) if (TP + FN) != 0 else 0
    accuracy = round(((TP / len(labeled_gt)) * 100), 0) if len(labeled_gt) != 0 else 0
    f1_score = round((2 * (precision * recall / (precision + recall))), 0) if (precision + recall) !=0 else 0

    return TP, FP, FN, precision, recall, accuracy, f1_score

########################################################################################################################


save_path = '/home/ellentuane/Documents/IC/output_confusion_matriz/'
predicted = r'/home/ellentuane/Documents/IC/detected/todos'
labeled = '/home/ellentuane/Documents/IC/labeled/'
dir_img = '/home/ellentuane/Documents/IC/image/input'
confusion_matriz_result = []


count = 0
while count < len(os.listdir(predicted)):


    for file_predicted in os.listdir(predicted):
        f_p = file_predicted.split("_")

        for file_labeled in os.listdir(labeled):
            fl = file_labeled.split("_")
            fl = [t.replace('.txt', '') for t in fl]
            fl = fl[2]

            if f_p[1] != fl:
                pass
            else:
                for frame in os.listdir(dir_img):
                    fr = frame.split("_")
                    fr = fr[2]
                    if fl != fr:
                        pass
                    else:
                        bbox_TP = []
                        bbox_inter = []
                        cm = []

                        detected = bb_predicted(predicted, file_predicted)
                        img = cv2.imread(os.path.join(dir_img, frame))
                        im_h, im_w, _ = img.shape
                        ground_truth = bb_labeled(labeled, file_labeled, im_h, im_w)


                        a = 0
                        while a < len(detected):
                            for box_d in detected:
                                for box_l in ground_truth:
                                    iou_result = intersection_over_union(box_l, box_d)
                                    if iou_result > 30:
                                        box_d.append(iou_result)
                                        box_l.append(iou_result)
                                        bbox_TP.append(box_d)
                                        bbox_inter.append([box_d, box_l])

                                    else:
                                        pass
                                a += 1

                        bbox_FN = False_Negative(ground_truth)
                        #print(bbox_FN)

                        for i in ground_truth:
                            # bb_labeled
                            #xx, yy, ww, hh = i[0][0], i[0][1], i[0][2], i[0][3]
                            img2 = cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 2)

                        for det in detected:
                            img3 = cv2.rectangle(img2, (det[0], det[1]), (det[2], det[3]), (100, 100, 0), 2)

                        for fpositivo in bbox_FP:
                            #x_fp, y_fp, w_fp, h_fp = fpositivo[0], fpositivo[1], fpositivo[2], fpositivo[3]
                            img4 = cv2.rectangle(img3, (fpositivo[0], fpositivo[1]), (fpositivo[2], fpositivo[3]),
                                                 (255, 0, 0), 2)

                            #BGR

                        for fnegativo in bbox_FN:
                            #x_fn, y_fn, w_fn, h_fn = fnegativo[0], fnegativo[1], fnegativo[2], fnegativo[3]
                            img = cv2.rectangle(img, (fnegativo[0], fnegativo[1]), (fnegativo[2], fnegativo[3]),
                                                (0, 0, 255), 2)

                        cv2.imwrite(f"{save_path}/{f_p[2]}_{frame}", img4)

                        cm = confusion_matriz(bbox_TP, bbox_FP, bbox_FN, ground_truth)
                        confusion_matriz_result.append([fr, f_p[2],  cm[0], cm[1], cm[2], cm[3], cm[4], cm[5], cm[6],
                                                        len(detected), len(ground_truth)])

                        break
                count += 1

df = pd.DataFrame(confusion_matriz_result, columns=['frame', 'net_size', 'TP', 'FP', 'FN', 'precision', 'recall',
                                                    'accuracy', 'f1_score', 'total_detected', 'total_labeled'])
df.to_csv('/home/ellentuane/Documents/IC/output_confusion_matriz/YOLO_confusion_matriz_.csv', index=False)
########################################################################################################################

#CALCULAR A DIFERENÇA INTERQUARTIL DO TAMANHO DOS OBJETOS PARA DETECTAR OUTLIERS

