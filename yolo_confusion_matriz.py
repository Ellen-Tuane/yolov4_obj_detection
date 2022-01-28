import os
import cv2
from bounding_boxes import BoundingBoxes
from confusion_matriz import ConfusionMatriz
from confusion_matriz_metrics import ConfusionMatrizMetrics
import pandas as pd


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
                rectangle_predicted = BoundingBoxes.bbox_coordinates_to_rectangle(int(j[3]), int(j[5]),
                                                                                  int(j[7]), int(j[9]))
                bbox_pred.append([rectangle_predicted.left_x1, rectangle_predicted.top_y1, rectangle_predicted.left_x2,
                                  rectangle_predicted.top_y2])
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
            rectangle_labeled = BoundingBoxes.yolo_annotation_to_rectangle(x, y, w, h, im_height, im_width)
            bbox_label.append([rectangle_labeled.left_x1, rectangle_labeled.top_y1,
                              rectangle_labeled.left_x2, rectangle_labeled.top_y2])

    return bbox_label


save_path = '/home/ellentuane/Documents/IC/output_confusion_matriz/'
predicted = r'/home/ellentuane/Documents/IC/detected/todos/'
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
                        detected = bb_predicted(predicted, file_predicted)
                        img = cv2.imread(os.path.join(dir_img, frame))
                        im_h, im_w, _ = img.shape
                        ground_truth = bb_labeled(labeled, file_labeled, im_h, im_w)

                        tp = ConfusionMatriz.true_positive(ground_truth, detected, 30)
                        fp = ConfusionMatriz.false_positive(tp, detected)
                        fn = ConfusionMatriz.false_negative(tp, ground_truth)

                        cm = ConfusionMatrizMetrics.confusionMatrixMetrics(len(tp), len(fp), len(fn), len(ground_truth))

                        confusion_matriz_result.append([fr, f_p[2], len(tp), len(fp), len(fn), cm.precision, cm.recall, cm.accuracy, cm.f1_score,
                                                        len(detected), len(ground_truth)])

                        for xx in tp:
                            img1 = cv2.rectangle(img, (xx[0][0], xx[0][1]), (xx[0][2], xx[0][3]), (255, 255, 0), 2)
                            img1 = cv2.rectangle(img, (xx[1][0], xx[1][1]), (xx[1][2], xx[1][3]), (0, 255, 0), 2)

                        for yy in fp:
                            img1 = cv2.rectangle(img, (yy[0], yy[1]), (yy[2], yy[3]), (0, 0, 255), 2)

                        for o in fn:
                            img1 = cv2.rectangle(img, (o[0], o[1]), (o[2], o[3]), (255, 0, 255), 2)

                        cv2.imwrite(f"{save_path}/{f_p[2]}_{frame}", img1)


                        break
                count += 1





df = pd.DataFrame(confusion_matriz_result, columns=['frame', 'net_size', 'TP', 'FP', 'FN', 'precision', 'recall',
                                                    'accuracy', 'f1_score', 'total_detected', 'total_labeled'])
df.to_csv('/home/ellentuane/Documents/IC/output_confusion_matriz/YOLO_confusion_matriz_.csv', index=False)

