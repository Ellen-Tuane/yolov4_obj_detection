import os
import cv2
from model_performance.bounding_boxes import BoundingBoxes
from model_performance.confusion_matriz import ConfusionMatriz
from model_performance.confusion_matriz_metrics import ConfusionMatrizMetrics
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
                rectangle_predicted = BoundingBoxes.bbox_to_rectangle(int(j[3]), int(j[5]),
                                                                                  int(j[7]), int(j[9]))
                bbox_pred.append([rectangle_predicted[0], rectangle_predicted[1], rectangle_predicted[2],
                                  rectangle_predicted[3]])
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
            rectangle_labeled = BoundingBoxes.yolo_annotation_to_bbox(x, y, w, h, im_height, im_width)
            bbox_label.append([rectangle_labeled[0], rectangle_labeled[1],
                              rectangle_labeled[2], rectangle_labeled[2]])

    return bbox_label


save_path = '/home/ellentuane/Documents/IC/output_confusion_matriz/city'
predicted = '/home/ellentuane/Documents/IC/output_confusion_matriz/city/city_frame_detections_results'
labeled = '/home/ellentuane/Documents/IC/output_confusion_matriz/city/city_labels'
dir_img = '/home/ellentuane/Documents/IC/output_confusion_matriz/city/city_frame'
confusion_matriz_result = []

count = 0
while count < len(os.listdir(predicted)):
    for file_predicted in os.listdir(predicted):
        predicted_name = file_predicted.split("_")[:3]
        for file_labeled in os.listdir(labeled):
            label_name = file_labeled.split("_")[1]
            if predicted_name[1] != label_name:
                pass
            else:
                for frame in os.listdir(dir_img):
                    frame_name = frame.split("_")[-2]
                    if label_name != frame_name:
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

                        confusion_matriz_result.append([frame_name, predicted_name[2], len(tp), len(fp), len(fn),
                                                        cm.precision, cm.recall, cm.accuracy, cm.f1_score,
                                                        len(detected), len(ground_truth)])
                        tp1 = []
                        for tps in tp:
                            tp1.append(tps[1])
                        img = BoundingBoxes.draw_bounding_boxes_confusion_matriz(img, ground_truth, (0, 100, 0))
                        img = BoundingBoxes.draw_bounding_boxes_confusion_matriz(img, tp1, (0, 255, 0))
                        img = BoundingBoxes.draw_bounding_boxes_confusion_matriz(img, fp, (255, 0, 0))
                        img = BoundingBoxes.draw_bounding_boxes_confusion_matriz(img, fn, (0, 0, 255))
                        predicted_name = '_'.join(predicted_name)
                        cv2.imwrite(f"{save_path}/{predicted_name}.jpg", img)
                        break
                count += 1
df = pd.DataFrame(confusion_matriz_result, columns=['frame', 'net_size', 'TP', 'FP', 'FN', 'precision', 'recall',
                                                    'accuracy', 'f1_score', 'total_detected', 'total_labeled'])
df.to_csv(f"{save_path}/soccer_confusion_matriz_.csv", index=False)
