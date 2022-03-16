class ConfusionMatriz:
    def __init__(self, true_positive, false_positive, false_negative):
        self.true_positive = int(true_positive)  #This is an instance in which the classifier predicted positive when
        # the truth is indeed positive, that is, a detection for which IoU ≥ α
        self.false_positive = int(false_positive)  #This is a wrong positive detection, that is, a detection
        # for which IoU < α
        self.false_negative = int(false_negative)  #This is an actual instance that is not detected by the classifier

    #this function needs the (left_x1, top_y1, left_x2, top_y2) bbox configuration to work correctly
    @staticmethod
    def intersection_over_union(bbox_a, bbox_b):
        xA = max(bbox_a[0], bbox_b[0])
        yA = max(bbox_a[1], bbox_b[1])
        xB = min(bbox_a[2], bbox_b[2])
        yB = min(bbox_a[3], bbox_b[3])
        # compute the area of intersection rectangle
        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        if inter_area > 0:
            # compute the area of both the prediction and ground-truth
            # rectangles
            bbox_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
            bbox_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the intersection area
            iou = round((inter_area / float(bbox_a_area + bbox_b_area - inter_area)) * 100, 2)
        else:
            # return the intersection over union value
            iou = 0
        return iou

    @staticmethod
    def true_positive(ground_truth_bbox, detected_bbox, threshold):
        bbox_tp = []
        count = 0

        while count < len(detected_bbox):
            for box_d in detected_bbox:
                for box_l in ground_truth_bbox:
                    iou_result = ConfusionMatriz.intersection_over_union(box_l, box_d)

                    if iou_result > threshold:
                        bbox_tp.append([box_l, box_d])
                    else:
                        pass
                count += 1
        return bbox_tp

    @staticmethod
    def false_positive(true_positive_bbox, detected_bbox):
        false_positive_bbox = []
        tp_detected_bb = []

        for tp_detected in true_positive_bbox:
            tp_detected_bb.append(tp_detected[1])

        for i in detected_bbox:
            if i not in tp_detected_bb:
                false_positive_bbox.append(i)
            else:
                pass
        return false_positive_bbox

    @staticmethod
    def false_negative(true_positive_bbox, ground_truth_bbox):
        false_negative_bbox = []
        tp_gt_bbox = []

        for tp_gt in true_positive_bbox:
            tp_gt_bbox.append(tp_gt[0])

        for i in ground_truth_bbox:
            if i not in tp_gt_bbox:
                false_negative_bbox.append(i)
            else:
                pass
        return false_negative_bbox
