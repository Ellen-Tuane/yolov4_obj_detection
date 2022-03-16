class ConfusionMatrizMetrics:

    def __init__(self, precision, recall, accuracy, f1_score):
        self.precision = int(precision)
        self.recall = int(recall)
        self.accuracy = int(accuracy)
        self.f1_score = int(f1_score)

    @staticmethod
    def precision(true_positive, false_positive):
        return round(((true_positive / (true_positive + false_positive)) * 100), 0) \
            if (true_positive + false_positive) != 0 else 0

    @staticmethod
    def recall(true_positive, false_negative):
        return round((true_positive / (true_positive + false_negative) * 100), 0) \
            if (true_positive + false_negative) != 0 else 0

    @staticmethod
    def accuracy(true_positive, ground_truth):
        return round(((true_positive / ground_truth) * 100), 0) if ground_truth != 0 else 0

    @staticmethod
    def f1_score(precision, recall):
        return round((2 * (precision * recall / (precision + recall))), 0) if (precision + recall) != 0 else 0

    @classmethod
    def confusionMatrixMetrics(cls, true_positive, false_positive, false_negative, len_ground_truth):
        precision = ConfusionMatrizMetrics.precision(true_positive, false_positive)
        recall = ConfusionMatrizMetrics.recall(true_positive, false_negative)
        accuracy = ConfusionMatrizMetrics.accuracy(true_positive, len_ground_truth)
        f1_score = ConfusionMatrizMetrics.f1_score(precision, recall)

        return ConfusionMatrizMetrics(precision, recall, accuracy, f1_score)
