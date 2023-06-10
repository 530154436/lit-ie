from typing import List


def _precision_recall_fscore(pred_sum, tp_sum, true_sum):
    recall = tp_sum / true_sum if true_sum > 0 else 0.0
    precision = tp_sum / pred_sum if pred_sum > 0 else 0.0

    if recall + precision == 0.0:
        f_score = 0.0
    else:
        f_score = 2 * recall * precision / (recall + precision)

    return precision, recall, f_score


def extract_tp_actual_correct(y_true: List[set], y_pred: List[set]):
    entities_true = set()
    entities_pred = set()
    for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        for d in y_t:
            entities_true.add((i, d))
        for d in y_p:
            entities_pred.add((i, d))

    tp_sum = len(entities_true & entities_pred)
    pred_sum = len(entities_pred)
    true_sum = len(entities_true)
    return pred_sum, tp_sum, true_sum
