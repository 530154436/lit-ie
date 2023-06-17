from .precision_recall_fscore import (
    _precision_recall_fscore,
    extract_tp_actual_correct,
    extract_tp_actual_correct_for_event
)
from ..base import Metric


class ExtractionScore(Metric):

    def __init__(self, average="micro"):
        self.average = average
        self.reset()

    def update(self, y_true, y_pred):
        pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true, y_pred)
        self.pred_sum += pred_sum
        self.tp_sum += tp_sum
        self.true_sum += true_sum

    def value(self):
        return _precision_recall_fscore(self.pred_sum, self.tp_sum, self.true_sum)

    def name(self):
        return "extraction_score"

    def reset(self):
        self.pred_sum = 0
        self.tp_sum = 0
        self.true_sum = 0


class EventExtractionScore(Metric):

    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        ex, ey, ez, ax, ay, az = extract_tp_actual_correct_for_event(y_true, y_pred)
        self.ex += ex
        self.ey += ey
        self.ez += ez

        self.ax += ax
        self.ay += ay
        self.az += az

    def value(self):
        return {
            "event": _precision_recall_fscore(self.ey, self.ex, self.ez),
            "argu": _precision_recall_fscore(self.ay, self.ax, self.az)
        }

    def name(self):
        return "event_extraction_score"

    def reset(self):
        # 事件级别
        self.ex = 0
        self.ey = 0
        self.ez = 0

        # 论元级别
        self.ax = 0
        self.ay = 0
        self.az = 0


if __name__ == "__main__":
    metric = ExtractionScore()
    p = [{("a", 1), ("a", 2), ("b", 3)}, {("a", 4), ("c", 5)}]
    t = [{("a", 1), ("a", 4), ("b", 3)}, {("a", 2), ("c", 5)}]
    metric.update(t, p)
    print(metric.value())
