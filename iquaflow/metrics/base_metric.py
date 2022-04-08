from typing import Any


class Metric:
    """
    Metric generic object that is used as a class interface
    """

    def __init__(self):
        self.name = "base_metric"

    def apply(self, predictions: str, gt_path: str) -> Any:
        """
        Aplies the metric to the prediction of the run

        Args:
            predictions: str. Path of the predictions
            gt_path: str. Path to the ground truth
        """
        return {"metric": 8}
