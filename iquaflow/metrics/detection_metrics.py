from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from iquaflow.metrics import Metric

coco_eval_metrics_names = [
    "ap_0.5-0.95_all",
    "ap_0.5_all",
    "ap_0.75_all",
    "ap_0.5-0.95_small",
    "ap_0.5-0.95_medium",
    "ap_0.5-0.95_large",
    "ar_0.5-0.95_all_det_1",
    "ar_0.5-0.95_all_det_10",
    "ar_0.5-0.95_all",
    "ar_0.5-0.95_small",
    "ar_0.5-0.95_medium",
    "ar_0.5-0.95_large",
]


class BBDetectionMetrics(Metric):
    def __init__(self) -> None:
        """
        Bounding Box COCO metrics.
        """
        self.metric_names = coco_eval_metrics_names

    def apply(self, predictions: str, gt_path: str) -> Any:
        """
        Aplies the metric to the prediction of the run

        Args:
            predictions: str. Path of the predictions
            gt_path: str. Path to the ground truth
        """
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(predictions)
        img_ids = sorted(coco_gt.getImgIds())
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.img_ids = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return {k: v for k, v in zip(self.metric_names, coco_eval.stats)}
