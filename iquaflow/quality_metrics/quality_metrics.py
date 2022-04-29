import json
import os
from typing import Any, List, Optional

from iquaflow.metrics import Metric
from iquaflow.quality_metrics.regressor import Regressor, parse_params_cfg


class QualityMetrics(Metric):
    def __init__(self) -> None:
        parser = parse_params_cfg(default_cfg_path="config.cfg")
        args, uk_args = parser.parse_known_args()  # [] or use defaults
        # dict_args = vars(args)  # for debugging
        self.regressor = Regressor(args)
        self.metric_names = list(self.regressor.modifier_params.keys())
        # self.metric_names.append("path")  # include path in results? #warning: commented cause of core error, can't log_metric a string of path

    def read_files(self, predictions: str, gt_path: Optional[str] = None) -> Any:
        # old
        """
        ds_folder=os.path.dirname(gt_path)
        images_folder=os.listdir(ds_folder)[0]
        abs_images_folder=ds_folder+'/'+images_folder
        image_files=os.listdir(abs_images_folder)
        """
        json_file = open(predictions)
        output = json.load(json_file)
        # artifacts_path = os.path.dirname(predictions)
        """
        tmp_val_ds_output = output["val_ds_output"]
        val_ds_output = os.path.join(
            artifacts_path,
            os.path.basename(os.path.dirname(tmp_val_ds_output)),
            os.path.basename(tmp_val_ds_output),
        )
        images_path = val_ds_output
        """
        images_path = output["val_ds_output"]
        # read images from input path
        image_files = os.listdir(images_path)
        for idx, image_name in enumerate(image_files):
            image_files[idx] = images_path + "/" + image_name  # abs_images_folder
        return image_files

    def deploy_stats(self, image_files: List[str]) -> Any:
        # run regressor deploy to get stats
        return self.regressor.deploy(image_files)

    def get_results(self, stats: Any) -> Any:
        """
        # group results with metrics
        results = []
        for idx, image_path in enumerate(image_files):
            result = {
                k: v for k, v in zip(self.metric_names, [stats[idx], image_path])
            }  # must be list for zipping dict
            results.append(result)
        """
        # unify images stats by average -> considering single parameter
        avg_stats = sum(stats) / len(stats)
        results = {k: v for k, v in zip(self.metric_names, [avg_stats])}
        return results

    def apply(self, predictions: str, gt_path: Optional[str] = None) -> Any:
        image_files = self.read_files(predictions, gt_path)
        stats = self.deploy_stats(image_files)
        results = self.get_results(stats)
        return results
