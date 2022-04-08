import os
import urllib.request
from typing import Optional

from iquaflow.quality_metrics.quality_metrics import QualityMetrics
from iquaflow.quality_metrics.regressor import Regressor, parse_params_cfg


class RERMetrics(QualityMetrics):
    def __init__(self, config_filename: Optional[str] = None) -> None:
        if config_filename is None:
            current_path = os.path.dirname(os.path.realpath(__file__))
            config_filename = os.path.join(current_path, "cfgs_example/config_rer.cfg")
        parser = parse_params_cfg(default_cfg_path=config_filename)
        args, uk_args = parser.parse_known_args()  # [] or use defaults
        # dict_args = vars(args)  # for debugging
        self.regressor = Regressor(args)
        # check if checkpoint exists
        if not os.path.exists(self.regressor.checkpoint_path):
            checkpoint_url = "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/metrics_jun/checkpoint_rer_epoch16.pth"
            urllib.request.urlretrieve(checkpoint_url, self.regressor.checkpoint_path)
        self.metric_names = list(self.regressor.modifier_params.keys())
        # self.metric_names.append("path")  # include path in results? #warning: commented cause of core error, can't log_metric a string of path
        self.metric_names.remove("dataset")  # remove default key for rer/snr
