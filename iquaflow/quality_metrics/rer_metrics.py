import os
import urllib.request

from iquaflow.quality_metrics.quality_metrics import QualityMetrics
from iquaflow.quality_metrics.regressor import Regressor, parse_params_cfg


class RERMetrics(QualityMetrics):
    def __init__(
        self,
        cfg_path: str = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "cfgs_example/config_rer.cfg"
        ),
        checkpoint_url: str = "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/metrics_jun/checkpoint_rer_epoch16.pth",
    ) -> None:
        parser = parse_params_cfg(default_cfg_path=cfg_path)
        args, uk_args = parser.parse_known_args()  # [] or use defaults
        # dict_args = vars(args)  # for debugging
        self.regressor = Regressor(args)
        # check if checkpoint exists
        if not os.path.exists(self.regressor.checkpoint_path):
            urllib.request.urlretrieve(checkpoint_url, self.regressor.checkpoint_path)
        self.metric_names = list(self.regressor.modifier_params.keys())
        # self.metric_names.append("path")  # include path in results? #warning: commented cause of core error, can't log_metric a string of path
        self.metric_names.remove("dataset")  # remove default key for rer/snr
