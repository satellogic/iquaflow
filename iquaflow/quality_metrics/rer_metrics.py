import os
import urllib.request
from typing import Any, Optional

from iquaflow.quality_metrics.quality_metrics import QualityMetrics
from iquaflow.quality_metrics.regressor import Regressor, parse_params_cfg


class RERMetrics(QualityMetrics):
    def __init__(
        self,
        cfg_path: str = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "cfgs_best/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.001_numregs40_rer_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3.cfg",
        ),
        checkpoint_url: str = "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/rer/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.001_numregs40_rer_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3/checkpoint_epoch6.pth",
        input_size: Optional[int] = None,
    ) -> None:
        if input_size is not None:
            dict_ckpts = self.get_dict_ckpts()
            list_keys = [
                key for key in dict_ckpts.keys() if f"{input_size}-{input_size}" in key
            ]
            if len(list_keys) > 0:
                key = list_keys[0]
                cfg_path = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), key
                )
                checkpoint_url = dict_ckpts[key]
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

    def get_dict_ckpts(self) -> Any:
        dict_ckpts = {
            # "cfgs_example/config_rer.cfg": "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/metrics_jun/checkpoint_rer_epoch16.pth",
            "cfgs_best/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.001_numregs40_rer_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3.cfg": "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/rer/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.001_numregs40_rer_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3/checkpoint_epoch6.pth",
            "cfgs_best/test_AerialImageDataset_batchsize16_lr0.01_weightdecay1e-05_numregs40_rer_epochs200_numcrops20_splits0.445-0.112_inputsize512-512_momentum0.9_softthreshold0.3.cfg": "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/rer/test_AerialImageDataset_batchsize16_lr0.01_weightdecay1e-05_numregs40_rer_epochs200_numcrops20_splits0.445-0.112_inputsize512-512_momentum0.9_softthreshold0.3/checkpoint_epoch3.pth",
            "cfgs_best/test_AerialImageDataset_batchsize256_lr0.001_weightdecay0.0001_numregs40_rer_epochs200_numcrops100_splits0.445-0.112_inputsize128-128_momentum0.9_softthreshold0.3.cfg": "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/rer/test_AerialImageDataset_batchsize256_lr0.001_weightdecay0.0001_numregs40_rer_epochs200_numcrops100_splits0.445-0.112_inputsize128-128_momentum0.9_softthreshold0.3/checkpoint_epoch15.pth",
            "cfgs_best/test_AerialImageDataset_batchsize64_lr0.01_weightdecay0.0001_numregs40_rer_epochs200_numcrops50_splits0.445-0.112_inputsize256-256_momentum0.9_softthreshold0.3.cfg": "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/rer/test_AerialImageDataset_batchsize64_lr0.01_weightdecay0.0001_numregs40_rer_epochs200_numcrops50_splits0.445-0.112_inputsize256-256_momentum0.9_softthreshold0.3/checkpoint_epoch3.pth",
        }
        return dict_ckpts
