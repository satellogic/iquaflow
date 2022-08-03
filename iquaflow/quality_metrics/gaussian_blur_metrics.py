import os
import urllib.request

from iquaflow.quality_metrics.quality_metrics import QualityMetrics
from iquaflow.quality_metrics.regressor import Regressor, parse_params_cfg

"""
cfgs_example/config_gaussian.cfg
https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/metrics_jun/checkpoint_gaussian_epoch193.pth
cfgs_best/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.0001_numregs50_sigma_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3.cfg
https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/blur/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.0001_numregs50_sigma_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3/checkpoint_epoch68.pth
cfgs_best/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.0001_numregs50_sigma_epochs200_numcrops20_splits0.445-0.112_inputsize512-512_momentum0.9_softthreshold0.3.cfg
https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/blur/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.0001_numregs50_sigma_epochs200_numcrops20_splits0.445-0.112_inputsize512-512_momentum0.9_softthreshold0.3/checkpoint_epoch19.pth
cfgs_best/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.001_numregs50_sigma_epochs200_numcrops50_splits0.445-0.112_inputsize256-256_momentum0.9_softthreshold0.3.cfg
https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/blur/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.001_numregs50_sigma_epochs200_numcrops50_splits0.445-0.112_inputsize256-256_momentum0.9_softthreshold0.3/checkpoint_epoch11.pth
"""


class GaussianBlurMetrics(QualityMetrics):
    def __init__(
        self,
        cfg_path: str = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "cfgs_best/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.0001_numregs50_sigma_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3.cfg",
        ),
        checkpoint_url: str = "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/blur/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.0001_numregs50_sigma_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3/checkpoint_epoch68.pth",
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
