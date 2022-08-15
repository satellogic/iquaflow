import os
from typing import Any, List, Optional

import numpy as np

from iquaflow.quality_metrics import (
    GaussianBlurMetrics,
    GSDMetrics,
    NoiseSharpnessMetrics,
    QualityMetrics,
    RERMetrics,
    SNRMetrics,
)


class ScoreMetrics(QualityMetrics):
    def __init__(
        self,
        metric_names: Optional[List[str]] = None,
        objectives: Optional[List[float]] = None,
        config_filenames: Optional[List[str]] = None,
        ranges: Optional[List[List[float]]] = None,
        weights: Optional[List[float]] = None,
        default_checkpoint_urls: Optional[List[str]] = None,
        input_size: Optional[int] = None,
    ) -> None:
        if metric_names is None:
            metric_names = ["sigma", "sharpness", "scale", "snr", "rer"]
        if objectives is None:
            [
                1.0,  # minimize
                1.0,  # n/a
                0.3,  # minimize
                15,  # minimize
                0.55,  # maximize
            ]
        if config_filenames is None:
            [
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "cfgs_best/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.0001_numregs50_sigma_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3.cfg",
                ),
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "cfgs_best/test_AerialImageDataset_batchsize4_lr0.01_weightdecay0.0001_numregs9_sharpness_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3.cfg",
                ),
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "cfgs_best/test_AerialImageDataset_batchsize16_lr0.01_weightdecay1e-05_numregs10_scale_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3.cfg",
                ),
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "cfgs_best/test_AerialImageDataset_batchsize8_lr0.01_weightdecay0.001_numregs40_snr_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3.cfg",
                ),
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "cfgs_best/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.001_numregs40_rer_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3.cfg",
                ),
            ],
        if ranges is None:
            [
                [1.0, 2.5],
                [1.0, 10.0],
                [15, 30],
                [15, 55],
                [0.15, 0.55],
            ]
        if weights is None:
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        if default_checkpoint_urls is None:
            default_checkpoint_urls = [
                    "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/blur/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.0001_numregs50_sigma_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3/checkpoint_epoch68.pth",
                    "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/sharpness/test_AerialImageDataset_batchsize4_lr0.01_weightdecay0.0001_numregs9_sharpness_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3/checkpoint_epoch143.pth",
                    "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/scale/test_AerialImageDataset_batchsize16_lr0.01_weightdecay1e-05_numregs10_scale_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3/checkpoint_epoch0.pth",
                    "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/rer/test_AerialImageDataset_batchsize16_lr0.01_weightdecay0.001_numregs40_rer_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3/checkpoint_epoch6.pth",
                    "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/training-results-whole/snr/test_AerialImageDataset_batchsize8_lr0.01_weightdecay0.001_numregs40_snr_epochs200_numcrops10_splits0.445-0.112_inputsize1024-1024_momentum0.9_softthreshold0.3/checkpoint_epoch148.pth",
                ]
        self.default_checkpoint_urls = default_checkpoint_urls
        self.input_size = input_size
        self.metric_names = metric_names
        self.config_filenames = config_filenames
        self.objectives = objectives
        self.ranges = ranges
        self.weights = weights

        """  # do not init (need too much memory to keep all QMRNets on init)
        self.regressors = []
        self.submetric_names = []
        for idx, metric in enumerate(metric_names):
            parser = parse_params_cfg(default_cfg_path=config_filenames[idx])
            args, uk_args = parser.parse_known_args()
            self.regressors.append(Regressor(args))
            # check if checkpoint exists
            if not os.path.exists(self.regressors[idx].checkpoint_path):
                checkpoint_url = self.default_checkpoint_urls[idx]
                urllib.request.urlretrieve(
                    checkpoint_url, self.regressors[idx].checkpoint_path
                )
            self.submetric_names.append(
                list(self.regressors[idx].modifier_params.keys())
            )
        """

    def calc_score(self, stats: List[Any]) -> Any:
        score = 0.0
        metric_scores = []
        for idx, metric in enumerate(self.metric_names):
            minmax = np.abs(np.max(self.ranges[idx]) - np.min(self.ranges[idx]))  # type: ignore
            avg_stats = sum(stats[idx]) / len(stats[idx])
            abs_avg_stats_diff = np.abs(
                self.objectives[idx] - avg_stats  # type: ignore
            )  # error w/ respect objective value
            metric_score = (minmax - abs_avg_stats_diff) / minmax
            metric_scores.append(metric_score)
            weighted_metric_score = metric_score * self.weights[idx]
            score += weighted_metric_score
        return score, metric_scores

    def deploy_stats(self, image_files: List[str]) -> Any:
        stats = []
        for idx, metric in enumerate(self.metric_names):
            """
            # run regressor deploy to get stats
            stats.append(self.regressors[idx].deploy(image_files))
            """
            # init regressor instance
            quality_metric: Optional[Any] = None
            if metric == "sigma":
                if self.input_size is None:
                    quality_metric = GaussianBlurMetrics(
                        self.config_filenames[idx], self.default_checkpoint_urls[idx]  # type: ignore
                    )
                else:
                    quality_metric = GaussianBlurMetrics(self.input_size)  # type: ignore
            elif metric == "sharpness":
                if self.input_size is None:
                    quality_metric = NoiseSharpnessMetrics(
                        self.config_filenames[idx], self.default_checkpoint_urls[idx]  # type: ignore
                    )
                else:
                    quality_metric = NoiseSharpnessMetrics(self.input_size)  # type: ignore
            elif metric == "scale":
                if self.input_size is None:
                    quality_metric = GSDMetrics(
                        self.config_filenames[idx], self.default_checkpoint_urls[idx]  # type: ignore
                    )
                else:
                    quality_metric = GSDMetrics(self.input_size)  # type: ignore
            elif metric == "rer":
                if self.input_size is None:
                    quality_metric = RERMetrics(
                        self.config_filenames[idx], self.default_checkpoint_urls[idx]  # type: ignore
                    )
                else:
                    quality_metric = RERMetrics(self.input_size)  # type: ignore
            elif metric == "snr":
                if self.input_size is None:
                    quality_metric = SNRMetrics(
                        self.config_filenames[idx], self.default_checkpoint_urls[idx]  # type: ignore
                    )
                else:
                    quality_metric = SNRMetrics(self.input_size)  # type: ignore
            else:
                continue
            # deploy regressor instance
            stats.append(quality_metric.regressor.deploy(image_files))
        return stats

    def get_results(self, stats: Any) -> Any:  # override function
        score, metric_scores = self.calc_score(stats)
        results = {k: v for k, v in zip(self.metric_names, metric_scores)}
        results["score"] = score  # add total (weighted) score
        return results

    def apply(
        self, predictions: str, gt_path: Optional[str] = None
    ) -> Any:  # override function
        image_files = self.read_files(predictions, gt_path)
        stats = self.deploy_stats(image_files)
        results = self.get_results(stats)
        return results
