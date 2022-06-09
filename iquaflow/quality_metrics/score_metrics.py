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
        metric_names: List[str] = ["sigma", "sharpness", "scale", "snr", "rer"],
        objectives: List[float] = [
            1.0,  # minimize
            1.0,  # n/a
            0.3,  # minimize
            15,  # minimize
            0.55,  # maximize
        ],
        config_filenames: List[str] = [
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "cfgs_example/config_gaussian.cfg",
            ),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "cfgs_example/config_sharpness.cfg",
            ),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "cfgs_example/config_scale.cfg",
            ),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "cfgs_example/config_snr.cfg",
            ),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "cfgs_example/config_rer.cfg",
            ),
        ],
        ranges: List[List[float]] = [
            [1.0, 2.5],
            [1.0, 10.0],
            [15, 30],
            [15, 55],
            [0.15, 0.55],
        ],
        weights: List[float] = [0.2, 0.2, 0.2, 0.2, 0.2],
        default_checkpoint_urls: List[str] = [
            "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/metrics_jun/checkpoint_gaussian_epoch193.pth",
            "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/metrics_jun/checkpoint_sharpness_epoch133.pth",
            "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/metrics_jun/checkpoint_scale.pth",
            "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/metrics_jun/checkpoint_rer_epoch16.pth",
            "https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/models/regressor/metrics_jun/checkpoint_snr_epoch49.pth",
        ],
    ) -> None:
        self.metric_names = metric_names
        self.config_filenames = config_filenames
        self.objectives = objectives
        self.ranges = ranges
        self.weights = weights
        self.default_checkpoint_urls = default_checkpoint_urls
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
            minmax = np.abs(np.max(self.ranges[idx]) - np.min(self.ranges[idx]))
            avg_stats = sum(stats[idx]) / len(stats[idx])
            abs_avg_stats_diff = np.abs(
                self.objectives[idx] - avg_stats
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
                quality_metric = GaussianBlurMetrics(
                    self.config_filenames[idx], self.default_checkpoint_urls[idx]
                )
            elif metric == "sharpness":
                quality_metric = NoiseSharpnessMetrics(
                    self.config_filenames[idx], self.default_checkpoint_urls[idx]
                )
            elif metric == "scale":
                quality_metric = GSDMetrics(
                    self.config_filenames[idx], self.default_checkpoint_urls[idx]
                )
            elif metric == "rer":
                quality_metric = RERMetrics(
                    self.config_filenames[idx], self.default_checkpoint_urls[idx]
                )
            elif metric == "snr":
                quality_metric = SNRMetrics(
                    self.config_filenames[idx], self.default_checkpoint_urls[idx]
                )
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
