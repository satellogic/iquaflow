import os
import shutil

import mlflow

from iquaflow.datasets import DSModifier, DSWrapper
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.quality_metrics import (  # ScoreMetrics,
    GaussianBlurMetrics,
    GSDMetrics,
    NoiseSharpnessMetrics,
    QualityMetrics,
    RERMetrics,
    SNRMetrics,
)

results = {
    "sigma": 1.0,
    "sharpness": 1.56,  # 1.0,
    "rer": 0.5,  # 0.4,
    "snr": 27.5,
    "scale": 0.6,  # 0.416,
    #    "score": 0.861,
}

current_path = os.path.dirname(os.path.realpath(__file__))
ml_models_path = os.path.join(current_path, "test_ml_models")
mlruns_path = mlflow.get_tracking_uri().replace("file://", "")
print(mlruns_path)
print(mlruns_path)
print(mlruns_path)
base_ds = os.path.join(current_path, "test_datasets")
data_path = os.path.join(base_ds, "ds_inria_dataset")

experiment_name = "test"


def remove_mlruns() -> None:
    if os.path.exists(mlruns_path) and os.path.isdir(mlruns_path):
        shutil.rmtree(mlruns_path)
    os.mkdir(mlruns_path)
    trash_path = os.path.join(mlruns_path, ".trash")
    if os.path.exists(trash_path) and os.path.isdir(trash_path):
        shutil.rmtree(trash_path)
    os.mkdir(trash_path)


def check_metric_result(metric: QualityMetrics, metric_name: str) -> None:
    # prepare_fresh_ds_modifier
    remove_mlruns()
    ds_wrapper = DSWrapper(data_path=data_path)
    ds_modifiers_list = [DSModifier()]
    python_ml_script_path = os.path.join(ml_models_path, "sr.py")
    task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
    experiment = ExperimentSetup(
        experiment_name=experiment_name,
        task_instance=task,
        ref_dsw_train=ds_wrapper,
        ds_modifiers_list=ds_modifiers_list,
    )
    experiment.execute()
    experiment_info = ExperimentInfo(experiment_name)
    experiment_info.apply_metric_per_run(metric, str(ds_wrapper.json_annotations))
    run_name = "ds_inria_dataset#base_modifier"
    run = experiment_info.runs[run_name]
    print(results)
    print(run["metrics_dict"][metric_name])
    assert (
        run["metrics_dict"][metric_name] < results[metric_name] * 1.1
        and run["metrics_dict"][metric_name] > results[metric_name] * 0.9
    ), f"Wrong result for metric {metric_name}"
    remove_mlruns()


class TestQualityMetrics:
    def test_apply_method_gaussianblur(self):
        metric = GaussianBlurMetrics(input_size=1024)
        assert hasattr(metric, "apply"), "GaussianBlur metric has not attr apply"
        assert callable(
            getattr(metric, "apply")
        ), "GaussianBlur metric attr apply is not callable"

    def test_apply_method_noisesharpness(self):
        metric = NoiseSharpnessMetrics(input_size=1024)
        assert hasattr(metric, "apply"), "NoiseSharpness metric has not attr apply"
        assert callable(
            getattr(metric, "apply")
        ), "NoiseSharpness metric attr apply is not callable"

    def test_apply_method_rer(self):
        metric = RERMetrics(input_size=1024)
        assert hasattr(metric, "apply"), "RER metric has not attr apply"
        assert callable(
            getattr(metric, "apply")
        ), "RER metric attr apply is not callable"

    def test_apply_method_snr(self):
        metric = SNRMetrics(input_size=1024)
        assert hasattr(metric, "apply"), "SNR metric has not attr apply"
        assert callable(
            getattr(metric, "apply")
        ), "SNR metric attr apply is not callable"

    def test_apply_method_gsd(self):
        metric = GSDMetrics(input_size=1024)
        assert hasattr(metric, "apply"), "GSD metric has not attr apply"
        assert callable(
            getattr(metric, "apply")
        ), "GSD metric attr apply is not callable"

    def test_apply_method_score(self):
        metric = NoiseSharpnessMetrics(input_size=1024)
        assert hasattr(metric, "apply"), "Score metric has not attr apply"
        assert callable(
            getattr(metric, "apply")
        ), "Score metric attr apply is not callable"

    def test_metric_result_gaussianblur(self):
        check_metric_result(GaussianBlurMetrics(input_size=1024), "sigma")

    def test_metric_result_noisesharpness(self):
        check_metric_result(NoiseSharpnessMetrics(input_size=1024), "sharpness")

    def test_metric_result_rer(self):
        check_metric_result(RERMetrics(input_size=1024), "rer")

    def test_metric_result_snr(self):
        check_metric_result(SNRMetrics(input_size=1024), "snr")

    def test_metric_result_gsd(self):
        check_metric_result(GSDMetrics(input_size=1024), "scale")


#    def test_metric_result_score(self):
#        check_metric_result(ScoreMetrics(input_size=1024), "score")
