import os
import shutil

from iquaflow.datasets import DSModifier, DSWrapper
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.quality_metrics import (
    GaussianBlurMetrics,
    GSDMetrics,
    NoiseSharpnessMetrics,
    RERMetrics,
    SNRMetrics,
)

# PATHS and MOCK script
dataset_name = "ds_inria_dataset"
dataset_images_folder = "images"
current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current_path))  # "/iquaflow-/"
tests_path = os.path.join(root_path, "tests")
ml_models_path = os.path.join(tests_path, "test_ml_models")
mlruns_path = os.path.join(current_path, "mlruns")
base_ds = os.path.join(tests_path, "test_datasets")
ds_path = os.path.join(base_ds, dataset_name)
data_path = os.path.join(ds_path, dataset_images_folder)
mock_model_script_name = "sr.py"
mock_extra_args = {
    "trainds": [ds_path],
    "traindsinput": [data_path],
    "valds": [ds_path],
    "valdsinput": [data_path],
    # "outputpath": [mlruns_path], #directory to put output.json from mock sr.py
}


def remove_mlruns() -> None:
    if os.path.isdir(mlruns_path):
        shutil.rmtree(mlruns_path)
    prev = "/".join(mlruns_path.split("/")[:-2])
    prev = os.path.join(prev, "mlruns")
    if os.path.isdir(prev):
        shutil.rmtree(prev)


# RER CASE
remove_mlruns()
ds_wrapper = DSWrapper(data_path=data_path)
ds_modifiers_list = [DSModifier()]
python_ml_script_path = os.path.join(ml_models_path, mock_model_script_name)
task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
experiment_name = "test_quality_rer"
experiment = ExperimentSetup(
    experiment_name=experiment_name,
    task_instance=task,
    ref_dsw_train=ds_wrapper,
    ds_modifiers_list=ds_modifiers_list,
    ref_dsw_val=ds_wrapper,
    repetitions=1,
    extra_train_params=mock_extra_args,
)
experiment.execute()
experiment_info = ExperimentInfo(experiment_name)
metric = RERMetrics()
results_run = experiment_info.apply_metric_per_run_no_mlflow(
    metric, str(ds_wrapper.json_annotations)
)  # check experiment_info.py:42 if run.data.tags["mlflow.runName"] exists
run_name = "images_short#base_modifier"
run_exp = experiment_info.runs[run_name]
print(run_exp)

# SNR CASE
remove_mlruns()
ds_wrapper = DSWrapper(data_path=data_path)
ds_modifiers_list = [DSModifier()]
python_ml_script_path = os.path.join(ml_models_path, mock_model_script_name)
task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
experiment_name = "test_quality_snr"
experiment = ExperimentSetup(
    experiment_name=experiment_name,
    task_instance=task,
    ref_dsw_train=ds_wrapper,
    ds_modifiers_list=ds_modifiers_list,
    ref_dsw_val=ds_wrapper,
    repetitions=1,
    extra_train_params=mock_extra_args,
)
experiment.execute()
experiment_info = ExperimentInfo(experiment_name)
metric = SNRMetrics()  # type: ignore
results_run = experiment_info.apply_metric_per_run_no_mlflow(
    metric, str(ds_wrapper.json_annotations)
)  # check experiment_info.py:42 if run.data.tags["mlflow.runName"] exists
run_name = "images_short#base_modifier"
run_exp = experiment_info.runs[run_name]
print(run_exp)

# GAUSSIAN BLUR CASE
remove_mlruns()
ds_wrapper = DSWrapper(data_path=data_path)
ds_modifiers_list = [DSModifier()]
python_ml_script_path = os.path.join(ml_models_path, mock_model_script_name)
task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
experiment_name = "test_quality_gaussian"
experiment = ExperimentSetup(
    experiment_name=experiment_name,
    task_instance=task,
    ref_dsw_train=ds_wrapper,
    ds_modifiers_list=ds_modifiers_list,
    ref_dsw_val=ds_wrapper,
    repetitions=1,
    extra_train_params=mock_extra_args,
)
experiment.execute()
experiment_info = ExperimentInfo(experiment_name)
metric = GaussianBlurMetrics()  # type: ignore
results_run = experiment_info.apply_metric_per_run_no_mlflow(
    metric, str(ds_wrapper.json_annotations)
)  # check experiment_info.py:42 if run.data.tags["mlflow.runName"] exists
run_name = "images_short#base_modifier"
run_exp = experiment_info.runs[run_name]
print(run_exp)

# NOISE SHARPNESS CASE
remove_mlruns()
ds_wrapper = DSWrapper(data_path=data_path)
ds_modifiers_list = [DSModifier()]
python_ml_script_path = os.path.join(ml_models_path, mock_model_script_name)
task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
experiment_name = "test_quality_sharpness"
experiment = ExperimentSetup(
    experiment_name=experiment_name,
    task_instance=task,
    ref_dsw_train=ds_wrapper,
    ds_modifiers_list=ds_modifiers_list,
    ref_dsw_val=ds_wrapper,
    repetitions=1,
    extra_train_params=mock_extra_args,
)
experiment.execute()
experiment_info = ExperimentInfo(experiment_name)
metric = NoiseSharpnessMetrics()  # type: ignore
results_run = experiment_info.apply_metric_per_run_no_mlflow(
    metric, str(ds_wrapper.json_annotations)
)  # check experiment_info.py:42 if run.data.tags["mlflow.runName"] exists
run_name = "images_short#base_modifier"
run_exp = experiment_info.runs[run_name]
print(run_exp)

# GSD CASE
remove_mlruns()
ds_wrapper = DSWrapper(data_path=data_path)
ds_modifiers_list = [DSModifier()]
python_ml_script_path = os.path.join(ml_models_path, mock_model_script_name)
task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
experiment_name = "test_quality_gsd"
experiment = ExperimentSetup(
    experiment_name=experiment_name,
    task_instance=task,
    ref_dsw_train=ds_wrapper,
    ds_modifiers_list=ds_modifiers_list,
    ref_dsw_val=ds_wrapper,
    repetitions=1,
    extra_train_params=mock_extra_args,
)
experiment.execute()
experiment_info = ExperimentInfo(experiment_name)
metric = GSDMetrics()  # type: ignore
results_run = experiment_info.apply_metric_per_run_no_mlflow(
    metric, str(ds_wrapper.json_annotations)
)
run_name = "images_short#base_modifier"
run_exp = experiment_info.runs[run_name]
print(run_exp)
remove_mlruns()
