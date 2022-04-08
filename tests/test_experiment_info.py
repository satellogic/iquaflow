import os
import shutil

import mlflow
import pandas as pd

from iquaflow.datasets import DSModifier, DSWrapper
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.task_execution import PythonScriptTaskExecution

current_path = os.path.dirname(os.path.realpath(__file__))
ml_models_path = os.path.join(current_path, "test_ml_models")
python_ml_script_path = os.path.join(ml_models_path, "train.py")
mlruns_path = mlflow.get_tracking_uri().replace("file://", "")

base_ds = os.path.join(current_path, "test_datasets")
data_path = os.path.join(base_ds, "ds_coco_dataset")

experiment_name = "test"
rows = [
    [
        "ds_coco_dataset#base_modifier#base_modifier",
        "base_modifier#base_modifier",
        0.11046221254112044,
        0.007616041326960263,
        0.06699120597771174,
    ],
    [
        "ds_coco_dataset#base_modifier",
        "base_modifier",
        0.11046221254112044,
        0.007616041326960263,
        0.06699120597771174,
    ],
]
columns = ["name", "ds_modifier", "val_f1", "val_rmse", "train_rmse"]
df_reference = pd.DataFrame(rows, columns=columns)


def remove_mlruns() -> None:
    if os.path.isdir(mlruns_path):
        shutil.rmtree(mlruns_path)
    prev = "/".join(mlruns_path.split("/")[:-2])
    prev = os.path.join(prev, "mlruns")
    if os.path.isdir(prev):
        shutil.rmtree(prev)
    for modif_dir in os.listdir(data_path):
        if "#" in modif_dir:
            shutil.rmtree(os.path.join(data_path, modif_dir))
    os.makedirs(os.path.join(mlruns_path, ".trash"), exist_ok=True)


class TestExperimentInfo:
    def test_experiment_info_class(self):
        remove_mlruns()
        ds_wrapper = DSWrapper(data_path=data_path)
        ds_modifiers_list = [DSModifier(), DSModifier(DSModifier())]
        task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
        experiment = ExperimentSetup(
            experiment_name=experiment_name,
            task_instance=task,
            ref_dsw_train=ds_wrapper,
            ds_modifiers_list=ds_modifiers_list,
        )
        experiment.execute()

        experiment_info = ExperimentInfo(experiment_name)
        runs = experiment_info.runs

        run_name = "ds_coco_dataset#base_modifier#base_modifier"
        assert runs[run_name]["run_status"] == "FINISHED"
        assert (
            runs[run_name]["params_dict"]["ds_name"]
            == "ds_coco_dataset#base_modifier#base_modifier"
        )
        assert (
            runs[run_name]["params_dict"]["modifier"] == "base_modifier#base_modifier"
        )

        run_name = "ds_coco_dataset#base_modifier"
        assert runs[run_name]["run_status"] == "FINISHED"
        assert (
            runs[run_name]["params_dict"]["ds_name"] == "ds_coco_dataset#base_modifier"
        )
        assert runs[run_name]["params_dict"]["modifier"] == "base_modifier"
        remove_mlruns()

    def test_experiment_info_dataframe(self):
        remove_mlruns()
        ds_wrapper = DSWrapper(data_path=data_path)
        ds_modifiers_list = [DSModifier(), DSModifier(DSModifier())]
        task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
        experiment = ExperimentSetup(
            experiment_name=experiment_name,
            task_instance=task,
            ref_dsw_train=ds_wrapper,
            ds_modifiers_list=ds_modifiers_list,
        )
        experiment.execute()

        experiment_info = ExperimentInfo(experiment_name)
        df = experiment_info.get_df(
            ds_params=["modifier"],
            metrics=["val_f1", "val_rmse", "train_rmse"],
            fields_to_float_lst=["val_f1", "val_rmse", "train_rmse"],
        )

        assert df_reference.equals(df)
        remove_mlruns()

    def test_experiment_info_grouped_dataframe(self):
        remove_mlruns()
        ds_wrapper = DSWrapper(data_path=data_path)
        ds_modifiers_list = [DSModifier(), DSModifier(DSModifier())]
        task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
        experiment = ExperimentSetup(
            experiment_name=experiment_name,
            task_instance=task,
            ref_dsw_train=ds_wrapper,
            ds_modifiers_list=ds_modifiers_list,
            repetitions=8,
        )
        experiment.execute()

        experiment_info = ExperimentInfo(experiment_name)
        df = experiment_info.get_df(
            ds_params=["modifier"],
            metrics=["val_f1", "val_rmse", "train_rmse"],
            grouped=["ds_modifier"],
        )

        df_reference = pd.DataFrame(rows, columns=columns)
        df_reference = pd.concat([df_reference] * 8)
        df_reference = df_reference.groupby("ds_modifier").agg(
            {
                "val_f1": ["mean", "std"],
                "val_rmse": ["mean", "std"],
                "train_rmse": ["mean", "std"],
            }
        )
        assert df_reference.equals(df)
        remove_mlruns()
