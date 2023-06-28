import os
import shutil
import tempfile
from typing import Any, Dict

import mlflow

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


def remove_mlruns() -> None:
    if os.path.isdir(mlruns_path):
        shutil.rmtree(mlruns_path)
    prev = "/".join(mlruns_path.split("/")[:-2])
    prev = os.path.join(prev, "mlruns")
    if os.path.isdir(prev):
        shutil.rmtree(prev)
    os.makedirs(os.path.join(mlruns_path, ".trash"), exist_ok=True)


def get_mlflow_run_info() -> Dict[str, Any]:
    tracking_client = mlflow.tracking.MlflowClient()
    experiment_id = tracking_client.get_experiment_by_name(
        experiment_name
    ).experiment_id
    runs = {}  # type: Dict[str, Any]
    for run_info in tracking_client.list_run_infos(experiment_id):
        run_status = run_info.status
        run_id = run_info.run_id
        run = tracking_client.get_run(run_id)
        run_name = run.data.tags["mlflow.runName"]
        params_dict = run.data.params
        metrics_dict = run.data.metrics
        artifact_path = run_info.artifact_uri.replace("file://", "")

        runs[run_name] = {}
        runs[run_name]["run_status"] = run_status
        runs[run_name]["params_dict"] = params_dict
        runs[run_name]["metrics_dict"] = metrics_dict
        runs[run_name]["artifact_path"] = artifact_path

    return runs


class TestExperimentSetup:
    def test_run(self):
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

        runs = get_mlflow_run_info()
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

    def test_run_multiple_hyperparams(self):
        remove_mlruns()
        ds_wrapper = DSWrapper(data_path=data_path)
        ds_modifiers_list = [DSModifier()]
        task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
        experiment = ExperimentSetup(
            experiment_name=experiment_name,
            task_instance=task,
            ref_dsw_train=ds_wrapper,
            ds_modifiers_list=ds_modifiers_list,
            extra_train_params={"lr": [1e-5, 1e-6]},
        )
        experiment.execute()
        tracking_client = mlflow.tracking.MlflowClient()
        experiment_id = tracking_client.get_experiment_by_name(
            experiment_name
        ).experiment_id
        assert (
            len(tracking_client.list_run_infos(experiment_id)) == 2
        ), "Wrong number of runs, expected 2"
        remove_mlruns()

    def test_run_with_validation_ds(self):
        remove_mlruns()
        ds_modifiers_list = [DSModifier()]
        task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)

        with tempfile.TemporaryDirectory() as out_path:
            vali_data_path = os.path.join(out_path, "valids")
            shutil.copytree(data_path, vali_data_path)

            experiment = ExperimentSetup(
                experiment_name=experiment_name,
                task_instance=task,
                ref_dsw_train=DSWrapper(data_path=data_path),
                ref_dsw_val=DSWrapper(data_path=vali_data_path),
                ds_modifiers_list=ds_modifiers_list,
            )

            experiment.execute()
            experiment_info = ExperimentInfo(experiment_name)
            params_dict = experiment_info.runs["ds_coco_dataset#base_modifier"][
                "params_dict"
            ]
            assert (
                "ds_name_val" in params_dict and "modifier_val" in params_dict
            ), "Missing validation dataset logged parameters in mlflow"

        remove_mlruns()


class TestMLFlowURI:
    def test_mlflow_tracking_registry_uri(self):
        default_tracking_uri = mlflow.get_tracking_uri()
        default_registry_uri = mlflow.get_registry_uri()

        with tempfile.TemporaryDirectory() as out_path:
            python_ml_script_path = os.path.join(out_path, "train.py")
            open(python_ml_script_path, "a").close()  # create empty file
            open(
                os.path.join(out_path, "annotations.json"), "a"
            ).close()  # create empty file

            ds_wrapper = DSWrapper(data_path=out_path)
            task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
            _ = ExperimentSetup(
                experiment_name="experiment_name",
                task_instance=task,
                ref_dsw_train=ds_wrapper,
                ds_modifiers_list=[],
                cloud_options={"tracking_uri": "", "registry_uri": ""},
            )

            assert (
                mlflow.get_tracking_uri() == default_tracking_uri
            ), "Unexpected default tracking_uri"
            assert (
                mlflow.get_registry_uri() == default_registry_uri
            ), "Unexpected default registry_uri"

            tracking_uri = default_tracking_uri
            registry_uri = default_registry_uri
            _ = ExperimentSetup(
                experiment_name="experiment_name",
                task_instance=task,
                ref_dsw_train=ds_wrapper,
                ds_modifiers_list=[],
                cloud_options={
                    "tracking_uri": tracking_uri,
                    "registry_uri": registry_uri,
                },
            )

            assert mlflow.get_tracking_uri() == tracking_uri, "Unexpected tracking_uri"
            assert mlflow.get_registry_uri() == registry_uri, "Unexpected registry_uri"

            tracking_uri = default_tracking_uri
            registry_uri = default_registry_uri
            _ = ExperimentSetup(
                experiment_name="experiment_name",
                task_instance=task,
                ref_dsw_train=ds_wrapper,
                ds_modifiers_list=[],
                cloud_options={
                    "tracking_uri": tracking_uri,
                    "registry_uri": registry_uri,
                },
            )

            assert mlflow.get_tracking_uri() == tracking_uri, "Unexpected tracking_uri"
            assert (
                mlflow.get_registry_uri() == registry_uri
            ), "registry_uri should be the same as tracking_uri when set to default"
