import os
import shutil
from typing import Any, Dict, Tuple

import mlflow
import pytest

from iquaflow.experiments.task_execution import (
    PythonScriptTaskExecution,
    SageMakerConfig,
    SageMakerEstimatorFactory,
    SageMakerTaskExecution,
)

current_path = os.path.dirname(os.path.realpath(__file__))
ml_models_path = os.path.join(current_path, "test_ml_models")
python_ml_script_path = os.path.join(ml_models_path, "train.py")
mlruns_path = mlflow.get_tracking_uri().replace("file://", "")

experiment_name = "test"


def get_mlflow_run_info() -> Tuple[str, Dict[str, Any], Dict[str, Any], str]:
    tracking_client = mlflow.tracking.MlflowClient()
    experiment_id = tracking_client.get_experiment_by_name(
        experiment_name
    ).experiment_id
    run_info = tracking_client.list_run_infos(experiment_id)[0]
    run_status = run_info.status
    run_id = run_info.run_id
    run = tracking_client.get_run(run_id)
    params_dict = run.data.params
    metrics_dict = run.data.metrics
    artifact_path = run_info.artifact_uri.replace("file://", "")
    return run_status, params_dict, metrics_dict, artifact_path


def remove_mlruns() -> None:
    if os.path.isdir(mlruns_path):
        shutil.rmtree(mlruns_path)
    prev = "/".join(mlruns_path.split("/")[:-2])
    prev = os.path.join(prev, "mlruns")
    if os.path.isdir(prev):
        shutil.rmtree(prev)
    os.makedirs(os.path.join(mlruns_path, ".trash"), exist_ok=True)


def get_sagemaker_estimator_args() -> Dict[str, Any]:
    return {
        "entry_point": "training.py",
        "source_dir": "use-case-sr",
        "role": None,
        "framework_version": "1.8.1",
        "py_version": "py3",
        "instance_count": 1,
        "instance_type": "local_gpu",
    }


class TestPythonScriptTaskExecution:
    def test_run_without_training_params(self):
        remove_mlruns()
        task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
        task.train_val(
            experiment_name=experiment_name,
            run_name="test_run",
            train_ds="train_ds",
            val_ds="val_ds",
        )
        run_status, params_dict, metrics_dict, artifact_path = get_mlflow_run_info()
        assert run_status == "FINISHED"
        assert params_dict["train_ds"] == "train_ds"
        remove_mlruns()

    def test_run_with_training_params(self):
        remove_mlruns()
        task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)
        task.train_val(
            experiment_name=experiment_name,
            run_name="test_run",
            train_ds="train_ds",
            val_ds="val_ds",
            mlargs={"lr": 1e-6},
        )
        run_status, params_dict, metrics_dict, artifact_path = get_mlflow_run_info()
        assert run_status == "FINISHED"
        assert params_dict["train_ds"] == "train_ds"
        assert float(params_dict["lr"]) == 1e-6
        remove_mlruns()

    def test_python_script_does_not_exist(self):
        remove_mlruns()
        with pytest.raises(FileNotFoundError):
            PythonScriptTaskExecution(model_script_path="not_exist")
        remove_mlruns()


class TestSageMakerTaskExecution:
    def test_sagemaker_task_exec_attributes(self):
        remove_mlruns()
        smc = SageMakerConfig()
        for attr in ["sagemaker_session", "bucket", "role", "vpc_config"]:
            assert hasattr(smc, attr), f"SageMakerConfig, Missing attribute {attr}"

        from sagemaker.pytorch import PyTorch

        args_dict = get_sagemaker_estimator_args()
        smf = SageMakerEstimatorFactory(PyTorch, args_dict)
        for attr in ["estimator_instancer", "args_dict"]:
            assert hasattr(
                smf, attr
            ), f"SageMakerEstimatorFactory, Missing attribute {attr}"
        assert (
            smf.args_dict == args_dict
        ), "Unexpected args_dict in SageMakerEstimatorFactory"

        task = SageMakerTaskExecution(smf)
        for attr in ["sagemaker_session", "bucket", "role", "vpc_config"]:
            assert hasattr(
                task, attr
            ), f"SageMakerTaskExecution, Missing attribute {attr}"
        for attr in ["estimator_instancer", "args_dict"]:
            assert hasattr(
                task.estimator_factory, attr
            ), f'SageMakerTaskExecution.estimator_factory, Missing attribute "estimator_factory."+{attr}'
        assert (
            task.estimator_factory.args_dict == args_dict
        ), "Unexpected args_dict in SageMakerTaskExecution"
        remove_mlruns()
