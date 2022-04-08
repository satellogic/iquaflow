import errno
import glob
import json
import numbers
import os
import random
import subprocess
import tarfile
import tempfile
from typing import Any, Dict, List, Union

import mlflow
import sagemaker

from iquaflow.aws_utils import download_s3_folder, treat_path_string_start
from iquaflow.datasets import DSWrapper


def get_hash() -> str:
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    hash = "".join([random.choice(characters) for _ in range(4)])
    hash += "_"
    hash += "".join([random.choice(characters) for _ in range(4)])
    return hash


class TaskExecution:
    """
    This Object is the abstract class for executing a black box ML model. Essentialy, it wraps the black box ML model
    and make it run under a MLFlow experiment context. Additionaly, it logs the model outputs as MLFlow prams, metrics and artifacts.
    Concrete implementations of this class has knowledge of how executiing thne training of the model depending the framework is being used.

    Attributes:
        experiment_name: str. Name of the experiment
        run_name: str.Name of the run
    """

    def __init__(self, tmp_dir: Union[str, None] = None) -> None:
        self.experiment_name = ""
        self.run_name = ""
        self.tmp_dir = tmp_dir

        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)

    def generate_experiment(self, experiment_name: str, run_name: str) -> None:
        """Generates MLFlow experiment
        Creates an MLflow experiment if it does not exist.

        Args:
        experiment_name: str. Name of the experiment
        run_name: str.Name of the run
        """
        self.experiment_name = experiment_name
        e = mlflow.get_experiment_by_name(self.experiment_name)
        if e is None:
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)
        self.run_name = run_name

    def train_val(
        self,
        experiment_name: str,
        run_name: str,
        train_ds: Union[str, DSWrapper],
        val_ds: Union[str, DSWrapper] = "",
        mlargs: Dict[str, Any] = {},
        mlflow_monitoring: bool = False,
    ) -> None:
        """ML Training
        Executes the ML model under a MLflow context and providing a temporary output folder for temporal results.
        Afeter the model is trained results are logged in MLflow

        Args:
        experiment_name: str. Name of the experiment
        run_name: str.Name of the run
        train_ds: str of dataset path or DSWrapper. Training dataset
        val_ds: str of dataset path, DSWrappe. Optional. Validation dataset
        mlargs: named parameters. Optional. Additional training parameters
        """
        self.mlflow_monitoring = mlflow_monitoring

        self.generate_experiment(experiment_name, run_name)
        with tempfile.TemporaryDirectory() as output_path:
            if self.tmp_dir:
                output_path = os.path.join(self.tmp_dir, get_hash())
            os.makedirs(output_path, exist_ok=True)
            with mlflow.start_run(run_name=self.run_name) as run:
                train_ds_path = self._log_dataset_info(train_ds)
                val_ds_path = self._log_dataset_info(val_ds, sufix="_val")
                if "testds" in mlargs:
                    mlargs["testds"] = self._log_dataset_info(
                        mlargs["testds"], sufix="_test"
                    )
                if self.mlflow_monitoring:
                    mlargs["mlfuri"] = mlflow.get_tracking_uri()
                    mlargs["mlfexpid"] = run.info.experiment_id
                    mlargs["mlfrunid"] = run.info.run_id
                self._execute(train_ds_path, val_ds_path, output_path, mlargs=mlargs)
                self._gather_results(output_path)

    def _log_dataset_info(self, ds: Union[str, DSWrapper], sufix: str = "") -> str:
        """Log Dataset
        It logs the dataset parameters if it is instance of DSWrapper

        Args:
            ds: str of dataset path or DSWrapper
            sufix: str to add in the keynames (used in val_ds so that is different than tran_ds keys)

        Returns:
            The path of the dataset
        """
        if isinstance(ds, DSWrapper):
            ds_json_dict = ds.log_parameters()
            self.log_json(ds_json_dict, sufix=sufix)
            return ds.data_path
        else:
            return ds

    def _execute(
        self,
        train_ds: str,
        val_ds: str,
        output_path: str,
        mlargs: Dict[str, Any] = {},
    ) -> None:
        """Executes the Ml model

        This method should be extended for any concrete implementation of the class. Essentially it executes the ML model.

        Args:
        train_ds: str of dataset path or DSWrapper. Training dataset
        val_ds: str of dataset path, DSWrappe. Optional. Validation dataset
        output_path: str. Temporary path where the results are placed
        mlargs: named parameters. Optional. Additional training parameters
        """
        pass

    def _gather_results(self, path: str) -> None:
        """Data gathering
        It gathers ML model output and log the information to MLFlow

        Args:
        path: str where the outputs are stored
        """
        self.gather_json_values(path)
        self.gather_artifacts(path)

    def gather_json_values(self, path: str) -> None:
        """
        Reads all json files located in the path and logs them to MLFlow

        Args;
        path: str where the outputs are stored
        """
        json_files = glob.glob(os.path.join(path, "*.json"))
        for json_path in json_files:
            if os.path.basename(json_path) == "results.json":
                with open(json_path, "r") as j:
                    json_dict = json.loads(j.read())
                self.log_json(json_dict)

    def gather_artifacts(self, path: str) -> None:
        """
        Logs all artifacts to MLflow
        """
        mlflow.log_artifacts(path)

    def log_json(self, json_dict: Dict[str, Any], sufix: str = "") -> None:
        """
        Read each key and value from the json. Then log each one as a parameter or a metric. If values is a number it logs it as a parameter.
        Otherwise, if the value is a list, it logs it a s metric.

        Args:
        json_dict: dict
        sufix: str to add in the keynames (used in val_ds so that is different than tran_ds keys)
        """

        for k, v in json_dict.items():
            if isinstance(v, numbers.Number) or isinstance(v, str):
                mlflow.log_param(k + sufix, v)
            else:
                if isinstance(v, list) and len(v) >= 1:
                    first_element = v[0]
                    if isinstance(first_element, list) or isinstance(
                        first_element, tuple
                    ):
                        for step, value in v:
                            mlflow.log_metric(key=k + sufix, value=value, step=step)
                    else:
                        for step, value in enumerate(v):
                            mlflow.log_metric(key=k + sufix, value=value, step=step)


class PythonScriptTaskExecution(TaskExecution):
    """
    This concrete class extends from TaskExecution. This implementation knows how to start a training of a ML model that is in a Python script.

    Args:
        model_script_path: str. Path of the Python script
        tmp_dir: str. Set it to change a new temporary folder. Default is None which will point to the /tmp directory.

    Attributes:
        model_script_path: str. Path of the Python script
        python_script_format: str. Command to execute the Python script training method
    """

    def __init__(self, model_script_path: str, tmp_dir: Union[str, None] = None):
        TaskExecution.__init__(self, tmp_dir)
        if not os.path.isfile(model_script_path):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), model_script_path
            )
        self.model_script_path = model_script_path
        self.python_script_format = "python {} --trainds {} --outputpath {} {}"

    def _exec_subprocess(self, script: str) -> Any:

        p = subprocess.Popen(
            script,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        if p.stdout is not None:
            for stdout_line in iter(p.stdout.readline, ""):
                yield stdout_line
            p.stdout.close()

        return_code = p.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, script)

    def _execute(
        self,
        train_ds: str,
        val_ds: str,
        output_path: str,
        mlargs: Dict[str, Any] = {},
    ) -> None:
        if val_ds != "":
            mlargs.update({"valds": val_ds})
        script = self.python_script_format.format(
            self.model_script_path,
            train_ds,
            output_path,
            " ".join(self._transform_arguments(mlargs)),
        )

        # for line in self._exec_subprocess(script):
        #     print(line, end="")
        print(script)
        process = subprocess.Popen(
            script, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        stdout, stderr = process.communicate()

        print("****** IQF subprocess --stdout-- *********")
        print(stdout.decode())
        print("****** IQF subprocess --stderr-- *********")
        print(stderr.decode())

    def _transform_arguments(self, mlargs: Dict[str, Any] = {}) -> List[str]:
        return ["--{} {}".format(k, v) for k, v in mlargs.items()]


class SageMakerConfig:
    """
    This wrapps a sagemaker session, bucket and role.
    """

    def __init__(self) -> None:

        try:

            self.sagemaker_session = sagemaker.Session()
            self.bucket = self.sagemaker_session.default_bucket()
            self.role = sagemaker.get_execution_role()

        except Exception as e:

            print(f"{e}")
            print(
                "WARNING > You are not in SageMaker environment. You should execute this in the cloud "
            )
            self.sagemaker_session = None
            self.bucket = None
            self.role = None

        self.vpc_config = {
            "SecurityGroupIds": [
                "sg-123456abc",
            ],
            "Subnets": [
                "subnet-123456abc",
            ],
        }


class SageMakerEstimatorFactory:
    """
    Objects of this can generate Sagemaker Estimators.

    Args:
        estimator_instancer: An estimator class, can also be Pytorch from sagemaker.pytorch or similar
        args_dict: Dict[str,Any]. are all the arguments needed to construct the estimator
    """

    def __init__(self, estimator_instancer: Any, args_dict: Dict[str, Any]) -> None:
        self.estimator_instancer = estimator_instancer
        self.args_dict = args_dict

    def gen_estimator(self, hyperparameters: Dict[str, Any]) -> Any:

        self.args_dict["hyperparameters"] = hyperparameters

        return self.estimator_instancer(**self.args_dict)


class SageMakerTaskExecution(TaskExecution):
    """
    This concrete class extends from TaskExecution.
    This implementation knows how to start a training of a ML model that is wrapped with a SageMaker Estimator.

    Args:
        estimator_factory: SageMakerEstimatorFactory. Used to construct a new Estimator every time the task is executed with new hyperparameters.
        push_image_to_ecr: bool. Whether if the docker image is pushed from locally available.

    Attributes:
        estimator_factory: SageMakerEstimatorFactory. Same as argument.
        push_image_to_ecr: bool. Same as argument.
        sagemaker_session: Sagemaker.Session. sagemaker_session. This is retrived by calling the SageMakerConfig().
        bucket: boto3.bucket. bucket. This is retrived by calling the SageMakerConfig().
        role: role. role. This is retrived by calling the SageMakerConfig().
        vpc_config: Dict. vpc_config. This is retrived by calling the SageMakerConfig().
    """

    def __init__(
        self,
        estimator_factory: SageMakerEstimatorFactory,
        exec_locally: bool = False,
        push_image_to_ecr: bool = False,
        tmp_dir: Union[str, None] = None,
    ) -> None:

        self.exec_locally = exec_locally

        TaskExecution.__init__(self, tmp_dir)
        self.estimator_factory = estimator_factory
        self.push_image_to_ecr = push_image_to_ecr

        smc = SageMakerConfig()
        self.sagemaker_session = smc.sagemaker_session
        self.bucket = smc.bucket
        self.role = smc.role
        self.vpc_config = smc.vpc_config

        if push_image_to_ecr:
            print("WARNING: push_image_to_ecr is not yet supported")
            # mlflow.sagemaker.push_image_to_ecr(image=image_name)

    def _execute(
        self,
        train_ds: str,
        val_ds: str,
        output_path: str,
        mlargs: Dict[str, Any] = {},
    ) -> None:
        """
        "SM_CHANNEL_TRAINDS :",train_s3_path
        "SM_CHANNEL_VALDS :",val_s3_path
        "SM_OUTPUT_DATA_DIR :", out_s3_path
        """

        hyp_dict = {
            "trainds": train_ds,
            "valds": val_ds,
            "outputpath": output_path,
        }

        # s3 routes
        train_s3_path, val_s3_path, out_s3_path = [
            os.path.join("s3://" + self.bucket, treat_path_string_start(data_path))
            for data_path in [
                train_ds,
                val_ds,
                output_path,
            ]
        ]

        inputs_dict = {"trainds": train_s3_path}

        if val_ds == "":
            del hyp_dict["valds"]
        else:
            inputs_dict["valds"] = val_s3_path

        if self.exec_locally:
            hyp_dict = {**hyp_dict, **mlargs}
        else:
            # in this case the inputs are s3:// in the estimator.fit()
            hyp_dict = mlargs

        self.estimator_factory.args_dict["output_path"] = out_s3_path

        sagemaker_estimator = self.estimator_factory.gen_estimator(hyp_dict)

        sagemaker_estimator.fit(inputs=inputs_dict)

        download_s3_folder(
            self.bucket,
            treat_path_string_start(output_path),
            local_dir=output_path,
        )

        tar = tarfile.open(
            glob.glob(os.path.join(output_path, "*/output/output.tar.gz"))[0], "r:gz"
        )

        tar.extractall(path=output_path)

        tar.close()
