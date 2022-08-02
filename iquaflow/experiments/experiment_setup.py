import itertools
from typing import Any, Dict, List, Optional

import mlflow

from iquaflow.aws_utils import treat_path_string_start, upload_objects
from iquaflow.datasets import DSModifier, DSWrapper

from .task_execution import TaskExecution


class ExperimentSetup:
    """
    This object defines the experiment loop. It helps the user to automatize many training executions with differents datasets modifiers

    Args:
        experiment_name: str. Name of the experiment
        task_instance: TaskExecution. Instance of TaskExecution
        ref_dsw_train: DSWrapper. Reference training DSWRapper that the experiment bases on
        ds_modifiers_list: list. List of DSModifier
        ref_dsw_val: DSWrapper. Optional. Reference validation DSWRapper that the experiment bases on
        extra_train_params: dict. Extra training paramters for the ML model
        bucket_name: str. If set, modified data (by IQF-modifiers) will be uploaded to the bucket.
        tracking_uri: str. trackingURI for mlflow. default is local to the ./mlflow folder
        registry_uri: str. registryURI for mlflow. default is local to the ./mlflow folder

    Attributes:
        experiment_name: str. Name of the experiment
        task_instance: TaskExecution. Instance of TaskExecution
        ref_dsw_train: DSWrapper. Reference training DSWRapper that the experiment bases on
        ds_modifiers_list: list. List of DSModifier
        ref_dsw_val: DSWrapper. Optional. Reference validation DSWRapper that the experiment bases on
        extra_train_params: dict. Extra training paramters for the ML model
        cloud_options: dict. This are the options related with the cloud such as sagemaker. Values are:

            - bucket_name: str. If set, modified data (by IQF-modifiers) will be uploaded to the bucket.
            - tracking_uri: str. trackingURI for mlflow. default is local to the ./mlflow folder
            - registry_uri: str. registryURI for mlflow. default is local to the ./mlflow folder
            - upload_num_threads: int. Number of threads used for uploading data to bucket.

    """

    def __init__(
        self,
        experiment_name: str,
        task_instance: TaskExecution,
        ref_dsw_train: DSWrapper,
        ds_modifiers_list: List[DSModifier],
        ref_dsw_val: Optional[DSWrapper] = None,
        ref_dsw_test: Optional[DSWrapper] = None,
        extra_train_params: Dict[str, Any] = {},
        repetitions: int = 1,
        cloud_options: Dict[str, Any] = {},
        mlflow_monitoring: bool = False,
    ) -> None:
        self.experiment_name = experiment_name
        self.task_instance = task_instance
        self.extra_train_params = extra_train_params
        self.ds_modifiers_list = ds_modifiers_list
        self.ref_dsw_train = ref_dsw_train
        self.ref_dsw_val = ref_dsw_val
        self.ref_dsw_test = ref_dsw_test
        self.repetitions = repetitions
        self.mlflow_monitoring = mlflow_monitoring

        for reserved_word in ["testds", "mlfuri", "mlfexpid", "mlfrunid"]:
            assert (
                reserved_word not in extra_train_params
            ), "Attention, {} is a reserved word. It cannot be used as extra_train_params key"

        cloud_options_defaults = {
            "bucket_name": None,
            "tracking_uri": "",
            "registry_uri": "",
            "upload_num_threads": 10,
        }

        cloud_options = {
            k: (cloud_options[k] if k in cloud_options else cloud_options_defaults[k])
            for k in cloud_options_defaults
        }

        self.bucket_name = cloud_options["bucket_name"]
        self.tracking_uri = cloud_options["tracking_uri"]
        self.registry_uri = cloud_options["registry_uri"]
        self.upload_num_threads = cloud_options["upload_num_threads"]

        mlflow.set_tracking_uri(
            (
                mlflow.get_tracking_uri()
                if self.tracking_uri == ""
                else self.tracking_uri
            )
        )

        mlflow.set_registry_uri(
            (
                self.tracking_uri  # copy tracking when set to default
                if self.registry_uri == ""
                else self.registry_uri
            )
        )

    def _to_bucket_if_needed(self, data_path_lst: List[str]) -> None:
        """
        Uploads datasets made from modifiers to the bucket.
        Useful for cases such as when launching the training in SageMaker.

        Args:
            data_path_lst: List[str]. The list of datasets to upload.
        """
        if self.bucket_name:
            for data_path in data_path_lst:
                # remove initial ./
                data_path = treat_path_string_start(data_path)
                upload_objects(
                    bucket_name=self.bucket_name,
                    root_path=data_path,
                    root_pth_bucket=data_path,
                    upload_num_threads=self.upload_num_threads,
                )

    def execute(self) -> None:
        """Experiment execution
        It trains the ML model by using Task Execution for each of the DSModifers of the list
        This includes:

            * Executing the modifiers
            * Uploading data to bucket when needed
            * Executing the TaskExecution
        """
        # Looped once for the executing the modifiers
        # Then give the chance to upload to a bucket
        # Finally loop them again for executing the actual experiments
        for ds_modifier in self.ds_modifiers_list:
            ds_wrapper_modified_train = ds_modifier.modify_ds_wrapper(
                ds_wrapper=self.ref_dsw_train
            )

            to_bucket_if_needed_lst = [ds_wrapper_modified_train.data_path]

            if self.ref_dsw_test is not None:

                ds_wrapper_modified_test = ds_modifier.modify_ds_wrapper(
                    ds_wrapper=self.ref_dsw_test
                )

                to_bucket_if_needed_lst.append(ds_wrapper_modified_test.data_path)

            if self.ref_dsw_val is not None:

                ds_wrapper_modified_val = ds_modifier.modify_ds_wrapper(
                    ds_wrapper=self.ref_dsw_val
                )

                to_bucket_if_needed_lst.append(ds_wrapper_modified_val.data_path)

            self._to_bucket_if_needed(to_bucket_if_needed_lst)

            # Build combinations of argument variations
            combo = list(
                itertools.product(
                    *[self.extra_train_params[key] for key in self.extra_train_params]
                )
            )
            for combo_el in combo:
                curr_dict = {
                    key: combo_el[ii] for ii, key in enumerate(self.extra_train_params)
                }

                if self.ref_dsw_test is not None:
                    curr_dict.update({"testds": ds_wrapper_modified_test.data_path})

                for t in range(self.repetitions):

                    self.task_instance.train_val(
                        experiment_name=self.experiment_name,
                        run_name=ds_wrapper_modified_train.ds_name,
                        train_ds=ds_wrapper_modified_train,
                        val_ds=(
                            ds_wrapper_modified_val
                            if self.ref_dsw_val is not None
                            else ""
                        ),
                        mlargs=curr_dict,
                        mlflow_monitoring=self.mlflow_monitoring,
                    )
