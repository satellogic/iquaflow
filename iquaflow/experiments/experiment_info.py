import json
import os
import tempfile
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd

from iquaflow.aws_utils import download_s3_filename
from iquaflow.metrics import Metric


class ExperimentInfo:
    """
    This objects allows the user to manage the experiment information. It simplifies the access to MLFlow and allows to apply new metrics to previous executed
    experiments.

    Args:
        experiment_name: str. Name of the experiment


    Attributes:
        experiment_name: str. Name of the experiment

    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.get_mlflow_run_info()

    def get_mlflow_run_info(self) -> Dict[str, Any]:
        """
        It gathers the experiment information ina a python dictionary.
        """
        tracking_client = mlflow.tracking.MlflowClient()
        experiment_id = tracking_client.get_experiment_by_name(
            self.experiment_name
        ).experiment_id
        self.experiment_id = experiment_id
        runs = {}  # type: Dict[str, Any]
        for run_info in tracking_client.list_run_infos(experiment_id):
            run_status = run_info.status
            run_id = run_info.run_id
            run = tracking_client.get_run(run_id)
            run_name = run.data.tags["mlflow.runName"]
            params_dict = run.data.params
            metrics_dict = run.data.metrics
            artifact_path = run_info.artifact_uri.replace("file://", "")
            if run_name in runs.keys():
                names_qty = len(
                    [
                        k
                        for k in runs.keys()
                        if k.startswith(run_name)
                        and (len(k) > len(run_name) and k[len(run_name)] == "_")
                    ]
                )
                run_name = run_name + "_{}".format(names_qty)
            runs[run_name] = {}
            runs[run_name]["run_id"] = run_id
            runs[run_name]["run_status"] = run_status
            runs[run_name]["params_dict"] = params_dict
            runs[run_name]["metrics_dict"] = metrics_dict
            runs[run_name]["artifact_path"] = artifact_path
            runs[run_name]["output_pred_path"] = os.path.join(
                artifact_path, "output.json"
            )
        self.runs = runs
        return runs

    def apply_metric_per_run(self, metric_: Metric, gt_path: str) -> Dict[str, Any]:
        """
        Applies a new metric to previously executed experiments.
        
        Args:
            metric_: Metric. metric to be applied
            gt_path: str. ground truth to apply metric
        """
        for k, run in self.runs.items():
            results = metric_.apply(run["output_pred_path"], gt_path)

            with mlflow.start_run(run_id=run["run_id"]):
                for k, v in results.items():
                    mlflow.log_metric(k, v)
            # run.update(results)
        self.get_mlflow_run_info()
        return self.runs

    def apply_metric_per_run_no_mlflow(self, metric_: Metric, gt_path: str) -> Any:
        """
        Applies a new metric to previously executed experiments.
        
        Args:
            metric_: Metric. metric to be applied
            gt_path: str. ground truth to apply metric
        """
        results_run = []
        for k, run in self.runs.items():
            results = metric_.apply(run["output_pred_path"], gt_path)
            results_run.append(results)
            """
            for results_img in results:
                with mlflow.start_run(run_id=run["run_id"]):
                    for u, v in results_img.items():
                        if u is not "path": #warning: patch of a core error, can't log_metric a string
                            mlflow.log_metric(u, v)
            """
            self.runs[k]["metrics_dict"] = results  # this has all image metric results
            # run.update(results)
        # self.get_mlflow_run_info()
        return results_run  # results_run

    def get_df(
        self,
        ds_params: List[str],
        metrics: List[str],
        grouped: Optional[List[str]] = None,
        dropna: Optional[bool] = True,
        fields_to_float_lst: Optional[List[str]] = None,
        fields_to_int_lst: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        rows = []
        for run_name, run_dict in self.runs.items():
            row = []
            if run_dict["run_status"] == "FINISHED":
                row.append(run_name)
                for ds_p in ds_params:
                    value = run_dict["params_dict"].get(ds_p, None)
                    row.append(value)
                for m_p in metrics:
                    value = run_dict["metrics_dict"].get(m_p, None)
                    if not value:
                        value = run_dict["params_dict"].get(m_p, None)
                        value = value or run_dict.get(m_p, None)
                    row.append(value)
            rows.append(row)
        columns = ["name"]
        columns += ["ds_" + ds_p if "ds_" not in ds_p else ds_p for ds_p in ds_params]
        columns += metrics
        df = pd.DataFrame(rows, columns=columns)
        if grouped is not None and len(grouped) > 0:
            grouped = ["ds_" + ds_p if "ds_" not in ds_p else ds_p for ds_p in grouped]
            df = df.groupby(grouped).agg({m: ["mean", "std"] for m in metrics})
        if dropna:
            df = df.dropna()
        if fields_to_float_lst:
            df[fields_to_float_lst] = df[fields_to_float_lst].astype(
                "float", errors="ignore"
            )
        if fields_to_int_lst:
            df[fields_to_int_lst] = df[fields_to_int_lst].astype("int", errors="ignore")
        return df

    def get_results_dict(self) -> Dict[str, List[Any]]:
        """
        It returs a dictionary of the results.json for all runs of the experiment
        """
        results_dict = {"example": [{"ex1": [0]}]}
        for run_name, run_dict in self.runs.items():
            if run_dict["run_status"] == "FINISHED":
                with tempfile.TemporaryDirectory() as tmpdir:
                    if run_dict["output_pred_path"].startswith("s3://"):
                        json_fn = os.path.join(tmpdir, "tmp.json")
                        bucket_fn_pieces = (
                            run_dict["output_pred_path"]
                            .replace("s3://", "")
                            .split(os.sep)
                        )
                        bucket_name = bucket_fn_pieces[0]
                        key_fn = (
                            (os.sep)
                            .join(bucket_fn_pieces[1:])
                            .replace("output.json", "results.json")
                        )
                        download_s3_filename(
                            bucket_name,
                            key_fn,
                            json_fn,
                        )
                    else:
                        json_fn = run_dict["output_pred_path"]
                    with open(json_fn) as json_file:
                        results_dict[run_name] = json.load(json_file)

        if "example" in results_dict:
            del results_dict["example"]

        return results_dict
