import os
import tempfile
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns

from iquaflow.experiments import ExperimentVisual

def first_sample_df() -> pd.DataFrame:
    """
    A basic sample dataframe.
    """
    fmri = sns.load_dataset("fmri")
    df1 = fmri.loc[:, ["timepoint", "event", "signal"]]
    df1["ds_modifier"] = df1["timepoint"]
    df1["min_size"] = df1["event"]
    df1["val_rmse"] = df1["signal"]
    df1 = df1.loc[:, ["ds_modifier", "min_size", "val_rmse"]]
    df1 = df1.groupby(["ds_modifier", "min_size"]).agg({"val_rmse": ["mean", "std"]})
    return df1


def sample_df(option: str) -> pd.DataFrame:
    """
    This is a function that generates sample dataframes.
    """
    if option == "ROC":

        def synthetic_roc(x: List[np.array], rad: float) -> List[float]:
            return [
                (rad - xel ** 2) ** 0.5 + (np.random.rand() - 0.5) * 0.05 for xel in x
            ]

        def gen_df(x: List[np.array], rad: float) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "Precision": x,
                    "Recall": synthetic_roc(x, rad),
                    "ds_modifier": [rad for _ in range(len(x))],
                }
            )

        df1 = pd.concat(
            [
                gen_df([r / 10 for r in range(9)], rad)
                for rad in [1, 1, 1, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
            ]
        ).set_index("ds_modifier")
    elif option == "agg1":
        df1 = first_sample_df()
    elif option == "adapted1":
        df1 = first_sample_df()
        df1 = ExperimentVisual(df1)._adapt_agg(only_mean=True)
    elif option == "agg2":
        fmri = sns.load_dataset("fmri")
        df1 = fmri.loc[:, ["timepoint", "event", "signal"]]
        df1["ds_modifier"] = df1["timepoint"]
        df1["val_rmse"] = df1["signal"]
        df1 = df1.loc[:, ["ds_modifier", "val_rmse"]]
        df1 = df1.groupby(["ds_modifier"]).agg({"val_rmse": ["mean", "std"]})
    else:
        df1 = first_sample_df()
    return df1