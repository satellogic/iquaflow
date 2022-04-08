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
                (rad - xel**2) ** 0.5 + (np.random.rand() - 0.5) * 0.05 for xel in x
            ]

        def gen_df(x: List[np.array], rad: float) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "Precision": x,
                    "Recall": synthetic_roc(x, rad),
                    "rad": [rad for _ in range(len(x))],
                }
            )

        rad_lst = [1, 1, 1, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
        df1 = pd.concat([gen_df([r / 10 for r in range(9)], rad) for rad in rad_lst])
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


# def set_unique_idx(df:pd.DataFrame) -> pd.DataFrame:
#     df['new_index']=[i for i in range(len(df))]
#     return df.set_index('new_index')


class TestExperimentVisual:
    def test_experiment_visual_class(self):

        # Making df1 for testing...
        df1 = sample_df("agg1")

        with tempfile.TemporaryDirectory() as out_path:
            #
            # Other kind of variables to plot (different than std and mean)
            #
            ev = ExperimentVisual(df1, os.path.join(out_path, "lineplot1.png"))
            assert ev.df.equals(
                df1
            ), "dataframe does not correspond to the assigned dataframe"
            assert ev.out_fullfn == os.path.join(
                out_path, "lineplot1.png"
            ), "out_fullfn does not correspond to the assigned out_fullfn"
            ev.visualize(title="title")
            assert (
                os.path.exists(ev.out_fullfn) if ev.out_fullfn else False
            ), "Failed to save lineplot1.png"
            #
            ev = ExperimentVisual(df1, os.path.join(out_path, "bars1.png"))
            ev.visualize(plot_kind="bars", title="title")
            assert (
                os.path.exists(ev.out_fullfn) if ev.out_fullfn else False
            ), "Failed to save bars1.png"
            #
            ev = ExperimentVisual(df1, os.path.join(out_path, "scatter1.png"))
            ev.visualize(plot_kind="scatter", title="title")
            assert (
                os.path.exists(ev.out_fullfn) if ev.out_fullfn else False
            ), "Failed to save scatter1.png"

    def test_experiment_visual_class_single_var(self):

        # Second test with a different dataframe
        df1 = sample_df("agg1")
        df1 = ExperimentVisual(df1)._adapt_agg(
            only_mean=True
        )  # adapt the inner df1 with default var

        with tempfile.TemporaryDirectory() as out_path:

            ev = ExperimentVisual(df1, os.path.join(out_path, "lineplot2.png"))
            assert ev.df.equals(
                df1
            ), "dataframe does not correspond to the assigned dataframe"
            assert ev.out_fullfn == os.path.join(
                out_path, "lineplot2.png"
            ), "out_fullfn does not correspond to the assigned out_fullfn"
            ev.visualize(title="title")
            assert (
                os.path.exists(ev.out_fullfn) if ev.out_fullfn else False
            ), "Failed to save lineplot2.png"
            #
            ev = ExperimentVisual(df1, os.path.join(out_path, "bars2.png"))
            ev.visualize(plot_kind="bars", title="title")
            assert (
                os.path.exists(ev.out_fullfn) if ev.out_fullfn else False
            ), "Failed to save bars2.png"
            #
            ev = ExperimentVisual(df1, os.path.join(out_path, "scatter2.png"))
            ev.visualize(plot_kind="scatter", title="title")
            assert (
                os.path.exists(ev.out_fullfn) if ev.out_fullfn else False
            ), "Failed to save scatter2.png"
            #
            # Test that there is no output when out_fullfilename set to None
            ev = ExperimentVisual(df1)
            ev.visualize()
            assert not ev.out_fullfn, "out_fullfn should be None but it is not"

    def test_experiment_visual_class_std_mean(self):

        # Making df1 for testing...
        df1 = sample_df("agg2")

        with tempfile.TemporaryDirectory() as out_path:

            ev = ExperimentVisual(df1, os.path.join(out_path, "lineplot_meanstd1.png"))
            assert ev.df.equals(
                df1
            ), "dataframe does not correspond to the assigned dataframe"
            assert ev.out_fullfn == os.path.join(
                out_path, "lineplot_meanstd1.png"
            ), "out_fullfn does not correspond to the assigned out_fullfn"
            ev.visualize(title="title", plot_mean_std=True)
            assert (
                os.path.exists(ev.out_fullfn) if ev.out_fullfn else False
            ), "Failed to save lineplot_meanstd1.png"
            #
            ev = ExperimentVisual(df1, os.path.join(out_path, "bars_meanstd1.png"))
            ev.visualize(plot_kind="bars", title="title", plot_mean_std=True)
            assert (
                os.path.exists(ev.out_fullfn) if ev.out_fullfn else False
            ), "Failed to save bars1_meanstd.png"

    def test_experiment_visual_class_precision_recall(self):

        # Making df1 for testing ROC curve
        df1 = sample_df("ROC")

        with tempfile.TemporaryDirectory() as out_path:

            ev = ExperimentVisual(df1, os.path.join(out_path, "ROC.png"))
            assert ev.df.equals(
                df1
            ), "dataframe does not correspond to the assigned dataframe"
            assert ev.out_fullfn == os.path.join(
                out_path, "ROC.png"
            ), "out_fullfn does not correspond to the assigned out_fullfn"
            ev.visualize(
                xvar="Precision", yvar="Recall", legend_var="rad", title="title"
            )
            assert (
                os.path.exists(ev.out_fullfn) if ev.out_fullfn else False
            ), "Failed to save ROC.png"
