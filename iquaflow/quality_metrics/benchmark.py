import operator
import os
import pickle
from glob import glob
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_eval_metrics() -> Tuple[Dict[str, str], List[str]]:
    metric_tags = {
        "losses": "Loss(BCELoss)",
        "precs": "Precision",
        "recs": "Recall",
        "accs": "Accuracy",
        "fscores": "Fscore",
        "medRs": "medR",
        "Rk1s": "Recall@1",
        "Rk5s": "Recall@5",
        "Rk10s": "Recall@10",
        "AUCs": "AUC",
        "precs_k1": "Precision@1",
        "precs_k5": "Precision@5",
        "precs_k10": "Precision@10",
        "recs_k1": "Recall@1",
        "recs_k5": "Recall@5",
        "recs_k10": "Recall@10",
        "accs_k1": "Accuracy@1",
        "accs_k5": "Accuracy@5",
        "accs_k10": "Accuracy@10",
    }
    metric_list = list(metric_tags.keys())
    return metric_tags, metric_list


def plot_1d(
    values: Any,
    name: str,
    path: str,
    axes: List[str],
    labels: List[str],
    fsize: Tuple[int, int] = (5, 5),
    plot_type: str = "plot",
    limit: int = 20,
) -> None:
    plt.figure(figsize=fsize)
    colors = [
        "blue",
        "orange",
        "red",
        "green",
        "gold",
        "darkviolet",
        "cyan",
        "grey",
        "black",
        "lime",
        "magenta",
    ]
    # exit if values is empty or nan
    if len(values) == 0:
        return None
    """  # if results are nan, discard plot
    if np.isnan(np.nanmin(np.vstack(np.array(values)))) or np.isnan(
        np.nanmax(np.vstack(np.array(values)))
    ):
        return None
    """
    # limit and colors
    values[limit:] = []
    labels[limit:] = []
    labelsdiff = len(labels) - len(colors)
    for num in range(labelsdiff):
        colors.append(colors[-(len(colors) - num)])
    # plot
    figs = []
    if plot_type == "bar":
        plt.bar(np.arange(len(values)), values, color=colors)
        plt.xticks(range(0, len(values)), labels, rotation=10, fontsize=6)
        plt.ylim(
            (
                np.nanmin(values) - np.nanstd(values),
                np.nanmax(values) + np.nanstd(values),
            )
        )
    elif plot_type == "boxplot":
        fig = plt.boxplot(values)
        plt.xticks(range(0, len(values)), labels, rotation=10, fontsize=6)
    else:  # plot
        for idx, value in enumerate(values):
            (fig,) = plt.plot(value, color=colors[idx])
            figs.append(fig)
        plt.legend(handles=figs, labels=labels)
        plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    # save figure and clean plt
    plot_name = path + "/" + name + ".png"
    plt.savefig(plot_name)
    plt.clf()


def get_topk(vals: Any, k: int = 3, reverse: bool = True) -> List[Any]:
    if k > len(vals):
        k = len(vals)
    if reverse is True:
        bestvals = [np.max(row) for row in vals]
        indexed = list(enumerate(bestvals))  # attach indices to the list
        top_k = sorted(indexed, key=operator.itemgetter(1))[-k:]
        return list(reversed([i for i, v in top_k]))  # max first, min last
    else:
        bestvals = [np.min(row) for row in vals]
        indexed = list(enumerate(bestvals))  # attach indices to the list
        top_k = sorted(indexed, key=operator.itemgetter(1))[-k:]
        return list([i for i, v in top_k])  # min first, max last


def get_values_last(vals: Any, epochs: int = 10) -> Tuple[Any, Any]:
    # filter last 10 epochs values
    vals_last = [
        row[-epochs : len(row)] if hasattr(row, "__len__") else row for row in vals
    ]
    # get mean of last values
    vals_last_mean = [np.mean(row) for row in vals_last]
    return vals_last, vals_last_mean


def get_values_top(vals: Any, tags: Any, k: int = 11) -> Tuple[Any, Any]:
    top_k_idx = get_topk(vals, k)
    vals_top = [vals[idx] for idx in top_k_idx]
    vals_tags_top = [
        tags[idx] if len(tags) >= len(top_k_idx) else tags for idx in top_k_idx
    ]
    return vals_top, vals_tags_top


def formatvals(vals: Any, tostr: bool = True, round_decimals: int = 5) -> List[Any]:
    newvals = []
    for val in vals:
        if tostr:
            newval = str(val)  # str(round(val,round_decimals))
        else:
            newval = val  # round(val,round_decimals)
        newvals.append(newval)
    return newvals


def plot_parameter(
    parameter_name: str, train_values: Any, val_values: Any, tags: Any, output_path: str
) -> None:
    plot_1d(
        train_values,
        "train_" + parameter_name,
        output_path,
        ["epoch", parameter_name],
        tags,
        (10, 10),
        "plot",
        len(train_values),
    )
    plot_1d(
        val_values,
        "val_" + parameter_name,
        output_path,
        ["epoch", parameter_name],
        tags,
        (10, 10),
        "plot",
        len(val_values),
    )
    # filter last 10 epochs values
    train_values_last, train_values_last_mean = get_values_last(train_values, 10)
    val_values_last, val_values_last_mean = get_values_last(val_values, 10)
    # filter top k (avoid large benchmark plot comparison)
    train_values_top, train_tags_top = get_values_top(train_values_last, tags, 11)
    val_values_top, val_tags_top = get_values_top(val_values_last, tags, 11)

    print(f"Plotting {parameter_name}")
    # Boxplots per epoch
    plot_1d(
        train_values_top,
        "box_train_" + parameter_name,
        output_path,
        ["epoch", parameter_name],
        train_tags_top,
        (10, 10),
        "boxplot",
        len(train_values_top),
    )
    plot_1d(
        val_values_top,
        "box_val_" + parameter_name,
        output_path,
        ["epoch", parameter_name],
        val_tags_top,
        (10, 10),
        "boxplot",
        len(val_values_top),
    )
    # Barplots per epoch
    plot_1d(
        train_values_last_mean,
        "bar_train_" + parameter_name,
        output_path,
        ["epoch", parameter_name],
        tags,
        (10, 10),
        "bar",
        len(train_values_last_mean),
    )
    plot_1d(
        val_values_last_mean,
        "bar_val_" + parameter_name,
        output_path,
        ["epoch", parameter_name],
        tags,
        (10, 10),
        "bar",
        len(val_values_last_mean),
    )


def plot_benchmark_whole_pkl(pkl_files: List[str], output_path_root: str) -> None:
    experiment_tags: Dict[str, Any] = {}
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            print("loading on benchmark: " + pkl_file)
            data = pickle.load(f)
            tag = os.path.basename(os.path.dirname(pkl_file))
            experiment_tags[tag] = {}
            experiment_tags[tag]["train"] = data[0]
            experiment_tags[tag]["val"] = data[1]
    experiment_list = list(experiment_tags.keys())
    metric_tags, metric_list = get_eval_metrics()

    df_train = pd.DataFrame(index=experiment_list, columns=metric_list)
    df_val = pd.DataFrame(index=experiment_list, columns=metric_list)
    for metric in metric_list:
        train_values = [
            experiment_tags[tag]["train"][metric] for tag in experiment_list
        ]
        val_values = [experiment_tags[tag]["val"][metric] for tag in experiment_list]
        # get unique values
        train_values_last, train_values_last_mean = get_values_last(train_values, 10)
        val_values_last, val_values_last_mean = get_values_last(val_values, 10)
        # to dataframe
        df_train[metric] = train_values_last_mean
        df_val[metric] = val_values_last_mean
    df_train.to_csv(os.path.join(output_path_root, "df_train.csv"))
    df_val.to_csv(os.path.join(output_path_root, "df_val.csv"))
    print("Wrote df_train.csv and df_val.csv")
    """
    for metric in metric_list:
        tags_idx = [
            idx
            for idx, tag in enumerate(experiment_list)
            if metric in list(experiment_tags[tag]["train"].keys())
            and metric in list(experiment_tags[tag]["val"].keys())
        ]
        selected_tags = list(np.asarray(experiment_list)[tags_idx])
        train_values = [experiment_tags[tag]["train"][metric] for tag in selected_tags]
        val_values = [experiment_tags[tag]["val"][metric] for tag in selected_tags]
        plot_parameter(
            parameter_name=metric_tags[metric],
            train_values=train_values,
            val_values=val_values,
            tags=selected_tags,
            output_path=output_path_root,
        )

    """


def write_pkl_from_csv(csv_file: str) -> str:
    print("Writing pkl from " + csv_file)
    train_dict_results_whole = {}
    val_dict_results_whole = {}
    folder = os.path.dirname(csv_file)
    results_array = np.loadtxt(csv_file, delimiter=",")
    metric_tags, metric_list = get_eval_metrics()
    for idx in range(0, len(metric_list) * 2, 2):
        if len(results_array) > idx + 1:
            metric_idx = int(idx * 0.5)  # every 2 so divide by 2 and floor
            metric = metric_list[metric_idx]
            train_dict_results_whole[metric] = results_array[idx]
            val_dict_results_whole[metric] = results_array[idx + 1]
    # write pkl
    results_pkl = os.path.join(folder, "results.pkl")
    with open(results_pkl, "wb") as f:
        pickle.dump([train_dict_results_whole, val_dict_results_whole], f)
    return results_pkl


def plot_single(output_path: str) -> None:
    pkl_file = os.path.join(output_path, "results.pkl")
    csv_file = os.path.join(output_path, "stats.csv")
    if not os.path.exists(pkl_file):
        pkl_file = write_pkl_from_csv(csv_file)
    pkl_files = [pkl_file]
    plot_benchmark_whole_pkl(pkl_files, output_path)


def plot_benchmark(output_path_root: str = "tmp-/") -> None:
    csv_files = glob(os.path.join(output_path_root, "*/") + "stats.csv", recursive=True)
    pkl_files = []
    for csv_file in csv_files:
        folder = os.path.dirname(csv_file)
        pkl_file = os.path.join(folder, "results.pkl")
        if not os.path.exists(csv_file) and not os.path.exists(pkl_file):
            pass
        elif not os.path.exists(pkl_file):
            pkl_file = write_pkl_from_csv(csv_file)
        pkl_files.append(pkl_file)
    plot_benchmark_whole_pkl(pkl_files, output_path_root)


if __name__ == "__main__":
    plot_benchmark("tmp-/")
