import operator
import os
from glob import glob
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


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
    values[limit:] = []
    labels[limit:] = []
    labelsdiff = len(labels) - len(colors)
    for num in range(labelsdiff):
        colors.append(colors[-(len(colors) - num)])
    figs = []
    if plot_type == "bar":
        plt.bar(np.arange(len(values)), values, color=colors)
        plt.xticks(range(0, len(values)), labels, rotation=10, fontsize=6)
        plt.ylim((np.min(values) - np.std(values), np.max(values) + np.std(values)))
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
    plt.savefig(path + "/" + name + ".png")
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
    )
    plot_1d(
        val_values,
        "val_" + parameter_name,
        output_path,
        ["epoch", parameter_name],
        tags,
        (10, 10),
    )
    # filter last 10 epochs values
    train_values_last = [
        row[-10 : len(row)] if hasattr(row, "__len__") else row for row in train_values
    ]
    val_values_last = [
        row[-10 : len(row)] if hasattr(row, "__len__") else row for row in val_values
    ]
    # get mean of last values
    train_values_last_mean = [np.mean(row) for row in train_values_last]
    val_values_last_mean = [np.mean(row) for row in val_values_last]
    # filter top k (avoid large benchmark plot comparison)
    print(parameter_name)
    train_top_k_idx = get_topk(train_values_last, 11)
    train_values_top = [train_values_last[idx] for idx in train_top_k_idx]
    train_tags_top = [
        tags[idx] if len(tags) >= len(train_top_k_idx) else tags
        for idx in train_top_k_idx
    ]
    val_top_k_idx = get_topk(val_values_last, 11)
    val_values_top = [val_values_last[idx] for idx in val_top_k_idx]
    val_tags_top = [
        tags[idx] if len(tags) >= len(val_top_k_idx) else tags for idx in val_top_k_idx
    ]
    # Boxplots per epoch
    plot_1d(
        train_values_top,
        "box_train_" + parameter_name,
        output_path,
        ["epoch", parameter_name],
        train_tags_top,
        (10, 10),
        "boxplot",
    )
    plot_1d(
        val_values_top,
        "box_val_" + parameter_name,
        output_path,
        ["epoch", parameter_name],
        val_tags_top,
        (10, 10),
        "boxplot",
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
    )
    plot_1d(
        val_values_last_mean,
        "bar_val_" + parameter_name,
        output_path,
        ["epoch", parameter_name],
        tags,
        (10, 10),
        "bar",
    )


def plot_benchmark_whole(csv_files: List[str], output_path_root: str) -> None:
    dict_tags = {}
    train_losses = []
    val_losses = []
    train_precs = []
    val_precs = []
    train_recs = []
    val_recs = []
    train_accs = []
    val_accs = []
    train_fscores = []
    val_fscores = []
    train_medRs = []
    val_medRs = []
    train_recs_k1 = []
    val_recs_k1 = []
    train_recs_k5 = []
    val_recs_k5 = []
    train_recs_k10 = []
    val_recs_k10 = []
    for file in csv_files:
        print("loading on benchmark: " + file)
        tag = os.path.basename(os.path.dirname(file))
        results_array = np.loadtxt(file, delimiter=",")
        if len(results_array) > 1:
            train_losses.append(results_array[0])  # train_losses
            val_losses.append(results_array[1])  # val_losses
            dict_tags[tag] = ["Loss"]
        if len(results_array) > 3:
            train_precs.append(results_array[2])  # train_precs
            val_precs.append(results_array[3])  # val_precs
            dict_tags[tag].append("Precision")
        if len(results_array) > 5:
            train_recs.append(results_array[4])  # train_recs
            val_recs.append(results_array[5])  # val_recs
            dict_tags[tag].append("Recall")
        if len(results_array) > 7:
            train_accs.append(results_array[6])  # train_accs
            val_accs.append(results_array[7])  # val_accs
            dict_tags[tag].append("Accuracy")
        if len(results_array) > 9:
            train_fscores.append(results_array[8])  # train_fscores
            val_fscores.append(results_array[9])  # val_fscores
            dict_tags[tag].append("Fscore")
        if len(results_array) > 11:
            train_medRs.append(results_array[10])  # train_medRs
            val_medRs.append(results_array[11])  # val_medRs
            dict_tags[tag].append("medR")
        if len(results_array) > 13:
            train_recs_k1.append(results_array[12])  # train_recs_k1
            val_recs_k1.append(results_array[13])  # val_recs_k1
            dict_tags[tag].append("R@1")
        if len(results_array) > 15:
            train_recs_k5.append(results_array[14])  # train_recs_k5
            val_recs_k5.append(results_array[15])  # val_recs_k5
            dict_tags[tag].append("R@5")
        if len(results_array) > 17:
            train_recs_k10.append(results_array[16])  # train_recs_k10
            val_recs_k10.append(results_array[17])  # val_recs_k10
            dict_tags[tag].append("R@10")
    tags = list(dict_tags.keys())
    # Line plots per epoch
    if len(train_losses):
        tags_idx = [idx for idx, tag in enumerate(tags) if "Loss" in dict_tags[tag]]
        selected_tags = list(np.asarray(tags)[tags_idx])
        plot_parameter(
            parameter_name="Loss(BCELoss)",
            train_values=train_losses,
            val_values=val_losses,
            tags=selected_tags,
            output_path=output_path_root,
        )

    if len(train_precs):
        tags_idx = [
            idx for idx, tag in enumerate(tags) if "Precision" in dict_tags[tag]
        ]
        selected_tags = list(np.asarray(tags)[tags_idx])
        plot_parameter(
            parameter_name="Precision",
            train_values=train_precs,
            val_values=val_precs,
            tags=selected_tags,
            output_path=output_path_root,
        )

    if len(train_recs):
        tags_idx = [idx for idx, tag in enumerate(tags) if "Recall" in dict_tags[tag]]
        selected_tags = list(np.asarray(tags)[tags_idx])
        plot_parameter(
            parameter_name="Recall",
            train_values=train_recs,
            val_values=val_recs,
            tags=selected_tags,
            output_path=output_path_root,
        )

    if len(train_accs):
        tags_idx = [idx for idx, tag in enumerate(tags) if "Accuracy" in dict_tags[tag]]
        selected_tags = list(np.asarray(tags)[tags_idx])
        plot_parameter(
            parameter_name="Accuracy",
            train_values=train_accs,
            val_values=val_accs,
            tags=selected_tags,
            output_path=output_path_root,
        )

    if len(train_fscores):
        tags_idx = [idx for idx, tag in enumerate(tags) if "Fscore" in dict_tags[tag]]
        selected_tags = list(np.asarray(tags)[tags_idx])
        plot_parameter(
            parameter_name="Fscore",
            train_values=train_fscores,
            val_values=val_fscores,
            tags=selected_tags,
            output_path=output_path_root,
        )
    if len(train_medRs):
        tags_idx = [idx for idx, tag in enumerate(tags) if "medR" in dict_tags[tag]]
        selected_tags = list(np.asarray(tags)[tags_idx])
        plot_parameter(
            parameter_name="medR",
            train_values=train_medRs,
            val_values=val_medRs,
            tags=selected_tags,
            output_path=output_path_root,
        )
    if len(train_recs_k1):
        tags_idx = [idx for idx, tag in enumerate(tags) if "R@1" in dict_tags[tag]]
        selected_tags = list(np.asarray(tags)[tags_idx])
        plot_parameter(
            parameter_name="R@1",
            train_values=train_recs_k1,
            val_values=val_recs_k1,
            tags=selected_tags,
            output_path=output_path_root,
        )
    if len(train_recs_k5):
        tags_idx = [idx for idx, tag in enumerate(tags) if "R@5" in dict_tags[tag]]
        selected_tags = list(np.asarray(tags)[tags_idx])
        plot_parameter(
            parameter_name="R@5",
            train_values=train_recs_k5,
            val_values=val_recs_k5,
            tags=selected_tags,
            output_path=output_path_root,
        )
    if len(train_recs_k10):
        tags_idx = [idx for idx, tag in enumerate(tags) if "R@10" in dict_tags[tag]]
        selected_tags = list(np.asarray(tags)[tags_idx])
        plot_parameter(
            parameter_name="R@10",
            train_values=train_recs_k10,
            val_values=val_recs_k10,
            tags=selected_tags,
            output_path=output_path_root,
        )


def plot_single(output_path: str) -> None:
    csv_files = [output_path + "/" + "stats.csv"]
    plot_benchmark_whole(csv_files, output_path)


def plot_benchmark(output_path_root: str = "tmp-/") -> None:
    csv_files = glob(os.path.join(output_path_root, "*/") + "stats.csv", recursive=True)
    plot_benchmark_whole(csv_files, output_path_root)


if __name__ == "__main__":
    plot_benchmark("tmp-/")
