import argparse
import json
import os
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np

detections_path = "coco_detect.json"


def train_epoch(scale: float = 1.0) -> Any:
    rmse = np.random.uniform(0, 1.5) * scale
    return rmse


def eval_epoch(last_train: float, last_f1: float) -> Tuple[float, float]:
    rmse = np.random.uniform(0, last_train)
    f1 = last_f1 * 1.01
    return rmse, f1


def train(
    train_ds: str, val_ds: str, output_path: str, epochs: int, lr: float = 0.01
) -> None:
    np.random.seed(88)
    scale = 0.995
    val_f1 = 0.1
    train_loss = []
    val_loss = []
    f1 = []
    for e in range(epochs):
        train_rmse = train_epoch(scale)
        eval_rmse, val_f1 = eval_epoch(train_rmse, val_f1)
        scale *= scale

        train_loss.append(train_rmse)
        val_loss.append(eval_rmse)
        f1.append(val_f1)

    output_json = {
        "train_rmse": train_loss,
        "val_rmse": val_loss,
        "val_f1": f1,
        "epochs": epochs,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "lr": lr,
    }

    plt.plot(range(epochs), f1)
    plt.title("Test f1")
    plt.xlabel("epochs")
    plt.ylabel("f1")
    plots_path = os.path.join(output_path, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    plt.savefig(os.path.join(plots_path, "f1.png"))

    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(output_json, f)

    # Detections
    current_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(current_path, detections_path)) as json_file:
        detections = json.load(json_file)

    with open(os.path.join(output_path, "output.json"), "w") as f:
        json.dump(detections, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainds")
    parser.add_argument("--valds")
    parser.add_argument("--outputpath")
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--lr", default=0.1)

    args = parser.parse_args()
    train_ds = args.trainds
    val_ds = args.valds
    output_path = args.outputpath
    epochs = int(args.epochs)
    lr = float(args.lr)

    train(train_ds, val_ds, output_path, epochs, lr=float(args.lr))
