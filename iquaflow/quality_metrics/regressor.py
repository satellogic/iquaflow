# matplotlib inline
import argparse
import configparser
import glob
import json
import os
import pickle
import shutil
import time
from bisect import bisect_right
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

from iquaflow.datasets import (
    DSModifier,
    DSModifier_blur,
    DSModifier_gsd,
    DSModifier_rer,
    DSModifier_sharpness,
    DSModifier_snr,
)
from iquaflow.quality_metrics.benchmark import plot_benchmark, plot_single
from iquaflow.quality_metrics.dataloader import Dataset
from iquaflow.quality_metrics.tools import (
    MultiHead,
    circ3d_pad,
    create_network,
    force_rgb,
    get_accuracy,
    get_accuracy_k,
    get_AUC,
    get_fscore,
    get_median_rank,
    get_precision,
    get_precision_k,
    get_recall,
    get_recall_k,
    get_recall_rate,
    get_tensor_crop_transform,
    soft2hard,
)


def parse_params_cfg(default_cfg_path: str = "config.cfg") -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "testing_argument", nargs="?", default="tests/test_regressor.py"
    )
    parser.add_argument("--cfg_path", type=str, default=default_cfg_path)
    parser.add_argument("--overwrite_modifiers", default=False, action="store_true")
    parser.add_argument("--overwrite_crops", default=False, action="store_true")
    parser.add_argument("--plot_only", default=False, action="store_true")
    parser.add_argument("--data_only", default=False, action="store_true")
    parser.add_argument("--validate_only", default=False, action="store_true")
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--seed", type=str, default=str(np.random.randint(12345)))
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--save_mat", default=False, action="store_true")
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
    )
    """  # changed parge_args by parse_known_args
    if parser.prog == "pytest":
        parser.add_argument("--mypy", default="tests", type=str)
        parser.add_argument("-vv", default=True, action="store_true")

    if parser.prog == "ipykernel_launcher":
        parser.add_argument("-f", default=True, action="store_true")
    """
    tmp_args, uk_args = parser.parse_known_args()
    cfg_path = tmp_args.cfg_path

    config = configparser.ConfigParser()
    config.read(cfg_path)
    parser.add_argument("--trainid", default=config["RUN"]["trainid"])
    parser.add_argument(
        "--trainds", default=config["PATHS"]["trainds"]
    )  # inria-aid, "xview", "ds_coco_dataset"
    parser.add_argument("--outputpath", default=config["PATHS"]["outputpath"])
    parser.add_argument("--traindsinput", default=config["PATHS"]["traindsinput"])
    parser.add_argument("--valds", default=config["PATHS"]["valds"])
    parser.add_argument("--valdsinput", default=config["PATHS"]["valdsinput"])
    if config.has_option("HYPERPARAMS", "modifier_params"):
        parser.add_argument(
            "--modifier_params",
            default=eval(config["HYPERPARAMS"]["modifier_params"]),
        )  # for numpy commands (e.g. np.linspace(min,max,num_reg))
    else:
        parser.add_argument(
            "--modifier_params",
            default=json.loads(config["HYPERPARAMS"]["modifier_params"]),
        )  # dict format
    parser.add_argument(
        "--num_regs", default=json.loads(config["HYPERPARAMS"]["num_regs"])
    )
    parser.add_argument("--epochs", default=json.loads(config["HYPERPARAMS"]["epochs"]))
    parser.add_argument(
        "--num_crops", default=json.loads(config["HYPERPARAMS"]["num_crops"])
    )
    parser.add_argument("--splits", default=json.loads(config["HYPERPARAMS"]["splits"]))
    parser.add_argument(
        "--input_size", default=json.loads(config["HYPERPARAMS"]["input_size"])
    )
    parser.add_argument(
        "--batch_size", default=json.loads(config["HYPERPARAMS"]["batch_size"])
    )  # samples per batch (* NUM_CROPS * DEFAULT_SIGMAS). (eg. 2*4*4)
    parser.add_argument("--lr", default=json.loads(config["HYPERPARAMS"]["lr"]))
    parser.add_argument(
        "--momentum", default=json.loads(config["HYPERPARAMS"]["momentum"])
    )
    parser.add_argument(
        "--weight_decay", default=json.loads(config["HYPERPARAMS"]["weight_decay"])
    )
    parser.add_argument(
        "--workers", default=json.loads(config["HYPERPARAMS"]["workers"])
    )
    parser.add_argument(
        "--data_shuffle", default=eval(config["HYPERPARAMS"]["data_shuffle"])
    )
    if config.has_option("HYPERPARAMS", "soft_threshold"):
        parser.add_argument(
            "--soft_threshold",
            default=json.loads(config["HYPERPARAMS"]["soft_threshold"]),
        )
    else:
        parser.add_argument(
            "--soft_threshold",
            default=0.3,
        )
    if config.has_option("HYPERPARAMS", "network"):
        parser.add_argument(
            "--network",
            default=json.loads(config["HYPERPARAMS"]["network"]),
        )
    else:
        parser.add_argument(
            "--network",
            default="resnet18",
        )
    if config.has_option("HYPERPARAMS", "pretrained"):
        parser.add_argument(
            "--pretrained",
            default=eval(config["HYPERPARAMS"]["pretrained"]),
        )
    else:
        parser.add_argument(
            "--pretrained",
            default=True,
        )
    tmp_args, uk_args = parser.parse_known_args()
    outputpath = tmp_args.outputpath
    trainid = tmp_args.trainid
    ckpt_folder = os.path.join(outputpath, trainid)
    ckpt_cfg_path = os.path.join(ckpt_folder, os.path.basename(cfg_path))

    if not os.path.exists(outputpath):  # create main output folder
        os.mkdir(outputpath)
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    if not os.path.exists(ckpt_cfg_path):  # new
        shutil.copyfile(cfg_path, ckpt_cfg_path)
    else:
        os.remove(ckpt_cfg_path)  # overwrite old
        shutil.copyfile(cfg_path, ckpt_cfg_path)

    return parser


def get_modifiers_from_params(
    modifier_params: Dict[str, Any],
    parametizable_params: List[str] = ["sigma", "scale", "sharpness", "snr", "rer"],
) -> Tuple[Any, Any]:
    ds_modifiers = []
    if len(modifier_params.items()) == 0:
        ds_modifiers.append(DSModifier())
    else:
        for key, elem in modifier_params.items():
            if key not in parametizable_params:
                continue

            for gidx in range(len(elem)):
                if key == "sigma":
                    ds_modifiers.append(
                        DSModifier_blur(params={key: modifier_params[key][gidx]})
                    )
                elif key == "scale":
                    if (
                        "interpolation" not in modifier_params.keys()
                    ):  # default rescale interpolation
                        interpolation = 2
                    else:
                        interpolation = modifier_params["interpolation"]
                    if (
                        "resol" not in modifier_params.keys()
                    ):  # default rescale interpolation
                        resol = 0.3
                    else:
                        resol = modifier_params["resol"]
                    ds_modifiers.append(
                        DSModifier_gsd(
                            params={
                                key: modifier_params[key][gidx],
                                "interpolation": interpolation,
                                "resol": resol,
                            }
                        )
                    )
                    # restore param to meters per pixel
                    gsd = resol * modifier_params[key][gidx]
                    modifier_params[key][gidx] = gsd
                elif key == "sharpness":
                    ds_modifiers.append(
                        DSModifier_sharpness(params={key: modifier_params[key][gidx]})
                    )
                elif key == "rer":
                    if "dataset" not in modifier_params.keys():
                        modifier = DSModifier_rer(
                            params={key: modifier_params[key][gidx], "dataset": "xview"}
                        )
                    else:
                        modifier = DSModifier_rer(
                            params={
                                key: modifier_params[key][gidx],
                                "dataset": modifier_params["dataset"],
                            }
                        )
                    if modifier.init_RER > modifier_params[key][gidx]:
                        ds_modifiers.append(modifier)
                    # else:
                    #    ds_modifiers.append(DSModifier())
                elif key == "snr":
                    if "dataset" not in modifier_params.keys():
                        modifier = DSModifier_snr(  # type: ignore
                            params={key: modifier_params[key][gidx], "dataset": "xview"}
                        )
                    else:
                        modifier = DSModifier_snr(  # type: ignore
                            params={
                                key: modifier_params[key][gidx],
                                "dataset": modifier_params["dataset"],
                            }
                        )
                    if modifier.init_SNR > modifier_params[key][gidx]:  # type: ignore
                        ds_modifiers.append(modifier)
                    # else:
                    #    ds_modifiers.append(DSModifier())

    return ds_modifiers, modifier_params


def get_regression_interval_classes(
    modifier_params: Dict[str, Any],
    num_regs: List[int],
    parametizable_params: List[str] = ["sigma", "scale", "sharpness", "snr", "rer"],
) -> Dict[str, Any]:
    yclasses = {}
    params = list(modifier_params.keys())
    for idx, param in enumerate(params):
        if param not in parametizable_params:
            continue
        param_items = modifier_params[param]
        if type(param_items) == np.ndarray:
            if num_regs[idx] == len(param_items):
                yclasses[param] = param_items
            else:
                yclasses[param] = np.linspace(
                    np.min(param_items), np.max(param_items), num_regs[idx]
                )
        else:
            yclasses[param] = param_items
    return yclasses


class Regressor:
    def __init__(self, args: Any) -> None:
        self.train_ds = args.trainds
        self.train_ds_input = args.traindsinput
        self.val_ds = args.valds
        self.val_ds_input = args.valdsinput
        self.epochs = int(args.epochs)
        self.validate_only = args.validate_only
        self.lr = float(args.lr)
        self.momentum = float(args.momentum)
        self.weight_decay = float(args.weight_decay)
        self.batch_size = int(args.batch_size)
        self.network = args.network
        self.pretrained = args.pretrained

        # Get Regressor Params from dicts
        self.modifier_params = args.modifier_params
        self.num_regs = list(args.num_regs)
        self.num_crops = args.num_crops  # used only by Dataset

        # Implemented params
        self.parametizable_params: List[str] = [
            "sigma",
            "scale",
            "sharpness",
            "snr",
            "rer",
        ]
        # Define modifiers and regression intervals
        self.ds_modifiers, self.modifier_params = get_modifiers_from_params(
            self.modifier_params, self.parametizable_params
        )  # train case
        # self.ds_modifiers = [DSModifier()] # deploy case
        self.params = [
            key
            for key, value in self.modifier_params.items()
            if (type(value) == np.ndarray) & (key in self.parametizable_params)
        ]  # list(self.modifier_params.keys())
        self.dict_params = {par: idx for idx, par in enumerate(self.params)}
        self.yclasses = get_regression_interval_classes(
            self.modifier_params, self.num_regs, self.parametizable_params
        )
        self.crop_size = args.input_size
        self.tCROP = get_tensor_crop_transform(self.crop_size, "random")
        self.cCROP = get_tensor_crop_transform(self.crop_size, "center")
        # Create Network
        if len(self.params) == 1:  # Single Head
            self.net = create_network(self.network, self.pretrained, self.num_regs[0])
        else:  # MultiHead
            self.net = MultiHead(
                create_network(self.network, self.pretrained), self.num_regs
            )
        # Training HyperParams
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.criterion = torch.nn.BCELoss()
        self.soft_threshold = args.soft_threshold

        # Output Paths
        self.output_path = args.outputpath
        self.train_id = args.trainid
        self.output_path = os.path.join(self.output_path, self.train_id)
        self.checkpoint_name = "checkpoint" + "_epoch" + str(self.epochs) + ".pth"
        self.checkpoint_path = os.path.join(
            self.output_path, self.checkpoint_name
        )  # add join names for params

        # DEBUG Options
        self.debug = args.debug
        self.save_mat = args.save_mat
        if self.debug:
            print(self.net)

        # GPU Options
        self.cuda = args.cuda
        self.gpus = args.gpus
        self.seed = args.seed
        if self.cuda:
            print("=> use gpu id: '{}'".format(self.gpus))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
            if not torch.cuda.is_available():
                raise Exception(
                    "No GPU found or Wrong gpu id, please run without --cuda"
                )
        if self.cuda:
            print("Random Seed: ", self.seed)
            torch.cuda.manual_seed(self.seed)

        if self.cuda:
            self.criterion = self.criterion.cuda()
            self.net = self.net.cuda()

    def train_val(self, train_loader: Any, val_loader: Any) -> Any:
        best_loss = np.inf
        best_prec = 0.0
        best_rec = 0.0
        train_dict_results_whole = {}  # type: ignore
        (
            train_dict_results_whole["losses"],
            train_dict_results_whole["precs"],
            train_dict_results_whole["recs"],
            train_dict_results_whole["accs"],
            train_dict_results_whole["fscores"],
            train_dict_results_whole["medRs"],
            train_dict_results_whole["Rk1s"],
            train_dict_results_whole["Rk5s"],
            train_dict_results_whole["Rk10s"],
            train_dict_results_whole["AUCs"],
            train_dict_results_whole["precs_k1"],
            train_dict_results_whole["precs_k5"],
            train_dict_results_whole["precs_k10"],
            train_dict_results_whole["recs_k1"],
            train_dict_results_whole["recs_k5"],
            train_dict_results_whole["recs_k10"],
            train_dict_results_whole["accs_k1"],
            train_dict_results_whole["accs_k5"],
            train_dict_results_whole["accs_k10"],
        ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
        val_dict_results_whole = {}  # type: ignore
        (
            val_dict_results_whole["losses"],
            val_dict_results_whole["precs"],
            val_dict_results_whole["recs"],
            val_dict_results_whole["accs"],
            val_dict_results_whole["fscores"],
            val_dict_results_whole["medRs"],
            val_dict_results_whole["Rk1s"],
            val_dict_results_whole["Rk5s"],
            val_dict_results_whole["Rk10s"],
            val_dict_results_whole["AUCs"],
            val_dict_results_whole["precs_k1"],
            val_dict_results_whole["precs_k5"],
            val_dict_results_whole["precs_k10"],
            val_dict_results_whole["recs_k1"],
            val_dict_results_whole["recs_k5"],
            val_dict_results_whole["recs_k10"],
            val_dict_results_whole["accs_k1"],
            val_dict_results_whole["accs_k5"],
            val_dict_results_whole["accs_k10"],
        ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])

        for epoch in range(self.epochs):  # epoch
            # TRAIN
            (
                train_dict_results_epoch,
                train_dict_results_batch,
                train_dict_data_batch,
            ) = self.train_val_epoch(train_loader, epoch, False)
            # VALIDATE
            (
                val_dict_results_epoch,
                val_dict_results_batch,
                val_dict_data_batch,
            ) = self.train_val_epoch(val_loader, epoch, True)

            # CHECK BEST EPOCH CHECKPOINT
            if (
                val_dict_results_epoch["epoch_loss"] is None
                and train_dict_results_epoch["epoch_loss"] is None
            ):
                print("Error: Validation and Training losses are None.")
                raise
            is_best = (val_dict_results_epoch["epoch_loss"] < best_loss) & (
                (val_dict_results_epoch["epoch_prec"] > best_prec)
                | (val_dict_results_epoch["epoch_rec"] > best_rec)
            )
            if is_best:
                best_loss = val_dict_results_epoch["epoch_loss"]
                best_prec = val_dict_results_epoch["epoch_prec"]
                best_rec = val_dict_results_epoch["epoch_rec"]
                # remove previous checkpoint
                if os.path.exists(
                    self.checkpoint_path
                ):  # 'checkpoint_path' in dir(self) &
                    os.remove(self.checkpoint_path)
                # save checkpoint
                self.checkpoint_name = "checkpoint" + "_epoch" + str(epoch) + ".pth"
                self.checkpoint_path = os.path.join(
                    self.output_path, self.checkpoint_name
                )
                print("Found best model, saving checkpoint " + self.checkpoint_path)
                torch.save(self.net, self.checkpoint_path)  # self.net.state_dict()
                # SAVE TOP/WORST K
                if self.debug:
                    # copy top K files for that epoch
                    output_path_top = os.path.join(
                        self.output_path, "top_epoch" + str(epoch)
                    )
                    if not os.path.exists(output_path_top):
                        os.mkdir(output_path_top)
                    for idx, filename in enumerate(val_dict_results_epoch["topK"]):
                        file_newpath = os.path.join(
                            output_path_top,
                            "rank"
                            + str(val_dict_results_epoch["topK_ranks"][idx])
                            + "_"
                            + os.path.basename(
                                os.path.dirname(os.path.dirname(filename))
                            )
                            + "_"
                            + os.path.basename(filename),
                        )
                        shutil.copyfile(filename, file_newpath)
                    # copy worst K files for that epoch
                    output_path_worst = os.path.join(
                        self.output_path, "worst_epoch" + str(epoch)
                    )
                    if not os.path.exists(output_path_worst):
                        os.mkdir(output_path_worst)
                    for idx, filename in enumerate(val_dict_results_epoch["worstK"]):
                        file_newpath = os.path.join(
                            output_path_worst,
                            "rank"
                            + str(val_dict_results_epoch["worstK_ranks"][idx])
                            + "_"
                            + os.path.basename(
                                os.path.dirname(os.path.dirname(filename))
                            )
                            + "_"
                            + os.path.basename(filename),
                        )
                        shutil.copyfile(filename, file_newpath)

            # append all results per epoch
            train_dict_results_whole["losses"].append(
                train_dict_results_epoch["epoch_loss"]
            )
            train_dict_results_whole["precs"].append(
                train_dict_results_epoch["epoch_prec"]
            )
            train_dict_results_whole["recs"].append(
                train_dict_results_epoch["epoch_rec"]
            )
            train_dict_results_whole["accs"].append(
                train_dict_results_epoch["epoch_acc"]
            )
            train_dict_results_whole["fscores"].append(
                train_dict_results_epoch["epoch_fscore"]
            )
            train_dict_results_whole["medRs"].append(
                train_dict_results_epoch["epoch_medR"]
            )
            train_dict_results_whole["Rk1s"].append(
                train_dict_results_epoch["epoch_Rk1"]
            )
            train_dict_results_whole["Rk5s"].append(
                train_dict_results_epoch["epoch_Rk5"]
            )
            train_dict_results_whole["Rk10s"].append(
                train_dict_results_epoch["epoch_Rk10"]
            )
            train_dict_results_whole["AUCs"].append(
                train_dict_results_epoch["epoch_AUC"]
            )
            train_dict_results_whole["precs_k1"].append(
                train_dict_results_epoch["epoch_prec_k1"]
            )
            train_dict_results_whole["precs_k5"].append(
                train_dict_results_epoch["epoch_prec_k5"]
            )
            train_dict_results_whole["precs_k10"].append(
                train_dict_results_epoch["epoch_prec_k10"]
            )
            train_dict_results_whole["recs_k1"].append(
                train_dict_results_epoch["epoch_rec_k1"]
            )
            train_dict_results_whole["recs_k5"].append(
                train_dict_results_epoch["epoch_rec_k5"]
            )
            train_dict_results_whole["recs_k10"].append(
                train_dict_results_epoch["epoch_rec_k10"]
            )
            train_dict_results_whole["accs_k1"].append(
                train_dict_results_epoch["epoch_acc_k1"]
            )
            train_dict_results_whole["accs_k5"].append(
                train_dict_results_epoch["epoch_acc_k5"]
            )
            train_dict_results_whole["accs_k10"].append(
                train_dict_results_epoch["epoch_acc_k10"]
            )
            val_dict_results_whole["losses"].append(
                val_dict_results_epoch["epoch_loss"]
            )
            val_dict_results_whole["precs"].append(val_dict_results_epoch["epoch_prec"])
            val_dict_results_whole["recs"].append(val_dict_results_epoch["epoch_rec"])
            val_dict_results_whole["accs"].append(val_dict_results_epoch["epoch_acc"])
            val_dict_results_whole["fscores"].append(
                val_dict_results_epoch["epoch_fscore"]
            )
            val_dict_results_whole["medRs"].append(val_dict_results_epoch["epoch_medR"])
            val_dict_results_whole["Rk1s"].append(val_dict_results_epoch["epoch_Rk1"])
            val_dict_results_whole["Rk5s"].append(val_dict_results_epoch["epoch_Rk5"])
            val_dict_results_whole["Rk10s"].append(val_dict_results_epoch["epoch_Rk10"])
            val_dict_results_whole["AUCs"].append(val_dict_results_epoch["epoch_AUC"])
            val_dict_results_whole["precs_k1"].append(
                val_dict_results_epoch["epoch_prec_k1"]
            )
            val_dict_results_whole["precs_k5"].append(
                val_dict_results_epoch["epoch_prec_k5"]
            )
            val_dict_results_whole["precs_k10"].append(
                val_dict_results_epoch["epoch_prec_k10"]
            )
            val_dict_results_whole["recs_k1"].append(
                val_dict_results_epoch["epoch_rec_k1"]
            )
            val_dict_results_whole["recs_k5"].append(
                val_dict_results_epoch["epoch_rec_k5"]
            )
            val_dict_results_whole["recs_k10"].append(
                val_dict_results_epoch["epoch_rec_k10"]
            )
            val_dict_results_whole["accs_k1"].append(
                val_dict_results_epoch["epoch_acc_k1"]
            )
            val_dict_results_whole["accs_k5"].append(
                val_dict_results_epoch["epoch_acc_k5"]
            )
            val_dict_results_whole["accs_k10"].append(
                val_dict_results_epoch["epoch_acc_k10"]
            )

            # save pkl
            dict_results_epoch_pkl = os.path.join(self.output_path, "results.pkl")
            with open(dict_results_epoch_pkl, "wb") as f:
                pickle.dump([train_dict_results_whole, val_dict_results_whole], f)
            # save csv
            np.savetxt(
                os.path.join(self.output_path, "stats.csv"),
                np.asarray(
                    [
                        train_dict_results_whole["losses"],
                        val_dict_results_whole["losses"],
                        train_dict_results_whole["precs"],
                        val_dict_results_whole["precs"],
                        train_dict_results_whole["recs"],
                        val_dict_results_whole["recs"],
                        train_dict_results_whole["accs"],
                        val_dict_results_whole["accs"],
                        train_dict_results_whole["fscores"],
                        val_dict_results_whole["fscores"],
                        train_dict_results_whole["medRs"],
                        val_dict_results_whole["medRs"],
                        train_dict_results_whole["Rk1s"],
                        val_dict_results_whole["Rk1s"],
                        train_dict_results_whole["Rk5s"],
                        val_dict_results_whole["Rk5s"],
                        train_dict_results_whole["Rk10s"],
                        val_dict_results_whole["Rk10s"],
                        train_dict_results_whole["AUCs"],
                        val_dict_results_whole["AUCs"],
                        train_dict_results_whole["precs_k1"],
                        val_dict_results_whole["precs_k1"],
                        train_dict_results_whole["precs_k5"],
                        val_dict_results_whole["precs_k5"],
                        train_dict_results_whole["precs_k10"],
                        val_dict_results_whole["precs_k10"],
                        train_dict_results_whole["recs_k1"],
                        val_dict_results_whole["recs_k1"],
                        train_dict_results_whole["recs_k5"],
                        val_dict_results_whole["recs_k5"],
                        train_dict_results_whole["recs_k10"],
                        val_dict_results_whole["recs_k10"],
                        train_dict_results_whole["accs_k1"],
                        val_dict_results_whole["accs_k1"],
                        train_dict_results_whole["accs_k5"],
                        val_dict_results_whole["accs_k5"],
                        train_dict_results_whole["accs_k10"],
                        val_dict_results_whole["accs_k10"],
                    ]
                ),
                delimiter=",",
            )
            if self.save_mat:
                dict_results_batch_pkl = os.path.join(
                    self.output_path, f"results_epoch{epoch}.pkl"
                )
                with open(dict_results_batch_pkl, "wb") as f:
                    pickle.dump([train_dict_results_batch, val_dict_results_batch], f)
                dict_data_batch_pkl = os.path.join(
                    self.output_path, f"data_epoch{epoch}.pkl"
                )
                with open(dict_data_batch_pkl, "wb") as f:
                    pickle.dump([train_dict_data_batch, val_dict_data_batch], f)

        plot_single(self.output_path)
        # plot_benchmark(os.path.dirname(self.output_path))

    def load_ckpt(self) -> bool:
        if not os.path.exists(
            self.checkpoint_path
        ):  # if doesnt exist, list all and read latest
            list_of_checkpoints = glob.glob(self.output_path + "/*.pth")
            if len(list_of_checkpoints) > 0:
                self.checkpoint_path = max(list_of_checkpoints, key=os.path.getctime)
            else:
                self.checkpoint_path = None  # type: ignore

        if self.checkpoint_path:
            self.net = torch.load(self.checkpoint_path)  # type: ignore
            # self.net.load_state_dict(torch.load(self.checkpoint_path))
            return True
        else:
            return False

    def deploy(self, image_files: Any) -> Any:  # load checkpoint and run an image path
        # load latest checkpoint
        loaded_ckpt = self.load_ckpt()
        if loaded_ckpt is False:
            print("Could not find any checkpoint, closing deploy")
            return []

        # print(image_files)

        """ # Center crop
        # prepare data
        x = []
        for idx in range(len(image_files)):
            filename = image_files[idx]
            filename_noext = os.path.splitext(os.path.basename(filename))[0]
            # image = io.imread(fname=filename)
            image = Image.open(filename)
            image_tensor = transforms.functional.to_tensor(image).unsqueeze_(0)
            preproc_image = self.cCROP(
                image_tensor
            )  # todo: maybe replace this deploy by several crops and select most frequent?
            preproc_image=torch.unsqueeze(force_rgb(preproc_image[0,:]),dim=0)
            x.append(preproc_image)
            save_image(preproc_image, os.path.join(self.output_path,os.path.basename(filename)))
        x = torch.cat(x, dim=0)
        """
        # N Random Crops
        x = []
        for idx in range(len(image_files)):
            filename = image_files[idx]
            # filename_noext = os.path.splitext(os.path.basename(filename))[0]
            image = Image.open(filename)
            image_tensor = transforms.functional.to_tensor(image).unsqueeze_(0)
            if (
                image_tensor.shape[2] < self.crop_size[0]
                or image_tensor.shape[3] < self.crop_size[1]
            ):
                image_tensor = torch.tensor(
                    circ3d_pad(image_tensor.squeeze(), self.crop_size)
                ).unsqueeze(0)
            if (
                image_tensor.shape[2] != self.crop_size[0]
            ):  # whole satellite image or crop?
                xx = []

                for cidx in range(self.num_crops):
                    # print("Generating crop ("+str(cidx+1)+"/"+str(self.num_crops)+")")
                    preproc_image = self.tCROP(image_tensor)
                    preproc_image = torch.unsqueeze(
                        force_rgb(preproc_image[0, :]), dim=0
                    )
                    xx.append(preproc_image)
                    save_image(
                        preproc_image,
                        os.path.join(self.output_path, os.path.basename(filename)),
                    )
                xx = torch.cat(xx, dim=0)  # type: ignore
                x.append(xx)
            else:
                x.append(image_tensor)
        # run for each image crop
        reg_values = dict((par, []) for par in self.params)  # type: ignore
        for idx, crops in enumerate(x):
            pred = self.net(crops)
            if len(self.params) == 1:  # Single head prediction
                par = self.params[0]
                pmax = []
                for i, prediction in enumerate(pred):
                    pmax.append(pred[i].argmax())
                reg_values[par].append(
                    max(set(pmax), key=pmax.count)
                )  # save reg index values with most occurencies (for each image crops, select most common param in pred list)
            else:  # Multi head prediction
                for pidx, par in enumerate(self.params):
                    pmax = []
                    for i, prediction in enumerate(pred[pidx]):
                        pmax.append(pred[pidx][i].argmax())
                    reg_values[par].append(
                        max(set(pmax), key=pmax.count)
                    )  # save reg index values with most occurencies (for each image crops, select most common param in pred list)
        # prepare output_json
        if len(self.params) == 1:
            output_values = []
            par = self.params[0]
            for i, regval in enumerate(reg_values[par]):
                value = self.yclasses[par][regval]
                output_values.append(value)
                output_json = {par: value, "path": image_files[i]}
                print(" pred " + str(output_json))
        else:
            output_values = [[] for par in self.params]
            for i, crops in enumerate(x):
                output_json = {}
                for pidx, par in enumerate(self.params):
                    regval = reg_values[par][i]
                    value = self.yclasses[par][regval]
                    output_json[par] = value
                    output_values[pidx].append(value)
                output_json["path"] = image_files[i]
                print(" pred " + str(output_json))
        return output_values

    def deploy_single(
        self, image_tensor: Any
    ) -> Any:  # load checkpoint and run an image tensor
        # load latest checkpoint
        if not os.path.exists(
            self.checkpoint_path
        ):  # if doesnt exist, list all and read latest
            list_of_checkpoints = glob.glob(self.output_path + "/*.pth")
            if len(list_of_checkpoints) > 0:
                self.checkpoint_path = max(list_of_checkpoints, key=os.path.getctime)
            else:
                self.checkpoint_path = None  # type: ignore

        if self.checkpoint_path:
            self.net = torch.load(self.checkpoint_path)  # type: ignore
            # self.net.load_state_dict(torch.load(self.checkpoint_path))
        else:
            print("Could not find any checkpoint, closing deploy...")
            return []

        return self.net(image_tensor)

    def train_val_epoch(
        self, dataset_loader: Any, epoch: Any, validate: bool = False
    ) -> Any:
        dict_results_batch = {}
        (
            dict_results_batch["losses"],
            dict_results_batch["precs"],
            dict_results_batch["recs"],
            dict_results_batch["accs"],
            dict_results_batch["fscores"],
            dict_results_batch["medRs"],
            dict_results_batch["Rk1s"],
            dict_results_batch["Rk5s"],
            dict_results_batch["Rk10s"],
            dict_results_batch["AUCs"],
            dict_results_batch["precs_k1"],
            dict_results_batch["precs_k5"],
            dict_results_batch["precs_k10"],
            dict_results_batch["recs_k1"],
            dict_results_batch["recs_k5"],
            dict_results_batch["recs_k10"],
            dict_results_batch["accs_k1"],
            dict_results_batch["accs_k5"],
            dict_results_batch["accs_k10"],
        ) = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
        dict_results_epoch = {}
        (
            dict_results_epoch["epoch_loss"],
            dict_results_epoch["epoch_prec"],
            dict_results_epoch["epoch_rec"],
            dict_results_epoch["epoch_acc"],
            dict_results_epoch["epoch_fscore"],
            dict_results_epoch["epoch_medR"],
            dict_results_epoch["epoch_Rk1"],
            dict_results_epoch["epoch_Rk5"],
            dict_results_epoch["epoch_Rk10"],
            dict_results_epoch["epoch_AUC"],
            dict_results_epoch["epoch_prec_k1"],
            dict_results_epoch["epoch_prec_k5"],
            dict_results_epoch["epoch_prec_k10"],
            dict_results_epoch["epoch_rec_k1"],
            dict_results_epoch["epoch_rec_k5"],
            dict_results_epoch["epoch_rec_k10"],
            dict_results_epoch["epoch_acc_k1"],
            dict_results_epoch["epoch_acc_k5"],
            dict_results_epoch["epoch_acc_k10"],
        ) = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
        dict_data_batch = {}  # type: ignore
        (
            dict_data_batch["filenames"],
            dict_data_batch["ranks"],
            dict_data_batch["predictions"],
            dict_data_batch["targets"],
        ) = (
            [],
            [],
            [],
            [],
        )

        # set training type (gradients)
        if validate is False:
            torch.enable_grad()
            self.net.train()
        else:
            torch.no_grad()
            self.net.eval()

        # validate_only case, exit if training
        if validate is False and self.validate_only is True:
            return (dict_results_epoch, dict_results_batch, dict_data_batch)

        # join regression batch for crops and modifiers
        # xbatches=[x for bix,(x,y) in enumerate(dataset_loader)]
        # ybatches=[y for bix,(x,y) in enumerate(dataset_loader)]

        for bidx, (filename, param, x, y) in enumerate(dataset_loader):
            if validate is False:
                x.requires_grad = True
            # transform sigmas to regression intervals (yreg)
            param = [
                self.params[0] if par == "" else par for par in param
            ]  # if param is empty, set to first param in params list
            yreg = torch.stack(
                [
                    torch.tensor(
                        bisect_right(self.yclasses[param[i]], y[i]) - 1,
                        dtype=torch.long,
                    )
                    for i in range(len(y))
                ]
            )
            yreg = Variable(yreg)
            if self.cuda:
                yreg = yreg.cuda()
                x = x.cuda()
            prediction = self.net(x)  # input x and predict based on x, [b,:,:,:]

            # calculate prediction metrics
            if len(self.params) == 1:  # single head
                target = torch.eye(self.num_regs[0])[yreg]
                pred = torch.nn.Sigmoid()(prediction)
                if self.cuda:
                    target = target.cuda()
                # calc loss
                loss = self.criterion(pred, target)  # yreg as alternative (classes)

                # output encoding (threshold output and compute) to get TP,FP...
                output_hard = soft2hard(prediction, self.soft_threshold)
                prec = get_precision(output_hard, target)
                rec = get_recall(output_hard, target)
                acc = get_accuracy(output_hard, target)
                fscore = get_fscore(output_hard, target)
                medR, rank = get_median_rank(prediction, target)
                Rk1 = get_recall_rate(prediction, target, 1)
                Rk5 = get_recall_rate(prediction, target, 5)
                Rk10 = get_recall_rate(prediction, target, 10)
                AUC = get_AUC(output_hard, target)
                prec_k1 = get_precision_k(output_hard, target, 1, self.soft_threshold)
                prec_k5 = get_precision_k(output_hard, target, 5, self.soft_threshold)
                prec_k10 = get_precision_k(output_hard, target, 10, self.soft_threshold)
                rec_k1 = get_recall_k(output_hard, target, 1, self.soft_threshold)
                rec_k5 = get_recall_k(output_hard, target, 5, self.soft_threshold)
                rec_k10 = get_recall_k(output_hard, target, 10, self.soft_threshold)
                acc_k1 = get_accuracy_k(output_hard, target, 1, self.soft_threshold)
                acc_k5 = get_accuracy_k(output_hard, target, 5, self.soft_threshold)
                acc_k10 = get_accuracy_k(output_hard, target, 10, self.soft_threshold)
                """
                par = self.params[0]
                for i, tgt in enumerate(target):
                    output_json = {par: self.yclasses[par][prediction[i].argmax()]}
                    target_json = {par: self.yclasses[par][target[i].argmax()]}
                    # print("target "+str(target_json)+" pred "+str(output_json)+" batch "+str(bidx+1))
                """
                if self.save_mat:
                    dict_data_batch["predictions"].append(
                        prediction
                    )  # note: prediction before sigmoid
                    dict_data_batch["targets"].append(target)
            else:  # multihead
                loss = 0.0  # compute losses differently for each head
                (
                    pprec,
                    preca,
                    pacc,
                    pfscore,
                    pmedR,
                    pRk1,
                    pRk5,
                    pRk10,
                    pAUC,
                    pprec_k1,
                    pprec_k5,
                    pprec_k10,
                    preca_k1,
                    preca_k5,
                    preca_k10,
                    pacc_k1,
                    pacc_k5,
                    pacc_k10,
                ) = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                param_ids = [self.dict_params[par] for par in param]
                pprediction = []
                ptarget = []
                for pidx, par in enumerate(self.params):
                    param_indices = [
                        ppidx for ppidx, pid in enumerate(param_ids) if pid == pidx
                    ]
                    if len(param_indices) == 0:
                        continue
                    param_yreg = yreg[param_indices]
                    param_target = torch.eye(self.num_regs[pidx])[
                        param_yreg
                    ]  # one-hot encoding
                    if self.cuda:
                        param_target = param_target.cuda()
                    param_prediction = prediction[pidx][param_indices]
                    param_pred = torch.nn.Sigmoid()(param_prediction)
                    param_loss = self.criterion(
                        param_pred, param_target
                    )  # one loss for each param

                    loss += param_loss  # final loss is sum of all param BCE losses
                    # todo: check if this loss computation works, or losses need to be adapted to each param/head?

                    # output encoding (threshold output and compute) to get TP,FP...
                    output_hard = soft2hard(param_prediction, self.soft_threshold)
                    param_prec = get_precision(output_hard, param_target)
                    param_rec = get_recall(output_hard, param_target)
                    param_acc = get_accuracy(output_hard, param_target)
                    param_fscore = get_fscore(output_hard, param_target)
                    param_medR, rank = get_median_rank(param_prediction, param_target)
                    param_Rk1 = get_recall_rate(param_prediction, param_target, 1)
                    param_Rk5 = get_recall_rate(param_prediction, param_target, 5)
                    param_Rk10 = get_recall_rate(param_prediction, param_target, 10)
                    param_AUC = get_AUC(output_hard, param_target)
                    param_prec_k1 = get_precision_k(
                        output_hard, param_target, 1, self.soft_threshold
                    )
                    param_prec_k5 = get_precision_k(
                        output_hard, param_target, 5, self.soft_threshold
                    )
                    param_prec_k10 = get_precision_k(
                        output_hard, param_target, 10, self.soft_threshold
                    )
                    param_reca_k1 = get_recall_k(
                        output_hard, param_target, 1, self.soft_threshold
                    )
                    param_reca_k5 = get_recall_k(
                        output_hard, param_target, 5, self.soft_threshold
                    )
                    param_reca_k10 = get_recall_k(
                        output_hard, param_target, 10, self.soft_threshold
                    )
                    param_acc_k1 = get_accuracy_k(
                        output_hard, param_target, 1, self.soft_threshold
                    )
                    param_acc_k5 = get_accuracy_k(
                        output_hard, param_target, 5, self.soft_threshold
                    )
                    param_acc_k10 = get_accuracy_k(
                        output_hard, param_target, 10, self.soft_threshold
                    )
                    # append metrics per head
                    pprec.append(param_prec)
                    preca.append(param_rec)
                    pacc.append(param_acc)
                    pfscore.append(param_fscore)
                    pmedR.append(param_medR)
                    pRk1.append(param_Rk1)
                    pRk5.append(param_Rk5)
                    pRk10.append(param_Rk10)
                    pAUC.append(param_AUC)
                    pprec_k1.append(param_prec_k1)
                    pprec_k5.append(param_prec_k5)
                    pprec_k10.append(param_prec_k10)
                    preca_k1.append(param_reca_k1)
                    preca_k5.append(param_reca_k5)
                    preca_k10.append(param_reca_k10)
                    pacc_k1.append(param_acc_k1)
                    pacc_k5.append(param_acc_k5)
                    pacc_k10.append(param_acc_k10)
                    """
                    # print json values (converted from onehot to param interval values)
                    for i, tgt in enumerate(param_target):
                        output_json = {par: self.yclasses[par][param_pred[i].argmax()]}
                        target_json = {
                            par: self.yclasses[par][param_target[i].argmax()]
                        }
                        # print("target "+str(target_json)+" pred "+str(output_json)+" batch "+str(bidx+1))
                    """
                    if self.save_mat:
                        pprediction.append(param_prediction)
                        ptarget.append(param_target)

                if self.save_mat:
                    dict_data_batch["predictions"].append(
                        pprediction
                    )  # note: prediction before sigmoid
                    dict_data_batch["targets"].append(ptarget)

                # mean of each metric head
                (
                    prec,
                    rec,
                    acc,
                    fscore,
                    medR,
                    Rk1,
                    Rk5,
                    Rk10,
                    AUC,
                    prec_k1,
                    prec_k5,
                    prec_k10,
                    rec_k1,
                    rec_k5,
                    rec_k10,
                    acc_k1,
                    acc_k5,
                    acc_k10,
                ) = (
                    np.nanmean(pprec),
                    np.nanmean(preca),
                    np.nanmean(pacc),
                    np.nanmean(pfscore),
                    np.nanmean(pmedR),
                    np.nanmean(pRk1),
                    np.nanmean(pRk5),
                    np.nanmean(pRk10),
                    np.nanmean(pAUC),
                    np.nanmean(pprec_k1),
                    np.nanmean(pprec_k5),
                    np.nanmean(pprec_k10),
                    np.nanmean(preca_k1),
                    np.nanmean(preca_k5),
                    np.nanmean(preca_k10),
                    np.nanmean(pacc_k1),
                    np.nanmean(pacc_k5),
                    np.nanmean(pacc_k10),
                )

            # append results for each batch
            (
                dict_results_batch["precs"],
                dict_results_batch["recs"],
                dict_results_batch["accs"],
                dict_results_batch["fscores"],
                dict_results_batch["medRs"],
                dict_results_batch["Rk1s"],
                dict_results_batch["Rk5s"],
                dict_results_batch["Rk10s"],
                dict_results_batch["AUCs"],
                dict_results_batch["precs_k1"],
                dict_results_batch["precs_k5"],
                dict_results_batch["precs_k10"],
                dict_results_batch["recs_k1"],
                dict_results_batch["recs_k5"],
                dict_results_batch["recs_k10"],
                dict_results_batch["accs_k1"],
                dict_results_batch["accs_k5"],
                dict_results_batch["accs_k10"],
            ) = (
                np.append(dict_results_batch["precs"], prec),
                np.append(dict_results_batch["recs"], rec),
                np.append(dict_results_batch["accs"], acc),
                np.append(dict_results_batch["fscores"], fscore),
                np.append(dict_results_batch["medRs"], medR),
                np.append(dict_results_batch["Rk1s"], Rk1),
                np.append(dict_results_batch["Rk5s"], Rk5),
                np.append(dict_results_batch["Rk10s"], Rk10),
                np.append(dict_results_batch["AUCs"], AUC),
                np.append(dict_results_batch["precs_k1"], prec_k1),
                np.append(dict_results_batch["precs_k5"], prec_k5),
                np.append(dict_results_batch["precs_k10"], prec_k10),
                np.append(dict_results_batch["recs_k1"], rec_k1),
                np.append(dict_results_batch["recs_k5"], rec_k5),
                np.append(dict_results_batch["recs_k10"], rec_k10),
                np.append(dict_results_batch["accs_k1"], acc_k1),
                np.append(dict_results_batch["accs_k5"], acc_k5),
                np.append(dict_results_batch["accs_k10"], acc_k10),
            )
            if self.cuda:
                dict_results_batch["losses"] = np.append(
                    dict_results_batch["losses"], loss.data.cpu().numpy()
                )
            else:
                dict_results_batch["losses"] = np.append(
                    dict_results_batch["losses"], loss.data.numpy()
                )

            # Debug batch results
            if self.debug:
                prefix = "Train " if validate is False else "Val "
                print_debug_batch = "\t".join(
                    [
                        f"{key}={dict_results_batch[key][-1]:.3f}"
                        for key in list(dict_results_batch.keys())
                    ]
                )
                print(
                    prefix
                    + "Step "
                    + str(epoch)
                    + " Batch "
                    + str(bidx)
                    + "/"
                    + str(dataset_loader.__len__())
                    + ": "
                    + print_debug_batch
                )

            # append filenames and ranks
            dict_data_batch["filenames"].append(filename)
            dict_data_batch["ranks"].append(rank)

            # backprop (train case only)
            if validate is False:
                self.optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients

        # calc epoch results
        if len(dict_results_batch["losses"]) > 0:
            (
                dict_results_epoch["epoch_loss"],
                dict_results_epoch["epoch_prec"],
                dict_results_epoch["epoch_rec"],
                dict_results_epoch["epoch_acc"],
                dict_results_epoch["epoch_fscore"],
                dict_results_epoch["epoch_medR"],
                dict_results_epoch["epoch_Rk1"],
                dict_results_epoch["epoch_Rk5"],
                dict_results_epoch["epoch_Rk10"],
                dict_results_epoch["epoch_AUC"],
                dict_results_epoch["epoch_prec_k1"],
                dict_results_epoch["epoch_prec_k5"],
                dict_results_epoch["epoch_prec_k10"],
                dict_results_epoch["epoch_rec_k1"],
                dict_results_epoch["epoch_rec_k5"],
                dict_results_epoch["epoch_rec_k10"],
                dict_results_epoch["epoch_acc_k1"],
                dict_results_epoch["epoch_acc_k5"],
                dict_results_epoch["epoch_acc_k10"],
            ) = (
                np.nanmean(dict_results_batch["losses"]),
                np.nanmean(dict_results_batch["precs"]),
                np.nanmean(dict_results_batch["recs"]),
                np.nanmean(dict_results_batch["accs"]),
                np.nanmean(dict_results_batch["fscores"]),
                np.nanmean(dict_results_batch["medRs"]),
                np.nanmean(dict_results_batch["Rk1s"]),
                np.nanmean(dict_results_batch["Rk5s"]),
                np.nanmean(dict_results_batch["Rk10s"]),
                np.nanmean(dict_results_batch["AUCs"]),
                np.nanmean(dict_results_batch["precs_k1"]),
                np.nanmean(dict_results_batch["precs_k5"]),
                np.nanmean(dict_results_batch["precs_k10"]),
                np.nanmean(dict_results_batch["recs_k1"]),
                np.nanmean(dict_results_batch["recs_k5"]),
                np.nanmean(dict_results_batch["recs_k10"]),
                np.nanmean(dict_results_batch["accs_k1"]),
                np.nanmean(dict_results_batch["accs_k5"]),
                np.nanmean(dict_results_batch["accs_k10"]),
            )

        # print log
        print_debug_epoch = "\t".join(
            [
                f"{key}={dict_results_epoch[key]:.3f}"
                for key in list(dict_results_epoch.keys())
            ]
        )
        prefix = "Train " if validate is False else "Val "
        print(prefix + "Step " + str(epoch) + ": " + print_debug_epoch)

        # Get top and worst K crop filenames
        if self.debug:
            filenames_stacked = np.array(
                [col for row in dict_data_batch["filenames"] for col in row]
            )
            ranks_stacked = np.array(
                [col for row in dict_data_batch["ranks"] for col in row]
            )
            order_ranks = np.argsort(ranks_stacked)
            Ktop = self.batch_size  # default 32
            dict_results_epoch["topK"] = filenames_stacked[order_ranks][:Ktop]
            dict_results_epoch["worstK"] = filenames_stacked[order_ranks][::-1][:Ktop]
            dict_results_epoch["topK_ranks"] = ranks_stacked[order_ranks][:Ktop]
            dict_results_epoch["worstK_ranks"] = ranks_stacked[order_ranks][::-1][:Ktop]
        else:
            (
                dict_results_epoch["topK"],
                dict_results_epoch["worstK"],
                dict_results_epoch["topK_ranks"],
                dict_results_epoch["worstK_ranks"],
            ) = ([], [], [], [])

        # clean memory if not debugging mats
        if not self.save_mat:
            dict_results_batch = {}
            dict_data_batch = {}

        return (dict_results_epoch, dict_results_batch, dict_data_batch)


if __name__ == "__main__":
    parser = parse_params_cfg()
    args, uk_args = parser.parse_known_args()
    print(args)
    print("Preparing Regressor")
    reg = Regressor(args)

    # plot stats
    stats_file = os.path.join(reg.output_path, "stats.csv")
    if args.plot_only is True or os.path.exists(stats_file):
        results_array = np.loadtxt(stats_file, delimiter=",")
        if len(np.shape(results_array)) == 2:
            diff_epoch = reg.epochs - np.shape(results_array)[1]
        else:
            diff_epoch = -1  # file invalid (empty results file)
        if diff_epoch == 0:
            print("Found " + stats_file + " plotting...")
            plot_single(reg.output_path)
            plot_benchmark(os.path.dirname(reg.output_path))
            exit()
        else:
            print("Found " + stats_file + " but not all epoch, re-training")
            args.resume = False  # train instead of deploy
            # reg.epochs = reg.epochs - diff_epoch # stats will not be appended so run all again

    # TRAIN+VAL (depending if checkpoint exists)
    if args.resume is False:
        val_dataset = Dataset(
            "test",
            args.valds,
            args.valdsinput,
            args.num_crops,
            args.input_size,
            args.splits[1],
        )  # set num_crops as split proportion
        val_dataset.__modify__(reg.ds_modifiers, args.overwrite_modifiers)
        val_dataset.__crop__(args.overwrite_crops)
        print(
            "Prepared Validation Dataset "
            + "("
            + str(val_dataset.__len__())
            + ") images x modifiers x crops"
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=reg.batch_size,
            shuffle=args.data_shuffle,
            num_workers=args.workers,
            pin_memory=True,
        )
        if (
            args.validate_only is True and args.data_only is True
        ):  # validate data only condition
            exit()
        train_dataset = Dataset(
            "train",
            args.trainds,
            args.traindsinput,
            args.num_crops,
            args.input_size,
            args.splits[0],
        )  # set num_crops as split proportion
        train_dataset.__modify__(reg.ds_modifiers, args.overwrite_modifiers)
        train_dataset.__crop__(args.overwrite_crops)
        print(
            "Prepared Train Dataset "
            + "("
            + str(train_dataset.__len__())
            + ") images x modifiers x crops"
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=reg.batch_size,
            shuffle=args.data_shuffle,
            num_workers=args.workers,
            pin_memory=True,
        )
        if args.data_only is True:  # data only condition
            exit()
        if (
            os.path.exists(reg.checkpoint_path) is False
        ):  # force train if checkpoint does not exist
            reg.validate_only = False
        else:  # load model if checkpoint exists
            loaded_ckpt = reg.load_ckpt()
        # Train and validate regressor
        reg.train_val(train_loader, val_loader)
    else:
        # DEPLOY
        gt_path = args.valdsinput
        image_paths = os.listdir(gt_path)
        image_files = []
        for idx, image_name in enumerate(image_paths):
            image_files.append(gt_path + "/" + image_name)  # abs_images_folder
        start_time = time.time()
        reg.deploy(image_files)
        print("--- %s seconds ---" % (time.time() - start_time))
