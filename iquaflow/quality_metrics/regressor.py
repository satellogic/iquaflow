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

from iquaflow.datasets import (
    DSModifier,
    DSModifier_blur,
    DSModifier_gsd,
    DSModifier_rer,
    DSModifier_sharpness,
    DSModifier_snr,
)
from iquaflow.quality_metrics.benchmark import (
    get_eval_metrics,
    plot_benchmark,
    plot_single,
)
from iquaflow.quality_metrics.dataloader import Dataset
from iquaflow.quality_metrics.tools import (  # circ3d_pad,; force_rgb,
    MultiHead,
    create_network,
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
    if config.has_option("HYPERPARAMS", "patience_stopping"):
        parser.add_argument(
            "--patience_stopping",
            default=json.loads(config["HYPERPARAMS"]["patience_stopping"]),
        )
    else:
        parser.add_argument(
            "--patience_stopping",
            default=10,
        )
    if config.has_option("HYPERPARAMS", "patience_step"):
        parser.add_argument(
            "--patience_step",
            default=json.loads(config["HYPERPARAMS"]["patience_step"]),
        )
    else:
        parser.add_argument(
            "--patience_step",
            default=0.5,
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


def get_parametizable_keys() -> List[str]:
    return ["sigma", "scale", "sharpness", "snr", "rer"]


def get_non_parametizable_keys() -> List[str]:
    return ["dataset", "interpolation", "resol"]


def get_modifiers_from_params(
    modifier_params: Dict[str, Any],
    parametizable_params: List[str] = get_parametizable_keys(),
    non_parametizable_params: List[str] = get_non_parametizable_keys(),
) -> Tuple[Any, Any]:
    ds_modifiers = []
    if len(modifier_params.items()) == 0:
        ds_modifiers.append(DSModifier())
    else:
        for key, elem in modifier_params.items():
            if (key not in parametizable_params) and (
                key not in non_parametizable_params
            ):
                print(
                    f"{key} not found in parametizable list, make sure your modifier is included in iquaflow.datasets imports and in parametizable_params"
                )
            elif key in non_parametizable_params:
                print(f"{key} not used as parameter")
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
                else:
                    ds_modifier_name = "DSModifier_" + key
                    if ds_modifier_name is globals():
                        ds_modifier = globals()[key](
                            params={key: modifier_params[key][gidx]}
                        )
                        ds_modifiers.append(ds_modifier)

    return ds_modifiers, modifier_params


def get_regression_interval_classes(
    modifier_params: Dict[str, Any],
    num_regs: List[int],
    parametizable_params: List[str] = get_parametizable_keys(),
    non_parametizable_params: List[str] = get_non_parametizable_keys(),
) -> Dict[str, Any]:
    yclasses = {}
    params = list(modifier_params.keys())
    for idx, param in enumerate(params):
        if (param not in parametizable_params) and (
            param not in non_parametizable_params
        ):
            print(
                f"{param} not found in parametizable list, make sure your modifier is included in iquaflow.datasets imports and in parametizable_params"
            )
        elif param in non_parametizable_params:
            print(f"{param} not used as parameter")
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
        # Define modifiers and regression intervals
        self.ds_modifiers, self.modifier_params = get_modifiers_from_params(
            self.modifier_params
        )  # train case
        # self.ds_modifiers = [DSModifier()] # deploy case
        self.params = [
            key
            for key, value in self.modifier_params.items()
            if (type(value) == np.ndarray) & (key in get_parametizable_keys())
        ]  # list(self.modifier_params.keys())
        self.dict_params = {par: idx for idx, par in enumerate(self.params)}
        self.yclasses = get_regression_interval_classes(
            self.modifier_params, self.num_regs
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
        self.patience_stopping = args.patience_stopping
        self.patience_step = args.patience_step
        self.patience_counter = 0

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
        self.device = torch.device(
            "cuda:" + ",".join(self.gpus) if torch.cuda.is_available() else "cpu"
        )
        self.net = self.net.to(self.device)  # set net to device

        if "cuda" in self.device.type:
            self.cuda = True

        if self.cuda:
            torch.cuda.empty_cache()  # empty cuda cache
            print("=> using gpu id: '{}'".format(self.gpus))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
            if not torch.cuda.is_available():
                raise Exception(
                    "No GPU found or Wrong gpu id, please run without --cuda"
                )
            print("Random Seed: ", self.seed)
            torch.cuda.manual_seed(self.seed)
            self.criterion = self.criterion.to(self.device)
            self.net = self.net.to(self.device)

    def train_val(self, train_loader: Any, val_loader: Any) -> Any:
        best_loss = np.inf
        best_fscore = 0.0
        train_dict_results_whole = {}  # type: ignore
        val_dict_results_whole = {}  # type: ignore
        metric_tags, metric_list = get_eval_metrics()
        for metric in metric_list:
            train_dict_results_whole[metric] = []
            val_dict_results_whole[metric] = []
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
                val_dict_results_epoch["losses"] is None
                and train_dict_results_epoch["losses"] is None
            ):
                print("Error: Validation and Training losses are None.")
                raise
            is_best = (val_dict_results_epoch["losses"] < best_loss) & (
                val_dict_results_epoch["fscores"] > best_fscore
            )
            if is_best:
                self.patience_counter = 0
                best_loss = val_dict_results_epoch["losses"]
                best_fscore = val_dict_results_epoch["fscores"]
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
            else:  # is_best false
                # EARLY STOPPING AND LR STEP
                self.patience_counter += 1
                if self.patience_counter >= self.patience_stopping:
                    break
                else:  # lower lr upon step and update optimizer
                    self.lr *= self.patience_step
                    self.optimizer = torch.optim.SGD(
                        self.net.parameters(),
                        lr=self.lr,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay,
                    )
            # append all results per epoch
            for metric in metric_list:
                train_dict_results_whole[metric].append(
                    train_dict_results_epoch[metric]
                )
                val_dict_results_whole[metric].append(val_dict_results_epoch[metric])

            # save pkl
            results_pkl = os.path.join(self.output_path, "results.pkl")
            with open(results_pkl, "wb") as f:
                pickle.dump([train_dict_results_whole, val_dict_results_whole], f)
            # save csv
            array_csv = np.vstack(
                np.asarray(
                    [
                        np.array(
                            [
                                train_dict_results_whole[metric],
                                val_dict_results_whole[metric],
                            ]
                        )
                        for metric in metric_list
                    ]
                )
            )
            np.savetxt(
                os.path.join(self.output_path, "stats.csv"),
                array_csv,
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
            self.net = self.net.to(self.device)
            if self.cuda:
                self.net = self.net.to(self.device)
            return True
        else:
            return False

    def deploy(
        self, image_files: Any, crop_type: str = "random"
    ) -> Any:  # load checkpoint and run an image path
        # load latest checkpoint
        loaded_ckpt = self.load_ckpt()
        if loaded_ckpt is False:
            print("Could not find any checkpoint, closing deploy")
            return []
        if len(image_files) == 0:
            print("Empty image list, closing deploy")
            return []
        # create dataset + dataloader instance
        testds = os.path.dirname(
            os.path.dirname(image_files[0])
        )  # path to dataset folder
        testdsinput = os.path.dirname(image_files[0])  # path to images folder
        default_img_size = Image.open(image_files[0]).size
        num_crops = int(self.crop_size / default_img_size[0])
        test_dataset = Dataset(
            "test",
            testds,
            testdsinput,
            num_crops,  # regressor-specific on training as well
            self.crop_size,  # regressor-specific on training as well
            1.0,
            default_img_size,
        )
        test_dataset.lists_files = image_files
        test_dataset.__crop__(False, True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        # prepare data
        dict_data_batch: Dict[str, Any] = {}
        (
            dict_data_batch["crop_filenames"],
            dict_data_batch["filenames"],
            dict_data_batch["predictions"],
            dict_data_batch["values"],
        ) = (
            [],
            [],
            [],
            [],
        )
        # loop run network per batch and save values to dict_data_batch
        reg_values: Dict[str, Any] = dict((par, []) for par in self.params)
        # set torch to not save gradients
        # torch.set_grad_enabled(False)  # this won't do it
        self.net.eval()
        with torch.no_grad():
            for bidx, (filename, param, x, y) in enumerate(test_loader):
                param = [
                    self.params[0] if par == "" else par for par in param
                ]  # if param is empty, set to first param in params list
                print(f"deploying {param} in {filename}")
                # data to device
                x = x.to(self.device)
                if self.cuda:
                    x = x.to(self.device)
                # run
                prediction = self.net(x)
                # pred = torch.nn.Sigmoid()(prediction)
                # get results as regressor parameter values
                if len(self.params) == 1:  # Single head prediction
                    par = self.params[0]
                    for i, param_prediction in enumerate(prediction):
                        argmax = param_prediction.argmax()
                        value = self.yclasses[par][argmax]
                        reg_values[par].append(value)
                else:
                    for pidx, par in enumerate(self.params):
                        for i, param_prediction in enumerate(prediction):
                            argmax = param_prediction[par].argmax()
                            value = self.yclasses[par][argmax]
                            reg_values[par].append(value)
                dict_data_batch["crop_filenames"].append(filename)
                dict_data_batch["predictions"].append(prediction)
                dict_data_batch["values"].append(reg_values)
                origin_filename = []
                for file in filename:
                    origin_filename.append(test_dataset.dict_crop_files_origin[file])
                dict_data_batch["filenames"].append(origin_filename)

        # unify filenames values (several crops, thus several results per image)
        output_values = []
        for image_filename in image_files:
            file_values = []
            for bidx, _ in enumerate(dict_data_batch["filenames"]):
                for cidx, crop_filename_origin in enumerate(
                    dict_data_batch["filenames"][bidx]
                ):
                    for pidx, par in enumerate(self.params):
                        if crop_filename_origin == image_filename:
                            file_values.append(
                                dict_data_batch["values"][bidx][par][cidx]
                            )
            mean_values = np.nanmean(np.array(file_values))
            output_values.append(mean_values)
        return output_values

    def deploy_single(
        self, image_tensor: Any
    ) -> Any:  # load checkpoint and run an image tensor
        loaded_ckpt = self.load_ckpt()
        if loaded_ckpt is False:
            print("Could not find any checkpoint, closing deploy")
            return []
        # move image_tensor to cuda
        image_tensor = image_tensor.to(self.device)
        if self.cuda:
            image_tensor = image_tensor.to(self.device)
        return self.net(image_tensor)

    def train_val_epoch(
        self, dataset_loader: Any, epoch: Any, validate: bool = False
    ) -> Any:
        metric_tags, metric_list = get_eval_metrics()
        dict_results_batch = {}
        dict_results_epoch = {}
        current_values = {}
        for metric in metric_list:
            dict_results_batch[metric] = np.array([])
            dict_results_epoch[metric] = np.nan
            current_values[metric] = np.nan
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
            torch.set_grad_enabled(True)
            self.net.train()
        else:
            torch.no_grad()
            torch.set_grad_enabled(False)
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
                yreg = yreg.to(self.device)
                x = x.to(self.device)
            prediction = self.net(x)  # input x and predict based on x, [b,:,:,:]

            # calculate prediction metrics
            if len(self.params) == 1:  # single head
                target = torch.eye(self.num_regs[0])[yreg]
                pred = torch.nn.Sigmoid()(prediction)
                if self.cuda:
                    target = target.to(self.device)
                # calc loss
                loss = self.criterion(pred, target)  # yreg as alternative (classes)

                # output encoding (threshold output and compute) to get TP,FP...
                output_hard = soft2hard(prediction, self.soft_threshold)
                current_values["precs"] = get_precision(output_hard, target)
                current_values["recs"] = get_recall(output_hard, target)
                current_values["accs"] = get_accuracy(output_hard, target)
                current_values["fscores"] = get_fscore(output_hard, target)
                current_values["medRs"], rank = get_median_rank(prediction, target)
                current_values["Rk1s"] = get_recall_rate(prediction, target, 1)
                current_values["Rk5s"] = get_recall_rate(prediction, target, 5)
                current_values["Rk10s"] = get_recall_rate(prediction, target, 10)
                current_values["AUCs"] = get_AUC(output_hard, target)
                current_values["precs_k1"] = get_precision_k(
                    output_hard, target, 1, self.soft_threshold
                )
                current_values["precs_k5"] = get_precision_k(
                    output_hard, target, 5, self.soft_threshold
                )
                current_values["precs_k10"] = get_precision_k(
                    output_hard, target, 10, self.soft_threshold
                )
                current_values["recs_k1"] = get_recall_k(
                    output_hard, target, 1, self.soft_threshold
                )
                current_values["recs_k5"] = get_recall_k(
                    output_hard, target, 5, self.soft_threshold
                )
                current_values["recs_k10"] = get_recall_k(
                    output_hard, target, 10, self.soft_threshold
                )
                current_values["accs_k1"] = get_accuracy_k(
                    output_hard, target, 1, self.soft_threshold
                )
                current_values["accs_k5"] = get_accuracy_k(
                    output_hard, target, 5, self.soft_threshold
                )
                current_values["accs_k10"] = get_accuracy_k(
                    output_hard, target, 10, self.soft_threshold
                )
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
                current_param_values = {}
                current_param_values_list = {}
                for metric in metric_list:
                    current_param_values[metric] = np.nan
                    current_param_values_list[metric] = np.array([])
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
                        param_target = param_target.to(self.device)
                    param_prediction = prediction[pidx][param_indices]
                    param_pred = torch.nn.Sigmoid()(param_prediction)
                    param_loss = self.criterion(
                        param_pred, param_target
                    )  # one loss for each param

                    loss += param_loss  # final loss is sum of all param BCE losses
                    # todo: check if this loss computation works, or losses need to be adapted to each param/head?

                    # output encoding (threshold output and compute) to get TP,FP...
                    output_hard = soft2hard(param_prediction, self.soft_threshold)
                    current_param_values["precs"] = get_precision(
                        output_hard, param_target
                    )
                    current_param_values["recs"] = get_recall(output_hard, param_target)
                    current_param_values["accs"] = get_accuracy(
                        output_hard, param_target
                    )
                    current_param_values["fscores"] = get_fscore(
                        output_hard, param_target
                    )
                    current_param_values["medRs"], rank = get_median_rank(
                        param_prediction, param_target
                    )
                    current_param_values["Rk1s"] = get_recall_rate(
                        param_prediction, param_target, 1
                    )
                    current_param_values["Rk5s"] = get_recall_rate(
                        param_prediction, param_target, 5
                    )
                    current_param_values["Rk10s"] = get_recall_rate(
                        param_prediction, param_target, 10
                    )
                    current_param_values["AUCs"] = get_AUC(output_hard, param_target)
                    current_param_values["precs_k1"] = get_precision_k(
                        output_hard, param_target, 1, self.soft_threshold
                    )
                    current_param_values["precs_k5"] = get_precision_k(
                        output_hard, param_target, 5, self.soft_threshold
                    )
                    current_param_values["precs_k10"] = get_precision_k(
                        output_hard, param_target, 10, self.soft_threshold
                    )
                    current_param_values["recs_k1"] = get_recall_k(
                        output_hard, param_target, 1, self.soft_threshold
                    )
                    current_param_values["recs_k5"] = get_recall_k(
                        output_hard, param_target, 5, self.soft_threshold
                    )
                    current_param_values["recs_k10"] = get_recall_k(
                        output_hard, param_target, 10, self.soft_threshold
                    )
                    current_param_values["accs_k1"] = get_accuracy_k(
                        output_hard, param_target, 1, self.soft_threshold
                    )
                    current_param_values["accs_k5"] = get_accuracy_k(
                        output_hard, param_target, 5, self.soft_threshold
                    )
                    current_param_values["accs_k10"] = get_accuracy_k(
                        output_hard, param_target, 10, self.soft_threshold
                    )
                    # append metrics per head
                    for metric in metric_list:
                        current_param_values_list[metric] = np.append(
                            current_param_values_list[metric],
                            current_param_values[metric],
                        )

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
                for metric in metric_list:
                    current_values[metric] = np.nanmean(
                        current_param_values_list[metric]
                    )

            # append results for each batch
            for metric in metric_list:
                if metric != "losses":
                    dict_results_batch[metric] = np.append(
                        dict_results_batch[metric], current_values[metric]
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
            for metric in metric_list:
                dict_results_epoch[metric] = np.nanmean(dict_results_batch[metric])

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
    args.num_crops = int(reg.crop_size / args.input_size)
    reg.num_crops = int(reg.crop_size / args.input_size)

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
