# matplotlib inline
import argparse
import configparser
import glob
import json
import os
import shutil
import time
from bisect import bisect_right
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.models as models
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
    circ3d_pad,
    force_rgb,
    get_accuracy,
    get_fscore,
    get_median_rank,
    get_precision,
    get_recall,
    get_recall_rate,
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
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--seed", type=str, default=str(np.random.randint(12345)))
    parser.add_argument("--debug", default=False, action="store_true")
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
    parser.add_argument("--data_shuffle", default=config["HYPERPARAMS"]["data_shuffle"])
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
    parser.add_argument(
        "--trainds", default=config["PATHS"]["trainds"]
    )  # inria-aid, "xview", "ds_coco_dataset"
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


class MultiHead(torch.nn.Module):  # deprecated
    def __init__(self, *modules: torch.nn.Module) -> None:
        super().__init__()
        self.modules = modules  # type: ignore

    def forward(self, inputs: Any) -> Any:
        return [module(inputs) for module in self.modules]  # type: ignore


class MultiHead_ResNet(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Module = models.resnet18(pretrained=True),
        *heads: torch.nn.Module
    ) -> None:
        super().__init__()
        self.network = network
        self.network.fc = torch.nn.Sequential()  # remove fc
        self.network.heads = heads  # type: ignore

    def forward(self, inputs: Any) -> Any:
        x = self.network(inputs)
        return [head(x) for head in self.network.heads]  # type: ignore


class Regressor:
    def __init__(self, args: Any) -> None:
        self.train_ds = args.trainds
        self.train_ds_input = args.traindsinput
        self.val_ds = args.valds
        self.val_ds_input = args.valdsinput
        self.epochs = int(args.epochs)
        self.lr = float(args.lr)
        self.momentum = float(args.momentum)
        self.weight_decay = float(args.weight_decay)
        self.batch_size = int(args.batch_size)

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
        self.tCROP = transforms.Compose(
            [
                transforms.RandomCrop(size=(self.crop_size[0], self.crop_size[1])),
            ]
        )  # define torch transform
        self.cCROP = transforms.Compose(
            [
                transforms.CenterCrop(size=(self.crop_size[0], self.crop_size[1])),
            ]
        )  # define torch transform
        # Create Network
        if len(self.params) == 1:  # Single Head
            self.net = models.resnet18(pretrained=True)
            self.net.fc = torch.nn.Linear(512, self.num_regs[0])
        else:  # MultiHead
            self.net = MultiHead_ResNet(
                models.resnet18(pretrained=True),
                *[
                    torch.nn.Linear(512, self.num_regs[idx])
                    for idx in range(len(self.params))
                ]
            )
            # self.net=models.resnet18(pretrained=True)
            # self.net.fc=MultiHead(*[torch.nn.Linear(512, self.num_regs[idx]) for idx in range(len(self.params))])

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

        # GPU Options
        self.cuda = args.cuda
        self.gpus = args.gpus
        self.seed = args.seed
        self.debug = args.debug
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

    # GPU CASE
    def train_val(self, train_loader: Any, val_loader: Any) -> Any:
        best_loss = np.inf
        best_prec = 0.0
        best_rec = 0.0
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
        for epoch in range(self.epochs):  # epoch
            (
                train_loss,
                train_prec,
                train_rec,
                train_acc,
                train_fscore,
                train_medR,
                train_rec_k1,
                train_rec_k5,
                train_rec_k10,
                train_topK,
                train_worstK,
                train_topK_ranks,
                train_worstK_ranks,
            ) = self.train(train_loader, epoch)
            (
                val_loss,
                val_prec,
                val_rec,
                val_acc,
                val_fscore,
                val_medR,
                val_rec_k1,
                val_rec_k5,
                val_rec_k10,
                val_topK,
                val_worstK,
                val_topK_ranks,
                val_worstK_ranks,
            ) = self.validate(val_loader, epoch)
            if val_loss is None or train_loss is None:
                import pdb

                pdb.set_trace()
            is_best = (val_loss < best_loss) & (
                (val_prec > best_prec) | (val_rec > best_rec)
            )
            if is_best:
                best_loss = val_loss
                best_prec = val_prec
                best_rec = val_rec
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

                # copy top K files for that epoch
                output_path_top = os.path.join(
                    self.output_path, "top_epoch" + str(epoch)
                )
                if not os.path.exists(output_path_top):
                    os.mkdir(output_path_top)
                for idx, filename in enumerate(val_topK):
                    file_newpath = os.path.join(
                        output_path_top,
                        "rank"
                        + str(val_topK_ranks[idx])
                        + "_"
                        + os.path.basename(os.path.dirname(os.path.dirname(filename)))
                        + "_"
                        + os.path.basename(filename),
                    )
                    shutil.copyfile(filename, file_newpath)
                output_path_worst = os.path.join(
                    self.output_path, "worst_epoch" + str(epoch)
                )
                if not os.path.exists(output_path_worst):
                    os.mkdir(output_path_worst)
                for filename in val_worstK:
                    file_newpath = os.path.join(
                        output_path_worst,
                        "rank"
                        + str(val_worstK_ranks[idx])
                        + "_"
                        + os.path.basename(os.path.dirname(os.path.dirname(filename)))
                        + "_"
                        + os.path.basename(filename),
                    )
                    shutil.copyfile(filename, file_newpath)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_precs.append(train_prec)
            val_precs.append(val_prec)
            train_recs.append(train_rec)
            val_recs.append(val_rec)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_fscores.append(train_fscore)
            val_fscores.append(val_fscore)
            train_medRs.append(train_medR)
            val_medRs.append(val_medR)
            train_recs_k1.append(train_rec_k1)
            val_recs_k1.append(val_rec_k1)
            train_recs_k5.append(train_rec_k5)
            val_recs_k5.append(val_rec_k5)
            train_recs_k10.append(train_rec_k10)
            val_recs_k10.append(val_rec_k10)
            print(
                "Train Loss: "
                + str(train_loss)
                + " Train Precision: "
                + str(train_prec)
                + " Train Recall: "
                + str(train_rec)
                + " Train Accuracy: "
                + str(train_acc)
                + " Train Fscore: "
                + str(train_fscore)
                + " Train medR: "
                + str(train_medR)
                + " Train R@1: "
                + str(train_rec_k1)
                + " Train R@5: "
                + str(train_rec_k5)
                + " Train R@10: "
                + str(train_rec_k10)
            )
            print(
                "Validation Loss: "
                + str(val_loss)
                + " Validation Precision: "
                + str(val_prec)
                + " Validation Recall: "
                + str(val_rec)
                + " Validation Accuracy: "
                + str(val_acc)
                + " Validation Fscore: "
                + str(val_fscore)
                + " Validation medR: "
                + str(val_medR)
                + " Validation R@1: "
                + str(val_rec_k1)
                + " Validation R@5: "
                + str(val_rec_k5)
                + " Validation R@10: "
                + str(val_rec_k10)
            )
            np.savetxt(
                os.path.join(self.output_path, "stats.csv"),
                np.asarray(
                    [
                        train_losses,
                        val_losses,
                        train_precs,
                        val_precs,
                        train_recs,
                        val_recs,
                        train_accs,
                        val_accs,
                        train_fscores,
                        val_fscores,
                        train_medRs,
                        val_medRs,
                        train_recs_k1,
                        val_recs_k1,
                        train_recs_k5,
                        val_recs_k5,
                        train_recs_k10,
                        val_recs_k10,
                    ]
                ),
                delimiter=",",
            )
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

    def train(self, train_loader: Any, epoch: Any) -> Any:
        self.net.train()  # train mode
        losses = np.array([])
        precs = np.array([])
        recs = np.array([])
        accs = np.array([])
        fscores = np.array([])
        medRs = np.array([])
        recs_k1 = np.array([])
        recs_k5 = np.array([])
        recs_k10 = np.array([])
        # xbatches=[x for bix,(x,y) in enumerate(train_loader)]
        # ybatches=[y for bix,(x,y) in enumerate(train_loader)]
        filenames = []
        ranks = []
        for bidx, (filename, param, x, y) in enumerate(
            train_loader
        ):  # ongoing: if net is outputting several outputs, separate target and prediction for each param and make sure evaluate that specific head
            # join regression batch for crops and modifiers
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

            if len(self.params) == 1:
                target = torch.eye(self.num_regs[0])[yreg]
                pred = torch.nn.Sigmoid()(prediction)
                if self.cuda:
                    target = target.cuda()
                # calc loss
                loss = self.criterion(pred, target)  # yreg as alternative (classes)

                # output to soft encoding (threshold output and compute) to get TP,FP...
                output_soft = soft2hard(prediction, self.soft_threshold)
                prec = get_precision(output_soft, target)
                rec = get_recall(output_soft, target)
                acc = get_accuracy(output_soft, target)
                fscore = get_fscore(output_soft, target)
                medR, rank = get_median_rank(prediction, target)
                rec_k1 = get_recall_rate(prediction, target, 1)
                rec_k5 = get_recall_rate(prediction, target, 5)
                rec_k10 = get_recall_rate(prediction, target, 10)
                if self.debug:  # Debug
                    print(
                        "Batch("
                        + str(bidx)
                        + "/"
                        + str(train_loader.__len__())
                        + ")"
                        + " Precision="
                        + str(float(prec))
                        + " Recall="
                        + str(float(rec))
                        + " Accuracy="
                        + str(float(acc))
                        + " F-Score="
                        + str(float(fscore))
                        + " medR="
                        + str(float(medR))
                        + " R@1="
                        + str(float(rec_k1))
                        + " R@5="
                        + str(float(rec_k5))
                        + " R@10="
                        + str(float(rec_k10))
                    )
                """
                par = self.params[0]
                for i, tgt in enumerate(target):
                    output_json = {par: self.yclasses[par][prediction[i].argmax()]}
                    target_json = {par: self.yclasses[par][target[i].argmax()]}
                    # print("target "+str(target_json)+" pred "+str(output_json)+" batch "+str(bidx+1))
                """
            else:
                # compute losses differently for each head
                loss = 0.0
                pprec = []
                preca = []
                pacc = []
                pfscore = []
                pmedR = []
                preca_k1 = []
                preca_k5 = []
                preca_k10 = []
                param_ids = [self.dict_params[par] for par in param]
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
                    # output to soft encoding (threshold output and compute) to get TP,FP...
                    output_soft = soft2hard(param_prediction, self.soft_threshold)
                    param_prec = get_precision(output_soft, param_target)
                    param_rec = get_recall(output_soft, param_target)
                    param_acc = get_accuracy(output_soft, param_target)
                    param_fscore = get_fscore(output_soft, param_target)
                    param_medR, rank = get_median_rank(param_prediction, param_target)
                    param_rec_k1 = get_recall_rate(param_prediction, param_target, 1)
                    param_rec_k5 = get_recall_rate(param_prediction, param_target, 5)
                    param_rec_k10 = get_recall_rate(param_prediction, param_target, 10)
                    pprec.append(param_prec)
                    preca.append(param_rec)
                    pacc.append(param_acc)
                    pfscore.append(param_fscore)
                    pmedR.append(param_medR)
                    preca_k1.append(param_rec_k1)
                    preca_k5.append(param_rec_k5)
                    preca_k10.append(param_rec_k10)
                    """
                    # print json values (converted from onehot to param interval values)
                    for i, tgt in enumerate(param_target):
                        output_json = {par: self.yclasses[par][param_pred[i].argmax()]}
                        target_json = {
                            par: self.yclasses[par][param_target[i].argmax()]
                        }
                        # print("target "+str(target_json)+" pred "+str(output_json)+" batch "+str(bidx+1))
                    """
                prec = np.nanmean(pprec)  # use mean or return separate precs?
                rec = np.nanmean(preca)
                acc = np.nanmean(pacc)
                fscore = np.nanmean(pfscore)
                medR = np.nanmean(pmedR)
                rec_k1 = np.nanmean(preca_k1)
                rec_k5 = np.nanmean(preca_k5)
                rec_k10 = np.nanmean(preca_k10)
                # Debug
                if self.debug:
                    print(
                        "Batch("
                        + str(bidx)
                        + "/"
                        + str(train_loader.__len__())
                        + ")"
                        + " Precision="
                        + str(float(prec))
                        + " Recall="
                        + str(float(rec))
                        + " Accuracy="
                        + str(float(acc))
                        + " F-Score="
                        + str(float(fscore))
                        + " medR="
                        + str(float(medR))
                        + " R@1="
                        + str(float(rec_k1))
                        + " R@5="
                        + str(float(rec_k5))
                        + " R@10="
                        + str(float(rec_k10))
                    )
            precs = np.append(precs, prec)
            recs = np.append(recs, rec)
            accs = np.append(accs, acc)
            fscores = np.append(fscores, fscore)
            medRs = np.append(medRs, medR)
            recs_k1 = np.append(recs_k1, rec_k1)
            recs_k5 = np.append(recs_k5, rec_k5)
            recs_k10 = np.append(recs_k10, rec_k10)
            if self.cuda:
                losses = np.append(losses, loss.data.cpu().numpy())
            else:
                losses = np.append(losses, loss.data.numpy())

            # backprop
            self.optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients

            # print("Debug (check sigma intervals (one-hot encoding) of target and prediction)")
            # print("Target=")
            # print(target.squeeze())
            # print("Prediction=")
            # print(pred.squeeze())

            filenames.append(filename)
            ranks.append(rank)

        epoch_loss = np.nanmean(losses)
        epoch_prec = np.nanmean(precs)
        epoch_rec = np.nanmean(recs)
        epoch_acc = np.nanmean(accs)
        epoch_fscore = np.nanmean(fscores)
        epoch_medR = np.nanmean(medRs)
        epoch_rec_k1 = np.nanmean(recs_k1)
        epoch_rec_k5 = np.nanmean(recs_k5)
        epoch_rec_k10 = np.nanmean(recs_k10)
        print(
            "Train Step = %d" % epoch
            + " Loss = %.4f" % epoch_loss
            + " Precision = %0.4f" % epoch_prec
            + " Accuracy = %.4f" % epoch_acc
            + " Recall = %.4f" % epoch_rec
            + " Fscore = %.4f" % epoch_fscore
            + " medR = %.4f" % epoch_medR
            + " R@1 = %.4f" % epoch_rec_k1
            + " R@5 = %.4f" % epoch_rec_k5
            + " R@10 = %.4f" % epoch_rec_k10
        )

        # Get top and worst K crop filenames
        filenames_stacked = np.array([col for row in filenames for col in row])
        ranks_stacked = np.array([col for row in ranks for col in row])
        order_ranks = np.argsort(ranks_stacked)
        Ktop = self.batch_size  # default 32
        topK = filenames_stacked[order_ranks][:Ktop]
        worstK = filenames_stacked[order_ranks][::-1][:Ktop]
        topK_ranks = ranks_stacked[order_ranks][:Ktop]
        worstK_ranks = ranks_stacked[order_ranks][::-1][:Ktop]
        return (
            epoch_loss,
            epoch_prec,
            epoch_rec,
            epoch_acc,
            epoch_fscore,
            epoch_medR,
            epoch_rec_k1,
            epoch_rec_k5,
            epoch_rec_k10,
            topK,
            worstK,
            topK_ranks,
            worstK_ranks,
        )

    def validate(self, val_loader: Any, epoch: Any) -> Any:
        with torch.no_grad():
            # val
            self.net.eval()  # val mode
            losses = np.array([])
            precs = np.array([])
            recs = np.array([])
            accs = np.array([])
            fscores = np.array([])
            medRs = np.array([])
            recs_k1 = np.array([])
            recs_k5 = np.array([])
            recs_k10 = np.array([])
            # xbatches=[x for bix,(x,y) in enumerate(val_loader)]
            # ybatches=[y for bix,(x,y) in enumerate(val_loader)]
            filenames = []
            ranks = []
            for bidx, (filename, param, x, y) in enumerate(val_loader):
                # join regression batch for crops and modifiers
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
                if len(self.params) == 1:
                    target = torch.eye(self.num_regs[0])[yreg]
                    pred = torch.nn.Sigmoid()(prediction)
                    if self.cuda:
                        target = target.cuda()
                    loss = self.criterion(pred, target)  # yreg as alternative (classes)
                    # output to soft encoding (threshold output and compute) to get TP,FP...
                    output_soft = soft2hard(prediction, self.soft_threshold)
                    prec = get_precision(output_soft, target)
                    rec = get_recall(output_soft, target)
                    acc = get_accuracy(output_soft, target)
                    fscore = get_fscore(output_soft, target)
                    medR, rank = get_median_rank(prediction, target)
                    rec_k1 = get_recall_rate(prediction, target, 1)
                    rec_k5 = get_recall_rate(prediction, target, 5)
                    rec_k10 = get_recall_rate(prediction, target, 10)
                    # Debug
                    if self.debug:
                        print(
                            "Batch("
                            + str(bidx)
                            + "/"
                            + str(val_loader.__len__())
                            + ")"
                            + " Precision="
                            + str(float(prec))
                            + " Recall="
                            + str(float(rec))
                            + " Accuracy="
                            + str(float(acc))
                            + " F-Score="
                            + str(float(fscore))
                            + " medR="
                            + str(float(medR))
                            + " R@1="
                            + str(float(rec_k1))
                            + " R@5="
                            + str(float(rec_k5))
                            + " R@10="
                            + str(float(rec_k10))
                        )
                    """
                    par = self.params[0]
                    for i, tgt in enumerate(target):
                        output_json = {par: self.yclasses[par][prediction[i].argmax()]}
                        target_json = {par: self.yclasses[par][target[i].argmax()]}
                        # print("target "+str(target_json)+" pred "+str(output_json)+" batch "+str(bidx+1))
                    """
                else:
                    # compute losses differently for each head
                    loss = 0.0
                    pprec = []
                    preca = []
                    pacc = []
                    pfscore = []
                    pmedR = []
                    preca_k1 = []
                    preca_k5 = []
                    preca_k10 = []
                    param_ids = [self.dict_params[par] for par in param]
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
                        param_prediction = prediction[pidx][param_indices]
                        param_pred = torch.nn.Sigmoid()(param_prediction)
                        if self.cuda:
                            param_target = param_target.cuda()
                        param_loss = self.criterion(
                            param_pred, param_target
                        )  # one loss for each param
                        loss += param_loss  # final loss is sum of all param BCE losses

                        # output to soft encoding (threshold output and compute) to get TP,FP...
                        output_soft = soft2hard(param_prediction, self.soft_threshold)
                        param_prec = get_precision(output_soft, param_target)
                        param_preca = get_recall(output_soft, param_target)
                        param_pacc = get_accuracy(output_soft, param_target)
                        param_pfscore = get_fscore(output_soft, param_target)
                        param_medR, rank = get_median_rank(
                            param_prediction, param_target
                        )
                        param_rec_k1 = get_recall_rate(
                            param_prediction, param_target, 1
                        )
                        param_rec_k5 = get_recall_rate(
                            param_prediction, param_target, 5
                        )
                        param_rec_k10 = get_recall_rate(
                            param_prediction, param_target, 10
                        )
                        pprec.append(param_prec)
                        preca.append(param_preca)
                        pacc.append(param_pacc)
                        pfscore.append(param_pfscore)
                        pmedR.append(param_medR)
                        preca_k1.append(param_rec_k1)
                        preca_k5.append(param_rec_k5)
                        preca_k10.append(param_rec_k10)
                        """
                        par = self.params[0]
                        # print json values (converted from onehot to param interval values)
                        for i, tgt in enumerate(param_target):
                            output_json = {
                                par: self.yclasses[par][param_pred[i].argmax()]
                            }
                            target_json = {
                                par: self.yclasses[par][param_target[i].argmax()]
                            }
                            # print("target "+str(target_json)+" pred "+str(output_json)+" batch "+str(bidx+1))
                        """
                    prec = np.nanmean(pprec)
                    rec = np.nanmean(preca)
                    acc = np.nanmean(pacc)
                    fscore = np.nanmean(pfscore)
                    medR = np.nanmean(pmedR)
                    rec_k1 = np.nanmean(preca_k1)
                    rec_k5 = np.nanmean(preca_k5)
                    rec_k10 = np.nanmean(preca_k10)
                    # Debug
                    if self.debug:
                        print(
                            "Batch("
                            + str(bidx)
                            + "/"
                            + str(val_loader.__len__())
                            + ")"
                            + " Precision="
                            + str(float(prec))
                            + " Recall="
                            + str(float(rec))
                            + " Accuracy="
                            + str(float(acc))
                            + " F-Score="
                            + str(float(fscore))
                            + " medR="
                            + str(float(medR))
                            + " R@1="
                            + str(float(rec_k1))
                            + " R@5="
                            + str(float(rec_k5))
                            + " R@10="
                            + str(float(rec_k10))
                        )
                precs = np.append(precs, prec)
                recs = np.append(recs, rec)
                accs = np.append(accs, acc)
                fscores = np.append(fscores, fscore)
                medRs = np.append(medRs, medR)
                recs_k1 = np.append(recs_k1, rec_k1)
                recs_k5 = np.append(recs_k5, rec_k5)
                recs_k10 = np.append(recs_k10, rec_k10)

                if self.cuda:
                    losses = np.append(losses, loss.data.cpu().numpy())
                else:
                    losses = np.append(losses, loss.data.numpy())

                filenames.append(filename)
                ranks.append(rank)

            epoch_loss = np.nanmean(losses)
            epoch_prec = np.nanmean(precs)
            epoch_rec = np.nanmean(recs)
            epoch_acc = np.nanmean(accs)
            epoch_fscore = np.nanmean(fscores)
            epoch_medR = np.nanmean(medRs)
            epoch_rec_k1 = np.nanmean(recs_k1)
            epoch_rec_k5 = np.nanmean(recs_k5)
            epoch_rec_k10 = np.nanmean(recs_k10)
            print(
                "Val Step = %d" % epoch
                + " Loss = %.4f" % epoch_loss
                + " Precision = %0.4f" % epoch_prec
                + " Accuracy = %.4f" % epoch_acc
                + " Recall = %.4f" % epoch_rec
                + " Fscore = %.4f" % epoch_fscore
                + " medR = %.4f" % epoch_medR
                + " R@1 = %.4f" % epoch_rec_k1
                + " R@5 = %.4f" % epoch_rec_k5
                + " R@10 = %.4f" % epoch_rec_k10
            )

            # Get top and worst K crop filenames
            filenames_stacked = np.array([col for row in filenames for col in row])
            ranks_stacked = np.array([col for row in ranks for col in row])
            order_ranks = np.argsort(ranks_stacked)
            Ktop = self.batch_size  # default 32
            topK = filenames_stacked[order_ranks][:Ktop]
            worstK = filenames_stacked[order_ranks][::-1][:Ktop]
            topK_ranks = ranks_stacked[order_ranks][:Ktop]
            worstK_ranks = ranks_stacked[order_ranks][::-1][:Ktop]

            return (
                epoch_loss,
                epoch_prec,
                epoch_rec,
                epoch_acc,
                epoch_fscore,
                epoch_medR,
                epoch_rec_k1,
                epoch_rec_k5,
                epoch_rec_k10,
                topK,
                worstK,
                topK_ranks,
                worstK_ranks,
            )


if __name__ == "__main__":
    parser = parse_params_cfg()
    args, uk_args = parser.parse_known_args()
    print(args)
    print("Preparing Regressor")
    reg = Regressor(args)

    # plot stats
    stats_file = reg.output_path + "/" + "stats.csv"
    if args.plot_only and os.path.exists(stats_file):
        results_array = np.loadtxt(stats_file, delimiter=",")
        diff_epoch = reg.epochs - len(results_array[0])
        if diff_epoch > 0:
            print("Found " + stats_file + " plotting...")
            plot_single(reg.output_path)
            plot_benchmark(os.path.dirname(reg.output_path))
            exit()
        else:
            print("Found " + stats_file + " but not all epoch, re-training")
            loaded_ckpt = reg.load_ckpt()
            args.resume = False  # train instead of deploy
            # reg.epochs = reg.epochs - diff_epoch # stats will not be appended so run all again

    # TRAIN+VAL (depending if checkpoint exists)
    if args.resume is False:
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
        if args.data_only is True:
            exit()
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
