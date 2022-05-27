from typing import Any

import torch
import torch.nn as nn

from iquaflow.quality_metrics import (
    GaussianBlurMetrics,
    GSDMetrics,
    NoiseSharpnessMetrics,
    RERMetrics,
    ScoreMetrics,
    SNRMetrics,
)

"""# usage:
1. replace your model argparser by argparse_regressor_loss():
    argparser = argparse_regressor_loss(argparsler)
2. when initializing your model losses, just define your QMRloss
    if opt.regressor_loss is not None:
        global quality_metric
        global quality_metric_criterion
        quality_metric , quality_metric_criterion = init_regressor_loss(opt)
        print("Using regressor loss")
3. during training loop, apply your regressor loss function, running the respecting QMR model (RER,SNR,GSD, Blur,Sharpness,Score etc.)
    if opt.regressor_loss is not None:
        regressor_loss, img_reg, pred_reg = apply_regressor_loss(img_hr,output,quality_metric,quality_metric_criterion,opt,loss,loss_spatial)
        loss = loss + regressor_loss
"""


def argparse_regressor_loss(
    argparser: Any,
) -> Any:  # Add QMRLoss params to your model argument parser (training execution)
    argparser.add_argument(
        "--regressor_loss", default=None, type=str, help="Regressor quality metric"
    )
    argparser.add_argument(
        "--regressor_criterion",
        default=None,
        type=str,
        help="Criterion for regressor quality metric loss",
    )
    argparser.add_argument(
        "--regressor_loss_factor",
        type=float,
        default=1.0,
        help="Constant to multiply by loss",
    )
    argparser.add_argument(
        "--regressor_zeroclamp",
        action="store_true",
        help="Clamp negative regressor losses to 0",
    )
    argparser.add_argument(
        "--regressor_onorm", action="store_true", help="Normalize to original loss mean"
    )
    argparser.add_argument(
        "--regressor_gt2onehot",
        action="store_true",
        help="Make HR regressor outputs to binary, upon max",
    )
    return argparser


def init_regressor_loss(
    opt: Any,
) -> Any:  # opt must be the output of argparse after parse_args(), containing "regressor_loss", "regressor_criterion" and "cuda"
    if opt.regressor_loss == "rer":
        quality_metric = RERMetrics()
    elif opt.regressor_loss == "snr":
        quality_metric = SNRMetrics()  # type: ignore
    elif opt.regressor_loss == "sigma":
        quality_metric = GaussianBlurMetrics()  # type: ignore
    elif opt.regressor_loss == "sharpness":
        quality_metric = NoiseSharpnessMetrics()  # type: ignore
    elif opt.regressor_loss == "scale":
        quality_metric = GSDMetrics()  # type: ignore
    elif opt.regressor_loss == "score":
        quality_metric = ScoreMetrics()  # type: ignore
    if opt.regressor_criterion is None:
        quality_metric_criterion = nn.BCELoss(reduction="mean")
    elif opt.regressor_criterion == "BCELoss":
        quality_metric_criterion = nn.BCELoss(reduction="none")
    elif opt.regressor_criterion == "BCELoss_mean":
        quality_metric_criterion = nn.BCELoss(reduction="mean")
    elif opt.regressor_criterion == "BCELoss_sum":
        quality_metric_criterion = nn.BCELoss(reduction="sum")
    elif opt.regressor_criterion == "L1Loss":
        quality_metric_criterion = nn.L1Loss(reduction="none")  # type: ignore
    elif opt.regressor_criterion == "L1Loss_mean":
        quality_metric_criterion = nn.L1Loss(reduction="mean")  # type: ignore
    elif opt.regressor_criterion == "L1Loss_sum":
        quality_metric_criterion = nn.L1Loss(reduction="sum")  # type: ignore
    elif opt.regressor_criterion == "MSELoss":
        quality_metric_criterion = nn.MSELoss(reduction="none")  # type: ignore
    elif opt.regressor_criterion == "MSELoss_mean":
        quality_metric_criterion = nn.MSELoss(reduction="mean")  # type: ignore
    elif opt.regressor_criterion == "MSELoss_sum":
        quality_metric_criterion = nn.MSELoss(reduction="sum")  # type: ignore
    quality_metric_criterion.eval()
    quality_metric.regressor.net.eval()
    if opt.cuda:
        quality_metric_criterion.cuda()
        quality_metric.regressor.net.cuda()
    return quality_metric, quality_metric_criterion


def apply_regressor_loss(
    gt: Any,
    pred: Any,
    quality_metric: Any,
    quality_metric_criterion: Any,
    opt: Any,
    loss: Any = None,
    loss_spatial: Any = None,
) -> Any:
    output_reg = quality_metric.regressor.net(pred)
    pred_reg = nn.Sigmoid()(output_reg)
    img_reg = quality_metric.regressor.net(gt)
    regressor_loss = quality_metric_criterion(pred_reg, img_reg.detach())
    print("Original Loss")
    print(loss)
    # checking other hyperparams
    if opt.regressor_gt2onehot is True:
        img_reg_bin = torch.zeros_like(img_reg)
        for idx, hot in enumerate(img_reg):
            img_reg_bin[idx, hot.argmax()] = torch.ones_like(
                img_reg_bin[idx, hot.argmax()]
            )
        regressor_loss = quality_metric_criterion(pred_reg, img_reg_bin.detach())
    if (opt.regressor_zeroclamp is True) and (regressor_loss < 0):
        regressor_loss = -regressor_loss * 0  # make 0 conserving tensor type
    if opt.regressor_loss_factor != 1.0:  # multiply by constant factor
        regressor_loss = regressor_loss * opt.regressor_loss_factor
    if opt.regressor_onorm is True:
        regressor_loss = regressor_loss * torch.mean(loss_spatial)
    print("Regressor Loss")
    print(regressor_loss)
    return regressor_loss, img_reg, pred_reg
