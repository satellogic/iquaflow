from typing import Any, List

import cv2
import numpy as np
import torch


def force_rgb(x: Any) -> Any:
    if x.shape[0] < 3:  # if grayscale, replicate channel to rgb
        x = torch.cat([x] * 3)
    elif (
        x.shape[0] > 3
    ):  # if >3 dimensions, use first 3 to discard other dimensions (e.g. depth)
        x = x[0:3]
    return x


def circ3d_pad(mat: Any, desired_shape: Any) -> Any:
    newmat = torch.zeros((mat.shape[0], desired_shape[0], desired_shape[1]))
    v_border = int((desired_shape[0] - mat.shape[1]) * 0.5)
    h_border = int((desired_shape[1] - mat.shape[2]) * 0.5)
    if mat.shape[0] > 1:
        for channel in range(mat.shape[0]):
            newmat_c = np.pad(
                mat[channel, :, :], (v_border, h_border), mode="symmetric"
            )
            vdiff = newmat.shape[1] - newmat_c.shape[0]
            hdiff = newmat.shape[2] - newmat_c.shape[1]
            if vdiff > 0:
                newmat_c = np.vstack((newmat_c, np.tile(newmat_c[[-1], :], (vdiff, 1))))
            if vdiff < 0:
                newmat_c = newmat_c[: newmat_c.shape[0] + vdiff, :]
            if hdiff > 0:
                newmat_c = np.hstack((newmat_c, np.tile(newmat_c[:, [-1]], (1, hdiff))))
            if hdiff < 0:
                newmat_c = newmat_c[:, : newmat_c.shape[1] + hdiff]
            newmat[channel, :, :] = torch.from_numpy(newmat_c)
    else:
        newmat_c = np.pad(mat, (v_border, h_border), mode="symmetric")
        vdiff = newmat.shape[1] - newmat_c.shape[0]
        hdiff = newmat.shape[2] - newmat_c.shape[1]
        if vdiff > 0:
            newmat_c = np.vstack((newmat_c, np.tile(newmat_c[[-1], :], (vdiff, 1))))
        if vdiff < 0:
            newmat_c = newmat_c[: newmat_c.shape[0] + vdiff, :]
        if hdiff > 0:
            newmat_c = np.hstack((newmat_c, np.tile(newmat_c[:, [-1]], (1, hdiff))))
        if hdiff < 0:
            newmat_c = newmat_c[:, : newmat_c.shape[1] + hdiff]
        newmat = torch.from_numpy(newmat_c)
    return newmat


def soft2hard(prediction: Any, threshold: float = 0.3) -> Any:
    output_hard = (torch.sigmoid(prediction) > threshold).float()
    for idx, hot in enumerate(output_hard):
        if hot.sum() == 0:
            hot[output_hard[idx].argmax()] = 1
            output_hard[idx] = hot
    return output_hard


def get_precision(output_soft: Any, target: Any, threshold: float = 0.5) -> Any:
    # calculate precision
    TP = float(torch.sum((output_soft >= threshold) & ((target == 1))))
    FP = float(torch.sum((output_soft >= threshold) & (target == 0)))
    if (TP + FP) > 0:
        prec = torch.tensor((TP) / (TP + FP))
    else:
        prec = torch.tensor(0.0)
    return prec


def get_accuracy(output_soft: Any, target: Any, threshold: float = 0.5) -> Any:
    # calculate precision
    TP = float(torch.sum((output_soft >= threshold) & ((target == 1))))
    FP = float(torch.sum((output_soft >= threshold) & (target == 0)))
    FN = float(torch.sum((output_soft == 0) & (target == 1)))
    if (TP + FP + FN) > 0:
        acc = torch.tensor((TP) / (TP + FP + FN))
    else:
        acc = torch.tensor(0.0)
    return acc


def get_recall(output_soft: Any, target: Any, threshold: float = 0.5) -> Any:
    # calculate precision
    TP = float(torch.sum((output_soft >= threshold) & ((target == 1))))
    FN = float(torch.sum((output_soft == 0) & (target == 1)))
    if (TP + FN) > 0:
        rec = torch.tensor((TP) / (TP + FN))
    else:
        rec = torch.tensor(0.0)
    return rec


def get_fscore(output_soft: Any, target: Any, threshold: float = 0.5) -> Any:
    rec = get_recall(output_soft, target, threshold)
    prec = get_precision(output_soft, target, threshold)
    fscore = (2 * rec * prec) / (rec + prec + np.finfo(np.float32).eps)
    return fscore


def get_median_rank(prediction: Any, target: Any) -> Any:
    ranks = []
    for idx, values in enumerate(prediction):
        idx_output = torch.argmax(prediction[idx])
        idx_target = torch.argmax(target[idx])
        rank = torch.abs(idx_output - idx_target)
        ranks.append(int(rank))
    ranks = np.array(ranks)
    medR = np.median(ranks)
    return medR, ranks


def get_recall_rate(prediction: Any, target: Any, K: int = 10) -> Any:
    ranks_inside_k = []
    for idx, values in enumerate(prediction):
        idx_output = torch.argmax(prediction[idx])
        idx_target = torch.argmax(target[idx])
        rank = torch.abs(idx_output - idx_target)
        if rank <= K:
            ranks_inside_k.append(1)
        else:
            ranks_inside_k.append(0)
    ranks_inside_k = np.array(ranks_inside_k)
    recall_rate_k = np.nanmean(ranks_inside_k)
    return recall_rate_k


def split_list(lists_files: List[str], split_percent: float = 1.0) -> List[str]:
    if split_percent > 0:  # if split is 0.2, select FIRST 20%
        lists_files = lists_files[: int(split_percent * len(lists_files))]
    else:  # if split is negative such as -0.2, select LAST 20%
        lists_files = lists_files[int(split_percent * len(lists_files)) :]
    return lists_files


def check_if_contains_edges(
    image: Any,
    lower_canny_thres: int = 0,
    upper_canny_thres: int = 100,
    percent_edges_threshold: float = 0.02,
) -> Any:
    """
    Checks there are well-defined edges for analyzing RER.
    We keep upper_canny_threshold high, so as to ensure the existance of defined edges.
    We keep percent_edges_threshold in low, because even only a few good edges should be enough to determine
    sharpness.
    """
    edges = cv2.Canny(image, lower_canny_thres, upper_canny_thres)
    return edges[edges == 255].size / edges.size > percent_edges_threshold


def check_if_contains_homogenous(
    image: Any,
    lower_canny_thres: float = 0,
    upper_canny_thres: Any = 30,
    percent_edges_threshold: Any = 0.25,
) -> Any:
    """
    Checks there are homogeneous regions sufficient for analyzing SNR.
    We keep upper_canny_threshold low, so as to include any possibility of edges that may be misinterpreted as noise
    We keep percent_edges_threshold in a middle range, since we can allow edges but want sufficient regions of
    homogeneity.
    """
    edges = cv2.Canny(image, lower_canny_thres, upper_canny_thres)
    return edges[edges == 255].size / edges.size < percent_edges_threshold
