from typing import Any, List

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score  # type: ignore


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
            hot[prediction[idx].argmax()] = 1
            output_hard[idx] = hot
    return output_hard


def soft2binary(prediction: Any, threshold: float = 0.3) -> Any:
    output_binary = torch.clone(prediction)
    output_binary[output_binary >= threshold] = 1
    output_binary[output_binary <= threshold] = 0
    for idx, hot in enumerate(output_binary):
        if hot.sum() == 0:
            hot[prediction[idx].argmax()] = 1
            output_binary[idx] = hot
    return output_binary


def get_precision(output: Any, target: Any, threshold: float = 0.5) -> Any:
    # calculate precision
    TP = float(torch.sum((output >= threshold) & ((target == 1))))
    FP = float(torch.sum((output >= threshold) & (target == 0)))
    if (TP + FP) > 0:
        prec = torch.tensor((TP) / (TP + FP))
    else:
        prec = torch.tensor(0.0)
    return prec


def get_accuracy(output: Any, target: Any, threshold: float = 0.5) -> Any:
    # calculate precision
    TP = float(torch.sum((output >= threshold) & ((target == 1))))
    FP = float(torch.sum((output >= threshold) & (target == 0)))
    FN = float(torch.sum((output == 0) & (target == 1)))
    if (TP + FP + FN) > 0:
        acc = torch.tensor((TP) / (TP + FP + FN))
    else:
        acc = torch.tensor(0.0)
    return acc


def get_recall(output: Any, target: Any, threshold: float = 0.5) -> Any:
    # calculate precision
    TP = float(torch.sum((output >= threshold) & ((target == 1))))
    FN = float(torch.sum((output == 0) & (target == 1)))
    if (TP + FN) > 0:
        rec = torch.tensor((TP) / (TP + FN))
    else:
        rec = torch.tensor(0.0)
    return rec


def get_fscore(output: Any, target: Any, threshold: float = 0.5) -> Any:
    rec = get_recall(output, target, threshold)
    prec = get_precision(output, target, threshold)
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


def filter_k(prediction: Any, target: Any, K: int = 10) -> Any:
    if target.shape[1] <= K:  # K is equal or bigger than number of intervals
        return prediction, target
    target_window = torch.zeros(target.shape[0], K, dtype=target.dtype)
    prediction_window = torch.zeros(prediction.shape[0], K, dtype=prediction.dtype)
    for idx, values in enumerate(target):
        idx_target = int(torch.argmax(target[idx]))
        Khalf = int(K / 2)  # for odd case, cast is floor
        if idx_target - Khalf <= 0:
            target_window[idx] = target[idx][0:K]
            prediction_window[idx] = prediction[idx][0:K]
        elif idx_target + Khalf >= len(target[idx]):
            target_window[idx] = target[idx][len(target[idx]) - K : len(target[idx])]
            prediction_window[idx] = prediction[idx][
                len(prediction[idx]) - K : len(prediction[idx])
            ]
        else:
            if (K % 2) == 0:  # pair
                target_window[idx] = target[idx][
                    idx_target - Khalf : idx_target + Khalf
                ]
                prediction_window[idx] = prediction[idx][
                    idx_target - Khalf : idx_target + Khalf
                ]
            else:  # odd
                target_window[idx] = target[idx][
                    idx_target - Khalf : idx_target + Khalf + 1
                ]
                prediction_window[idx] = prediction[idx][
                    idx_target - Khalf : idx_target + Khalf + 1
                ]
    return prediction_window, target_window


def get_precision_k(
    output: Any, target: Any, K: int = 10, threshold: float = 0.5
) -> Any:
    output_window, target_window = filter_k(output, target, K)
    if K == 1:
        precision_k = torch.sum(output_window) / len(output_window)
    else:
        precision_k = get_precision(output_window, target_window, threshold)
    return float(precision_k)


def get_recall_k(output: Any, target: Any, K: int = 10, threshold: float = 0.5) -> Any:
    output_window, target_window = filter_k(output, target, K)
    if K == 1:
        recall_k = torch.sum(output_window) / len(output_window)
    else:
        recall_k = get_recall(output_window, target_window, threshold)
    return float(recall_k)


def get_accuracy_k(
    output: Any, target: Any, K: int = 10, threshold: float = 0.5
) -> Any:
    output_window, target_window = filter_k(output, target, K)
    if K == 1:
        accuracy_k = torch.sum(output_window) / len(output_window)
    else:
        accuracy_k = get_accuracy(output_window, target_window, threshold)
    return float(accuracy_k)


def get_AUC(output: Any, target: Any) -> Any:
    AUCs = np.array([])
    for idx, values in enumerate(output):
        if target.is_cuda is True or output.is_cuda is True:
            AUC = roc_auc_score(target[idx].cpu(), output[idx].cpu())
        else:
            AUC = roc_auc_score(target[idx], output[idx])
        AUCs = np.append(AUCs, AUC)
    return np.nanmean(AUCs)


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
