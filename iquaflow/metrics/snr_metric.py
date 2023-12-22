import os
import warnings
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.stats import sigmaclip
from skimage.feature import local_binary_pattern
from skimage.util.shape import view_as_windows

from iquaflow.metrics import Metric

warnings.filterwarnings("ignore")

DEBUG_DIR_NAME = "./snr_hmg_thres_debug/"


class SNRBase:
    """
    Base class for the different methods to measure SNR. Includes functions that are used in more than one
    method.
    """

    def __init__(self, each_channel: bool = False, debug: bool = False):
        self.patches: Tuple[List[Any]] = ([],)
        self.patch_size = None
        self.img_median = None
        self.each_channel = each_channel
        self.debug = debug

    def apply_one_channel(self, image: np.array) -> Tuple[float, float]:
        pass

    def apply(self, image: np.array) -> float:
        """
        Iterates through the different channels of the image, calls the function
        that calculates the SNR for that channel. In the end it takes the mean
        of the SNRs of the channels and returns it together with the error estimate.
        """
        # self._check_uint8(image)
        snr_list = []
        std_list = []
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        for ch in range(image.shape[2]):
            snr, snr_std = self.apply_one_channel(image[..., ch])
            if self.debug:
                print(
                    f"Channel {ch} at median DN {self.img_median}: SNR={snr} +- {snr_std}"
                )
            if snr:
                snr_list.append(snr)
                std_list.append(snr_std)
            if self.debug and len(self.patches[0]) > 0:
                self.plot_patches_on_image(image[..., ch], f"test_ch{ch}", show=True)
        if len(snr_list) == 1:
            return snr_list[0], std_list[0]
        if len(snr_list) == 0:
            return np.nan, np.nan
        snr_list = np.array(snr_list)
        std_list = np.array(std_list)
        if self.each_channel:
            out = {
                f"channel {i}": (snr, err)
                for i, (snr, err) in enumerate(zip(snr_list, std_list))
            }
            out["mean"] = np.nanmean(snr_list), max(
                np.sqrt(np.sum(std_list[~np.isnan(std_list)] ** 2))
                / len(std_list[~np.isnan(std_list)]),
                np.nanstd(snr_list),
            )
            return out
        return np.nanmean(snr_list), max(
            np.sqrt(np.sum(std_list[~np.isnan(std_list)] ** 2))
            / len(std_list[~np.isnan(std_list)]),
            np.nanstd(snr_list),
        )

    def plot_patches_on_image(self, img: np.array, fn: str, show: bool = False) -> None:
        """
        Plot the image and the good patches.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        im = np.where(img == 0, np.nan, img)
        ax.imshow(im)
        for p in np.argwhere(self.patches) * self.patch_size:
            rect = mpl_patches.Rectangle(
                (p[1], p[0]),
                self.patch_size,
                self.patch_size,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

        os.makedirs(os.path.dirname(DEBUG_DIR_NAME), exist_ok=True)
        plt.savefig(os.path.join(DEBUG_DIR_NAME, f"{fn}.png"))
        if show:
            plt.show()
        plt.close()

    def _check_uint8(self, image: np.array) -> None:
        if image.dtype != np.uint8:
            raise TypeError("Expecting np.uint8, received {}".format(image.dtype))


class SNRHomogeneousArea(SNRBase):
    """
    Implementation of the Homogeneous Area method to measure SNR.
    This method uses local binary patterns (LBP) to find flat patches with the given size
    in the image. It measures the SNR of each flat patch as SNR = mean/std of the pixel values.
    It bins the measurements into DNs according to the mean pixel value, and calculates the mean
    SNR for each bin. It returns the SNR of the bin that corresponds to the median pixel value of
    the image.

    Args:
        patch_size: the size of patches in pixels
        stride: optional, default equals to the patch_size
        radius: radius for LBP, default is 1
        lbp_threshold: the threshold to select good patches, default is 0.55
    """

    def __init__(
        self,
        patch_size: int = 5,
        stride: Optional[int] = None,
        radius: int = 1,
        lbp_threshold: float = 0.6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size  # type: ignore
        self.stride = stride or self.patch_size
        self.radius = radius
        self.lbp_threshold = lbp_threshold

    def find_patches_lbp(
        self,
        image: np.array,
        window_size: int,
        stride: int,
        radius: int,
        lbp_threshold: float,
    ) -> np.array:
        """
        Function that searches for patches that are considered flat.
        It first calculates the local binary pattern for the image using the given
        radius. Then it checks each image patch defined by the window_size
        and stride and checks the ratio of the pixels that have lbp values in the
        given range. If that ratio is higher than the lbp_threshold, the patch is
        good added to self.patches.

        Args:
            image: the main image where we search for patches
            window_size: the size of the patch:
            stride: the stride  to go from one patch to another
            radius: the radius for the local binary pattern function
            lbp_threshold: the threshold above which we considere a patch to be good
        """
        img_height, img_width = image.shape
        METHOD = "uniform"
        n_points = 8 * radius
        # The highest and lowest lbp values are considered as flat pattern.
        limit1 = radius - 1
        limit2 = n_points - radius + 1
        lbp = local_binary_pattern(image, n_points, radius, METHOD)

        lbp_windows = view_as_windows(lbp, (window_size, window_size), stride)
        r = np.logical_or(lbp_windows >= limit2, lbp_windows <= limit1).sum(
            axis=(2, 3)
        ) / (lbp_windows.shape[2] * lbp_windows.shape[3])

        self.patches = np.logical_and(r >= lbp_threshold, r < 1)

    def get_snr_value(self, img: np.array) -> Tuple[Optional[float], Optional[float]]:
        img = img.astype(float)
        img[img == 0] = np.nan
        self.img_median = np.nanmedian(img)

        if self.patches.sum() < 10:  # type: ignore
            if self.debug:
                print("Not enough patches")
            return None, None

        if self.debug:
            print(
                f"Median value: {self.img_median}, dynamic range: {np.nanmin(img)} - {np.nanmax(img)}"
            )
        m = np.mean(
            view_as_windows(img, (self.patch_size, self.patch_size), self.stride),
            axis=(3, 2),
        )
        s = np.std(
            view_as_windows(img, (self.patch_size, self.patch_size), self.stride),
            axis=(3, 2),
            ddof=1,
        )

        m_ = m[self.patches].astype(int).flatten()
        s_ = s[self.patches].flatten()
        s_[s_ == 0] = np.nan
        self.snr = m_ / s_
        self.snrs = {i: self.snr[np.argwhere(m_ == i)].flatten() for i in range(65536)}
        if self.snr[np.argwhere(m_ == self.img_median)].flatten().shape[0] < 3:
            if self.debug:
                print(f"Not enough SNR values at median value of {self.img_median}")
                print(self.snr[np.argwhere(m_ == self.img_median)].flatten())
            return np.nan, np.nan

        return np.nanmean(
            self.snr[np.argwhere(m_ == self.img_median)].flatten()
        ), np.nanstd(self.snr[np.argwhere(m_ == self.img_median)].flatten())

    def apply_one_channel(self, image: np.array) -> Any:
        self.find_patches_lbp(
            image, self.patch_size, self.stride, self.radius, self.lbp_threshold  # type: ignore
        )

        return self.get_snr_value(image)


class SNRHomogeneousBlocks(SNRBase):
    """
    Implements the Homogeneous Blocks algorithm for measuring SNR.
    During this method, all patches with the given patch size and stride are considered.
    The mean and standard deviation of the pixel values are calculated for each patch.
    The std values are binned to DNs according to the patch mean value. Then the bin corresponding
    to the image median pixel value is selected, and the noise is defined as the median value of
    the std values in that bin (after applying sigmaclipping). The SNR is defined as the image
    median pixel value divided by the noise.

    Args:
        patch_size:
        stride: optional, default equals to patch_size
    """

    def __init__(
        self, patch_size: int = 3, stride: Optional[int] = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size  # type: ignore
        self.stride = self.patch_size if not stride else stride

    def get_snr_value(self, img: np.array) -> Tuple[float, float]:
        # img_width, img_height = img.shape
        # if img.dtype == np.uint8:
        #     self.snrs: dict[int,list] = {i: [] for i in range(256)}
        # elif img.dtype == np.uint16:
        #     self.snrs: dict[int,list] = {i: [] for i in range(65536)}

        self.img_median = np.median(img[img > 0], axis=None)
        img = img.astype(float)
        img = np.where(img == 0, np.nan, img)
        # m = block_reduce(img, (self.patch_size, self.patch_size), np.nanmean).astype(int).flatten()
        # s = block_reduce(img, (self.patch_size, self.patch_size), np.nanstd).flatten()

        m = np.rint(
            np.mean(
                view_as_windows(img, (self.patch_size, self.patch_size), self.stride),
                axis=(3, 2),
            )
        ).flatten()
        s = np.std(
            view_as_windows(img, (self.patch_size, self.patch_size), self.stride),
            axis=(3, 2),
            ddof=1,
        ).flatten()
        if self.debug:
            self.snrs = (m, s)
        l_var = s[np.argwhere(m == self.img_median)]
        ll, low, up = sigmaclip(l_var[~np.isnan(l_var)], high=2)
        noise, sigma_noise = np.nanmedian(ll), np.nanstd(ll)

        return self.img_median / noise, self.img_median / (noise * noise) * sigma_noise

    def apply_one_channel(self, image: np.array) -> Tuple[float, float]:
        return self.get_snr_value(image)


def snr_function_from_array(
    img: np.array,
    ext: str = "tif",
    method: str = "HB",
    params: dict = {"patch_size": 3, "debug": False},  # type: ignore
) -> Tuple[Any, Any]:
    """
    Similar to snr_function_from_fn
    """
    if img.shape[0] < 5:
        img = np.moveaxis(img, 0, -1)
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    if method == "HA":
        SNR = SNRHomogeneousArea(**params)
    elif method == "HB":
        SNR = SNRHomogeneousBlocks(**params)  # type: ignore
    return SNR.apply(img)  # type: ignore


def snr_function_from_fn(
    image: str,
    ext: str = "tif",
    method: str = "HB",
    params: dict = {"patch_size": 3, "debug": False},  # type: ignore
) -> Tuple[Any, Any]:
    """
    Generic function to apply either SNR algorithm for an image.

    Args:
        image_path: the path to your image
        ext: the extension of the image
        method: the algorithm you want to use. Can be HA for SNRHomogeneousArea or HB for SNRHomogeneousBlocks
        params: the arguments that need to be passed to the algorithm. These can be the following:

            - HA algorithm:

                -- patch_size: the size of patches, default is 15
                -- stride: optional, default equals to the patch_size
                -- radius: radius for LBP, default is 1
                -- lbp_threshold: the threshold to select good patches, default is 0.55
                -- each_channel: return the SNR for each channel. If False, only returns the mean value
                -- debug: if True, returns intermediate values, and plots the image with the good patches. Use carefully
                    for large images

            - HB algorithm:

                -- patch_size: the size of patches, default is 3
                -- stride: optional, default equals to patch_size
                -- each_channel: return the SNR for each channel. If False, only returns the mean value
                -- debug: if True, returns intermediate values
    """
    if ext == "tif":
        with rasterio.open(image, "r") as data:
            img = data.read()
    else:
        img = cv2.imread(image)
    if img.shape[0] < 5:
        img = np.moveaxis(img, 0, -1)
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    if method == "HA":
        SNR = SNRHomogeneousArea(**params)
    elif method == "HB":
        SNR = SNRHomogeneousBlocks(**params)  # type: ignore
    return SNR.apply(img)  # type: ignore


class SNRMetric(Metric):
    def __init__(
        self, experiment_info: Any, ext: str = "tif", method: str = "HB", **kwargs: Any
    ) -> None:
        self.experiment_info = experiment_info
        self.ext = ext
        self.metric_names = ["snr_median", "snr_mean", "snr_std"]
        self.kwargs = kwargs
        self.method = method

    def apply(self, predictions: str, gt_path: str) -> Dict[str, float]:
        # These are actually attributes from ds_wrapper
        self.data_path = os.path.dirname(gt_path)
        self.parent_folder = os.path.dirname(self.data_path)
        # predictions be like /mlruns/1/6f1b6d86e42d402aa96665e63c44ef91/artifacts'
        guessed_run_id = predictions.split(os.sep)[-3]
        modifier_subfold = [
            k
            for k in self.experiment_info.runs
            if self.experiment_info.runs[k]["run_id"] == guessed_run_id
        ][0]
        glob_crit = os.path.join(
            self.parent_folder, modifier_subfold, "*", f"*.{self.ext}"
        )
        pred_fn_lst = glob(glob_crit)
        snrlst = []
        for pred_fn in pred_fn_lst:
            snr, _ = snr_function_from_fn(pred_fn, self.ext, self.method, **self.kwargs)
            snrlst.append(snr)
        return {
            k: v
            for k, v in zip(
                self.metric_names,
                [np.nanmedian(snrlst), np.nanmean(snrlst), np.nanstd(snrlst)],
            )
        }
