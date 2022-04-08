import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import signal
from scipy.stats import norm
from torchvision import transforms

from iquaflow.datasets import DSModifier, DSModifier_dir

logger = logging.getLogger(__name__)


class DSModifier_rer(DSModifier_dir):
    """
    Class that modifies a dataset in a folder compressing it with JPG encoding at given quality.

    Args:
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier (at least 'quality')

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """

    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        get_initial_from_dataset: Optional[str] = None,
        params: Dict[str, Any] = {"rer": 0.2, "dataset": "xview"},
    ):
        self.name = f"rer{params['rer']}_modifier"
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})
        self.BlurImageModifier = BlurImage()
        if get_initial_from_dataset is not None:
            self.init_RER = self.BlurImageModifier.get_initial_values_from_json(
                get_initial_from_dataset
            )
        elif "dataset" in self.params.keys():
            self.init_RER = self.BlurImageModifier.get_initial_values_from_json(
                self.params["dataset"]
            )
        else:
            self.init_RER = self.params["initial_rer"]

    def _mod_img(self, img: np.array) -> np.array:
        par = self.params["rer"]
        # image_tensor = transforms.functional.to_tensor(img)  # .unsqueeze_(0)
        proc_img = self.BlurImageModifier.apply_blur_to_image(
            img, self.init_RER, desired_RER=par
        )
        rec_img = np.asarray(transforms.functional.to_pil_image(proc_img))
        return rec_img


class BlurImage:
    """
    Class that can be used by dataset modifier for modifying images within datsets.
    Applies a certain blur to achieve desired blur.
    """

    def __init__(self, kernel_size: int = 15):
        self.kernel_size = kernel_size
        self.initial_RER = 0.54

    def get_initial_values_from_json(self, name: str) -> float:
        current_path = os.path.dirname(os.path.realpath(__file__))
        with open(current_path + "/dataset_labels/RER/{}_RER.json".format(name)) as f:
            json_dict = json.load(f)
            self.initial_RER = json_dict["RER"]
            return self.initial_RER

    def _check_uint8(self, image: np.array) -> None:
        if image.dtype != np.uint8:
            raise TypeError("Expecting np.uint8, received {}".format(image.dtype()))

    def _make_gaussian_kernel_from_sigma(self, sigma: float) -> np.array:
        half_kernel_size = math.floor(self.kernel_size / 2)

        x, y = np.meshgrid(
            np.linspace(-half_kernel_size, half_kernel_size, self.kernel_size),
            np.linspace(-half_kernel_size, half_kernel_size, self.kernel_size),
        )
        dst = np.sqrt(x * x + y * y)
        mu = 0.0
        gaussian = (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-((dst - mu) ** 2 / (2.0 * sigma**2)))
        )
        gaussian /= np.sum(gaussian)

        return gaussian

    def _return_lsf_from_rer(self, RER: float, threshold: float = 0.001) -> List[Any]:
        half_kernel_size = math.floor(self.kernel_size / 2)
        x_axis = np.arange(-half_kernel_size, half_kernel_size, 0.01)
        sigma = 1
        lsf_guess = norm.pdf(x_axis, 0, sigma)
        RER_guess = self._return_rer_from_lsf(x_axis, lsf_guess)
        while abs(RER - RER_guess) > threshold:
            sigma += RER_guess - RER
            lsf_guess = norm.pdf(x_axis, 0, sigma)
            RER_guess = self._return_rer_from_lsf(x_axis, lsf_guess)
        x_axis_guess = np.linspace(-half_kernel_size, half_kernel_size, len(lsf_guess))
        return [x_axis_guess, lsf_guess, sigma]

    def where_value(self, array: Any, value: Any) -> Any:
        return np.abs(array - value).argmin()

    def _return_rer_from_lsf(self, x_axis: Any, lsf: Any) -> Any:
        max_value_edge_response = np.sum(lsf)
        left_edge_response = (
            np.sum(lsf[: self.where_value(x_axis, -0.5)]) / max_value_edge_response
        )
        right_edge_response = (
            np.sum(lsf[: self.where_value(x_axis, 0.5)]) / max_value_edge_response
        )
        RER = float((right_edge_response - left_edge_response) / (0.5 - (-0.5)))
        return RER

    def _calculate_sigma_for_kernel(self, image_RER: Any, desired_RER: Any) -> Any:
        image_x_axis, image_lsf, image_sigma = self._return_lsf_from_rer(image_RER)
        desired_x_axis, desired_lsf, desired_sigma = self._return_lsf_from_rer(
            desired_RER
        )
        if desired_sigma > image_sigma:
            sigma_kernel = np.sqrt(desired_sigma**2 - image_sigma**2)
        else:
            sigma_kernel = 0
        return sigma_kernel

    def apply_blur_to_image(
        self,
        image: np.array,
        image_RER: Optional[float] = None,
        desired_RER: Optional[float] = None,
    ) -> np.array:
        """
        Applies a certain amount of blur to an image to achieve desired amount of blur.

        Given in image with a measured amount of blur (defined by the RER, the Relative Edge Response),
        we determine the blur kernel necessary to blur the image to a desired level of blur (again measured
        as RER).

        On accuracy of reaching desired RER:
        RER varies between bands, but accounting for this, taking the average RER of an image, we can expect
        the new RER to be with +/- 0.04 of the actual value.  Therefore, when aiming to have an RER within some range
        say, between 0.35-0.43, it is best to choose a middle value, such as 0.39.

        TODO:Future work will reduce this uncertainty by first correcting edge-overshoot in images.

        Args:
            image(np.ndarray, uint8): rgb image to be blurred
            image_RER(float): a number between 0-1, corresponding to the measured RER of the image
            desired_RER(float): a number between 0-1, corresponding to the desired RER of the image

        Return:
            np.ndarray, uint8: image with blur corresponding to desired RER value
        """
        self._check_uint8(image)
        if image_RER is None:
            image_RER = self.initial_RER
        if desired_RER is None:
            logger.warning("No desired RER set, setting to default of 0.2")
            desired_RER = 0.2

        sigma_kernel = self._calculate_sigma_for_kernel(image_RER, desired_RER)
        kernel = self._make_gaussian_kernel_from_sigma(sigma_kernel)
        result = np.zeros_like(image)
        for band in range(image.shape[2]):
            result[:, :, band] = signal.convolve2d(
                image[:, :, band], kernel, mode="same"
            )
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
