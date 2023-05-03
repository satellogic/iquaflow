import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
from torchvision import transforms

from iquaflow.datasets import DSModifier, DSModifier_dir

logger = logging.getLogger(__name__)


class DSModifier_snr(DSModifier_dir):
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
        params: Dict[str, Any] = {"snr": 20, "dataset": "xview"},
    ):
        self.name = f"snr{params['snr']}_modifier"
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})
        self.NoisyImageModifier = AddNoise()
        if get_initial_from_dataset is not None:
            self.init_SNR = self.NoisyImageModifier.get_initial_values_from_json(
                get_initial_from_dataset
            )
        elif "dataset" in self.params.keys():
            self.init_SNR = self.NoisyImageModifier.get_initial_values_from_json(
                self.params["dataset"]
            )
        else:
            self.init_SNR = self.params["initial_snr"]

    def _mod_img(self, img: np.array) -> np.array:
        par = self.params["snr"]
        # image_tensor = transforms.functional.to_tensor(img)  # .unsqueeze_(0)
        proc_img = self.NoisyImageModifier.increase_image_gaussian_noise(
            img, self.init_SNR, desired_snr=par
        )
        rec_img = np.asarray(proc_img) # np.asarray(transforms.functional.to_pil_image(proc_img))
        return rec_img


class AddNoise:
    """
    Class that can be used by dataset modifier for modifying images within datsets.
    Class to reduce SNR (increase level of noise) in an image from one known value to another.

    Note about Gaussian vs Poisson noise:
    Assuming the image already has some level of Poisson noise, adding Gaussian instead of applying Poisson
    noise results in a comparable DN vs SNR curve.  Applying a known amount of Poisson noise is challenging,
    one must account for details of the sensor including capture bit depth and full electron well values.
    And on datasets from other companies, this information is not readily accessible.  Furthermore, the equations
    for applying Poisson noise are less easily invertible to determine the exact amount of noise to apply to achieve
    the desired result.  For these reasons, so far we only have the option to apply Gaussian noise.
    """

    def __init__(self, initial_snr: float = 31.0):
        self.initial_snr = initial_snr

    def get_initial_values_from_json(self, name: Any) -> Any:
        current_path = os.path.dirname(os.path.realpath(__file__))
        with open(current_path + "/dataset_labels/SNR/{}_SNR.json".format(name)) as f:
            json_dict = json.load(f)
            self.initial_snr = json_dict["SNR"]
            return self.initial_snr

    def _check_uint8(self, image: Any) -> Any:
        if image.dtype != np.uint8:
            raise TypeError("Expecting np.uint8, received {}".format(image.dtype()))

    def _add_gaussian_noise(
        self, image: np.array, mean: float = 0.0, sigma: float = 1.0
    ) -> np.array:
        gaussian = np.random.normal(mean, sigma, (image.shape[0], image.shape[1]))
        image = np.clip(image + gaussian, 0, 255).astype(np.uint8)
        return image

    def increase_image_gaussian_noise(
        self,
        image: np.array,
        original_snr: Optional[float] = None,
        desired_snr: Optional[float] = None,
    ) -> np.array:
        """
        Decreases image SNR from known original amount to new desired value by adding Gaussian noise.

        On the accuracy of reaching the desired SNR:
        The new SNR of the image will be within desired SNR value +/-3, so when aiming to have the SNR within
        some range say, between 25-30, it is best to choose a middle value, such as 27.5.

        Args:
            image(np.ndarray, uint8): rgb image to be modified
            original_snr(int or float): a number, corresponding to the measured SNR of the image
            desired_snr(int or float): a number, corresponding to the desired SNR of the image

        Return:
            np.ndarray, uint8: image with an SNR corresponding to desired SNR value +/-3

        """
        self._check_uint8(image)
        if original_snr is None:
            original_snr = self.initial_snr
        if desired_snr is None:
            logger.warning("No desired SNR set, setting to default of 20")
            desired_snr = 20

        # use mean value as signal
        mean = np.mean(image)
        sigma_image = mean / original_snr
        sigma_desired = mean / desired_snr
        if sigma_desired > sigma_image:
            sigma_to_add = np.sqrt(sigma_desired**2 - sigma_image**2)
        else:
            sigma_to_add = 0.0

        # add gaussian noise
        result = np.zeros_like(image)
        for band in range(image.shape[2]):
            result[:, :, band] = self._add_gaussian_noise(
                image[:, :, band], mean=0, sigma=sigma_to_add
            )
        return result
