from typing import Any, Dict, Optional

import numpy as np
from torchvision import transforms

from iquaflow.datasets import DSModifier, DSModifier_dir


class DSModifier_blur(DSModifier_dir):
    """
    Class that modifies a dataset in a folder compressing it with Gaussian blurring at given sigma.

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
        params: Dict[str, Any] = {"sigma": 1},
    ):
        self.name = f"blur{params['sigma']}_modifier"
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})

    def _mod_img(self, img: np.array) -> np.array:
        par = self.params["sigma"]
        image_tensor = transforms.functional.to_tensor(img)  # .unsqueeze_(0)
        tGAUSSIAN = transforms.Compose(
            [
                transforms.GaussianBlur(kernel_size=(7, 7), sigma=par),
            ]
        )
        proc_img = tGAUSSIAN(image_tensor)
        rec_img = np.asarray(transforms.functional.to_pil_image(proc_img))
        return rec_img
