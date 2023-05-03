from typing import Any, Dict, Optional

import numpy as np
from torchvision import transforms

from iquaflow.datasets import DSModifier, DSModifier_dir


class DSModifier_sharpness(DSModifier_dir):
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
        params: Dict[str, Any] = {"sharpness": 2},
    ):
        self.name = f"sharpness{params['sharpness']}_modifier"
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})

    def _mod_img(self, img: np.array) -> np.array:
        par = self.params["sharpness"]
        image_tensor = transforms.functional.to_tensor(img).squeeze(0)
        proc_img = transforms.functional.adjust_sharpness(
            img=image_tensor, sharpness_factor=par
        )
        rec_img = proc_img.numpy() # np.asarray(transforms.functional.to_pil_image(proc_img))
        return rec_img
