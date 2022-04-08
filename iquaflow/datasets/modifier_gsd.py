from typing import Any, Dict, Optional

import numpy as np
from torchvision import transforms

from iquaflow.datasets import DSModifier, DSModifier_dir


class DSModifier_gsd(DSModifier_dir):
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
        params: Dict[str, Any] = {"scale": 1.0, "interpolation": 2, "resol": 0.3},
    ):
        self.params: Dict[str, Any] = params
        self.params["gsd"] = params["resol"] * params["scale"]
        self.name = f"gsd{self.params['gsd']}_modifier"
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})

    def _mod_img(self, img: np.array) -> np.array:
        resize_scale = self.params["scale"]
        interpolation = int(
            self.params["interpolation"]
        )  # type of resize interpolation
        image_tensor = transforms.functional.to_tensor(img)  # .unsqueeze_(0)
        resol_size = (
            int(image_tensor.shape[1] * resize_scale),
            int(image_tensor.shape[2] * resize_scale),
        )
        tRESOL = transforms.Compose(
            [
                transforms.Resize(
                    size=resol_size,
                    interpolation=transforms.functional._interpolation_modes_from_int(
                        interpolation
                    ),
                )
            ]
        )
        proc_img = tRESOL(image_tensor)
        rec_img = np.asarray(transforms.functional.to_pil_image(proc_img))
        return rec_img
