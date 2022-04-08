from typing import Any, Dict, Optional

import numpy as np

from iquaflow.datasets import DSModifier, DSModifier_dir


class DSModifier_quant(DSModifier_dir):
    """
    Class that modifies a dataset in a folder reducing the bit depth of the images.

    Args:
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier (at least 'bits')

    Attributes:
        name: str. Name of the modifier
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
    """

    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {"bits": 4},
    ):
        self.name = f"quant{params['bits']}_modifier"
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})

    def _mod_img(self, img: np.array) -> np.array:
        rec_img = img.copy()
        rec_img = rec_img & (0xFF << (8 - self.params["bits"]))
        return rec_img
