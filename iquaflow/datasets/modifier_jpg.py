from typing import Any, Dict, Optional

import cv2
import numpy as np

from iquaflow.datasets import DSModifier, DSModifier_dir


class DSModifier_jpg(DSModifier_dir):
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
        params: Dict[str, Any] = {"quality": 65},
    ):
        self.name = f"jpg{params['quality']}_modifier"
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})

    def _mod_img(self, img: np.array) -> np.array:
        par = [cv2.IMWRITE_JPEG_QUALITY, self.params["quality"]]
        retval, tmpenc = cv2.imencode(".jpg", img, par)
        # size_proc+= np.size(tmpenc)
        rec_img = cv2.imdecode(tmpenc, -1)
        return rec_img
