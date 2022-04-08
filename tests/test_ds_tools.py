import os
import shutil
import tempfile
from glob import glob

from iquaflow.datasets import DSModifier_dir, DSWrapper
from iquaflow.sanity import Rasterize


class TestDsTools:
    def test_mask_annotations_dir(self):

        current_path = os.path.dirname(os.path.realpath(__file__))
        orig_path = os.path.join(current_path, "test_datasets", "ds_geo_dataset")

        with tempfile.TemporaryDirectory() as tmp_dir:

            data_path = os.path.join(tmp_dir, "data_path")
            annotfn = os.path.join(data_path, "annots.geojson")
            mask_path = os.path.join(data_path, "images_mask")

            shutil.copytree(orig_path, data_path)

            # Rasterize
            _ = Rasterize(annotfn)

            ds_wrapper = DSWrapper(data_path)

            mod = DSModifier_dir()

            mod.modify_ds_wrapper(ds_wrapper)

            assert (
                ds_wrapper.mask_annotations_dir == mask_path
            ), f"Unexpected mask path {ds_wrapper.mask_annotations_dir}, should be {mask_path}"

            for img_fn in glob(os.path.join(data_path, "images", "*")):
                msk_fn = os.path.join(
                    data_path, "images_mask", os.path.basename(img_fn) + "_mask"
                )
                img_fn_dest = os.path.join(
                    data_path + "#dir_modifier", "images", os.path.basename(img_fn)
                )
                msk_fn_dest = os.path.join(
                    data_path + "#dir_modifier",
                    "images_mask",
                    os.path.basename(img_fn) + "_mask",
                )
                assert os.path.exists(
                    msk_fn
                ), f"File [{msk_fn}] should exist in the origin dataset"
                assert os.path.exists(
                    msk_fn
                ), f"File [{img_fn_dest}] should exist in the destination modified dataset"
                assert os.path.exists(
                    msk_fn
                ), f"File [{msk_fn_dest}] should exist in the destination modified dataset"
