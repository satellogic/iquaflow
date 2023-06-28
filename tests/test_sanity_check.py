import json
import os
import shutil
import tempfile
from glob import glob

import cv2
import geopandas as gpd
from shapely.geometry import Polygon

from iquaflow.datasets import DSWrapper
from iquaflow.sanity import Rasterize, SanityCheck, clean_list


def read_json(filename: str) -> bool:
    with open(filename) as json_file:
        _ = json.load(json_file)
    return True


class TestSanityCheck:
    def test_ds_wrapper_integration(self):
        ds_path = os.path.join("./tests", "test_datasets", "ds_coco_dataset")
        ds_wrapper = DSWrapper(data_path=ds_path)
        with tempfile.TemporaryDirectory() as out_path:
            dss = SanityCheck(ds_wrapper=ds_wrapper, output_path=out_path)
            assert dss.data_path == ds_path

    def test_init(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(current_path, "test_datasets", "ds_coco_dataset")

        with tempfile.TemporaryDirectory() as output_path:
            assert os.path.exists(data_path), data_path + " data_path does not exist"
            assert os.path.exists(output_path), "output_path does not exist"
            assert [el for el in glob(os.path.join(data_path, "*.*json"))][
                0
            ], "Could not retrieve the json / geojson file"
            annotfn = [el for el in glob(os.path.join(data_path, "*.*json"))][0]
            assert read_json(
                annotfn
            ), "Could not read the annotation ground-truth file as json"
            sc = SanityCheck(data_path=data_path, output_path=output_path)
            assert sc.isgeo == annotfn.endswith(
                "geojson"
            ), "is is geo only if annotfn ends with geojson"
            assert sc.data_path == data_path, "missing data_path"
            assert sc.output_path == output_path, "missing output_path"
            assert os.path.exists(
                os.path.join(data_path, "images")
            ), "dataset images does not exist"
            assert os.listdir(
                os.path.join(data_path, "images")
            ), "cannot retrieve list of images within dataset"

    def test_geojson(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        orig_path = os.path.join(current_path, "test_datasets", "ds_geo_dataset")

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = os.path.join(tmp_dir, "data_path")
            output_path = os.path.join(tmp_dir, "output_path")

            assert os.path.exists(orig_path), "orig_path does not exist"
            assert not os.path.exists(
                data_path
            ), "data_path does exists and it shoudn't yet"

            shutil.copytree(orig_path, data_path)

            sc = SanityCheck(data_path=data_path, output_path=output_path)
            assert sc.isgeo, "it should be detected as geojson type of annotations"

            # missing required attribute
            gdf = gpd.read_file(sc.annotfn)
            gdf.loc[0, ["image_filename"]] = None
            gdf_nan = sc._rm_geo_nan(gdf)
            assert list(gdf_nan.index.values.tolist()) == [
                i for i in range(len(gdf.index.values.tolist())) if i != 0
            ], "Remove Nan did not respond as expected"

            # empty geometry
            gdf = gpd.read_file(sc.annotfn)
            gdf.loc[1, ["geometry"]] = None
            gdf_empty = sc._rm_geo_empty(gdf)
            assert list(gdf_empty.index.values.tolist()) == [
                i for i in range(len(gdf.index.values.tolist())) if i != 1
            ], "Remove empty geometries did not respond as expected"

            # Invalid geometry
            gdf = gpd.read_file(sc.annotfn)
            gdf.loc[2, ["geometry"]] = Polygon(
                [
                    (0, 0),
                    (0, 3),
                    (3, 3),
                    (3, 0),
                    (2, 0),
                    (2, 2),
                    (1, 2),
                    (1, 1),
                    (2, 1),
                    (2, 0),
                    (0, 0),
                ]
            )
            gdf_valid = sc._rm_geo_invalid(gdf, try_fix_geoms=False)
            assert list(gdf_valid.index.values.tolist()) == [
                i for i in range(len(gdf.index.values.tolist())) if i != 2
            ], "Remove invalid geometries did not respond as expected"

            # Invalid geometry - fix geoms
            gdf = gpd.read_file(sc.annotfn)
            gdf.loc[2, ["geometry"]] = Polygon(
                [
                    (0, 0),
                    (0, 3),
                    (3, 3),
                    (3, 0),
                    (2, 0),
                    (2, 2),
                    (1, 2),
                    (1, 1),
                    (2, 1),
                    (2, 0),
                    (0, 0),
                ]
            )
            assert list(
                sc._rm_geo_invalid(gdf, try_fix_geoms=True).index.values.tolist()
            ) == list(gdf.index.values.tolist()), "Fix geometries does not work"

            # corrupt some annots for final test
            gdf = gpd.read_file(sc.annotfn)
            gdf.loc[0, ["image_filename"]] = None
            gdf.loc[1, ["geometry"]] = None
            gdf.loc[2, ["geometry"]] = Polygon(
                [
                    (0, 0),
                    (0, 3),
                    (3, 3),
                    (3, 0),
                    (2, 0),
                    (2, 2),
                    (1, 2),
                    (1, 1),
                    (2, 1),
                    (2, 0),
                    (0, 0),
                ]
            )
            gdf.to_file(sc.annotfn, driver="GeoJSON")

            # Autofix
            sc.autofix(
                missing=True,
                empty_geom=True,
                invalid_geom=True,
                try_fix_geoms=True,
            )
            assert os.path.exists(
                os.path.join(sc.output_path, "images")
            ), "images folder does not exist"
            assert os.path.exists(
                os.path.join(sc.output_path, sc.out_annotfn)
            ), "annots file does not exist"
            # initial geojson asserts
            assert (
                len(gpd.read_file(sc.annotfn).index.values.tolist()) == 15
            ), "There should be 15 rows in the dataframe"
            assert (
                len(gpd.read_file(sc.annotfn)["image_filename"].unique()) == 10
            ), "There should be 10 images in the dataframe"

            # sanitized geojson asserts
            assert (
                len(gpd.read_file(sc.out_annotfn).index.values.tolist()) == 13
            ), "There should be 13 rows in the dataframe"
            assert (
                len(gpd.read_file(sc.out_annotfn)["image_filename"].unique()) == 8
            ), "There should be 8 images in the dataframe"
            # TODO test image corruption sanitization HERE. will be done in 'autofix'

    def test_check_imgs_dup(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(current_path, "test_datasets", "ds_coco_dataset")

        with tempfile.TemporaryDirectory() as output_path:
            sc = SanityCheck(data_path=data_path, output_path=output_path)
            with open(sc.annotfn) as json_file:
                gt = json.load(json_file)

            imgs_lst = gt["images"]
            imgs_lst.append(imgs_lst[1].copy())
            imgs_lst[-1]["file_name"] = "100000522418.jpg"
            imgs_lst.append(imgs_lst[2].copy())
            imgs_lst[-1]["id"] = 884613

            problems = sc.check_imgs_dup(imgs_lst)

            assert (
                len(problems) == 2
            ), "SanityCheck.check_imgs_dup() error list size mismatch"
            assert (
                problems[0]["err_sbi"] == 10
                and problems[0]["err_code"] == "ERR_JSON_IMG_ID_DUP"
            ), "Duplicate img id check failed"
            assert (
                problems[1]["err_sbi"] == 11
                and problems[1]["err_code"] == "ERR_JSON_IMG_FNAME_DUP"
            ), "Duplicate img file_name check failed"

            clean_list(imgs_lst, [p["err_sbi"] for p in problems])

            problems = sc.check_imgs_dup(imgs_lst)
            assert problems == [], "Clean duplicated images failed"

    def test_err_image_type(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(current_path, "test_datasets", "ds_coco_dataset")

        with tempfile.TemporaryDirectory() as output_path:
            sc = SanityCheck(data_path=data_path, output_path=output_path)
            with open(sc.annotfn) as json_file:
                gt = json.load(json_file)

            imgs_lst = gt["images"]
            imgs_lst[1]["id"] = 666
            imgs_lst[1]["file_name"] = "666.rar"

            assert not sc.err_image_type(
                imgs_lst[0]
            ), "SanityCheck.err_image_type() failed with valid file extension"
            assert sc.err_image_type(
                imgs_lst[1]
            ), "SanityCheck.err_image_type() failed with invalid file extension"

    def test_check_coco_annotation(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(current_path, "test_datasets", "ds_coco_dataset")

        with tempfile.TemporaryDirectory() as output_path:
            sc = SanityCheck(data_path=data_path, output_path=output_path)
            with open(sc.annotfn) as json_file:
                gt = json.load(json_file)

            anns = gt["annotations"]
            anns[0]["bbox"].append(2.5)
            anns[1]["bbox"][0] = "a"
            anns[2]["bbox"][1] = -2.5

            anns[3]["segmentation"][0][0] = "a"
            anns[4]["segmentation"][0][0] = -5.5
            anns[5]["segmentation"][0].append(34)

            assert not SanityCheck.check_coco_annotation(
                anns[6]
            ), "SanityCheck.check_coco_annotation() failed on valid annotation"
            assert SanityCheck.check_coco_annotation(anns[0]) == [
                "ERR_JSON_ANN_BBOX_LEN"
            ], "SanityCheck.check_coco_annotation() LEN BBOX check failed"
            assert SanityCheck.check_coco_annotation(anns[1]) == [
                "ERR_JSON_ANN_BBOX_TYPE"
            ], "SanityCheck.check_coco_annotation() TYPE BBOX check failed"
            assert SanityCheck.check_coco_annotation(anns[2]) == [
                "ERR_JSON_ANN_BBOX_NEG"
            ], "SanityCheck.check_coco_annotation() NEG BBOX check failed"

            assert SanityCheck.check_coco_annotation(anns[3]) == [
                "ERR_JSON_ANN_SEG_TYPE"
            ], "SanityCheck.check_coco_annotation() TYPE SEGMENTATION check failed"
            assert SanityCheck.check_coco_annotation(anns[4]) == [
                "ERR_JSON_ANN_SEG_NEG"
            ], "SanityCheck.check_coco_annotation() NEG SEGMENTATION check failed"
            assert SanityCheck.check_coco_annotation(anns[5]) == [
                "ERR_JSON_ANN_SEG_PAR"
            ], "SanityCheck.check_coco_annotation() PAR SEGMENTATION check failed"

    def test_check_annotations(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(current_path, "test_datasets", "ds_coco_dataset")

        with tempfile.TemporaryDirectory() as output_path:
            sc = SanityCheck(data_path=data_path, output_path=output_path)
            problems = sc.check_annotations()
            assert (
                problems == []
            ), "SanityCheck.check_annotations() failed on valid COCO json"

    def test_fix_coco_image_size(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(current_path, "test_datasets", "ds_coco_dataset")

        with tempfile.TemporaryDirectory() as output_folder:
            sc = SanityCheck(data_path, output_folder)
            with open(sc.annotfn) as json_file:
                gt = json.load(json_file)

            sc.fix_coco_image_size(gt["images"][0:2], data_path + "/images")
            assert [(img["height"], img["width"]) for img in gt["images"][0:2]] == [
                (360, 640),
                (480, 640),
            ], "SanityCheck.fix_coco_image_size() failed."

    def test_rasterize(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        orig_path = os.path.join(current_path, "test_datasets", "ds_geo_dataset")

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = os.path.join(tmp_dir, "data_path")
            mask_path = os.path.join(data_path, "images_mask")
            annotfn = os.path.join(data_path, "annots.geojson")

            shutil.copytree(orig_path, data_path)

            # Before rasterize assesment
            assert not os.path.exists(
                mask_path
            ), f"mask_path [{mask_path}] should not exist yet"
            assert os.path.exists(annotfn), f"annotfn [{annotfn}] should exist"

            # Rasterize
            _ = Rasterize(annotfn)

            # After rasterize assesment
            assert os.path.exists(
                mask_path
            ), f"mask_path [{mask_path}] should exist after rasterization"

            for img_fn in glob(os.path.join(data_path, "images", "*")):
                msk_fn = os.path.join(
                    data_path, "images_mask", os.path.basename(img_fn) + "_mask"
                )
                assert os.path.exists(msk_fn), f"File [{msk_fn}] should exist"
                mask_arr = cv2.imread(msk_fn)
                assert (
                    mask_arr.shape[-2] == cv2.imread(img_fn).shape[-2]
                ), f"{msk_fn} AND {img_fn} should have the same width and high"
                assert (
                    "int" in mask_arr.dtype.name
                ), f"The type of the array {msk_fn} should be int or uint"


def test_clean_list():
    l1 = [1, "dos", "tres", 4, 5]
    to_clean = [0, 4, 2, 4]
    cleaned = clean_list(l1, to_clean)
    assert cleaned == ["dos", 4], "clean_list() error"
