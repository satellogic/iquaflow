import json
import os
import shutil
import tempfile

import geopandas as gpd
from shapely.geometry import Point, Polygon, box

from iquaflow.datasets import DSWrapper
from iquaflow.ds_stats import DsStats
from iquaflow.sanity import Rasterize


class TestDsStats:
    @staticmethod
    def test_ds_wrapper_integration():
        ds_path = os.path.join("./tests", "test_datasets", "ds_coco_dataset")
        ds_wrapper = DSWrapper(data_path=ds_path)
        with tempfile.TemporaryDirectory() as out_path:
            dss = DsStats(ds_wrapper=ds_wrapper, output_path=out_path)
            assert dss.data_path == ds_path

    @staticmethod
    def test_perform_stats_coco():
        ds_path = os.path.join("./tests", "test_datasets", "ds_coco_dataset")
        with tempfile.TemporaryDirectory() as out_path:
            dss = DsStats(data_path=ds_path, output_path=out_path)
            stats = dss.perform_stats(False)
            assert stats[0]["obj"] == "images", "missing json images stats"
            assert (
                stats[0]["stats"]["mean_height"] == 496.2
            ), "json images mean_height error"
            assert (
                stats[0]["stats"]["mean_width"] == 580.2
            ), "json images mean_width error"
            with open(dss.annotfns[0], "r") as src:
                gt = json.load(src)
            lstlst = dss.mask_stats(gt)
            assert (
                "background" == lstlst[0][0]
                and "motorcycle" == lstlst[0][1]
                and "microwave" == lstlst[0][-1]
            ), "Mask stats does not work as expected"
            assert (
                0.6045334161699303 == lstlst[1][0]
                and 0.017041966175426138 == lstlst[1][-1]
            ), "Mask stats does not work as expected"

            assert stats[1]["obj"] == "class_histo", "missing class balance histo"
            assert stats[1]["stats"]["cat_ids"] == [
                1,
                2,
                4,
                17,
                21,
                27,
                28,
                31,
                44,
                46,
                47,
                49,
                50,
                51,
                55,
                61,
                64,
                67,
                72,
                74,
                76,
                78,
                79,
                81,
                82,
                85,
            ], "class balance stats err"
            assert stats[1]["stats"]["cat_names"] == [
                "person",
                "bicycle",
                "motorcycle",
                "cat",
                "cow",
                "backpack",
                "umbrella",
                "handbag",
                "bottle",
                "wine glass",
                "cup",
                "knife",
                "spoon",
                "bowl",
                "orange",
                "cake",
                "potted plant",
                "dining table",
                "tv",
                "mouse",
                "keyboard",
                "microwave",
                "oven",
                "sink",
                "refrigerator",
                "clock",
            ], "class balance stats err"
            assert stats[1]["stats"]["cat_tags_count"] == [
                32,
                1,
                1,
                1,
                9,
                1,
                1,
                1,
                14,
                1,
                8,
                5,
                2,
                7,
                7,
                1,
                1,
                2,
                8,
                9,
                6,
                1,
                6,
                3,
                1,
                1,
            ], "class balance stats err"
            assert os.path.exists(
                os.path.join(
                    out_path, "images", "coco_annotations.json_class_histo.png"
                )
            ), "missing class balance histo .png"
            assert os.path.exists(
                os.path.join(
                    out_path, "images", "coco_annotations.json_class_area_histo.png"
                )
            ), "missing class area coverage histo .png"
            assert (
                stats[3]["obj"] == "bbox_aspect_ratio_histo"
            ), "missing bbox aspect ratio histo"
            assert stats[3]["stats"]["ar_counts"] == [
                57,
                33,
                17,
                13,
                4,
                0,
                1,
                3,
                0,
                2,
            ], "bbox aspect ratio histo err"
            assert os.path.exists(
                os.path.join(
                    out_path,
                    "images",
                    "coco_annotations.json_bbox_aspect_ratio_histo.png",
                )
            ), "missing bbox aspect ratio histo .png"
            assert stats[4]["obj"] == "bbox_area_histo", "missing bbox area histo"
            assert stats[4]["stats"]["ar_counts"] == [
                108,
                7,
                5,
                4,
                3,
                1,
                1,
                0,
                0,
                1,
            ], "bbox area histo err"
            assert os.path.exists(
                os.path.join(
                    out_path, "images", "coco_annotations.json_bbox_area_histo.png"
                )
            ), "missing bbox area histo .png"
            assert (
                stats[5]["obj"] == "images_aspect_ratio_histo"
            ), "missing images aspect ratio histo"
            assert stats[5]["stats"]["ar_counts"] == [
                2,
                1,
                0,
                0,
                0,
                0,
                4,
                2,
                0,
                1,
            ], "images aspect ratio histo err"
            assert os.path.exists(
                os.path.join(
                    out_path,
                    "images",
                    "coco_annotations.json_imgs_aspect_ratio_histo.png",
                )
            ), "missing images aspect ratio histo .png"
            assert stats[6]["obj"] == "images_area_histo", "missing images area histo"
            assert stats[6]["stats"]["ar_counts"] == [
                1,
                0,
                0,
                1,
                0,
                2,
                0,
                5,
                0,
                1,
            ], "images area histo err"
            assert os.path.exists(
                os.path.join(
                    out_path, "images", "coco_annotations.json_imgs_area_histo.png"
                )
            ), "missing images area histo .png"

    @staticmethod
    def test_perform_stats_geojson():
        ds_path = os.path.join("./tests", "test_datasets", "ds_geo_dataset")
        with tempfile.TemporaryDirectory() as out_path:
            dss = DsStats(data_path=ds_path, output_path=out_path)
            stats = dss.perform_stats(False)
            assert (
                stats[0]["stats"]["mean_height"] == 90
            ), "json images mean_height error"
            assert stats[0]["stats"]["mean_width"] == 90, "json images mean_width error"
            assert dss._area_pol(box(0, 0, 2, 2)) == 4, "Area calculation is wrong"
            triangle = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
            assert dss._calc_bbox(triangle) == box(*triangle.bounds), "Polygon is wrong"
            assert dss._calc_minrotrect(
                Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
            ) == box(0, 0, 1, 1), "minimum rotated rectangle is wrong"
            h, w, _ = dss._calc_rectangle_stats(triangle)
            assert h == 1, "High is wrong"
            assert w == 1, "Width is wrong"
            rot = dss._calc_minrotrect(triangle)
            _, _, a = dss._calc_rectangle_stats(rot)
            assert a in [45, 45 - 180], "Angle is wrong"
            p = Point(5, 5).buffer(1.0)
            c = dss._compute_compactness(p)
            assert type(c) == float, "Wrong compactness"
            assert dss.annotfns[0].endswith(
                ".geojson"
            ), "Expected .geojson from the geo_dataset"
            gdf = gpd.read_file(dss.annotfns[0])
            geo_stats = {
                "file": "./tests/test_datasets/ds_geo_dataset/annots.geojson",
                "obj": "geojson",
                "stats": {},
            }
            assert dss._dataframe_basic_stats(gdf, "area") == [
                {"min": None},
                {"mean": None},
                {"max": None},
            ], "area stats are wrong"
            stats = dss.perform_stats(False)
            stats_file = os.path.join(dss.output_path, "stats.json")
            assert os.path.exists(
                stats_file
            ), "For some reason the stats.json was not written"
            assert [key for key in geo_stats] in [
                [key for key in st] for st in stats
            ], "Geostats are not the ones expected"
            with open(stats_file) as json_file:
                stats_fromfile = json.load(json_file)
            assert (
                stats == stats_fromfile
            ), "returned stats are not equal to the written stats"

    @staticmethod
    def test_perform_stats_geojson_plots():
        ds_path = os.path.join("./tests", "test_datasets", "ds_geo_dataset")
        out_path = os.path.join("./tests", "test_datasets", "ds_geo_dataset", "stats")
        os.makedirs(out_path, exist_ok=True)
        dss = DsStats(data_path=ds_path, output_path=out_path)
        _ = dss.perform_stats(False)

    @staticmethod
    def test_notebook_annots_summary():
        ds_path = os.path.join("./tests", "test_datasets", "ds_geo_dataset")
        fn = os.path.join(ds_path, "annots.geojson")
        df = gpd.read_file(fn)
        df["area"] = df.area
        df = df[["image_filename", "class_id", "area"]]
        with tempfile.TemporaryDirectory() as out_path:
            htmlfn = os.path.join(out_path, "myhtml.html")
            DsStats.notebook_annots_summary(
                df,
                export_html_filename=htmlfn,
                fields_to_include=["image_filename", "class_id", "area"],
                show_inline=False,
            )
            assert os.path.exists(htmlfn), "Did not manage to export the html filename"

    def test_rasterized_stats(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        orig_path = os.path.join(current_path, "test_datasets", "ds_geo_dataset")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "output_path")
            data_path = os.path.join(tmp_dir, "data_path")
            annotfn = os.path.join(data_path, "annots.geojson")

            shutil.copytree(orig_path, data_path)

            # Rasterize
            _ = Rasterize(annotfn)

            dss = DsStats(data_path=data_path, output_path=output_path)
            stats = dss.perform_stats(False)
            mask_stats = [el for el in stats if "mask_dir" in el]
            assert (
                len(mask_stats) == 1
            ), "There should be exacly ONE dictionary with key *mask_dir* within the list stats"
            assert (
                mask_stats[0]["obj"] == "area_coverage_by_class"
            ), "The obj key should be populated with this name: *area_coverage_by_class*"
            assert (
                mask_stats[0]["stats"][0] > 92.47 and mask_stats[0]["stats"][0] < 92.5
            ), "The result (amount) of category 0 is wrong in the mask_stats"
