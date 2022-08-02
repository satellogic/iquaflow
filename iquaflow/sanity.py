import json
import os
from glob import glob
from typing import Any, Dict, List, Optional

import cv2
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.mask import mask

from iquaflow.datasets import DSWrapper

REQUIRED_FIELDS = ["image_filename", "class_id", "geometry"]


class SanityCheck:
    """
    Perform sanity to image datasets and ground truth.
    It can either work as standalone class or with DSWrapper class.

    Args:
        data_path: the full path to the original dataset
        output_path: the full path to the output dataset
        annotfn: annotation full filename (either json or ge

    Attributes:
        data_path: the full path to the original dataset
        output_path: the full path to the output dataset
        annotfn: annotation full filename (either json or geojson)
    """

    def __init__(
        self,
        data_path: str = "",
        output_path: str = "",
        ds_wrapper: Optional[DSWrapper] = None,
    ):
        """
        Attributes:
            data_path: the full path to the original dataset
            output_path: the full path to the output dataset
            annotfn: annotation full filename (either json or geojson)
        """

        if ds_wrapper is not None:
            self.data_path = ds_wrapper.data_path
        else:
            self.data_path = data_path
        self.ds_wrapper = ds_wrapper
        self.output_path = output_path
        self.annotfns = [el for el in glob(os.path.join(self.data_path, "*.*json"))]
        self.annotfn = self.annotfns[0]
        self.isgeo = self.annotfn.endswith("geojson")
        self.out_annotfn = os.path.join(output_path, os.path.basename(self.annotfn))
        self.valid_img_extensions = [
            "jpg",
            "png",
            "tif",
        ]  # add or remove desired img file extensions

        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)

    def check_annotations(self) -> List[Dict[str, Any]]:
        """
        Check annotation jsons / geojsons and return problems list

        Returns:
            Dictionary with problems found
        """
        problems = []
        if not self.isgeo:
            for json_fn in self.annotfns:
                with open(json_fn) as json_file:
                    gt = json.load(json_file)
                    im_dup = self.check_imgs_dup(gt["images"])
                    problems += [
                        {
                            "err_file": json_fn,
                            "err_obj": "images",
                            "err_sbi": err["err_sbi"],
                            "err_code": err["err_code"],
                            "err_txt": err["err_txt"],
                        }
                        for err in im_dup
                    ]
                    for sbi, im in enumerate(gt["images"]):
                        if self.err_image_type(im):
                            problems.append(
                                {
                                    "err_file": json_fn,
                                    "err_obj": "images",
                                    "err_sbi": sbi,
                                    "err_code": "ERR_JSON_IMG_FILE_TYPE",
                                }
                            )
                        # elif ....
                    for sbi, an in enumerate(gt["annotations"]):
                        err_an = SanityCheck.check_coco_annotation(an)
                        problems += [
                            {
                                "err_file": json_fn,
                                "err_obj": "annotations",
                                "err_sbi": sbi,
                                "err_code": e_a,
                            }
                            for e_a in err_an
                        ]

        elif self.isgeo:
            for json_fn in self.annotfns:

                gt = gpd.read_file(json_fn)
                gt_image = [
                    {"file_name": el, "id": enu}
                    for enu, el in enumerate(gt["image_filename"])
                ]
                im_dup = self.check_imgs_dup(gt_image)
                problems += [
                    {
                        "err_file": json_fn,
                        "err_obj": "images",
                        "err_sbi": err["err_sbi"],
                        "err_code": err["err_code"],
                        "err_txt": err["err_txt"],
                    }
                    for err in im_dup
                ]
                for sbi, im in enumerate(gt_image):

                    if self.err_image_type(im):
                        problems.append(
                            {
                                "err_file": json_fn,
                                "err_obj": "images",
                                "err_sbi": sbi,
                                "err_code": "ERR_JSON_IMG_FILE_TYPE",
                            }
                        )
        return problems

    def fix_annotations(
        self,
        problems: List[Dict[str, Any]],
        missing: Optional[bool] = True,
        empty_geom: Optional[bool] = True,
        invalid_geom: Optional[bool] = True,
        try_fix_geoms: Optional[bool] = True,
        rasterize: Optional[bool] = False,
    ) -> None:
        """
        This will remove all corrupted samples following the logic in the flags.\n
        The new sanitized dataset is located in output_path attribute from the SanityCheck instance
        
        Args:
            problems: List of problems
            missing: When set, it removes any sample that has a Nan value in any of the required fields. Only used in geojeson-like annotations dataset. default True.
            empty_geom: When set, it removes samples with empty annotations. Only used in geojeson-like annotations dataset. default True.
            invalid_geom: When set, it removes samples with invalid geometries. Only used in geojeson-like annotations dataset. default True.
            try_fix_geoms: When set, it fixes geometries. Only used in geojeson-like annotations dataset. default True.
            rasterize: It rasterizes the dataset creating categorical masks following the class id and geometries of the dataset.
        """
        if not self.isgeo:
            for json_fn in self.annotfns:
                if rasterize:
                    print(
                        "Skiping rasterization since it is only supported for coco-like annotations for now."
                    )
                with open(json_fn) as json_file:
                    gt = json.load(json_file)
                    bad_imgs = [
                        err["err_sbi"]
                        for err in problems
                        if err["err_file"] == json_fn and err["err_obj"] == "images"
                    ]
                    clean_list(gt["images"], bad_imgs)
                    bad_anns = [
                        err["err_sbi"]
                        for err in problems
                        if err["err_file"] == json_fn
                        and err["err_obj"] == "annotations"
                    ]
                    clean_list(gt["annotations"], bad_anns)
                    bad_cats = [
                        err["err_sbi"]
                        for err in problems
                        if err["err_file"] == json_fn and err["err_obj"] == "categories"
                    ]
                    clean_list(gt["categories"], bad_cats)
                    fout = os.path.join(self.output_path, os.path.basename(json_fn))
                    with open(fout, "w") as outfile:
                        json.dump(gt, outfile)
        elif self.isgeo:
            for json_fn in self.annotfns:

                if rasterize:
                    _ = Rasterize(json_fn)

                gdf = gpd.read_file(json_fn)

                fout = os.path.join(self.output_path, os.path.basename(json_fn))

                bad_imgs = [
                    err["err_sbi"]
                    for err in problems
                    if err["err_file"] == json_fn and err["err_obj"] == "images"
                ]

                indexNames = gdf[gdf["image_filename"].isin(bad_imgs)].index

                gdf.drop(indexNames, inplace=True)

                print("Sanitizing geojson like dataset")
                if missing:
                    gdf = self._rm_geo_nan(gdf)
                if empty_geom:
                    gdf = self._rm_geo_empty(gdf)
                if invalid_geom:
                    gdf = self._rm_geo_invalid(gdf, try_fix_geoms=try_fix_geoms)
                # save sanitized annotations
                gdf.to_file(fout, driver="GeoJSON")
                # # copy remaining images
                # //TODO symlink
                # for imgfn in gdf["image_filename"].unique():
                #     shutil.copy(
                #         os.path.join(self.data_path, "images", imgfn),
                #         os.path.join(self.output_path, "images", imgfn),
                #     )

    def autofix(
        self,
        missing: Optional[bool] = True,
        empty_geom: Optional[bool] = True,
        invalid_geom: Optional[bool] = True,
        try_fix_geoms: Optional[bool] = True,
    ) -> None:
        """
        Autofix will:
            1. detect corrupted anotations or images.
            2. try to fix the corrupted cases.
            3. remove all corrupted samples that were not able to fix.
        The new sanitized dataset is located in output_path attribute from the SanityCheck instance
        
        Args:
            missing: When set, it removes any sample that has a Nan value in any of the required fields, Only used in geojeson-like annotations dataset, default True.
            empty_geom: When set, it removes samples with empty annotations, Only used in geojeson-like annotations dataset, default True.
            invalid_geom: When set, it removes samples with invalid geometries, Only used in geojeson-like annotations dataset, default True.
            try_fix_geoms: When set, it fixes geometries, Only used in geojeson-like annotations dataset, default True.
        """
        if not self.isgeo:
            problems = self.check_annotations()
            if problems:
                print(f"{len(problems)} Problems found:")
                print(problems)
            self.fix_annotations(problems)
        else:
            problems = self.check_annotations()
            if problems:
                print(f"{len(problems)} Problems found:")
                print(problems)
            self.fix_annotations(
                problems,
                missing=missing,
                empty_geom=empty_geom,
                invalid_geom=invalid_geom,
                try_fix_geoms=try_fix_geoms,
            )
            # TODO add fix for corrupted images HERE

    # GeoJSON related methods
    def _rm_geo_nan(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Removes all rows containing a Nan value in any of the Required field columns.
        
        Args:
            gdf: GeoDataFrame to treat.
        
        Returns:
            Treated GeoDataFrame with corrupted (nan corruption kind) rows ereased.
        """
        gdf = gdf.dropna(
            how="any", subset=[i for i in REQUIRED_FIELDS if i != "geometry"]
        )
        return gdf

    def _rm_geo_empty(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Removes all rows containing a empty geometries in any of the Required field columns.
        
        Args:
            gdf: GeoDataFrame to treat.
        
        Returns:
            Treated GeoDataFrame with empty geometries rows ereased.
        """
        gdf = gdf.dropna(how="any", subset=["geometry"])
        return gdf

    def _fix_geoms(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Try to fix geometries with buffer = 0
        
        Args:
            gdf: GeoDataFrame to treat.
        
        Returns:
            Treated GeoDataFrame.
        """
        gdf.loc[:, ["geometry"]] = gdf.buffer(0)
        return gdf

    def _rm_geo_invalid(
        self, gdf: gpd.GeoDataFrame, try_fix_geoms: Optional[bool] = True
    ) -> gpd.GeoDataFrame:
        """
        Removes all rows containing a valid geometries in any of the Required field columns.
        
        Args:
            gdf: GeoDataFrame to treat.
        
        Returns:
            Treated GeoDataFrame with valid geometries rows ereased.
        """
        if try_fix_geoms:
            gdf = self._fix_geoms(gdf)
        return gdf[gdf.is_valid]

    # COCO related methods
    def err_image_type(self, im: Dict[str, Any]) -> bool:
        """
        It checks if the image format is a valid image file format.
        None is also considered invalid.

        Args:
            im: containing 'file_name' key
        
        Returns:
            is valid.
        """
        # Treat None case:
        if im["file_name"]:
            ext = im["file_name"].split(".")[-1]
            return ext not in self.valid_img_extensions
        else:
            return False

    @staticmethod
    def check_coco_annotation(an: Dict[str, Any]) -> List[str]:
        """
        Check integrity of one coco annotation.

        Args:
            an: COCO style annotation.

        Returns:
            List of error codes 'ERR_JSON_ANN_xxxx'. Empty list if no errors.
        """
        err = []
        # Bounding Box errors
        if len(an["bbox"]) != 4:
            err.append("ERR_JSON_ANN_BBOX_LEN")
        if any([type(n) not in [int, float] for n in an["bbox"]]):
            err.append("ERR_JSON_ANN_BBOX_TYPE")
        elif any((b < 0 for b in an["bbox"])):
            err.append("ERR_JSON_ANN_BBOX_NEG")
        # Segmentation Polygon errors
        if "segmentation" in an:
            if "counts" in an["segmentation"]:  # counts type segmentation (iscrowd)
                if any(
                    (
                        type(b) is not int
                        for k in an["segmentation"]
                        for b in an["segmentation"][k]
                    )
                ):
                    err.append("ERR_JSON_ANN_SEG_TYPE")
                elif any(
                    (b < 0 for k in an["segmentation"] for b in an["segmentation"][k])
                ):
                    err.append("ERR_JSON_ANN_SEG_NEG")
            else:
                if any(
                    (
                        type(b) not in [int, float]
                        for seg in an["segmentation"]
                        for b in seg
                    )
                ):
                    err.append("ERR_JSON_ANN_SEG_TYPE")
                elif any((b < 0 for seg in an["segmentation"] for b in seg)):
                    err.append("ERR_JSON_ANN_SEG_NEG")
                if any(((len(seg) % 2) != 0 for seg in an["segmentation"])):
                    err.append("ERR_JSON_ANN_SEG_PAR")

        return err

    @staticmethod
    def check_imgs_dup(imgs_lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find duplicates in coco json images list

        Args:
            imgs_lst: List of images (coco json "images" field).
        
        Returns:
            problems: List of errors.
        """
        problems = []
        img_ids: Dict[int, str] = {}
        img_files: Dict[str, int] = {}

        for sbi, im in enumerate(imgs_lst):
            if im["id"] in img_ids:
                problems.append(
                    {
                        "err_sbi": sbi,
                        "err_code": "ERR_JSON_IMG_ID_DUP",
                        "err_txt": f'Duplicate img id {im["id"]} (file_name:"{im["file_name"]}")  already in list with file_name:"{img_ids[im["id"]]}"',
                    }
                )
            elif im["file_name"] in img_files:
                problems.append(
                    {
                        "err_sbi": sbi,
                        "err_code": "ERR_JSON_IMG_FNAME_DUP",
                        "err_txt": f'Duplicate file_name "{im["file_name"]}" (id:"{im["id"]}") already in list with id:{img_files[im["file_name"]]}',
                    }
                )
            else:
                img_ids[im["id"]] = im["file_name"]
                img_files[im["file_name"]] = im["id"]

        return problems

    @staticmethod
    def fix_coco_image_size(imgs_lst: List[Dict[str, Any]], imgs_dir: str) -> None:
        """
        Fix height and width in coco json images list.

        Args:
            imgs_lst: List of images (coco json "images" field).
            imgs_dir: Path of image folder.
        """
        for image in imgs_lst:
            im_array = cv2.imread(imgs_dir + "/" + image["file_name"])
            image["height"] = im_array.shape[0]
            image["width"] = im_array.shape[1]


def clean_list(lst: List[Any], sbis_to_clean: List[int]) -> List[Any]:
    """
    Removes multiple elements of a list

    Args:
        lst: List to filter.
        sbis_to_clean: List of indexes to delete.

    Returns:
        Remaining elements
    """
    sbis_to_clean = list(set(sbis_to_clean))  # remove duplicates
    sbis_to_clean.sort(reverse=True)  # descending order for correct erase
    for i in sbis_to_clean:
        del lst[i]
    return lst


class Rasterize:
    """
    Generates raster masks from geometries in geojson annotations. Masks will be equal in size to its corresponding images.
    
    Args:
        geojson_fn: Annotations file in format of geojson
        main_df: Loaded dataframe from annoations file.

    Attributes:
        geojson_fn: Annotations file in format of geojson
        main_df: Loaded dataframe from annoations file.
    """

    def __init__(self, geojson_fn: str) -> None:
        """
        Attributes:
            geojson_fn (str): Annotations file in format of geojson
            main_df (str): Loaded dataframe from annoations file.
        """

        self.geojson_fn = geojson_fn

        self.main_df = gpd.read_file(self.geojson_fn)

        for req in REQUIRED_FIELDS:

            assert (
                req in self.main_df.columns
            ), f"Err: Required IQF convention field {req} not present in geodataframe."

        # for each image
        for ref_geotiff in self.main_df["image_filename"]:

            (
                img,
                list_of_geometries,
                list_of_class_ids,
                output_filename,
            ) = self._gen_new_mask(ref_geotiff)

            out_transform = None

            for class_id in set(list_of_class_ids):

                short_list_of_geometries = [
                    g
                    for g, c in zip(list_of_geometries, list_of_class_ids)
                    if c == class_id
                ]

                with rio.open(output_filename) as src:

                    ones_where_poly, out_transform = mask(
                        src,
                        short_list_of_geometries,
                        crop=False,
                        indexes=1,
                        nodata=1,  # Pixels within polygon are = to nodata
                    )

                    img[ones_where_poly == 1] = (
                        class_id * ones_where_poly[ones_where_poly == 1]
                    )

                    out_meta = src.meta.copy()

            self._write_img(img, out_transform, out_meta, output_filename)

    def _create_gtiff_out(self, ref_geotiff: str, output_filename: str) -> np.array:
        """
        Generate an associated mask file given an image.

        Args:
            ref_geotiff: Image filename.
            output_filename: Output mask filaname.
        
        Returns:
            The initial mask array that has been saved.
        """

        with rio.open(ref_geotiff) as src:

            Z = src.read(1)
            Z[:, :] = 255
            Z = Z.astype(rio.uint8)

        with rio.open(
            output_filename,
            "w",
            driver="GTiff",
            height=Z.shape[0],
            width=Z.shape[1],
            count=1,
            dtype=Z.dtype,
            crs=src.crs,
            transform=src.transform,
        ) as dst:

            dst.write(Z, 1)

        return Z

    def _gen_new_mask(self, ref_geotiff: str) -> Any:
        """
        Save an updated mask that has all its polygons with class ids in it.

        Args:
            ref_geotiff: Reference geotiff image.
        
        Returns:
            the image array, list of geometries and class ids, output_fn for the mask.
        """

        if not os.path.exists(ref_geotiff):
            ref_geotiff = glob(
                os.path.join(os.path.dirname(self.geojson_fn), "*", ref_geotiff)
            )[0]

        output_filename = os.path.join(
            os.path.dirname(ref_geotiff) + "_mask",
            os.path.basename(ref_geotiff) + "_mask",
        )

        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        # only annots belonging to the target image
        dataframe = self.main_df[
            self.main_df["image_filename"] == os.path.basename(ref_geotiff)
        ]

        list_of_geometries = dataframe["geometry"].values
        list_of_class_ids = list(dataframe["class_id"].values)

        # create output GT filename
        img = self._create_gtiff_out(ref_geotiff, output_filename)

        return img, list_of_geometries, list_of_class_ids, output_filename

    def _write_img(
        self, img: np.array, out_transform: Any, out_meta: Any, output_filename: str
    ) -> None:
        """
        It writes a raster image with the mask.
        """

        if out_transform is not None:

            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": img.shape[-2],
                    "width": img.shape[-1],
                    "transform": out_transform,
                    "dtype": "uint8",
                }
            )

        else:

            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": img.shape[-2],
                    "width": img.shape[-1],
                    "dtype": "uint8",
                }
            )

        with rio.open(output_filename, "w", **out_meta) as dest:

            dest.write(img, 1)
