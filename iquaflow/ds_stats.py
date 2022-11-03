import json
import math
import os
from glob import glob
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from easyimages import EasyImageList
from IPython.core.display import HTML, display
from shapely.geometry import Point, Polygon, box

from iquaflow.datasets import DSWrapper


class DsStats:
    """
    Perform stats to image datasets and annotations.
    It can either work as standalone class or with DSWrapper class.

    Args:
        data_path: The full path to the dataset to be analized
        output_path: the full path to the output (stats / plots)

    Attributes:
        data_path: The full path to the dataset to be analized
        output_path: the full path to the output (stats / plots)
    """

    def __init__(
        self,
        data_path: str = "",
        output_path: str = "",
        ds_wrapper: Optional[DSWrapper] = None,
    ):
        """
        Attributes:
            data_path: str or DSWrapper. The full path to the dataset to be analized
            output_path: the full path to the output (stats / plots)
        """

        if not output_path:
            output_path = os.path.join(data_path, "stats")

        if ds_wrapper is not None:
            self.data_path = ds_wrapper.data_path
        else:
            self.data_path = data_path

        self.ds_wrapper = ds_wrapper
        self.output_path = output_path
        self.annotfns = [el for el in glob(os.path.join(self.data_path, "*.*json"))]

        self.mask_subdir_lst = [
            os.path.join(self.data_path, el)
            for el in os.listdir(self.data_path)
            if all(
                (os.path.isdir(os.path.join(self.data_path, el)), el.endswith("_mask"))
            )
        ]

        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)

    def perform_stats(self, show_plots: Optional[bool] = True) -> List[Dict[str, Any]]:
        """
        Perform and plot stats on annotation jsons / geojsons and images

        Returns:
            Dictionary with statistics
        """
        stats = []

        if len(self.mask_subdir_lst) > 0:
            for mask_dir in self.mask_subdir_lst:
                msk_stats = {}
                total = 0
                for fn in glob(os.path.join(mask_dir, "*_mask")):
                    vals = cv2.imread(fn).flatten().tolist()
                    keys = set(vals)
                    for k in keys:
                        if k not in msk_stats:
                            msk_stats[k] = 0
                        msk_stats[k] += len([v for v in vals if v == k])
                        total += len([v for v in vals if v == k])

                bn_mask_dir = os.path.basename(mask_dir)

                stats.append(
                    {
                        "mask_dir": bn_mask_dir,
                        "obj": "area_coverage_by_class",
                        "stats": {
                            k: float(msk_stats[k]) * 100 / total for k in msk_stats
                        },
                    }
                )

                self._plot_hist_list(
                    [str(k) for k in msk_stats],
                    [float(msk_stats[k]) for k in msk_stats],
                    f"{bn_mask_dir} class area coverage",
                    f"{bn_mask_dir}_class_area_histo.png",
                    show_plots=show_plots,
                    xlabel="% Area coverage",
                )

        for json_fn in self.annotfns:
            if not json_fn.endswith("geojson"):
                with open(json_fn) as json_file:
                    gt = json.load(json_file)
                    fname = os.path.basename(json_fn)
                    coco_imgs_stats = self.coco_imgs_stats(gt["images"])
                    stats.append(
                        {"file": json_fn, "obj": "images", "stats": coco_imgs_stats}
                    )
                    coco_class_histo_d = self.coco_class_histo(
                        gt["annotations"], gt["categories"]
                    )
                    stats.append(
                        {
                            "file": json_fn,
                            "obj": "class_histo",
                            "stats": coco_class_histo_d,
                        }
                    )
                    self._plot_hist_list(
                        coco_class_histo_d["cat_names"],
                        coco_class_histo_d["cat_tags_count"],
                        f"{fname} class balance",
                        f"{fname}_class_histo.png",
                        show_plots=show_plots,
                    )
                    names, vals = self.mask_stats(gt)
                    stats.append(
                        {
                            "file": json_fn,
                            "obj": "area_coverage_by_class",
                            "stats": {str(k): v for k, v in zip(names, vals)},
                        }
                    )
                    self._plot_hist_list(
                        [str(n) for n in names],
                        [float(v) for v in vals],
                        f"{fname} class area coverage",
                        f"{fname}_class_area_histo.png",
                        show_plots=show_plots,
                    )

                    coco_bbox_aspect_ratio_histo_d = self.coco_bbox_aspect_ratio_histo(
                        gt["annotations"]
                    )
                    stats.append(
                        {
                            "file": json_fn,
                            "obj": "bbox_aspect_ratio_histo",
                            "stats": coco_bbox_aspect_ratio_histo_d,
                        }
                    )
                    self._plot_hist_coco(
                        coco_bbox_aspect_ratio_histo_d["aspect_ratios"],
                        "Bounding Box Aspect Ratios",
                        f"{fname}_bbox_aspect_ratio_histo.png",
                        show_plots=show_plots,
                    )

                    coco_bbox_area_histo_d = self.coco_bbox_area_histo(
                        gt["annotations"]
                    )
                    stats.append(
                        {
                            "file": json_fn,
                            "obj": "bbox_area_histo",
                            "stats": coco_bbox_area_histo_d,
                        }
                    )
                    self._plot_hist_coco(
                        coco_bbox_area_histo_d["areas"],
                        "Bounding Box Areas",
                        f"{fname}_bbox_area_histo.png",
                        show_plots=show_plots,
                    )

                    coco_imgs_aspect_ratio_histo_d = self.coco_imgs_aspect_ratio_histo(
                        gt["images"]
                    )
                    stats.append(
                        {
                            "file": json_fn,
                            "obj": "images_aspect_ratio_histo",
                            "stats": coco_imgs_aspect_ratio_histo_d,
                        }
                    )
                    self._plot_hist_coco(
                        coco_imgs_aspect_ratio_histo_d["aspect_ratios"],
                        "Images Aspect Ratios",
                        f"{fname}_imgs_aspect_ratio_histo.png",
                        show_plots=show_plots,
                    )

                    coco_imgs_area_histo_d = self.coco_imgs_area_histo(gt["images"])
                    stats.append(
                        {
                            "file": json_fn,
                            "obj": "images_area_histo",
                            "stats": coco_imgs_area_histo_d,
                        }
                    )
                    self._plot_hist_coco(
                        coco_imgs_area_histo_d["areas"],
                        "Image Areas",
                        f"{fname}_imgs_area_histo.png",
                        show_plots=show_plots,
                    )

            else:
                print("perform geojson stats here")
                gt = gpd.read_file(json_fn)
                gt_image = [
                    {"file_name": el, "id": enu}
                    for enu, el in enumerate(gt["image_filename"])
                ]
                imgs_stats = self.geojson_imgs_stats(gt_image)
                stats.append({"file": json_fn, "obj": "images", "stats": imgs_stats})

                # Geojson Annots stats:

                gt = self._add_operation(
                    gt, self._calc_minrotrect, "geometry", ["minrotrect"]
                )
                gt = self._add_operation(gt, self._calc_bbox, "geometry", ["bbox"])
                gt = self._add_operation(gt, self._area_pol, "geometry", ["area"])
                gt = self._add_operation(gt, self._area_pol, "bbox", ["bbx_area"])
                gt = self._add_operation(gt, self._area_pol, "minrotrect", ["rarea"])
                gt = self._add_operation(
                    gt,
                    self._calc_rectangle_stats,
                    "minrotrect",
                    ["rhigh", "rwidth", "rangle"],
                )
                gt = self._add_operation(
                    gt,
                    self._calc_rectangle_stats,
                    "bbox",
                    ["bbx_high", "bbx_width", "bbx_angle"],
                )
                gt = self._add_operation(
                    gt,
                    self._compute_compactness,
                    "geometry",
                    ["compactness"],
                )
                gt = self._add_operation(
                    gt,
                    self._calc_centroid,
                    "geometry",
                    ["x", "y"],
                )
                stats.append(
                    {
                        "file": json_fn,
                        "obj": "geojson",
                        "stats": {
                            key: self._dataframe_basic_stats(gt, key)
                            for key in [
                                "area",
                                "rarea",
                                "rhigh",
                                "rwidth",
                                "rangle",
                                "bbx_area",
                                "bbx_high",
                                "bbx_width",
                                "bbx_angle",
                                "compactness",
                                "x",
                                "y",
                            ]
                        },
                    }
                )
                for t, xl, yl, xa, ya, n in zip(
                    ["Bounding Box HW", "Rotated HW"],
                    ["width", "rwidth"],
                    ["high", "rhigh"],
                    ["bbx_width", "rwidth"],
                    ["bbx_high", "rhigh"],
                    ["Bounding Box HW", "rotatedHW.png"],
                ):
                    self._plot_scatter(
                        gt,
                        title=t,
                        xlabel=xl,
                        ylabel=yl,
                        xattr=xa,
                        yattr=ya,
                        name=n,
                        show_plots=show_plots,
                    )

                for t, m, n in zip(
                    [
                        "Area of the original geom",
                        "Area of the rotated bounding box",
                        "Area of the bounding box",
                        "Compactness",
                    ],
                    ["area", "rarea", "bbx_area", "compactness"],
                    [
                        "area_origgeom_hist.png",
                        "rotated_area_hist.png",
                        "bbx_area_hist.png",
                        "compactness_hist.png",
                    ],
                ):
                    self._plot_hist(
                        gt, title=t, main_attr=m, name=n, show_plots=show_plots
                    )

                self._plot_rose(gt, show_plots=show_plots)

        with open(os.path.join(self.output_path, "stats.json"), "w") as outfile:
            json.dump(stats, outfile)
        return stats

    @staticmethod
    def coco_imgs_stats(imgs_lst: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        calc stats in coco json images list

        Args:
            imgs_lst: List of images (coco json "images" field).

        Returns:
            stats: List of stats.
        """
        hs, ws = [], []

        for im in imgs_lst:
            hs.append(im["height"])
            ws.append(im["width"])

        return {"mean_height": mean(hs), "mean_width": mean(ws)}

    @staticmethod
    def mask_stats(coco_annots: Dict[Any, Any]) -> List[Union[List[str], List[float]]]:
        """
        A dictionary with ratio coverage (in area) per class.
        Background class has id -999

        Args:
            coco_annots: A dictionary with coco-format annotations
        """
        d = {"imgid": {-999: 0}}
        gt = coco_annots
        # Foreach image and foreach annot within image
        # add the annot area relative to the image area by class/category/kind
        for im in gt["images"]:

            image_id = im["id"]
            d[image_id] = {-999: 0}
            imarea = im["height"] * im["width"]
            an_lst = [an for an in gt["annotations"] if an["image_id"] == image_id]

            for annot in an_lst:

                if "segmentation" in annot:
                    kind = annot["category_id"]
                    seg = annot["segmentation"]

                    if "area" in annot:
                        area = annot["area"]
                    else:
                        area = Polygon([el for el in zip(seg[0::2], seg[1::2])]).area

                    if kind not in d[image_id]:
                        d[image_id][kind] = 0
                    d[image_id][kind] += area / imarea

        d_avg = {}

        for image_id in d:

            for kind in d[image_id]:

                if kind not in d_avg:
                    d_avg[kind] = 0
                if kind in d[image_id]:
                    d_avg[kind] += d[image_id][kind]

        d_avg_summary = {
            int(kind): d_avg[kind] / len([keyimg for keyimg in d]) for kind in d_avg
        }

        # background
        d_avg_summary[-999] = 1 - float(
            np.sum([d_avg_summary[kind] for kind in d_avg_summary])
        )

        name_lst = [
            str([el for el in gt["categories"] if el["id"] == kind][0]["name"])
            if kind != -999
            else "background"
            for kind in d_avg_summary
        ]

        return [name_lst, [float(d_avg_summary[k]) for k in d_avg_summary]]

    @staticmethod
    def coco_class_histo(
        ans: List[Dict[str, Any]], cats: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        compute class tags histogram for coco json

        Args:
            ans: List of annotations (coco json "annotations" field).
            cats: List of categories (coco json "categories" field).

        Returns:
            stats: Classes ids, names and tags count.
        """
        ans_cat = [an["category_id"] for an in ans]
        cats_k = list(set(ans_cat))
        cats_k.sort()
        ans_cc = [ans_cat.count(c) for c in cats_k]
        cat_id_to_name = {c["id"]: c["name"] for c in cats}
        cats_name = [cat_id_to_name[k] for k in cats_k]
        return {"cat_ids": cats_k, "cat_names": cats_name, "cat_tags_count": ans_cc}

    @staticmethod
    def coco_bbox_aspect_ratio_histo(ans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        compute bbox aspect ratio histogram for coco json

        Args:
            ans: List of annotations (coco json "annotations" field).

        Returns:
            stats: Aspect ratios, bin edges and counts.
        """
        ar = [an["bbox"][2] / an["bbox"][3] for an in ans if an["bbox"][3] > 0]
        hist_, bin_edges_ = np.histogram(ar)
        hist = [int(h) for h in hist_]
        bin_edges = [float(be) for be in bin_edges_]

        return {"aspect_ratios": ar, "bin_edges": bin_edges, "ar_counts": hist}

    @staticmethod
    def coco_imgs_aspect_ratio_histo(ims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        compute bbox aspect ratio histogram for coco json

        Args:
            ims: List of images (coco json "images" field).

        Returns:
            stats: Aspect ratios, bin edges and counts.
        """
        ar = [im["width"] / im["height"] for im in ims]
        hist_, bin_edges_ = np.histogram(ar)
        hist = [int(h) for h in hist_]
        bin_edges = [float(be) for be in bin_edges_]

        return {"aspect_ratios": ar, "bin_edges": bin_edges, "ar_counts": hist}

    @staticmethod
    def coco_bbox_area_histo(ans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        compute bbox area histogram for coco json

        Args:
            ans : List of annotations (coco json "annotations" field).

        Returns:
            stats: Areas, bin edges and counts.
        """
        ar = [an["bbox"][2] * an["bbox"][3] for an in ans]
        hist_, bin_edges_ = np.histogram(ar)
        hist = [int(h) for h in hist_]
        bin_edges = [float(be) for be in bin_edges_]
        return {"areas": ar, "bin_edges": bin_edges, "ar_counts": hist}

    @staticmethod
    def coco_imgs_area_histo(ims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        compute bbox area histogram for coco json

        Args:
            ims: List of images (coco json "images" field).
        Returns:
            stats: Areas, bin edges and counts.
        """
        ar = [im["width"] * im["height"] for im in ims]
        hist_, bin_edges_ = np.histogram(ar)
        hist = [int(h) for h in hist_]
        bin_edges = [float(be) for be in bin_edges_]
        return {"areas": ar, "bin_edges": bin_edges, "ar_counts": hist}

    def geojson_imgs_stats(self, imgs_lst: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        calc stats from images filename list

        Args:
            imgs_lst: List of image filenames (geojson "image_filename" field).
        Returns:
            stats: List of stats.
        """
        hs, ws = [], []

        for im in imgs_lst:
            imfn = os.path.join(self.data_path, "images", im["file_name"])
            height, width = cv2.imread(imfn).shape[0:2]
            hs.append(height)
            ws.append(width)

        return {"mean_height": mean(hs), "mean_width": mean(ws)}

    @staticmethod
    def _add_operation(
        gdf: gpd.GeoDataFrame,
        func: Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame],
        in_field: str,
        out_field_lst: List[str],
    ) -> gpd.GeoDataFrame:
        """
        This function returns the result of operation func on the in_field attribute from the geodataframe gdf.
        When the input element row from the geodataframe is None or similar, the return is also None for this element.

        Args:
            gdf: Input geodataframe to add info to.
            func: Function that will add further info to the dataframe
            in_field: Name of the Attribute or column used as an input of the operation
            out_field_lst: Name list of the Attributes or columns that will be added afterwards.

        Returns:
            The new geodataframe with added/modified information.
        """
        if len(out_field_lst) == 1:
            gdf.loc[:, out_field_lst] = [func(el) for el in gdf[in_field].values]
        else:
            vect = [func(el) for el in gdf[in_field].values]
            for enu, outfield in enumerate(out_field_lst):
                gdf.loc[:, outfield] = [v[enu] for v in vect]
        return gdf

    @staticmethod
    def _calc_minrotrect(feat: Polygon) -> Polygon:
        """
        Returns the best fitting rotated bounding box

        Args:
            feat: Input geometrical feature.

        Returns:
            Output geometrical rotated rectangle.
        """
        if feat:
            return feat.minimum_rotated_rectangle
        else:
            return None

    @staticmethod
    def _calc_bbox(feat: Polygon) -> Polygon:
        """
        Returns the best fitting bounding box

        Args:
            feat: Input geometrical feature.

        Returns:
            Output bounding box rectangle.
        """
        if feat:
            return box(*feat.bounds)
        else:
            return None

    @staticmethod
    def _area_pol(feat: Polygon) -> Union[float, None]:
        """
        Returns the area of the main geometry

        Args:
            feat: Input geometrical feature.

        Returns:
            area
        """
        if feat:
            return float(feat.area)
        else:
            return None

    @staticmethod
    def _compute_compactness(feat: Polygon) -> Union[float, None]:
        """
        Returns compactness of the polygon

        Args:
            feat: Input geometrical feature.

        Returns:
            compactness
        """
        area = feat.area
        perimeter = feat.length
        if feat:
            return float(4 * math.pi * area / (perimeter * perimeter))
        else:
            return None

    @staticmethod
    def _calc_rectangle_stats(feat: Polygon) -> Union[List[float], List[None]]:
        """
        It calculates basic statistics from a rectangle (expressed as Polygon)

        Args:
            feat: Input geometrical feature.

        Returns:
            high, width, angle from the rotated box
        """

        if feat:

            def angle(x1: float, y1: float, x2: float, y2: float) -> float:
                return math.atan2(y1 - y2, x1 - x2)

            box = feat

            # get minimum bounding box around polygon
            # box = poly.minimum_rotated_rectangle

            # get coordinates of polygon vertices
            x, y = box.exterior.coords.xy

            # get length of bounding box edges
            edge_length = (
                Point(x[0], y[0]).distance(Point(x[1], y[1])),
                Point(x[1], y[1]).distance(Point(x[2], y[2])),
            )

            # get length of polygon as the longest edge of the bounding box
            length = max(edge_length)

            # get width of polygon as the shortest edge of the bounding box
            width = min(edge_length)

            # Calculate max_angle
            id_max = [enu for enu, el in enumerate(edge_length) if el == length][0]

            if id_max == 0:
                angle_max_axis = angle(x[0], y[0], x[1], y[1])
            elif id_max == 1:
                angle_max_axis = angle(x[1], y[1], x[2], y[2])

            return [float(length), float(width), math.degrees(angle_max_axis)]
        else:
            return [None] * 3

    @staticmethod
    def _calc_centroid(feat: Polygon) -> Union[List[float], List[None]]:
        """
        It returns the x and y position of the centroid from the input Polygon

        Args:
            feat: Input geometrical feature.

        Returns:
            x,y centroid position
        """
        if feat:
            c = feat.centroid
            return [c.x, c.y]
        else:
            return [None] * 2

    @staticmethod
    def _dataframe_basic_stats(
        df: gpd.GeoDataFrame, field: str
    ) -> Union[List[Dict[str, float]], List[Dict[str, None]]]:
        """
        It returns the min, mean, max from a dataframe field

        Args:
            field: Geodataframe to count stats from
            field: Name of the attribute, column or field to calc stats from

        Returns:
            returns the min, mean, max from a dataframe field
        """
        if field in df:
            return [
                {"min": df[field].min()},
                {"mean": df[field].mean()},
                {"max": df[field].max()},
            ]
        else:
            return [
                {"min": None},
                {"mean": None},
                {"max": None},
            ]

    def _plot_scatter(
        self,
        df: gpd.GeoDataFrame,
        title: Optional[str] = "Centroids distribution",
        xlabel: Optional[str] = "centroid X",
        ylabel: Optional[str] = "centroid Y",
        xattr: Optional[str] = "x",
        yattr: Optional[str] = "y",
        name: Optional[str] = "centroids.png",
        show_plots: Optional[bool] = True,
    ) -> None:
        """
        It generates a png with a scatter plot
        """
        if not name:
            name = "centroids.png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.scatter(df[xattr], df[yattr], alpha=0.2)
        fig.savefig(os.path.join(self.output_path, "images", name))
        plt.close(fig)

    def _plot_hist(
        self,
        df: gpd.GeoDataFrame,
        title: Optional[str] = "Area from original geometry",
        main_attr: Optional[str] = "area",
        name: Optional[str] = "area_geom.png",
        show_plots: Optional[bool] = True,
    ) -> None:
        """
        It generates a png with a histogram plot
        """
        if not name:
            name = "area_geom.png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title(title)
        _ = ax.hist(df[main_attr])
        ax.grid(True)
        fig.savefig(os.path.join(self.output_path, "images", name))
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_hist_coco(
        self,
        data: List[float],
        title: str,
        name: str,
        show_plots: Optional[bool] = True,
    ) -> None:
        """
        It generates a png with a histogram plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title(title)
        _ = ax.hist(data)
        ax.grid(True)
        fig.savefig(os.path.join(self.output_path, "images", name))
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_hist_list(
        self,
        names: List[Any],
        counts: List[Union[int, float]],
        title: str,
        name: str,
        show_plots: Optional[bool] = True,
        xlabel: Optional[str] = "Tags count",
    ) -> None:
        """
        Generates a histogram plot from 2 lists, show if required and save in png
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(title)
        ax.barh(names, counts)
        ax.invert_yaxis()  # labels read top-to-bottom
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.grid(True)
        fig.savefig(os.path.join(self.output_path, "images", name))
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_rose(
        self,
        df: gpd.GeoDataFrame,
        title: Optional[str] = "Polar histogram of the rotated angles",
        main_attr: Optional[str] = "rangle",
        name: Optional[str] = "rangle_polarhist.png",
        show_plots: Optional[bool] = True,
    ) -> None:
        """
        It generates a png with a polar histogram plot (rose plot)
        """
        if not name:
            name = "area_geom.png"
        fig, ax = plt.subplots(
            1, 1, figsize=(10, 10), subplot_kw=dict(projection="polar")
        )
        ax.set_title(title)
        _ = self._circular_hist(ax, df[main_attr], density=True)
        fig.savefig(os.path.join(self.output_path, "images", name))
        if show_plots:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _circular_hist(
        ax: Any,
        x: Any,
        bins: Optional[int] = 16,
        density: Optional[bool] = True,
        offset: Optional[float] = 0.0,
        gaps: Optional[bool] = True,
    ) -> List[Any]:
        """
        Produce a circular histogram of angles on ax.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.PolarAxesSubplot
            axis instance created with subplot_kw=dict(projection='polar').

        x : array
            Angles to plot, expected in units of radians.

        bins : int, optional
            Defines the number of equal-width bins in the range. The default is 16.

        density : bool, optional
            If True plot frequency proportional to area. If False plot frequency
            proportional to radius. The default is True.

        offset : float, optional
            Sets the offset for the location of the 0 direction in units of
            radians. The default is 0.

        gaps : bool, optional
            Whether to allow gaps between bins. When gaps = False the bins are
            forced to partition the entire [-pi, pi] range. The default is True.

        Returns
        -------
        n : array or list of arrays
            The number of values in each bin.

        bins : array
            The edges of the bins.

        patches : `.BarContainer` or list of a single `.Polygon`
            Container of individual artists used to create the histogram
            or list of such containers if there are multiple input datasets.
        """
        if not bins:
            bins = 16
        # Wrap angles to [-pi, pi)
        x = (x + np.pi) % (2 * np.pi) - np.pi

        # Force bins to partition entire circle
        if not gaps:
            bins = np.linspace(-np.pi, np.pi, num=bins + 1)

        # Bin data and record counts
        n, bins_lst = np.histogram(x, bins=bins)

        # Compute width of each bin
        widths = np.diff(bins_lst)

        # By default plot frequency proportional to area
        if density:
            # Area to assign each bin
            area = n / x.size
            # Calculate corresponding bin radius
            radius = (area / np.pi) ** 0.5
        # Otherwise plot frequency proportional to radius
        else:
            radius = n

        # Plot data on ax
        patches = ax.bar(
            bins_lst[:-1],
            radius,
            zorder=1,
            align="edge",
            width=widths,
            edgecolor="C0",
            fill=False,
            linewidth=1,
        )

        # Set the direction of the zero angle
        ax.set_theta_offset(offset)

        # Remove ylabels for area plots (they are mostly obstructive)
        if density:
            ax.set_yticks([])

        return [n, bins, patches]

    @staticmethod
    def notebook_imgs_preview(
        data_path: str,
        sample: Optional[int] = None,
        size: Optional[int] = None,
    ) -> None:
        """
        A tool that can be used to visualize a summary of the images in a path

        Args:
            data_path: Path pointing to the images folder
            sample: number of samples to build the preview from
            size: The size of each sample.
        """
        # from easyimages import EasyImageList
        # setting default params:
        if sample is None:
            sample = 100
        if size is None:
            size = 100

        li = EasyImageList.from_folder(data_path)
        li.symlink_images()
        li.html(sample=sample, size=size)

    @staticmethod
    def notebook_annots_summary(
        df: gpd.pd.DataFrame,
        fields_to_include: List[str],
        export_html_filename: Optional[str] = None,
        show_inline: Optional[bool] = True,
    ) -> None:
        """
        A tool that can be used to visualize a interactive summary of the annotations

        Args:
            df: dataframe
            fields_to_include: which fields to consider (here one must about fields with complex structure such as 'geometry')
            export_html_filename: If indicated, it will export to html the interactive chart to the inputted filename.
            show_inline: By default is True, which indicates that it will be displayed inline in the notebook from where it is called.
        """
        try:
            jsonstr = df[fields_to_include].to_json(orient="records")
        except Exception as e:
            print(e)
            print(
                'Could not convert the dataframe to a json string.A trick to avoid the "the recursivity reached its limit" Error is to exclude \nconflictive attributes such as the "geometry" in the fields_to_include list argument'
            )
            return None

        HTML_TEMPLATE = """
                <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
                <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
                <facets-dive id="elem" height="600"></facets-dive>
                <script>
                var data = {jsonstr};
                document.querySelector("#elem").data = data;
                </script>"""
        html = HTML_TEMPLATE.format(jsonstr=jsonstr)
        page = HTML(html)

        if show_inline:
            display(page)

        if export_html_filename:
            with open(export_html_filename, "w") as f:
                f.write(page.data)
