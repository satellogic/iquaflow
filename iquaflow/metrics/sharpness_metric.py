import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import cv2
import geojson
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.fft import fft, fftfreq
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import curve_fit
from shapely.geometry import LineString, MultiPoint
from shapely.ops import snap
from skimage import feature
from skimage.transform import probabilistic_hough_line

from iquaflow.metrics import Metric

DEBUG_DIR = "./rer_debug"


def model_esf(x: np.array, a: float, b: float, c: float, d: float) -> np.array:
    """
    The model function to fit the ESF.
    """
    try:
        b_ = np.exp((x - b) / c)
        return a / (1 + b_) + d
    except Exception:
        pass


def gaussian(x: np.array, a: float, x0: float, sigma: float) -> np.array:
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


class SharpnessMeasure:
    """ "
    Measures the sharpness of an image.
    The implementation is mainly based of the paper of Cenci, L. et al. 2021: "Presenting a Semi-Automatic,
    Statistically-Based Approach to Assess the Sharpness Level of Optical Images from Natural Targets via the Edge
    Method. Case Study: The Landsat 8 OLI-L1T Data", DOI:  https://doi.org/10.3390/rs13081593

    We thank Luca Cenci and Valerio Pampanoni for their advise.
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
        edge_length: int = 5,
        edge_distance: int = 10,
        contrast_params: Dict[str, Dict[str, float]] = {
            "channel 0": {"alpha": 1.2, "beta": 0.3, "gamma": 1.2}
        },
        pixels_sampled: Optional[int] = None,
        r2_threshold: float = 0.995,
        snr_threshold: float = 50,
        get_rer: bool = True,
        get_fwhm: bool = True,
        get_mtf: bool = False,
        get_mtf_curve: bool = False,
        save_edges: bool = False,
        save_edges_to: str = "",
        debug: bool = False,
        calculate_mean: bool = False,
    ):

        """
        The implementation needs several parameters, where the best value can be quite different depending on the source
        of the image.
        :param window_size: If the image is large, the edge detector works better if we cut it to smaller windows. You
        need to define the window size here. If None is set, the whole image is passed to the edge detector at once.
        :param stride: The stride of the sliding windows. By default it is the same as the window size.
        :param edge_length: The length of the edges we want to evaluate.
        :param edge_distance: The minimum distance between to edges.
        :param contrast_params: A dictionary of dictionaries containing the alpha, beta, and gamma parameters to
        evaluate if an edge has sufficient contrast. The keys need to contain the channel number and the values are
        dictionaries containing the parameters.
        :param pixels_sampled: The number of pixels to be sampled on either sides of the edges. Minimum is 5. Default
        intents to create patches with a square shape
        :param r2_threshold: When fitting the constructed ESF with a theoretical curve, the R2 value of the fit has to
        be above this threshold to be evaluated.
        :param snr_threshold: The minimum value of esge SNR for an edge to be evaluated
        :param get_rer: If True, the RER value is calculated.
        :param get_fwhm: If True, the FWHM value is calculated.
        :param get_mtf: If True, the MTF is calculated, and its value at the Nyquist frequency and at the half Nyquist
        frequency is returned.
        :param debug: If True, some debugging messages and plots are shown.
        :param calculate_mean: If the image has multiple channels, all metrics are calculated independently for each
        channel. Setting this parameter as True, the mean values across channels is returned for each metric as well.

        """

        self.window_size = (window_size,)
        self.stride = self.window_size if stride is None else stride
        self.edge_length = edge_length
        self.edge_distance = edge_distance
        assert "channel 0" in contrast_params.keys()
        assert "alpha" in contrast_params["channel 0"].keys()
        self.contrast_params = contrast_params
        self.pixels_sampled = (
            int((self.edge_length + 6) // 2 + 1)
            if pixels_sampled is None
            else pixels_sampled
        )
        assert self.pixels_sampled >= 5
        self.r2_threshold = r2_threshold
        self.snr_threshold = snr_threshold
        self.get_rer = get_rer
        self.get_fwhm = get_fwhm
        self.get_mtf = get_mtf
        self.get_mtf_curve = get_mtf_curve
        if self.get_mtf_curve:
            self.raw_lsf_list: List[Any] = []
            self.mtf_curves: Dict[str, Any] = {}
        self.save_edges = save_edges
        if self.save_edges:
            self.save_edges_to = save_edges_to
            assert self.save_edges_to.endswith(".geojson")
            self.features: List[Any] = []
        self.debug = debug
        self.calculate_mean = calculate_mean
        self.patch_list: List[float] = []

        self.edge_snrs: List[Any] = []

    def apply(self, image: np.array) -> Any:
        """
        The main function to execute the measurement.
        :param image:
        :return:
        """
        results = {}
        # replace 0 values with np.nan values
        image = np.where(image == 0, np.nan, image)
        # if the image has multiple channels, each channel is evaluated individually
        if len(image.shape) > 2:
            for i in range(image.shape[2]):
                if f"channel {i}" in self.contrast_params.keys():
                    alpha = self.contrast_params[f"channel {i}"]["alpha"]
                    beta = self.contrast_params[f"channel {i}"]["beta"]
                    gamma = self.contrast_params[f"channel {i}"]["gamma"]
                else:
                    alpha = self.contrast_params["channel 0"]["alpha"]
                    beta = self.contrast_params["channel 0"]["beta"]
                    gamma = self.contrast_params["channel 0"]["gamma"]
                results[f"channel {i}"] = self.apply_to_one_channel(
                    image[:, :, i], (alpha, beta, gamma), i
                )
        else:
            alpha = self.contrast_params["channel 0"]["alpha"]
            beta = self.contrast_params["channel 0"]["beta"]
            gamma = self.contrast_params["channel 0"]["gamma"]
            results["channel 0"] = self.apply_to_one_channel(
                image, (alpha, beta, gamma), 0
            )

        # Calculate mean values of the metrics across channels
        if self.calculate_mean:
            out = {}  # type: ignore
            for ch in results.values():
                for metric, value in ch.items():
                    for direction, v in value.items():
                        k = f"{metric}_{direction}"
                        if k in out.keys():
                            out[k][0].append(v["mean"])
                            out[k][1].append(v["std"])
                        else:
                            out[k] = [[v["mean"]], [v["std"]]]
            results["mean"] = {}
            for k, v in out.items():
                if len(v[0]) <= 1:
                    results["mean"][k] = (v[0][0], v[1][0])
                else:
                    v0 = np.array(v[0])
                    v1 = np.array(v[1])
                    results["mean"][k] = (
                        np.nanmean(v0),
                        max(
                            np.sqrt(np.sum(v1[~np.isnan(v1)] ** 2))
                            / len(v1[~np.isnan(v1)]),
                            np.nanstd(v1),
                        ),
                    )

        if self.save_edges:
            feature_collection = geojson.FeatureCollection(self.features)
            with open(self.save_edges_to, "w") as file:
                geojson.dump(feature_collection, file)
        return results

    def apply_to_one_channel(
        self,
        image: np.array,
        patch_params: Tuple[float, float, float],
        channel_num: int,
    ) -> Any:
        """
        Runs the measurement one image channel at the the time.
        The following steps are executed:
        1. Find straight lines in the image with the required length
        2. Sort the lines by their angle
        3. Cut patches around the lines, and check if they have sufficient contrast
        4. Calculate the ESFs
        5. Calculate the required metrics.
        :param image:
        :param patch_params:
        :return:
        """
        assert len(image.shape) < 3

        output: Dict[str, Dict[str, Any]] = {}
        if self.get_rer:
            output["RER"] = {}

        if self.get_fwhm:
            output["FWHM"] = {}

        if self.get_mtf:
            output["MTF_NYQ"] = {}
            output["MTF_halfNYQ"] = {}

        edge_dict, line_dict = self.compose_edge_list(image, patch_params)

        for i, (kind, k) in enumerate(
            zip(
                ["vertical", "horizontal", "other"],
                ["X", "Y", "other"],
            )
        ):

            # self.edge_snrs.append([])
            # # Create edge_list
            # edge_list, line_list = self.get_edge_list(patches, good_lines[i])
            edge_list = edge_dict[kind]
            line_list = line_dict[kind]

            if self.get_mtf_curve:
                self.mtf_curves[k] = self.calculate_mtf_curve()

            # Calculate RER
            if self.get_rer:
                rer = self.calculate_rer(edge_list)

                if rer is not None:
                    output["RER"][k] = {
                        "mean": np.mean(rer),
                        "std": np.std(rer),
                        "length": np.size(rer),
                        "median": np.median(rer),
                        "IQR": np.subtract(*np.percentile(rer, [75, 25])),
                    }

            # Calculate FWHM
            if self.get_fwhm:
                fwhm = self.calculate_fwhm(edge_list)

                if fwhm is not None:
                    output["FWHM"][k] = {
                        "mean": np.mean(fwhm),
                        "std": np.std(fwhm),
                        "length": np.size(fwhm),
                        "median": np.median(fwhm),
                        "IQR": np.subtract(*np.percentile(fwhm, [75, 25])),
                    }
            # Calculate MTF
            if self.get_mtf:
                mtf = self.calculate_mtf(edge_list)
                if mtf is not None:
                    output["MTF_NYQ"][k] = {
                        "mean": np.mean(mtf[0]),
                        "std": np.std(mtf[0]),
                        "length": np.size(mtf[0]),
                        "median": np.median(mtf[0]),
                        "IQR": np.subtract(*np.percentile(mtf[0], [75, 25])),
                    }
                    output["MTF_halfNYQ"][k] = {
                        "mean": np.mean(mtf[1]),
                        "std": np.std(mtf[1]),
                        "length": np.size(mtf[1]),
                        "median": np.median(mtf[1]),
                        "IQR": np.subtract(*np.percentile(mtf[1], [75, 25])),
                    }

            if self.save_edges:
                for i, line in enumerate(line_list):
                    ls = geojson.LineString([(line[2], line[0]), (line[3], line[1])])
                    props = {"direction": k, "channel": f"channel {channel_num}"}
                    if self.get_rer:
                        props["RER"] = rer[i]
                    if self.get_fwhm:
                        props["FWHM"] = fwhm[i]
                    if self.get_mtf:
                        props["MTF_NYQ"] = mtf[0][i]
                        props["MTF_halfNYQ"] = mtf[1][i]
                    self.features.append(geojson.Feature(geometry=ls, properties=props))

        if self.get_mtf_curve:
            return self.mtf_curves
        # if self.debug:
        #     self._plot_good_patches(image, vertical_patches, horizontal_patches, other_patches, lines, good_lines)
        return output

    def compose_edge_list(
        self, image: np.array, patch_params: Tuple[float, float, float]
    ) -> Tuple[Any, Any]:
        edge_dict = {}
        line_dict = {}
        # Find straight lines in the image
        lines = self.get_lines(image)
        if lines is None:
            raise Exception(
                "Not a single line found on image. Try a different set of parameters, or check your image."
            )

        # Sort lines by angles
        vertical, horizontal, other = self.sort_angles(lines)

        # Find patches that are in agreement with the contrast parameters
        (
            vertical_patches,
            horizontal_patches,
            other_patches,
            good_lines,
        ) = self.find_good_patches(image, vertical, horizontal, other, patch_params)
        # Rotate horizontal patches
        horizontal_patches = [np.rot90(p) for p in horizontal_patches]

        for i, (patches, kind, k) in enumerate(
            zip(
                [vertical_patches, horizontal_patches, other_patches],
                ["vertical", "horizontal", "other"],
                ["X", "Y", "other"],
            )
        ):
            self.edge_snrs.append([])
            # Create edge_list
            edge_list, line_list = self.get_edge_list(patches, good_lines[i])
            edge_dict[kind] = edge_list
            line_dict[kind] = line_list
        return edge_dict, line_dict

    def get_lines(self, image: np.array) -> np.array:
        """
        Find suitable line in the image.
        Uses sliding windows to go through the images, and in each window calls the edge_detector function
        :param image: numpy array, only one channel
        :return: an array containing the detected lines
        """

        lines = np.empty(shape=[0, 4])

        # Uses sliding windows, because the Hough transform works better for smaller images, instead of
        # huge satellite images
        im_height, im_width = image.shape
        if self.window_size[0] is None:
            lines = self.edge_detector(image)
        else:
            step = self.stride[0]  # type: ignore
            for i in range(0, im_height, step):
                for j in range(0, im_width, step):
                    image_window = image[
                        i : min(i + self.window_size[0], image.shape[0]),
                        j : min(j + self.window_size[0], image.shape[1]),
                    ]
                    # if the window mostly contain np.nan values, don't run the edge detector
                    if (
                        np.isnan(image_window).sum()
                        >= 0.8 * image_window.shape[0] * image_window.shape[1]
                    ):
                        continue

                    lines_ = self.edge_detector(image_window)
                    if lines_ is None or len(lines_) == 0:
                        continue
                    # Correct the line coordinates to correspond to the coordinates of the whole image instead of the window.

                    lines_[:, 0::2] += j
                    lines_[:, 1::2] += i
                    lines = np.concatenate([lines, lines_], axis=0)
        return lines

    def edge_detector(
        self, image: np.array, threshold: int = 15, line_gap: int = 0
    ) -> np.array:
        """
        Runs the canny edge detector to find edges, then uses the Hough transform to find straight lines with the
        given parameters.
        Args:
            image: the input image
            threshold:
            min_line_length: the minimum length of the line in pixels
            line_gap: the line gap
        Return:
            numpy array with the line coordinates
        """
        canny = feature.canny(
            image,
            sigma=1,
            low_threshold=np.nanquantile(image, 0.99) * 0.1,
            high_threshold=np.nanquantile(image, 0.99) * 0.25,
        )
        # List of lines identified, lines in format ((x0, y0), (x1, y1)), indicating line start and end
        lines = probabilistic_hough_line(
            canny,
            threshold=threshold,
            line_length=self.edge_length,
            line_gap=line_gap,
            seed=42,
            theta=np.linspace(-np.pi / 2, np.pi / 2, 360 * 1, endpoint=False),
        )

        if len(lines) == 0:
            return None
        # Cuts the lines to segments with lengths of self.edge_length
        lines_ = self._sort_lines(lines)
        # Format: [x0, y0, x1, y1]
        return lines_

    def _sort_lines(self, lines: Any) -> np.array:
        """
        Given the list of lines found by the edge detectors, it checks the lengths of the lines, and cuts them up to
        segments with a length of the self.edge_length.
        :param lines:
        :return:
        """
        good_lines = []

        lines_ = [LineString(list(line)) for line in lines]
        for line in lines_:
            for j in range(self.edge_length):
                splitter = MultiPoint(
                    [
                        line.interpolate(j + (self.edge_length * i))
                        for i in range(int(line.length // self.edge_length + 1))
                    ]
                )
                cuts = snap(line, splitter, 1e-5)
                for i in range(1, len(cuts.coords)):
                    x0, y0 = round(cuts.coords.xy[0][i - 1]), round(
                        cuts.coords.xy[1][i - 1]
                    )
                    x1, y1 = round(cuts.coords.xy[0][i]), round(cuts.coords.xy[1][i])
                    ll = LineString([(x0, y0), (x1, y1)])
                    if ll.length >= self.edge_length - 1e-5:
                        good_lines.append([x0, y0, x1, y1])
        return np.array(good_lines)

    def sort_angles(self, lines: np.array) -> Tuple[Any, ...]:
        """
        Calculates the angle of the each of the lines relative to vertical.
        Sorts the lines into two groups:
            vertical (within +-15 degrees from vertical),
            horizontal (within +-15 degrees from horizontal),
            other.
        Args:
            lines: list of lines
        Returns:
            tuple of 3 list of lines, each with [x0,x1,y0,y1,theta]
        """
        horizontal = []
        vertical = []
        other = []

        for line in lines:
            x0, y0, x1, y1 = line
            # theta is relative to the vertical, y direction
            theta = np.rad2deg(np.arctan2(x1 - x0, y1 - y0)) % 180

            coords = np.array([x0, x1, y0, y1, theta])

            if (15 >= theta) or (180 - 15 <= theta):
                vertical.append(coords)
            elif 90 - 15 <= theta <= 90 + 15:
                horizontal.append(coords)
            else:
                other.append(coords)
        return (vertical, horizontal, other)

    def find_good_patches(
        self,
        image: np.array,
        vertical: List[np.array],
        horizontal: List[np.array],
        other: List[np.array],
        params: Tuple[float, float, float],
    ) -> Any:
        """
        Takes the line lists, for each line cuts a patch around the line. Then it checks if the patch complies with the
        contrast conditions defined by the parameters. If it complies, it adds the patch to the patch list.
        :param image:
        :param vertical: vertical line list
        :param horizontal: horizontal line list
        :param other: other line list
        :param params: contrast condition parameters
        :return: a Tuple with the patch lists in each directions, and a list of the good lines
        """

        alpha, beta, gamma = params
        vertical_patches: List[Any] = []
        horizontal_patches: List[Any] = []
        other_patches: List[Any] = []
        good_lines: List[Any] = []

        np.random.shuffle(vertical)
        np.random.shuffle(horizontal)
        np.random.shuffle(other)

        for lines, kind in zip([vertical, horizontal, other], ["v", "h", "o"]):
            good_lines.append([])
            masked_image = image.copy()
            for i, l in enumerate(lines):
                # adapt to array indexing
                y1_, y2_, x1_, x2_, theta = l

                x1, x2 = int(min(x1_, x2_)), int(max(x1_, x2_))
                y1, y2 = int(min(y1_, y2_)), int(max(y1_, y2_))

                # make sure the line is not too close to the image edge
                if (
                    x1 < 2 * self.pixels_sampled
                    or y1 < 2 * self.pixels_sampled
                    or x2 > image.shape[0] - 2 * self.pixels_sampled
                    or y2 > image.shape[1] - 2 * self.pixels_sampled
                ):
                    continue

                if kind == "v":
                    # for vertical lines the patch size is (line length + 6, 2*self.pixels_sampled)
                    patch = masked_image[
                        x1 - 3 : x1 + self.edge_length + 3,
                        y1 - self.pixels_sampled : y1 + self.pixels_sampled + 1,
                    ].copy()

                    # sample the dark and bright sides of the edge
                    DN1 = patch[3:-3, : self.pixels_sampled - 1]
                    DN2 = patch[3:-3, -(self.pixels_sampled - 1) :]

                elif kind == "h":
                    # for horizontal lines the patch size is (2*self.pixels_sampled, line length + 6)
                    patch = masked_image[
                        x1 - self.pixels_sampled : x1 + self.pixels_sampled + 1,
                        y1 - 3 : y1 + self.edge_length + 4,
                    ].copy()
                    # sample the dark and bright sides of the edge
                    DN1 = patch[: self.pixels_sampled - 1, 3:-3]
                    DN2 = patch[-(self.pixels_sampled - 1) :, 3:-3]
                else:
                    # for other angles, first a patch of (4*self.pixels_sampled, 4*self.pixels_sampled) is cut
                    # than the patch is rotated by the angle of theta to make it vertical
                    p = masked_image[
                        x1 - 2 * self.pixels_sampled : x2 + 2 * self.pixels_sampled + 1,
                        y1 - 2 * self.pixels_sampled : y2 + 2 * self.pixels_sampled + 1,
                    ].copy()
                    p_rot = ndimage.rotate(p, -theta)
                    _norm = np.linalg.norm([x2 - x1, y2 - y1])
                    patch = p_rot[
                        p_rot.shape[0] // 2
                        - int(_norm / 2)
                        - 3 : p_rot.shape[0] // 2
                        + int(_norm / 2)
                        + 4,
                        p_rot.shape[1] // 2
                        - self.pixels_sampled : p_rot.shape[1] // 2
                        + self.pixels_sampled
                        + 1,
                    ]

                    DN1 = patch[3:-3, : self.pixels_sampled - 1]
                    DN2 = patch[3:-3, -(self.pixels_sampled - 1) :]

                # print(np.isnan(patch).any())
                if np.isnan(patch).any():
                    continue

                if (
                    DN1.shape[0] == 0
                    or DN2.shape[0] == 0
                    or np.isnan(DN1).sum() > 0
                    or np.isnan(DN2).sum() > 1
                ):
                    continue

                if np.mean(DN1) > np.mean(DN2):
                    DNb = DN1
                    DNd = DN2
                else:
                    DNb = DN2
                    DNd = DN1

                # Check if patch complies contrast conditions
                if (
                    np.mean(DNb) / np.mean(DNd) > alpha
                    and np.std(DNb) / np.std(patch) < beta
                    and np.std(DNd) / np.std(patch) < beta
                    and np.quantile(DNb, 0.1) / np.quantile(DNd, 0.9) > gamma
                ):

                    # set nearby pixel values to nan in order to sample edges too close to each other
                    d = self.edge_distance // 2
                    masked_image[x1 - d : x2 + d, y1 - d : y2 + d] = np.nan

                    if kind == "v":
                        vertical_patches.append(patch)

                    elif kind == "h":
                        horizontal_patches.append(patch)

                    else:
                        other_patches.append(patch)

                    good_lines[-1].append(l)
        return (vertical_patches, horizontal_patches, other_patches, good_lines)

    def _plot_good_patches(
        self, image, v_patches, h_patches, o_patches, lines, good_lines
    ):
        fig, ax = plt.subplots(figsize=(100, 100))
        ax.imshow(image)
        print(f"good: {len(good_lines[-1])}")
        for line in lines:
            x1, y1, x2, y2 = line
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)
        for line in good_lines:
            x1, x2, y1, y2, _ = line
            ax.plot([x1, x2], [y1, y2], color="red", linewidth=2)

        fn = "good_lines.png"
        fig.savefig(os.path.join(DEBUG_DIR, fn))
        plt.clf()

    def get_edge_list(
        self, patch_list: List[Any], line_list: List[Any]
    ) -> Tuple[Any, Any]:
        """
        Create a list of the good edges using the provided patch list
        :param patch_list:
        :param lines:
        :return:
        """
        edge_list = []
        final_line_list = []

        for (patch, line) in zip(patch_list, line_list):
            _esf = self._get_edge(patch, line)
            if _esf is not None:
                edge_list.append(_esf[0])
                final_line_list.append(line)

        return edge_list, final_line_list

    def _get_edge(self, patch: np.array, line: np.array) -> np.array:
        """
        Return the ESF extracted from the provided patch, if possible.
        :param patch:
        :return:
        """
        # Calculate the edge locations
        edge_coeffs = self.fit_subpixel_edge(patch)
        # Calculate the ESF
        _esf = self.compute_esf(patch, edge_coeffs)
        if _esf is not None:
            self.patch_list.append(patch)
            return _esf, line

    def fit_subpixel_edge(self, patch: np.array) -> Any:
        """
        Calculates the exact edge location with a subpixel precison.
        :param patch:
        :return:
        """
        points = []
        # fit a cubic polynomial for each row
        # calculate the second derivative of the polynomial
        # where the second derivative is 0, that is the edge position at that row
        x = np.array(range(patch.shape[1]))
        y = np.moveaxis(patch[3:-3], 0, -1)
        coeffs = np.polyfit(x, y, 3)
        coeffs = np.moveaxis(coeffs, 0, -1)
        for c in coeffs:
            ffit = np.poly1d(c)
            d = ffit.deriv(2)
            points.append(-d.coeffs[1] / d.coeffs[0])
        # fit line to the derived points to find the equation of the edge
        edge_coeffs = np.poly1d(np.polyfit(list(range(len(points))), points, 1))
        return edge_coeffs

    def compute_esf(self, patch: np.array, edge_coeffs: Any) -> Any:
        """
        Constructs the ESF, the normalized ESF, and the LSF, checks their quality, and fits the theoretical functions.
        :param patch:
        :param edge_coeffs:
        :return:
        """
        # Check if edge is in the correct list
        if abs(90 - np.rad2deg(np.arctan(edge_coeffs[0]))) > 15:
            return None
        patch_h, patch_w = patch.shape

        # Pixel coordinates for the ESF
        x = np.arange(-patch_w // 2 + 1, patch_w // 2, 0.1)

        # pixel coordinates X direction
        x1 = np.arange(0, patch_w)
        # pixel coordinates Y direction
        x2 = np.arange(3, patch_h - 3, 0.5)

        # Placeholder for the transects
        edges = np.zeros((x2.shape[0], x.shape[0]))

        # Calculate the Y coordinates of each the sampling points of each of the transects
        transect_fit = np.array(
            [
                np.polynomial.Polynomial(
                    [k - (-edge_coeffs[1] * edge_coeffs[0]), -edge_coeffs[1]]
                )(x1)
                for k in x2
            ]
        )

        # Calculate the vectors pointing from the intersection of the edge and the transect to the sampling point
        m = np.array(np.meshgrid(edge_coeffs(x2), x1)).T
        v0 = m[..., 1] - m[..., 0]
        v1 = transect_fit - x2[:, None]
        v = np.array([v0, v1])

        # Calculate the distance between the sampling point and the edge
        d = v[0] / np.abs(v[0]) * np.linalg.norm(v, axis=0)
        # Make sure that the sampling point coordinates are in the patch
        m = np.sum(np.round(transect_fit).astype(int) <= 0, axis=1) + np.sum(
            np.round(transect_fit).astype(int) >= patch_h, axis=1
        )
        m = m == 0
        d = d[m]
        # For each transect, calculate the pixel coordinates of the sampling points, record their value and their
        # distance from the edge. Then fit a cubic spline function to the points.
        if not d.shape[0] == 0:
            idx = np.array(
                [
                    np.round(transect_fit[m]).astype(int),
                    np.vstack([x1] * x2.shape[0])[m],
                ]
            )
            # find the pixel value at the sampling point
            val = patch[idx[0], idx[1]].copy()

            if val.shape[0] > 0:
                val = np.take_along_axis(val, np.argsort(d, axis=1), axis=1)
                dists = np.take_along_axis(d, np.argsort(d, axis=1), axis=1)
                # for each transect fit a cubic spline on the sampled points, and resample the fit
                for i, (d, v) in enumerate(zip(dists, val)):
                    c = CubicSpline(d, v)
                    edges[i] = c(x)

        # Make sure there are no invalid edges before calculating the mean
        edges = edges[~np.all(edges == 0, axis=1)]
        # Make sure there are enough valid transects
        if edges.shape[0] < 5:
            return None
        # Calculate the mean ESF
        esf = np.mean(edges, axis=0)

        # Shift the esf so the edge is in the middle
        idx = np.argwhere((-3 <= x) & (x <= 3)).flatten()
        x_ = x[idx]
        esf_ = esf[idx]
        shift = x_.shape[0] // 2 - np.argmax(np.abs(esf_ - np.roll(esf_, 1))[1:])
        esf = np.roll(esf, shift)

        # calculate the LSF
        lsf = np.abs((esf - np.roll(esf, 1))[1:])
        shift = abs(shift)
        if shift > 0:
            x_ = x[shift + 1 : -shift].copy()
            lsf_ = lsf[shift:-shift].copy()
        else:
            x_ = x[1:].copy()
            lsf_ = lsf.copy()

        # Fit a Gaussian to the LSF
        lsf_popt = self._fit_function(gaussian, x_, lsf_, p0=[50, 0, 1])

        # Trim the wings of the ESF at +- 3 sigma of the fitted Gaussian
        if lsf_popt is not None:
            lim1 = round((abs(lsf_popt[1]) + lsf_popt[-1] * 3) * 2) / 2 + 0.5
            lim2 = round((abs(lsf_popt[1]) + lsf_popt[-1] * 5) * 2) / 2 + 0.5
        else:
            lim1 = x[-1]
            lim2 = x[-1]
        if lim1 <= 0:
            lim1 = x[-1]
        idx1 = np.argwhere((-lim1 <= x) & (x <= lim1)).flatten()
        idx2 = np.argwhere((-lim2 <= x) & (x <= lim2)).flatten()
        if lim2 >= x[-shift]:
            idx2 = idx2[shift:-shift]
        x_esf = x[idx1[1:]].copy()
        esf = esf[idx1[1:]].copy()
        x_lsf = x[idx2[1:]].copy()
        lsf = lsf[idx2[:-1]].copy()

        # Fit a Gaussian to the trimmed LSF
        lsf_popt = self._fit_function(gaussian, x_lsf, lsf, p0=[50, 0, 1])
        # Normalize the ESF
        esf_norm, v_min, v_max, esf_popt, esf_norm_popt = self.normalize_esf(esf, x_esf)
        # Evaluate the ESF
        if self.evaluate_esf(lsf_popt, esf_norm_popt, esf_norm, x_esf):
            # Return the all the values that were calculated for the edge
            # return [x_esf,x_lsf, esf, esf_popt, lsf, lsf_popt, esf_norm, esf_norm_popt]

            final_esf = model_esf(x_esf, *esf_popt)
            final_lsf = np.abs(np.diff(final_esf))
            if self.get_mtf_curve:
                # self.raw_lsf_list.append([x_, lsf_])
                m_x = np.arange(-self.pixels_sampled, self.pixels_sampled + 0.1, 0.1)
                m_esf = model_esf(m_x, *esf_popt)
                m_lsf = np.abs(np.diff(m_esf))
                self.raw_lsf_list.append([m_x[:-1], m_lsf])
            return [x_esf, final_esf, final_lsf, esf_norm_popt]
        else:
            return None

    def _fit_function(
        self, func: Any, x: np.array, data: np.array, p0: List[float]
    ) -> Any:
        if type(p0) == "list":
            p0 = np.array(p0).astype(np.float64)
        try:
            popt, pcov = curve_fit(func, x, data, p0=p0)
        except Exception:
            return None
        return popt

    def normalize_esf(self, esf: np.array, x: np.array) -> Tuple[Any, ...]:
        """
        Normalizes the ESF.
        :param esf:
        :param x:
        :return:
        """
        # Calculate initial values for curve fitting
        a0 = np.median(esf)
        d0 = np.quantile(esf, 0.05)
        # Fit curve
        esf_popt = self._fit_function(model_esf, x, esf, np.array([a0, 0.5, -0.5, d0]))
        if esf_popt is None:
            return None, None, None, None, None
        try:
            v_max = np.max(model_esf(np.arange(x[-1], x[-1] + 4, 0.1), *esf_popt))
            v_min = np.min(model_esf(np.arange(x[0] - 4, x[0], 0.1), *esf_popt))
        except Exception:
            return None, None, None, None, None

        if v_min is None or v_max is None:
            return None, None, None, None, None

        if v_max - v_min == 0:
            return None, None, None, None, None
        # Normalize ESF
        esf_norm = (esf - v_min) / (v_max - v_min)
        # Fit curve to normalized ESF
        esf_norm_popt = self._fit_function(model_esf, x, esf_norm, p0=[1, 0.5, 0.5, 0])
        return (esf_norm, v_min, v_max, esf_popt, esf_norm_popt)

    def evaluate_esf(
        self,
        lsf_popt: np.array,
        esf_norm_popt: np.array,
        esf_norm: np.array,
        x: np.array,
    ) -> bool:
        """
        Evaluate the quality of the ESF by calculating the edge SNR and checking if it is above the threshold, and
        by calculating the R2 value of the curve fitting, and also checking if it is above the threshold value.
        :param lsf_popt:
        :param esf_norm_popt:
        :param esf:
        :param esf_norm:
        :param v_min:
        :param v_max:
        :param x:
        :return:
        """
        if lsf_popt is None or esf_norm_popt is None:
            return False

        # Calculate R2 of the fit
        residuals = esf_norm - model_esf(x, *esf_norm_popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((esf_norm - np.mean(esf_norm)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        if r_squared < self.r2_threshold:
            return False

        # Calculate edge SNR
        new_lsf = np.diff(model_esf(x, *esf_norm_popt))
        idx = np.argwhere(new_lsf / np.max(new_lsf) < 0.05).flatten()
        if idx.shape[0] <= 5:
            return False
        try:
            idx_ = np.argwhere(np.diff(idx) > 1).flatten()[0]
        except Exception:
            return False
        x_left = idx[idx_]
        x_right = idx[idx_ + 1]
        _left = esf_norm[:x_left]
        _right = esf_norm[x_right:]
        if _left.shape[0] == 0 or _right.shape[0] == 0:
            return False
        noise1 = np.std(_left)
        noise2 = np.std(_right)

        noise = np.nanmean([noise1, noise2])
        if noise <= 0:
            # logging.debug("noise is not positive")
            return False
        edge_snr = 1 / noise

        if edge_snr >= self.snr_threshold and r_squared >= self.r2_threshold:
            self.edge_snrs[-1].append(edge_snr)
            return True

        return False

    def calculate_rer(self, edge_list: List[np.array]) -> np.array:
        """
        Given the list of edges, calculates the RER value for each edge.
        :param edge_list:
        :return:
        """
        rers = []
        for edge in edge_list:
            x, esf, lsf, esf_norm_popt = edge
            loc_50perc = x[np.argmin(np.abs(model_esf(x, *esf_norm_popt) - 0.5))]
            rer = model_esf(loc_50perc + 0.5, *esf_norm_popt) - model_esf(
                loc_50perc - 0.5, *esf_norm_popt
            )
            rers.append(rer)
        if len(rers) < 1:
            return None
        return np.array(rers)

    def calculate_fwhm(self, edge_list: List[np.array]) -> np.array:
        """
        Given the list of edges, calculates the FWHM value for each edge.
        :param edge_list:
        :return:
        """
        fwhms = []
        for edge in edge_list:
            x, esf, lsf, esf_norm_popt = edge
            try:
                # lsf_ = np.abs(np.diff(model_esf(x_esf, *esf_popt)))
                spline = UnivariateSpline(
                    np.arange(lsf.shape[0]), lsf - np.max(lsf) / 2, s=0
                )
                r1, r2 = spline.roots()
                fwhm = np.max([r1, r2]) - np.min([r1, r2])
                fwhms.append(fwhm / 10)
            except Exception:
                continue
        if len(fwhms) < 1:
            return None
        return np.array(fwhms)

    def calculate_mtf(self, edge_list: List[np.array]) -> np.array:
        """
        Given a list of edges, calculates the MTF curve for each edge, and returns the value at
        the Nyquist and half Nyquist frequencies.
        :param edge_list:
        :return:
        """
        mtf_nyq = []
        mtf_half_nyq = []
        for edge in edge_list:
            x, esf, lsf, esf_norm_popt = edge
            try:
                N = lsf.shape[0]
                T = 0.1
                yf = fft(lsf)
                xf = fftfreq(N, T)[: N // 2]
                mtf = 2.0 / N * np.abs(yf[0 : N // 2])
                mtf = mtf / mtf[0]
                idx = np.argwhere(xf < 1.5).flatten()
                p = CubicSpline(xf[idx], mtf[idx])
                mtf_nyq.append(p(0.5))
                mtf_half_nyq.append(p(0.25))

            except Exception:
                pass
        if len(mtf_nyq) < 1:
            return None
        return np.array(mtf_nyq), np.array(mtf_half_nyq)

    def calculate_mtf_curve(self) -> Any:
        """
        Calculates the average MTF curve of the image.
        It uses a saved list of LSFs. Calculates the MTF curve for each LSF separately, then the mean and the
        standard deviation of the curves.
        :return:
        """
        x_min = np.max([x[0] for x in np.array(self.raw_lsf_list)[:, 0]])
        x_max = np.min([x[-1] for x in np.array(self.raw_lsf_list)[:, 0]])
        lsfs = []
        mtfs = []
        for x, l in self.raw_lsf_list:
            idx = np.argwhere((x >= x_min) & (x <= x_max)).flatten()
            lsf = l[idx]
            lsfs.append(lsf)
            N = lsf.shape[0]
            T = 0.1
            yf = fft(lsf)
            xf = fftfreq(N, T)[: N // 2]
            mtf = 2.0 / N * np.abs(yf[0 : N // 2])
            mtf = mtf / mtf[0]
            mtfs.append(mtf)
        mean_mtf = np.mean(np.array(mtfs), axis=0)
        std_mtf = np.std(np.array(mtfs), axis=0)
        return {"cycles_per_pixel": xf, "mtf_mean": mean_mtf, "mtf_std": std_mtf}


def sharpness_function_from_array(
    img: np.array, metrics: List[str] = ["RER", "FWHM", "MTF"], **kwargs: Any
) -> Any:
    """
    Generic function to apply either SNR algorithm for an image.
    Args:
        image: a numpy array containing your image
        metrics: A list of the metrics you wish to calculate. Available metrics: RER, FWHM, MTF.
        kwargs: the SharpnessMeasure class has many tuneable parameters. If you wish to set any of them differently from
        the default, pass them here as keyword arguments.
    """

    if img.shape[0] < 5:
        img = np.moveaxis(img, 0, -1)
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    metrics = [s.upper() for s in metrics]
    kwargs["get_rer"] = "RER" in metrics
    kwargs["get_fwhm"] = "FWHM" in metrics
    kwargs["get_mtf"] = "MTF" in metrics
    sharpness = SharpnessMeasure(**kwargs)
    result = sharpness.apply(img)
    return result


def sharpness_function_from_fn(
    image: str,
    ext: str = "tif",
    metrics: List[str] = ["RER", "FWHM", "MTF"],
    **kwargs: Any,
) -> Any:
    """
    Generic function to apply either SNR algorithm for an image.
    Args:
        image: the path your image
        ext: the extension of your image
        metrics: A list of the metrics you wish to calculate. Available metrics: RER, FWHM, MTF.
        kwargs: the SharpnessMeasure class has many tuneable parameters. If you wish to set any of them differently from
        the default, pass them here as keyword arguments.
    """
    if ext == "tif":
        with rasterio.open(image, "r") as data:
            img = data.read()
    else:
        img = cv2.imread(image)
    if img.shape[0] < 5:
        img = np.moveaxis(img, 0, -1)
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    metrics = [s.upper() for s in metrics]
    kwargs["get_rer"] = "RER" in metrics
    kwargs["get_fwhm"] = "FWHM" in metrics
    kwargs["get_mtf"] = "MTF" in metrics
    sharpness = SharpnessMeasure(**kwargs)
    result = sharpness.apply(img)
    return result


def mtf_from_array(img: np.array, **kwargs: Any) -> Any:
    """
    Generic function to construct an average MTF curve of the image.
    Args:
        image: a numpy array containing your image
        kwargs: the SharpnessMeasure class has many tuneable parameters. If you wish to set any of them differently from
        the default, pass them here as keyword arguments.
    """

    if img.shape[0] < 5:
        img = np.moveaxis(img, 0, -1)
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    kwargs["get_rer"] = False
    kwargs["get_fwhm"] = False
    kwargs["get_mtf"] = False
    kwargs["get_mtf_curve"] = True
    sharpness = SharpnessMeasure(**kwargs)
    result = sharpness.apply(img)
    return result


class SharpnessMetric(Metric):
    def __init__(
        self,
        experiment_info: Any,
        ext: str = "tif",
        metrics: List[str] = ["RER", "FWHM", "MTF"],
        parallel: bool = True,
        njobs: Optional[int] = -1,
        **kwargs: Any,
    ) -> None:
        """
        The metric to measure sharpness within an Iquaflow experiment.
        :param experiment_info:
        :param ext:
        :param metrics:
        :param parallel:
        :param njobs:
        :param kwargs:
        """

        super().__init__()  # type: ignore
        self.experiment_info = experiment_info
        self.ext = ext
        self.metrics = [s.upper() for s in metrics]
        self.metric_names = []
        for metric in metrics:
            for direction in ["X", "Y", "other"]:
                if metric == "MTF":
                    self.metric_names.append(f"{metric}_NYQ_{direction}")
                    self.metric_names.append(f"{metric}_halfNYQ_{direction}")
                else:
                    self.metric_names.append(f"{metric}_{direction}")

        kwargs["calculate_mean"] = True
        self.parallel = parallel
        self.njobs = njobs
        self.kwargs = kwargs

    def apply(self, predictions: str, gt_path: str) -> Dict[str, float]:
        # These are actually attributes from ds_wrapper
        self.data_path = os.path.dirname(gt_path)
        self.parent_folder = os.path.dirname(self.data_path)
        # predictions be like /mlruns/1/6f1b6d86e42d402aa96665e63c44ef91/artifacts'
        guessed_run_id = predictions.split(os.sep)[-3]
        modifier_subfold = [
            k
            for k in self.experiment_info.runs
            if self.experiment_info.runs[k]["run_id"] == guessed_run_id
        ][0]
        glob_crit = os.path.join(
            self.parent_folder, modifier_subfold, "*", f"*.{self.ext}"
        )
        pred_fn_lst = glob(glob_crit)
        results: Dict[str, Any] = {}
        for m in self.metric_names:
            results[m] = []
        if self.parallel:
            r = Parallel(n_jobs=self.njobs)(
                delayed(sharpness_function_from_fn)(
                    pred_fn, self.ext, self.metrics, **self.kwargs
                )
                for pred_fn in pred_fn_lst
            )
            for result in r:
                for k, v in result["mean"].items():
                    results[k].append(v[0])

        else:
            for i, pred_fn in enumerate(pred_fn_lst):
                result = sharpness_function_from_fn(
                    pred_fn, self.ext, self.metrics, **self.kwargs
                )["mean"]

                for k, v in result.items():
                    results[k].append(v[0])
        return {k: round(np.nanmean(v), 3) for k, v in results.items()}
