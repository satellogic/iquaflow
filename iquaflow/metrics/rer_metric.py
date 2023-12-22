import math
import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy import interpolate, ndimage

from iquaflow.metrics import Metric

CONVOLUTION_PATCH_EXTRACTOR = "convolution-based"
CITY_PATCH_EXTRACTOR = "city"
BAOTOU_PATCH_EXTRACTOR = "Baotou-like"


def debug_plot(image, name="default"):
    plt.imshow(
        np.array(image),
        cmap=plt.cm.gray,
        vmin=0,
        vmax=int(np.iinfo(image.dtype).max),
        interpolation="none",
    )
    plt.savefig(os.path.join(f"rer_{name}.png"))
    plt.close()


class RERfunctions:
    def __init__(
        self, sr_edge_factor: int = 6, psf_size: int = 11, gaussian_sigma: int = 2
    ) -> None:
        self.sr_edge_factor = sr_edge_factor
        self.psf_size = psf_size
        self.gaussian_sigma = gaussian_sigma

    def rer(
        self, mtf: Any, list_of_patches: List[np.array], GSD_norm: int = 1
    ) -> np.float:
        list_rer = []
        for ix in range(len(list_of_patches)):
            if not self._check_patch(list_of_patches[ix]):
                continue
            # debug_plot(list_of_patches[ix], name="patch")

            result, edgeProfiles, _, _, _, _ = mtf.superresEdge(
                list_of_patches[ix], self.sr_edge_factor
            )
            if self.gaussian_sigma != 0:
                result = ndimage.gaussian_filter(result, sigma=self.gaussian_sigma)
            if 0 in result:
                continue
            if len(result.shape) == 0:
                continue
            else:
                # first check orientation of patch
                result_min = np.mean(result[: int(len(result) // self.sr_edge_factor)])
                result_max = np.mean(result[-int(len(result) // self.sr_edge_factor) :])
                if result_max < result_min:
                    result = np.flip(result)

                # normalize and calculate edge response
                result_min = np.mean(result[: int(len(result) // self.sr_edge_factor)])
                result = result - result_min
                result_max = np.mean(result[-int(len(result) // self.sr_edge_factor) :])
                result /= result_max
                loc_50perc = np.argmin(np.abs(result - 0.5))
                loc_upper = int(loc_50perc + self.sr_edge_factor / 2)
                loc_lower = int(loc_50perc - self.sr_edge_factor / 2)
                if loc_lower < 0 or loc_upper > len(result) - 1:
                    continue
                rer_value = (result[loc_upper] - result[loc_lower]) / GSD_norm
                list_rer.append(rer_value)
        if list_rer:
            return np.median(list_rer)
        else:
            return None

    def lsf_from_edge_response(
        self, xs: Any, edge_response: Any
    ) -> Tuple[np.array, np.array]:
        lsf = (edge_response - np.roll(edge_response, 1))[1:]
        xs_lsf = xs[1:]
        xs_lsf, lsf = self.center_lsf(xs_lsf, lsf)

        return xs_lsf, lsf

    def center_lsf(self, xs: Any, lsf: Any) -> Any:
        # Center lsf
        mass_loc = int(np.round(ndimage.measurements.center_of_mass(lsf))[0])
        cutoff = min(len(lsf) - 1 - mass_loc, mass_loc)
        lsf = lsf[mass_loc - cutoff : mass_loc + cutoff]
        xs = xs[mass_loc - cutoff : mass_loc + cutoff]

        # Trim lsf
        boundary = self.psf_size * self.sr_edge_factor
        cutoff2 = min(math.floor(boundary / 2), math.floor(len(lsf) / 2))
        mass_loc = np.where(lsf == np.amax(lsf))[0][0]
        lsf = lsf[mass_loc - cutoff2 : mass_loc + cutoff2]
        xs = xs[mass_loc - cutoff2 : mass_loc + cutoff2]

    def _check_patch(self, patch: np.array) -> bool:
        """
        If patch is found and has contrast greater than
        contrast_threshold, returns patch; else
        returns None.
        """
        if patch is None:
            return False

        else:
            # Check if good edge

            # Check if angle small enough
            max_angle = 15
            angle = MTF().get_angle(patch)
            if angle > max_angle:
                # Angle too large
                return False

            # Check for gradients, inhomogeneities (in practice, we see fluctuations due to noise)
            # we assume Poisson noise dominates
            inhomogeneity_threshold = int(np.sqrt(np.max(patch)))

            # Make sure edges of patch (columnwise) are constant
            range_in_first_column = np.max(patch[:, :1]) - np.min(patch[:, :1])
            range_in_last_column = np.max(patch[:, -2:]) - np.min(patch[:, -2:])
            if (
                range_in_first_column >= inhomogeneity_threshold
                or range_in_last_column >= inhomogeneity_threshold
            ):
                # Patch rejected due to edge inhomogeneities
                return False

            # Check that DN values in columns, rows are increasing, decreasing, or staying the same
            # as opposed to ex. increasing then decreasing, implying structure other than a straight edge
            def _check_differences(col_or_row: Any) -> bool:
                arr_differences = np.asarray(
                    [
                        (float(y) - float(x))
                        for x, y in zip(col_or_row[:-1], col_or_row[1:])
                    ]
                )
                if not (
                    _all_DNs_constant(arr_differences)
                    or _DNs_increasing(arr_differences)
                    or _DNs_decreasing(arr_differences)
                ):
                    return False
                else:
                    return True

            def _all_DNs_constant(array_of_diffs: np.array) -> bool:
                return all(abs(array_of_diffs) < inhomogeneity_threshold)

            def _DNs_increasing(array_of_diffs: np.array) -> bool:
                return all(array_of_diffs > -inhomogeneity_threshold)

            def _DNs_decreasing(array_of_diffs: np.array) -> bool:
                return all(array_of_diffs < inhomogeneity_threshold)

            for col in patch.T:
                if not _check_differences(col):
                    # Patch rejected due to column inhomogeneities
                    return False

            for row in patch:
                if not _check_differences(row):
                    # Patch rejected due to row inhomogeneities
                    return False

            # Check if over-saturated
            oversaturated_threshold = int(np.iinfo(patch.dtype).max * 0.96)
            if np.any(patch >= oversaturated_threshold):
                # Patch rejected for being oversaturated
                return False

            # Check patch has good contrast
            if self._good_contrast(patch):
                # Patch passed checks
                return True
            else:
                # Patch does not have sufficient contrast
                return False

    def _good_contrast(self, patch: np.array) -> Any:
        """
        Check contrast at diagonal corners of patch
        Return True if above threshold, False if not
        """
        contrast_threshold = 1.4
        corner_length = 2
        corner1 = np.mean(patch[:corner_length, :corner_length])
        corner2 = np.mean(patch[-corner_length:, -corner_length:])
        corner3 = np.mean(patch[:corner_length, -corner_length:])
        corner4 = np.mean(patch[-corner_length:, :corner_length])
        # Change 0 to 1 to avoid NaNs
        for array in [corner1, corner2, corner3, corner4]:
            if array == 0:
                array = 1

        return (
            (corner2 / corner1) > contrast_threshold
            or (corner1 / corner2) > contrast_threshold
        ) and (
            (corner3 / corner4) > contrast_threshold
            or (corner4 / corner3) > contrast_threshold
        )


class MtfData:
    def __init__(
        self,
        patch: np.array,
        lp_per_pix: Any,
        y: Any,
        nyquist_mtf: Any,
        edges: Any,
        edges_corrected: Any,
        patch_sr: np.array,
        angle: float,
    ) -> None:
        """

        :param patch: patch
        :param lp_per_pix: x-axis of MTF
        :param y: MTF estimation
        :param nyquist_mtf: MTF at nyquist
        :param edges1: Super resolved edge
        :param edges1_corrected: SR edge * cos(angle)
        :param patch_sr: Super resolved patch
        :param angle1: edge angle
        """
        self.patch = patch
        self.lp_per_pix = lp_per_pix
        self.y = y
        self.nyquist_mtf = nyquist_mtf
        self.edges = edges
        self.edges_corrected = edges_corrected
        self.patch_sr = patch_sr
        self.angle = angle


class MTF:
    def __init__(
        self,
        debug_folder: Optional[Any] = None,
        config: Optional[Any] = None,
        satellite_name: Optional[Any] = None,
        patch_extractor_method: str = "convolution-based",
        mtf_measurements_hor: Optional[Any] = None,
        mtf_measurements_vert: Optional[Any] = None,
    ) -> None:
        self.debug_folder = debug_folder
        self.config = config
        self.satellite_name = satellite_name
        self.patch_extractor_method = patch_extractor_method
        self.mtf_measurements_hor = mtf_measurements_hor
        self.mtf_measurements_vert = mtf_measurements_vert

    def return_patches_within_angle(
        self, patches: List[Any], lower_angle: float, higher_angle: float
    ) -> List[List[Any]]:
        """
        Takes in list of horizontal and vertical patches, lower and upper angle
        limits in degrees.
        Returns patches only in range lower_angle < patch angle < higher_angle.
        """

        def good_angle(theta: float) -> Any:
            """
            Checks if theta is within range lower_angle to higher_angle.
            Capable of handling if angle is not vertical, i.e. 0 location
            rotated at any of 0, 90, 180, 270 deg
            Arg
                theta: int or float, angle of edge in patch.
            Return
                True or False, whether within range or not, respectively.
            """
            return np.any(
                (lower_angle <= np.abs(theta - 180 * np.array([0, 0.5, 1, 1.5])))
                & (np.abs(theta - 180 * np.array([0, 0.5, 1, 1.5])) <= higher_angle)
            )

        return [
            [
                patches[0][i]
                for i in range(len(patches[0]))
                if good_angle(self.get_angle(patches[0][i]))
            ],
            [
                patches[1][i]
                for i in range(len(patches[1]))
                if good_angle(self.get_angle(patches[1][i]))
            ],
        ]

    def from_poly_to_angle(self, poly: np.array) -> np.float:
        """
        Input:
            poly: ndarray polynomial coefficient from linear fit
        Returns:
            theta: angle of line in radians
        """
        theta = np.arctan(poly[-2])
        return theta

    def get_angle(self, patch: np.array) -> Any:
        """
        Returns angle of single edge in patch, in degrees.
        If no edge, multiple edges, or poor quality, returns False.
        """
        # check patch has single edge
        if not self.is_patch_with_single_edge(patch):
            # Patch rejected due to edge quality
            return False

        # compute edge position
        edgePositions = np.array(list([self.findStepEdgeSubpix(x) for x in patch]))

        if edgePositions is None:
            # Edge positions not found
            return False

        # refine position by regression of a line
        _, p = self.fit_edges_with_line(edgePositions)

        theta = self.from_poly_to_angle(p)
        # value between 0-90 degrees
        return (
            int(np.rad2deg(theta))
            if np.rad2deg(theta) < 90
            else int(np.rad2deg(theta - np.pi / 2))
        )

    def fit_edges_with_line(self, edgePositions: np.array) -> Tuple[np.array, np.array]:
        """
        refine position by regression of a line
        Arg
            edgePositions: np.array, indices of location of steep gradient
        Return
            fitPositions: ndarray or poly1d, locations where polynomial
                          crosses LSFs
            p: ndarray, shape (deg + 1,) or (deg + 1, K):
                  Polynomial coefficients, highest power first.
        """
        p = np.polyfit(np.array(range(len(edgePositions))), edgePositions, 1)
        fitPositions = np.polyval(p, np.array(range(len(edgePositions))))
        return fitPositions, p

    def is_patch_with_single_edge(self, patch: Any) -> bool:
        """
        Checks patch has single edge.

        If no edge or multiple edge, returns False.
        """
        max_value = np.amax(patch)
        # canny filter to detect edges
        canny = cv2.Canny(
            patch.astype(np.uint8), 0.25 * max_value, 0.3 * max_value, apertureSize=3
        )
        # debug_plot(canny, name="canny")
        # Hough transform to yield r, theta of lines
        rho_resolution = 1
        theta_resolution = np.pi / 180
        accumulator_threshold_votes = 5
        lines = cv2.HoughLines(
            canny, rho_resolution, theta_resolution, accumulator_threshold_votes
        )

        # False if no lines or multiple lines found
        if lines is None or lines[0].shape[0] > 1:
            return False
        else:
            return True

    def findStepEdgeSubpix(self, x: Any, r: int = 3) -> Any:
        """Find the position in x that has highest gradient."""
        edge_int = np.int32(x)  # use signed int for doing calculations
        # Subtract roll - original to get difference between value in array and it's neighbor
        # (to linear order, the gradient), take abs value and find index of max
        return np.argmax(np.abs((np.roll(edge_int, -1) - edge_int))[:-1])

    def remove_saturated(self, edge: Any, k: Any) -> Any:
        # Discard saturated profiles if the have more than k pixels
        saturated = np.where(edge == 255)
        return len(saturated[0]) > k

    def superresEdge(
        self, edgeProfiles: np.array, n: int = 8, debug: bool = False
    ) -> Tuple[np.array, np.array, Any, np.array, np.array, np.array]:
        """
        Given a bunch of edge profiles, create an average
        profile that is n times the size based on the edge positions.
        Parameters
        ----------
        edgeProfiles : 2d ndarray
            an image containing a VERTICAL edge
        n : int
            super-resolution factor
        Returns
        -------
        result: 1d ndarray
                super resolved ESF
        im: 2d ndarray
            pixel precise re-aligned edges (useful for debug)
        theBins:
                subpixel bins used by the algorithm (useful for debug)
        correction: float
            spatial sampling factor due to the slant angle (must be <1)
            MTF is scaled by dividing the frequencies by this correction
        Notes
        -----
        It is very important that the image `edgeProfiles` contains only
        the vertical edge. Any other dominant edge will affect the estimation
        """
        # remove lines with nans
        edgeProfiles = [i for i in edgeProfiles if ~np.isnan(i).any()]
        edgeProfiles = [i for i in edgeProfiles if not self.remove_saturated(i, 0)]

        edgeProfiles_copy = edgeProfiles.copy()
        # error if too few line functions in patch
        if len(edgeProfiles) < 6:
            print("Number of edge profiles less than 6")
            return None, None, None, None, None, None

        # compute edge position
        edgePositions = np.array(
            list([self.findStepEdgeSubpix(x) for x in edgeProfiles])
        )

        if edgePositions is None:
            print("Edge positions not found")

        # refine position by regression of a line
        fitPositions, p = self.fit_edges_with_line(edgePositions)

        # check angle
        angle_radians = self.from_poly_to_angle(p)
        if np.isnan(angle_radians):
            print("Error computing angle")
            return None, None, None, None, None, None
        # correction of angle
        # http://www.imatest.com/wp-content/uploads/2015/02/Slanted-Edge_MTF_Stability_Repeatability.pdf
        # The edge samples collected along a line have a spacing smaller than 1px with respect to the slanted edge.
        # This oversampling factor corresponds to a dilation in frequency domain
        # compress_factor = np.cos(angle_radians)

        meanPos = np.floor(np.mean(fitPositions))
        shifts = np.floor(fitPositions) - meanPos
        bins = np.cast[int](np.modf(fitPositions)[0] * n)
        # logger.debug("bins: ", bins)
        edgeProfiles = [
            ndimage.shift(profile, -round(shft), mode="nearest", order=0)
            for profile, shft in zip(edgeProfiles, shifts)
        ]
        toAverage: Any = [[] for i in range(n)]
        for bin_i, profile in zip(bins, edgeProfiles):
            toAverage[bin_i].append(profile)

        # initialize array of zeros, will populate below
        result = np.zeros(len(edgeProfiles[0]) * n)

        #  Check if all bins have samples
        if not all(i for i in toAverage):
            # print("At least one empty bin (super resolution factor = %d).  Skipping patch."% n )
            return None, None, None, None, None, None

        # populate array result to create super-resolved edge
        for i in range(n):
            x = np.mean(toAverage[i], axis=0)
            result[n - i - 1 :: n] = x

        return (
            result,
            np.array(edgeProfiles),
            toAverage,
            angle_radians,
            p,
            edgeProfiles_copy,
        )

    def imatest_denoising_trick(self, LSF: np.array) -> np.array:
        """
        imatest_trick: Smooths the LSF (y) away from the edge to reduce noise as described in
        http://www.imatest.com/wp-content/uploads/2015/02/Slanted-Edge_MTF_Stability_Repeatability.pdf
        This code locates the position of the maximum and creates a mask of size [-sz/6, sz/6] around it
        where sz is the lenth of the vector y. Then smooths the mask with a gaussian with std=8.
        The result is the blending mask.
        Parameters
        ----------
        y: 1d ndarray
           the input LSF
        Returns
        ----------
        smoothed LSF
        """

        gy = ndimage.gaussian_filter(LSF, 8, mode="wrap")
        sz = len(LSF)

        # detect discontinuity and remove 1/3
        mx = np.argmax(gy)  # it was mx = np.argmax(y)
        mask = np.zeros(sz)
        mask[max(0, int(mx - sz / 6)) : min(sz - 1, int(mx + sz / 6))] = 1
        mask = ndimage.gaussian_filter(mask, 12)

        return mask * LSF + (1 - mask) * gy

    def MTF(
        self,
        x: Any,
        window: bool = True,
        imatest_trick: bool = True,
        dont_differentiate: bool = False,
    ) -> Any:
        """
        Compute the MTF of an edge scan function.

        Parameters
        ----------
        x : 1D ndarray
            Edge scan function.
        window : bool
            Whether to apply Hanning windowing to the input.
        imatest_trick : bool
            Smooth the LSF away from the edge to reduce noise as described in
            http://www.imatest.com/wp-content/uploads/2015/02/Slanted-Edge_MTF_Stability_Repeatability.pdf
        dont_differentiate : book
            Skips the differentiation

        Returns
        -------

        Notes
        -----
        The line spread function is the derivative of the edge scan function.  The
        FFT of the line spread function gives the MTF.

        See Also
        --------
        http://www.cis.rit.edu/research/thesis/bs/2001/perry/thesis.html
        """
        if dont_differentiate:
            y = x
        else:
            y = np.diff(x)
        # imatest denoising trick
        if imatest_trick:
            y = self.imatest_denoising_trick(y)
        if window:
            y = y * np.hanning(len(y))
        Y = np.fft.fft(y)
        return Y[: len(Y) // 2]

    def draw_estimated_line(self, x, angle, p, name):
        """
        Draw a red line with estimated edge in patch for debug
        """
        img = np.zeros((len(x), len(x[0]), 3))
        for i in range(len(x)):
            for j in range(len(x[0])):
                img[i, j, :] = x[i][j]
        start = (int(p[1]), 0)
        if angle > 0:
            end = (int(int(p[1]) + np.sin(np.deg2rad(angle)) * x.shape[0]), x.shape[0])
        else:
            end = (int(int(p[1]) + np.sin(np.deg2rad(angle)) * x.shape[0]), x.shape[0])
        cv2.line(img, end, start, (0, 00, 255))
        cv2.imwrite(name + ".png", img)
        return

    def estimateMTF_from_patch(self, x: Any, n: Any, figure_label: Any) -> Any:
        """
        wrapper for superresEdge and MTF.
        It validates if the ESF sampling is dense enough for the estimation,
        scales the MTF to account for the slant angle, and resamples the MTF
        """
        debug = True if self.debug_folder else False

        # compute the super-resolved edge
        edges, im, theBins, correction, p, edgeProfiles_copy = self.superresEdge(
            x, n=n, debug=debug
        )

        if edges is None:
            return None, None, None, None, None, None

        # recover the angle from the compression factor
        angle = np.rad2deg(correction)
        correction = np.cos(correction)
        x_edges = np.zeros((len(edgeProfiles_copy), len(edgeProfiles_copy[0]), 3))
        for i in range(len(edgeProfiles_copy)):
            for j in range(len(edgeProfiles_copy[0])):
                x_edges[i, j, :] = edgeProfiles_copy[i][j]

        # detect failure as unused bins
        binOccupancy = [len(b) for b in theBins]
        # if debug:
        #     logger.debug('the subpixel bins have %s elements each' % (binOccupancy))
        if min(binOccupancy) <= 0:
            if debug:
                print(
                    "FAILURE: the subpixel bins should have at least one sample bins: %s\n"
                    % (binOccupancy)
                    + "the angle of the edge (%d) is too small " % angle
                    + "and/or the edge lenght (%d) is too short" % im.shape[0]
                )
            else:
                import sys

                sys.stdout.write("X")

        # compute the MTF profile
        Y = self.MTF(edges)
        Y = Y / Y[0]

        # adjust the samping step accounting for the angle correction
        lpPerPix = (n * np.arange(len(Y)) / (2 * len(Y))) / correction

        # resample the MTF and accumulate in average
        # frequency samples for the resampling
        interpX = np.linspace(0, 1.5, 151)
        interpY = interpolate.interp1d(lpPerPix, np.abs(Y))(interpX)
        # sample the MTF at Nyquist
        NyquistMTF = interpolate.interp1d(lpPerPix, np.abs(Y))(0.5)

        print(("Successfully calculated MTF from patch"))

        return interpX, interpY, NyquistMTF, edges, im, angle

    def resample_edge_with_correction(
        self, edge: Any, angle_degrees: Any, figure_label: Any, dx: int = 0
    ) -> Any:
        """
        resamples a 1d signal applying an affine correction given by
           samples' = samples * correction  + dx
        where: correction = np.cos(angle_degrees/180*np.pi)
        retunrs the re-interpolated edge

        """
        N = len(edge)
        correction = np.cos(angle_degrees / 180 * np.pi)
        sampl = np.array(list(range(N))) * correction

        M = int(np.floor((N - 1) * correction))
        interpX = np.array(list(range(M))) - dx

        def interp(sampl: Any, minS: Any, maxS: Any, edge: Any, interpX: Any) -> Any:
            """
            local function that wraps scipy.interpolate.interp1d
            to handle boundary conditions when the requested samples (interpX)
            are outside the range of the input samples (sampl)

            Parameters
            ==========
            sampl: are the poistions of the input samples
            minS, maxS: are the min and max values of the samples in sampl.
            edge: are the values corresponding to sampl
            interpX: are the positions of the new samples

            Returns
            =======
            interpY: the values of the edge sampled at interpX
            """
            # mirror at the boundary
            interpX = [x if x <= maxS else 2 * maxS - x for x in interpX]
            interpX = [x if x >= minS else 2 * minS - x for x in interpX]

            # constant at the boundary
            # interpX = np.minimum(interpX,maxS)
            # interpX = np.maximum(interpX,minS)
            return interpolate.interp1d(sampl, edge)(interpX)

        # interpY = interpolate.interp1d(sampl, edge)(interpX)
        interpY = interp(sampl, 0, (N - 1) * correction, edge, interpX)

        return interpY

    def align_edges_subpix(
        self,
        edge1: Any,
        edge2: Any,
        angle2: Any,
        USE_IMATEST_TRICK: Any,
        figure_label: Any,
    ) -> Any:
        """
        Computes the displacement dx that aligns edge1 and edge2: two 1d ESF (edge spread functions).
        The algorithm computes the argmax correlation (with fourier) of derivaties of the signals (LSF).
        To reduce noise in the correlation a small gaussian filtering is applied to the LSF before correlation.
        The two LSF are also normalized before correlation.
        """

        sz = min(edge1.shape[0], edge2.shape[0])
        d1 = np.array(edge1)
        d2 = np.array(edge2)

        d1 = np.diff(d1)
        d2 = np.diff(d2)

        d1 = ndimage.gaussian_filter(d1, 1, mode="mirror")
        d2 = ndimage.gaussian_filter(d2, 1, mode="mirror")

        d1 = d1 / np.mean(d1)
        d2 = d2 / np.mean(d2)

        sz = min(d1.shape[0], d2.shape[0])
        d1 = d1[:sz]
        d2 = d2[:sz]

        dx = self.max_correlation(d1, d2, debug=False)

        return dx

    def max_correlation(
        self, a: np.array, b: np.array, debug: bool = False, phase: bool = False
    ) -> np.float:
        """
        computes the phase-correlation of a and b and uses it
        to compute and return (dx,dy) the subpixel shift between a and b
        debug (bool) shows the correlatio function
        if phase=True: uses phase-correlation instead of cross-correlation
        """

        sz = a.shape

        corr = np.abs(self.correlation(a, b, phase))
        corr = np.fft.ifftshift(
            ndimage.filters.gaussian_filter(np.fft.fftshift(corr), 1)
        )

        # position of the maximum
        mx = np.argmax(corr.flatten())
        # subpixel refinement of the maximum
        dx = self.parabola_refine(corr[(mx + np.arange(-1, 2)) % sz[0]])

        if mx > sz[0] / 2:
            mx = mx - sz[0]

        return mx + dx

    def correlation(self, a: np.array, b: np.array, phase: bool = False) -> np.array:
        """
        returns the COMPLEX cross-correlation (or phase-correlation) of a and b
          if phase=True computes :  F^{-1} (F(a) * F(b)^* )
          else :  F^{-1} (F(a) * F(b)^* / |F(a) * F(b)^*| )
        ATTENTION: it must return a complex valued image
        """

        if not (a.shape == b.shape):
            print("ERROR: images not the same size")
            return 0, 0

        fa = np.fft.fft(a)
        fb = np.fft.fft(b)

        corrF = fa * np.conj(fb)

        if phase:
            corrF = corrF / np.abs(corrF)
        pc = np.fft.ifft(corrF)
        return pc

    def parabola_refine(self, c: np.array) -> np.float:
        """
        maximum of parabolic interpolation
        interpolate a parabola through the points
        (-1,c[0]), (0,c[1]), (1,c[2])
        assuming that c[1]  >  c[0], c[2]

        I. E. Abdou,
        "Practical approach to the registration of multiple frames of video images"
        Proceedings of Visual Communications and Image Processing '99; (1998);
        doi: 10.1117/12.334685
        """
        return (c[2] - c[0]) / (2 * (2 * c[1] - c[0] - c[2]))

    def compute_aggregated_mtf_from_multiple_edges(
        self,
        edge_images_array: Any,
        figure_label: Any,
        USE_IMATEST_TRICK: bool = True,
        super_resolution_factor: int = 8,
    ) -> Any:
        """
        this function receives a list of slanted edges (vertically aligned) and produces the MTF

        Args:
            edge_images_array: list of numpy arrays containing the edges
            USE_IMATEST_TRICK=True: bool    use the imatest denoising trick
            display=True     : display intermediate results
            super_resolved=6 : change the super-resolution factor (minimum 6 because of the
                               resampling needed to aling the LSFs)

        Returns:
            return the mtf profile as two lists (interpX, interpY), in cycles per pixel
            use the functions display_mtf and  display_mtf_cycles_per_mm  to display the MTFs
        """
        # GENERATE THE INPUTS ESFs FOR EACH EDGE
        mtf_data = self.get_MTF_data_from_patches(
            edge_images_array, super_resolution_factor, figure_label
        )

        if mtf_data == []:
            print("Warning: Estimation based on a single sample")
            return None, None

        # Select reference edge (the first one), merge the edge's derivative instead of the edge
        edges1_corrected = mtf_data[0].edges_corrected
        edges_sum = np.diff(edges1_corrected)
        edges_sum /= np.mean(edges_sum)  # this also flips the edge
        if USE_IMATEST_TRICK:
            edges_sum = self.imatest_denoising_trick(edges_sum)
        edges_sum /= np.mean(edges_sum)  # this also flips the edge

        if len(mtf_data) == 1:
            edges2 = mtf_data[0].edges
            interpX, interpY = self.extract_MTF_from_single_edge(
                edges2, USE_IMATEST_TRICK, super_resolution_factor, figure_label
            )
            return interpX, interpY

        # PROCESS AND ACCUMULATE EACH EDGE
        for data in mtf_data[1:]:
            edges2 = data.edges

            edges2_corrected = data.edges_corrected
            angle2 = data.angle

            edges_sum, interpX, interpY = self.extract_MTF_from_many_edges(
                edges1_corrected,
                edges2_corrected,
                edges2,
                angle2,
                USE_IMATEST_TRICK,
                figure_label,
                edges_sum,
                super_resolution_factor,
            )

        return interpX, interpY

    def get_MTF_data_from_patches(
        self, edge_images_array: Any, super_resolution_factor: Any, figure_label: Any
    ) -> Any:
        mtf_data = []
        for i in range(len(edge_images_array)):  # for x in edge_images_array:
            patch = edge_images_array[i]
            try:
                figure_label_rect = figure_label + "_" + str(i) + "_"
                (
                    lpPerPix,
                    Y,
                    NyquistMTF,
                    edges,
                    patch_sr,
                    angle,
                ) = self.estimateMTF_from_patch(
                    patch, super_resolution_factor, figure_label_rect
                )

                if lpPerPix is None:
                    continue
            except TypeError:
                print("ERROR PROCESSING ONE PATCH. SKIPPING")
                continue

            # The edges are resampled to compensate for the compression factor due to the angle
            edges1_corrected = self.resample_edge_with_correction(
                edges, angle, figure_label
            )

            if abs(angle) > 1:
                mtf_data.append(
                    MtfData(
                        patch,
                        lpPerPix,
                        Y,
                        NyquistMTF,
                        edges,
                        edges1_corrected,
                        patch_sr,
                        angle,
                    )
                )
            else:
                print("Angle too small")

        return mtf_data

    def extract_MTF_from_single_edge(
        self,
        edges2: Any,
        USE_IMATEST_TRICK: Any,
        super_resolution_factor: Any,
        figure_label: Any,
    ) -> Any:
        dedges2 = np.diff(edges2)

        if USE_IMATEST_TRICK:
            dedges2 = self.imatest_denoising_trick(dedges2)
        dedges2 = dedges2 / np.mean(dedges2)  # this also flips the edge

        Y = self.MTF(dedges2, window=True, imatest_trick=False, dont_differentiate=True)
        Y = Y / Y[0]
        lpPerPix = super_resolution_factor * np.arange(len(Y)) / (2 * len(Y))
        interpX = np.linspace(0, 1.5, 151)
        interpY = interpolate.interp1d(lpPerPix, np.abs(Y))(interpX)

        return interpX, interpY

    def extract_MTF_from_many_edges(
        self,
        edges1_corrected: Any,
        edges2_corrected: Any,
        edges2: Any,
        angle2: Any,
        USE_IMATEST_TRICK: Any,
        figure_label: Any,
        edges_sum: Any,
        super_resolution_factor: Any,
    ) -> Any:
        # Align the edge to the reference with subpixel precision.
        dx2 = self.align_edges_subpix(
            edges1_corrected, edges2_corrected, angle2, USE_IMATEST_TRICK, figure_label
        )

        # The resulting transformation: scaling and shift is then
        # applied to resample once more the original signal.
        dedges2 = np.diff(edges2)
        dedges2 = dedges2 / np.mean(dedges2)  # this also flips the edge

        if USE_IMATEST_TRICK:
            dedges2 = self.imatest_denoising_trick(dedges2)
        dedges2 = dedges2 / np.mean(dedges2)  # this also flips the edge
        edges2_corrected_translated = self.resample_edge_with_correction(
            dedges2, angle2, figure_label, dx2
        )
        # this is the LSF

        # Then the different ALIGNED esf are averaged (trimming ends if needed)
        sz = min(edges_sum.shape[0], edges2_corrected_translated.shape[0])
        edges_sum = edges_sum[:sz] + edges2_corrected_translated[:sz]

        # compute the MTF profile for each individual contribution
        Y = self.MTF(
            edges2_corrected_translated,
            window=True,
            imatest_trick=False,
            dont_differentiate=True,
        )
        Y = Y / Y[0]
        interpX, interpY = self.scale_mtf(Y, super_resolution_factor)

        # compute the MTF profile for the aggregated curve

        Y = self.MTF(
            edges_sum, window=True, imatest_trick=True, dont_differentiate=True
        )
        Y = Y / Y[0]
        interpX, interpY = self.scale_mtf(Y, super_resolution_factor)

        return edges_sum, interpX, interpY

    def scale_mtf(
        self, Y: Any, super_resolution_factor: Any
    ) -> Tuple[np.array, np.array]:
        "Re scale MTF due to spectrum compression while super resolving"
        interpX = np.linspace(0, 1.5, 151)
        lpPerPix = super_resolution_factor * np.arange(len(Y)) / (2 * len(Y))

        # frequency samples for the resampling
        interpY = interpolate.interp1d(lpPerPix, np.abs(Y))(interpX)

        return interpX, interpY


class RERMetric(Metric):
    def __init__(
        self,
        experiment_info: Any,
        win: int = 16,
        stride: int = 16,
        ext: str = "png",
        n_jobs: int = 1,
    ) -> None:
        self.experiment_info = experiment_info
        self.win = win
        self.stride = stride
        self.metric_names = ["rer"]
        self.ext = ext
        self.n_jobs = n_jobs

    def _rer_metric(self, img: np.array, win: int, stride: int) -> np.float:
        self._check_uint8(img)

        mtf = MTF()
        iqf_funcs = RERfunctions(sr_edge_factor=4)

        patch_lst = []
        for i in range(0, img.shape[0] - win, stride):
            for j in range(0, img.shape[1] - win, stride):
                patch_lst.append(img[i : i + win, j : j + win])

        rer = iqf_funcs.rer(mtf, patch_lst)

        return rer

    def _parallel_rer(self, pred_fn: str) -> Tuple[Optional[int], Any]:
        pred = cv2.imread(pred_fn)

        if len(pred.shape) == 3:
            for i in range(pred.shape[-1]):
                r = self._rer_metric(pred[..., i], self.win, self.stride)
                if r:
                    return (i, r)
        else:
            r = self._rer_metric(pred, self.win, self.stride)
            if not np.isnan(r):
                return (0, r)
        return (None, 0.0)

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

        if len(pred_fn_lst) == 0:
            print(f"Warning, no results in: {glob_crit}")
            return {"rer": -999}

        sh = cv2.imread(pred_fn_lst[0]).shape

        results_lst = Parallel(n_jobs=self.n_jobs, verbose=30)(
            delayed(self._parallel_rer)(pred_fn) for pred_fn in pred_fn_lst
        )
        # unwrap results
        if any([result[0] is not None for result in results_lst]):
            stats_dict: Dict[int, List[Any]] = {0: [], 1: [], 2: []}
            for i, r in results_lst:
                if i is not None:
                    stats_dict[i].append(r)

            if len(sh) == 3:
                stats = {f"rer_{i}": np.median(stats_dict[i]) for i in range(sh[-1])}
            else:
                stats = {"rer": np.median(stats_dict[0])}

            self.metric_names = [k for k in stats]

        else:
            # all results returned were None
            stats = {"rer": -999}

        return stats

    def _check_uint8(self, image: np.array) -> None:
        if image.dtype != np.uint8:
            raise TypeError("Expecting np.uint8, received {}".format(image.dtype))
