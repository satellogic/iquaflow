############## IMPORTANT ##############
# temporary, does not work yet!


import numpy as np
import cv2
import math
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, UnivariateSpline, interp1d
from scipy import ndimage
from scipy.fft import fft, fftfreq
import os
import matplotlib.pyplot as plt
from medpy.filter.smoothing import anisotropic_diffusion
from skimage import feature
from skimage.transform import probabilistic_hough_line
import time
from joblib import Parallel, delayed
DEBUG_DIR = "/home/kati/projects/iqf/iquaflow/iquaflow/metrics/rer_debug"
import logging
from skimage.util import view_as_windows
# import os
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_DYNAMIC'] = 'FALSE'

def f(x, a, b, c, d):
    try:
        return a / (1 + np.exp((x - b) / c)) + d
    except FloatingPointError:
        pass

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

#np.seterr(all='raise', over='ignore')

class RERMetric:
    def __init__(self,
                 window_size,
                 stride = None,
                 edge_length=5,
                 alpha=1.3,
                 beta=1.0,
                 gamma=1.0,
                 pixels_sampled=7,
                 r2_threshold=0.995,
                 snr_threshold=50,
                 log_level='error',
                 debug_suffix=None,
                 parallel = True,
                 njobs=-1):
        try:
            self.log_level = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR, 'crititcal': logging.CRITICAL}[log_level]
        except:
            self.log_level = logging.ERROR
        self.debug_suffix = debug_suffix
        self.window_size = window_size,
        self.stride = self.window_size if stride is None else stride
        self.edge_length = edge_length
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        assert pixels_sampled >= 5
        self.pixels_sampled = pixels_sampled
        self.r2_threshold = r2_threshold
        self.snr_threshold = snr_threshold
        self.patch_list = []
        self.parallel  = parallel
        self.njobs = njobs

        logging.basicConfig(format='%(levelname)s:%(message)s', level=self.log_level)
        logging.getLogger().setLevel(self.log_level)

    def apply(self, image):
        results = {}
        image = np.where(image == 0, np.nan, image)
        # if the image has multiple channels, each channel is evaluated individually
        if len(image.shape) > 2:
            if self.parallel:
                r = Parallel(n_jobs=self.njobs)(delayed(self.apply_to_one_channel)(image[:, :, i]) for i in range(image.shape[2]))
                for i, r_ in enumerate(r):
                    results[f"channel {i}"] = r_
            else:
                for i in range(image.shape[2]):
                    results[f"channel {i}"] =  self.apply_to_one_channel(image[:, :, i])
        else:
            results["channel 0"] = self.apply_to_one_channel(image)
        return results

    def apply_to_one_channel(self, image, r2_threshold=0.995, snr_threshold=50):
        out = {'rer': {}, 'fwhm': {}, 'mtf': {}}
        rers = {'vertical': np.array([]), 'horizontal': np.array([]), 'other': np.array([])}
        fwhms = {'vertical': np.array([]), 'horizontal': np.array([]), 'other': np.array([])}
        mtfs_nyq = {'vertical': np.array([]), 'horizontal': np.array([]), 'other': np.array([])}
        mtfs_half_nyq = {'vertical': np.array([]), 'horizontal': np.array([]), 'other': np.array([])}
        lines = np.empty(shape=[0, 4])
        final_good_lines = {}

        for i in range(0, image.shape[0], self.stride[0]):
            for j in range(0, image.shape[1], self.stride[0]):
                image_window = image[i:min(i+self.window_size[0], image.shape[0]), j:min(j+self.window_size[0], image.shape[1])]
                if np.isnan(image_window).sum()>=0.8*image_window.shape[0]*image_window.shape[1]:
                    #print(np.isnan(image_window).sum())
                    continue
                lines_ = self.edge_detector(image_window, min_line_length=self.edge_length)
                if lines_ is None:
                    continue
                lines_[:,0::2] += j
                lines_[:,1::2] += i
                lines = np.concatenate([lines, lines_], axis=0)

        t1 = time.time()
        vertical, horizontal, other = self.sort_angles(lines)
        logging.debug(f"sort angles {time.time() - t1}")
        t1 = time.time()
        vertical_patches, horizontal_patches, other_patches, good_lines = self.find_good_patches(image, vertical,
                                                                                                 horizontal, other)
        logging.debug(f"find good patches {time.time() - t1}")


        horizontal_patches = [np.rot90(p) for p in horizontal_patches]
        for i, (patches, kind) in enumerate(zip([vertical_patches, horizontal_patches, other_patches], ['vertical', 'horizontal', 'other'])):

            logging.debug("---------")
            logging.debug(kind)
            t1 = time.time()
            edge_list, final_good_lines[kind] = self.get_edge_list(patches, good_lines[i])
            logging.debug(len(edge_list))
            t2 = time.time()
            logging.debug(f"get edge list {len(patches)} {kind} {t2-t1}")

            rer = self.calculate_rer(edge_list)

            if rer is not None:
                rers[kind] = np.concatenate([rers[kind], rer])

            fwhm = self.calculate_fwhm(edge_list)
            if fwhm is not None:
                fwhms[kind] = np.concatenate([fwhms[kind], fwhm])
            mtf = self.calculate_mtf(edge_list)
            if mtf is not None:
                mtfs_nyq[kind] = np.concatenate([mtfs_nyq[kind], mtf[0]])
                mtfs_half_nyq[kind] = np.concatenate([mtfs_half_nyq[kind], mtf[1]])
        for kind in ['vertical', 'horizontal', 'other']:
            out['rer'][kind] = (np.mean(rers[kind]), np.std(rers[kind]), np.size(rers[kind]))
            out['fwhm'][kind] = (np.mean(fwhms[kind]), np.std(fwhms[kind]), np.size(fwhms[kind]))
            out['mtf'][kind] = {'nyq': (np.mean(mtfs_nyq[kind]), np.std(mtfs_nyq[kind]), np.size(mtfs_nyq[kind])),
                                'half_nyq': (np.mean(mtfs_half_nyq[kind]), np.std(mtfs_half_nyq[kind]), np.size(mtfs_half_nyq[kind]))}
        if self.log_level==logging.INFO or self.log_level==logging.DEBUG:
            self._plot_good_patches(image, vertical_patches, horizontal_patches, other_patches, lines, good_lines, final_good_lines)
        return out

    def edge_detector(self, image, threshold=15, min_line_length=5, line_gap=0):
        """
        Runs the canny edge detector to find edges, then uses the Hough transform to find straight lines with the
        given parameters.
        Args:
            image: the input image
            threshold:
            min_line_length: the minimum length of the line in pixels
            line_gap: the line gap
        """
        canny = feature.canny(image, sigma=1, low_threshold=np.nanquantile(image, 0.99) * 0.1,
                              high_threshold=np.nanquantile(image, 0.99) * 0.25)
        # List of lines identified, lines in format ((x0, y0), (x1, y1)), indicating line start and end
        lines = probabilistic_hough_line(canny, threshold=threshold,
                                         line_length=min_line_length,
                                         line_gap=line_gap, seed=42, theta=np.linspace(-np.pi / 2, np.pi / 2, 360*1, endpoint=False))
        # reformatting to list of lists
        lines = [[p1[0], p1[1], p2[0], p2[1]] for p1, p2 in lines]
        lines = np.array(lines)
        if lines.shape[0] == 0:
            return None
       # return lines
        lens = np.linalg.norm([(lines[:, 0] - lines[:, 2]), (lines[:, 1] - lines[:, 3])], axis=0)
        lines_g = lines[lens == min_line_length]
        lines_ = lines[lens >  min_line_length]
        lens = lens[lens >  min_line_length]
        l = lens
        while True:

            v = np.array([(lines_[:, 2] - lines_[:, 0]), (lines_[:, 3] - lines_[:, 1])])
            vv = v / lens *  min_line_length
            new_p = (np.sign(vv) * np.ceil(np.abs(vv))).T + lines_[:, :2]
            lines_g1 = np.concatenate([lines_[:, :2], new_p], axis=1)
            lines_ = np.concatenate([new_p, lines_[:, 2:]], axis=1)
            lens = np.linalg.norm([(lines_[:, 0] - lines_[:, 2]), (lines_[:, 1] - lines_[:, 3])], axis=0)
            lines_g = np.concatenate([lines_g, lines_g1, lines_[(lens ==  min_line_length)]], axis=0)
          #  print(lines_g.shape)
            lines_ = lines_[lens >  min_line_length]
            lens = lens[lens >  min_line_length]
            if lines_.shape[0] == 0:
                break
        return lines_g

    def sort_angles(self, lines):
        """
        Calculates the angle of the each of the lines relative to vertical.
        Sorts the lines into two groups:
            vertical (within +-15 degrees from vertical),
            horizontal (within +-15 degrees from horizontal),
            other.
        Args:
            lines: list of lines
        Returns:
            3 list of lines, each with [x1,x2,y1,y2,theta]
        """
        horizontal = []
        vertical = []
        other = []

        for l in lines:
            y1, x1, y2, x2 = l
            theta = np.rad2deg(np.arctan2(y2 - y1, x2 - x1)) % 180

            coords = np.array([x1, x2, y1, y2, theta])

            if (15 >= theta) or (180 - 15 <= theta):
                vertical.append(coords)
            elif 90 - 15 <= theta <= 90 + 15:
                horizontal.append(coords)
            else:
                other.append(coords)
        return vertical, horizontal, other

    def find_good_patches(self, image, vertical, horizontal, other):

        vertical_patches = []
        horizontal_patches = []
        other_patches = []
        good_lines = []
        good_lines_vertical = []
        good_lines_horizontal = []
        good_lines_other = []
        for lines, kind in zip([vertical, horizontal, other], ['v', 'h', 'o']):
            for i, l in enumerate(lines):
                x1, x2, y1, y2, theta = l

                x1, x2 = int(min(x1, x2)), int(max(x1, x2))
                y1, y2 = int(min(y1, y2)), int(max(y1, y2))

                # make sure the line is not too close to the image edge
                if x1 < 2*self.pixels_sampled or \
                        y1 < 2*self.pixels_sampled or \
                        x2 > image.shape[0] - 2*self.pixels_sampled or \
                        y2 > image.shape[1] - 2*self.pixels_sampled:
                    continue

                if kind == 'v':
                    # for vertical lines the patch size is (line length + 6, 2*self.pixels_sampled)
                    patch = image[x1 - 3:x1 + self.edge_length + 4, y1 - self.pixels_sampled:y1  + self.pixels_sampled+1]
                    # sample the dark and bright sides of the edge
                    DN1 = patch[3:-3, :self.pixels_sampled - 3]
                    DN2 = patch[3:-3, -(self.pixels_sampled - 3):]

                elif kind == 'h':
                    # for horizontal lines the patch size is (2*self.pixels_sampled, line length + 6)
                    patch = image[x1 - self.pixels_sampled:x1  + self.pixels_sampled+1, y1 - 3:y1 + self.edge_length + 4]
                    # sample the dark and bright sides of the edge
                    DN1 = patch[:self.pixels_sampled - 3, 3:-3]
                    DN2 = patch[-(self.pixels_sampled - 3):, 3:-3]
                else:
                    # for other angles, first a patch of (4*self.pixels_sampled, 4*self.pixels_sampled) is cut
                    # than the patch is rotated by the angle of theta to make it vertical
                    p = image[x1 - 2 * self.pixels_sampled:x2 + 2 * self.pixels_sampled+1, y1 - 2 * self.pixels_sampled:y2 + 2 * self.pixels_sampled+1]
                    p_rot = ndimage.rotate(p, -theta)
                    _norm = np.linalg.norm([x2 - x1, y2 - y1])
                    patch = p_rot[p_rot.shape[0] // 2 - int(_norm/2) - 3:p_rot.shape[0] // 2 + int(_norm/2) + 4,
                            p_rot.shape[1] // 2 - self.pixels_sampled:p_rot.shape[1] // 2 + self.pixels_sampled+1]
                    DN1 = patch[3:-3, :self.pixels_sampled - 3]
                    DN2 = patch[3:-3, -(self.pixels_sampled - 3):]

                if DN1.shape[0] == 0 or DN2.shape[0] == 0 or np.isnan(DN1).sum()>0 or  np.isnan(DN2).sum()>1:
                    continue

                if np.mean(DN1) > np.mean(DN2):
                    DNb = DN1
                    DNd = DN2
                else:
                    DNb = DN2
                    DNd = DN1



                if np.mean(DNb) / np.mean(DNd) > self.alpha and np.std(DNb) / np.std(patch) < self.beta and np.std(DNd) / np.std(patch) < self.beta and np.quantile(DNb, 0.1) / np.quantile(DNd, 0.9) > self.gamma:
                    x1, x2, y1, y2, theta = l
                    if kind == 'v':
                        vertical_patches.append(patch)
                        good_lines_vertical.append((x1, x2, y1, y2, theta))
                    elif kind == 'h':
                        horizontal_patches.append(patch)
                        good_lines_horizontal.append((x1, x2, y1, y2, theta))
                    else:
                        other_patches.append(patch)
                        good_lines_other.append((x1, x2, y1, y2, theta))

                    good_lines.append((x1, x2, y1, y2, theta))

        return vertical_patches, horizontal_patches, other_patches, (good_lines_vertical, good_lines_horizontal, good_lines_other, good_lines)

    def _plot_good_patches(self, image, v_patches, h_patches, o_patches, lines, good_lines, final_good_lines):
        fig, ax = plt.subplots(figsize=(image.shape[0] // 20, image.shape[1] // 20))
        ax.imshow(image)
        for l in lines:
            x1, y1, x2, y2 = l
            ax.plot([x1, x2], [y1, y2], color='black', linewidth=5)
        for l in good_lines[-1]:
            x1, x2, y1, y2, _ = l
            ax.plot([y1, y2], [x1, x2], color='red', linewidth=3)
        for k,v in final_good_lines.items():
            for l in v:
                x1, x2, y1, y2, _ = l
                ax.plot([y1, y2], [x1, x2], color='green', linewidth=3)
        fn = "good_lines.png"
        if self.debug_suffix:
            fn = fn.split('.')[0] + self.debug_suffix + '.png'
        fig.savefig(os.path.join(DEBUG_DIR, fn))
        # plt.show()
        plt.clf()

        # tot = len(v_patches)
        # cols = 5
        # rows = tot // cols
        # rows += tot % cols
        # position = range(1, tot + 1)
        # fig = plt.figure(1, figsize=(2000,2000))
        # for j in range(tot):
        #     ax = fig.add_subplot(rows, cols, position[j])
        #     ax.imshow(v_patches[j])
        # fn = "vertical_edges.png"
        # if self.debug_suffix:
        #     fn = fn.split('.')[0] + self.debug_suffix + 'png'
        # fig.savefig(os.path.join(DEBUG_DIR, fn), bbox_inches='tight',dpi=100)
        # plt.show()
        # plt.clf()
        #
        # tot = len(h_patches)
        # cols = 5
        # rows = tot // cols
        # rows += tot % cols
        # position = range(1, tot + 1)
        # fig = plt.figure(1, figsize=(2000,2000))
        # for j in range(tot):
        #     ax = fig.add_subplot(rows, cols, position[j])
        #     ax.imshow(h_patches[j])
        # fn = 'horizontal_edges.png'
        # if self.debug_suffix:
        #     fn = fn.split('.')[0] + self.debug_suffix + 'png'
        # fig.savefig(os.path.join(DEBUG_DIR, fn), bbox_inches='tight',dpi=100)
        # plt.show()
        # plt.clf()
        #
        # tot = len(o_patches)
        # cols = 5
        # rows = tot // cols
        # rows += tot % cols
        # position = range(1, tot + 1)
        # fig = plt.figure(1, figsize=(200,200))
        # for j in range(tot):
        #     ax = fig.add_subplot(rows, cols, position[j])
        #     ax.imshow(o_patches[j])
        # fn = 'other_edges.png'
        # if self.debug_suffix:
        #     fn = fn.split('.')[0] + self.debug_suffix + 'png'
        # fig.savefig(os.path.join(DEBUG_DIR, fn))
        # plt.show()
        # plt.clf()

    def get_edge_list(self, patch_list, lines):
        edge_list = []
        if self.parallel:
            edge_list = Parallel(n_jobs=self.njobs)(delayed(self._get_edge)(patch) for patch in patch_list)

        else:
            for patch in patch_list:
                _esf = self._get_edge(patch)
                edge_list.append(_esf)

        return [x for x in edge_list if x is not None], [y for x,y in zip(edge_list, lines) if x is not None]

    def _get_edge(self, patch):
        edge_coeffs = self.fit_subpixel_edge(patch)
        _esf = self.compute_esf(patch, edge_coeffs)
        if _esf is not None:
            if self.log_level==logging.DEBUG:
                self.patch_list.append(patch)
            return _esf


    def calculate_rer(self, edge_list):
        rers = []
        x = np.arange(-4, 4, 0.1)
        for edge in edge_list:
            x, esf, esf_popt, lsf, lsf_popt, esf_norm, esf_norm_popt = edge
            loc_50perc = x[np.argmin(np.abs(f(x, *esf_norm_popt) - 0.5))]
            rer = f(loc_50perc+0.5, *esf_norm_popt) - f(loc_50perc-0.5, *esf_norm_popt)
            rers.append(rer)
           # print(rer)
        if len(rers) < 1:
            return None
        logging.info(f"RERS: {rers}")
        logging.info(f"{np.mean(rers), np.std(rers), np.median(rers), np.min(rers), np.max(rers), len(rers)}")
        return np.array(rers)
        # return np.mean(rers), np.std(rers), np.median(rers), len(rers), np.quantile(rers, 0), np.quantile(rers, 0.1), np.quantile(rers, 0.9), np.quantile(rers, 1.)

    def calculate_fwhm(self, edge_list):
        fwhms = []
        for edge in edge_list:
            x, esf, esf_popt, lsf, lsf_popt, esf_norm, esf_norm_popt = edge
            lsf = self._smooth_lsf(x, lsf)

            try:
                spline = UnivariateSpline(np.arange(lsf.shape[0]), lsf - np.max(lsf) / 2, s=0)
                r1, r2 = spline.roots()
                fwhm = np.max([r1, r2]) - np.min([r1, r2])

                fwhms.append(fwhm/10)
               # print(fwhm/10)
            except:
                pass
        if len(fwhms)<1:
            return None
        logging.info(f"FWHMs: {fwhms}")
        logging.info(f"{np.mean(fwhms), np.std(fwhms), np.median(fwhms), np.min(fwhms), np.max(fwhms), len(fwhms)}")
        return np.array(fwhms)
        #return np.mean(fwhms), np.std(fwhms), np.median(fwhms), len(fwhms)

    def _smooth_lsf(self, x, lsf):
        gy = ndimage.gaussian_filter(lsf, 8, mode='wrap')
        sz = len(lsf)
        #print(sz)
        # detect discontinuity and remove 1/3
        mx = np.argmax(gy)  # it was mx = np.argmax(y)
        mask = np.zeros(sz)

        mask[max(0, int(mx - 11)):min(sz - 1, int(mx +11))] = 1

        #print(mask)
        mask = ndimage.gaussian_filter(mask, 3)
        lsf_s = mask * lsf + (1 - mask) * gy
       # lsf_ss = lsf_s * np.hanning(len(lsf_s))
        return lsf_s

    def calculate_mtf(self, edge_list):
        mtf_nyq = []
        mtf_half_nyq = []
        for edge in edge_list:
            x, esf, esf_popt, lsf, lsf_popt, esf_norm, esf_norm_popt = edge
            lsf_s = self._smooth_lsf(x, lsf)

            try:
                N = lsf.shape[0]
                T = 0.1
                yf = fft(lsf_s)
                xf = fftfreq(N, T)[:N // 2]
                mtf = 2.0 / N * np.abs(yf[0:N // 2]) # ????
                mtf = mtf/mtf[0]
                idx = np.argwhere(xf < 1.5).flatten()
                p = interp1d(xf[idx], mtf[idx], kind='cubic')
                mtf_nyq.append(p(0.5))
                mtf_half_nyq.append(p(0.25))


                if self.log_level == logging.INFO:
                    fig, ax = plt.subplots()
                    ax.scatter(x, esf)
                    plt.show()
                    plt.clf()
                    fig, ax = plt.subplots()
                    ax.scatter(x, lsf)
                    ax.scatter(x, lsf_s, c='r')
                    ax.scatter(x, lsf_ss, c='g')
                    plt.show()
                    plt.clf()
                    #print(f"MTF no smoothing {p1(0.5), p1(0.25)}, smoothing {p(0.5), p(0.25)}, hanning {p2(0.5), p2(0.25)}")
                    fig, ax = plt.subplots()
                    ax.scatter(xf[idx], mtf[idx])
                    ax.plot(xf[idx], p(xf[idx]), ls='-')
                    ax.plot(xf[idx], p2(xf[idx]), ls='-', c='g')

                    plt.show()
                    plt.clf()

               # print(mtf[3])
            except:
                pass
        if len(mtf_nyq)<1:
            return None
        logging.info(f"MTF NYQ:  {mtf_nyq}")
        logging.info(f"{np.mean(mtf_nyq), np.std(mtf_nyq), np.median(mtf_nyq), np.min(mtf_nyq), np.max(mtf_nyq), len(mtf_nyq)}")
        logging.info(f"MTF half NYQ: {mtf_half_nyq}")
        logging.info(f"{np.mean(mtf_half_nyq), np.std(mtf_half_nyq), np.median(mtf_half_nyq), np.min(mtf_half_nyq), np.max(mtf_half_nyq), len(mtf_half_nyq)}")
        return np.array(mtf_nyq), np.array(mtf_half_nyq)
        #return np.mean(mtf_nyq), np.std(mtf_nyq), np.median(mtf_nyq), len(mtf_nyq)


    def fit_subpixel_edge(self, patch):
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

    def compute_esf(self, patch, edge_coeffs):
        patch_h, patch_w = patch.shape
        x = np.arange(-patch_w // 2 + 1, patch_w // 2 - 1, 0.1)
        x1 = np.arange(0, patch_w)
        x2 = np.arange(3, patch_h - 3, 0.5)
        edges = np.zeros((x2.shape[0], x.shape[0]))
        transect_fit = np.array(
            [np.polynomial.Polynomial([k - (-edge_coeffs[1] * edge_coeffs[0]), -edge_coeffs[1]])(x1) for k in x2])
        m = np.array(np.meshgrid(edge_coeffs(x2), x1)).T
        v0 = m[..., 1] - m[..., 0]
        v1 = transect_fit - x2[:, None]
        v = np.array([v0, v1])
        d = v[0] / np.abs(v[0]) * np.linalg.norm(v, axis=0)
        m = np.sum(np.round(transect_fit).astype(int) <= 0, axis=1) + np.sum(
            np.round(transect_fit).astype(int) >= patch_h, axis=1)
        m = m==0
        d = d[m]
        if not d.shape[0] == 0:
            idx = np.array([np.round(transect_fit[m]).astype(int), np.vstack([x1] * x2.shape[0])[m]])

            val = patch[idx[0], idx[1]]

            if val.shape[0] > 0:
                val = np.take_along_axis(val, np.argsort(d, axis=1), axis=1)
                dists = np.take_along_axis(d, np.argsort(d, axis=1), axis=1)
                for l, (d, v) in enumerate(zip(dists, val)):
                    c = CubicSpline(d, v)
                    edges[l] = c(x)


        edges = edges[~np.all(edges == 0, axis=1)]
        if edges.shape[0]<5:
            return None
        esf = np.mean(edges, axis=0)
        if self.log_level==logging.DEBUG:
            print("------------ esf -----------")
            fig, ax = plt.subplots()
            ax.scatter(x, esf)
            plt.show()
            plt.clf()
        idx = np.argwhere((-3 <= x) & (x <= 3)).flatten()
        x_ = x[idx]
        esf_ = esf[idx]
        shift = x_.shape[0] // 2 - np.argmax(np.abs(esf_ - np.roll(esf_, 1))[1:])
       # print(shift)
        esf = np.roll(esf, shift)
        if self.log_level==logging.DEBUG:
            fig, ax = plt.subplots()
            ax.scatter(x, esf)
            plt.show()
            plt.clf()
        lsf = np.abs((esf - np.roll(esf, 1))[1:])
        #lsf = np.gradient(esf)[1:]
        shift = abs(shift)
        if shift >0:
            x_ = x[shift+1:-shift]
            lsf_ = lsf[shift: -shift]
        else:
            x_ = x[1:]
            lsf_ = lsf
        # fig, ax = plt.subplots()
        # ax.scatter(x_, lsf_)
        # plt.show()
        # plt.clf()
        lsf_popt = self._fit_function(gaus, x_, lsf_, p0=[50, 0, 1])
        if lsf_popt is not None:
            lim = round((abs(lsf_popt[1]) + lsf_popt[-1]*3) * 2)/2 + 0.5
        else:
            lim = x[-1]
        if lim <= 0:
            lim = x[-1]
        idx = np.argwhere((-lim <= x) & (x <= lim)).flatten()
        x = x[idx[1:]]
        esf = esf[idx[1:]]
        lsf = lsf[idx[:-1]]

        lsf_popt = self._fit_function(gaus, x, lsf, p0=[50, 0, 1])
        esf_norm, v_min, v_max, esf_popt, esf_norm_popt = self.normalize_esf(esf, x)
        if self.evaluate_esf(lsf_popt, esf_norm_popt, esf, esf_norm, v_min, v_max, x):
            return [x, esf, esf_popt, lsf, lsf_popt, esf_norm, esf_norm_popt]
        else:
            return None

    def _fit_function(self, func, x, data, p0):
        if type(p0)=='list':
            p0 = np.array(p0).astype(np.float64)
        try:
            popt, pcov = curve_fit(func, x, data, p0=p0)
        except:
            return None
        return popt

    def normalize_esf(self, esf, x):
        a0 = np.median(esf)
        d0 = np.quantile(esf, 0.05)
        esf_popt = self._fit_function(f, x, esf, np.array([a0, 0.5, -0.5, d0]))
        if esf_popt is None:
            logging.debug("ESF fit failed")
            return None, None, None, None, None
        try:
            v_max = np.max(f(np.arange(x[-1], x[-1] + 4, 0.1), *esf_popt))
            v_min = np.mean(f(np.arange(x[0] - 4, x[0], 0.1), *esf_popt))
        except:
            ilogging.debug("min, max not found")
            return None, None, None, None, None


        if v_min is None or v_max is None:
            logging.debug("min, max not found")
            return None, None, None, None, None

        if v_max - v_min == 0:
            logging.debug("min, max not found")
            return None, None, None, None, None
        esf_norm = (esf - v_min) / (v_max - v_min)
        esf_norm_popt = self._fit_function(f, x, esf_norm, p0=[1, 0.5, 0.5, 0])
        return esf_norm, v_min, v_max, esf_popt, esf_norm_popt

    def evaluate_esf(self, lsf_popt,  esf_norm_popt, esf, esf_norm, v_min, v_max, x):
        if lsf_popt is None or esf_norm_popt is None:
            logging.debug("lsf or esf_norm fitting failed")
            return False
        a, b, c = lsf_popt
        #logging.debug(a,b,c)
        _left = esf[np.argwhere(x[1:] < b - 3 * c)]
        _right = esf[np.argwhere(x[1:] > b + 3 * c)]
        #logging.debug(_left.shape, _right.shape)
        if _left.shape[0]==0 or _right.shape[0]==0:
            # if _left.shape[0]==0:
            #     logging.debug("no left")
            #     logging.debug(b - 3 * c)
            # if _right.shape[0]==0:
            #     logging.debug("no right")
            #     logging.debug(b + 3 * c)
            logging.debug("left or right wing too small")
            return False
        noise1 = np.std(_left)
        noise2 = np.std(_right)

        noise = np.nanmean([noise1, noise2])
        if noise <= 0:
            logging.debug("noise is not positive")
            return False
        edge_snr = np.abs(v_min - v_max) / noise
       # logging.debug(f"snr {edge_snr}")


        # try:
        #     spline = UnivariateSpline(x[1:], lsf - np.max(lsf) / 2, s=0)
        #     r1, r2 = spline.roots()
        #     fwhm = np.max([r1, r2]) - np.min([r1, r2])
        # except:
        #     fwhm = 0


        residuals = esf_norm - f(x, *esf_norm_popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((esf_norm - np.mean(esf_norm)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        #logging.debug(edge_snr, r_squared)
        if self.log_level==logging.DEBUG:
            fig, ax = plt.subplots()
            ax.scatter(x, esf_norm)
            ax.plot(x, f(x, *esf_norm_popt))
            ax.title.set_text(f"{edge_snr}, {r_squared}")
            plt.show()
            plt.clf()
        if edge_snr > self.snr_threshold and r_squared > self.r2_threshold:

            return True

        if edge_snr <= self.snr_threshold:
            logging.debug("SNR too low")
        if r_squared <= self.r2_threshold:
            logging.debug("r2 is too low")
        return False


