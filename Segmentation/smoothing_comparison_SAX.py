"""
$ python3 compute_volumes_paramters_SAX.py  --help
usage: compute_volumes_paramters_SAX.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib:
import os

# imports 3rd party
import numpy as np
import nibabel as nib
import pylab as plt
import math
from math import factorial
from scipy.interpolate import splev, splrep, interp1d


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
       :param deriv:
       :param order:
       :param y:
       :param window_size:
       :param rate:
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    # precompute coefficients
    b = np.mat(
        [[k**i for i in order_range] for k in range(-half_window, half_window + 1)]
    )
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")


def get_list_of_dirs(target_dir, full_path=True):
    if full_path:
        dirs = sorted(
            [
                os.path.join(target_dir, d)
                for d in os.listdir(target_dir)
                if (
                    os.path.isdir(os.path.join(target_dir, d)) and not d.startswith(".")
                )
            ]
        )
    elif not full_path:
        dirs = sorted(
            [
                d
                for d in os.listdir(target_dir)
                if (
                    os.path.isdir(os.path.join(target_dir, d)) and not d.startswith(".")
                )
            ]
        )
    return dirs


def do_studies(study_IDs, nifti_dir):
    for study_ID_counter, study_ID in enumerate(study_IDs):
        print(f"do_studies() [{study_ID_counter}]: {study_ID}")
        subject_dir = os.path.join(nifti_dir, study_ID)
        filename_seg = os.path.join(subject_dir, "sa_seg_nnUnet.nii.gz")
        results_dir = os.path.join(subject_dir, "results_SAX")
        tt_sa_file = os.path.join(subject_dir, "sa_tt.txt")
        tt_sa = np.loadtxt(tt_sa_file)

        nim = nib.load(filename_seg)
        seg = nim.get_fdata()
        X, Y, Z, T = seg.shape
        dx, dy, dz = nim.header["pixdim"][1:4]
        volume_per_voxel = dx * dy * dz * 1e-3

        volume_LV = np.zeros(T)
        volume_RV = np.zeros(T)
        for fr in range(T):
            volume_LV[fr] = np.sum(seg[:, :, :, fr] == 1) * volume_per_voxel
            volume_RV[fr] = np.sum(seg[:, :, :, fr] == 3) * volume_per_voxel

        # =============================================================================
        # Smooth volume curve - savitzky_golay
        # =============================================================================
        window_size, poly_order = 13, 3
        volume_LV_SG = savitzky_golay(volume_LV, window_size, poly_order)
        volume_RV_SG = savitzky_golay(volume_RV, window_size, poly_order)

        # =============================================================================
        # Smooth volume curve - Spline
        # =============================================================================
        x = np.linspace(0, T - 1, T)
        # spl = splrep(x, volume_LV, s=50, per=True)
        num_points = 100
        xx = np.linspace(x.min(), x.max(), num_points)
        volume_LV_interp = interp1d(x, volume_LV, kind='linear')(xx)
        tt_sa_interp = interp1d(x, tt_sa, kind='linear')(xx)
        spl = splrep(xx, volume_LV_interp, s=150)
        volume_LV_spline = splev(xx, spl)

        x = np.linspace(0, T - 1, T)
        spl = splrep(x, volume_LV, s=100)
        volume_LV_spline2 = splev(x, spl)

        # tck, u = splprep(volume_RV.reshape(-1, 1).T)
        # u_new = np.linspace(u.min(), u.max(), T)
        # volume_RV_spline = splev(u_new, tck)[0]

        x = np.linspace(0, T - 1, T)
        spl = splrep(x, volume_RV, s=10, k=3, per=True)
        volume_RV_spline = splev(x, spl)

        # =============================================================================
        # Smooth volume curve - 1dinterp
        # =============================================================================
        x = np.linspace(0, T - 1, T)
        xx = np.linspace(np.min(x), np.max(x), T)
        itp = interp1d(x, volume_LV)
        volume_LV_interp1d = itp(xx)

        itp = interp1d(x, volume_RV)
        volume_RV_interp1d = itp(xx)

        # =============================================================================
        # plots
        # ============================================================================
        fig, ax = plt.subplots()
        ax.plot(tt_sa, volume_LV, "bo")
        # ax.plot(volume_LV, "b")
        # ax.plot(volume_LV_SG, "b")
        # ax.plot(volume_LV_spline, "m")
        ax.plot(tt_sa_interp, volume_LV_spline, "m")
        # ax.plot(volume_LV_interp1d, "-.g")
        # ax.plot(volume_LV_spline2, "k")
        # ax.plot(tt_sa, volume_LV_spline2, "k")

        ax.legend(["raw", "spline 100 points", "spline"])
        fig.savefig(
            os.path.join(
                "/data/Datasets/Flow/data_EF1/SAX_interpolation",
                "{}_LV.png".format(study_ID),
            )
        )

        # fig, ax = plt.subplots()
        # ax.plot(volume_RV, "ro")
        # ax.plot(volume_RV_SG, "b")
        # ax.plot(volume_RV_spline, "m")
        # ax.plot(volume_RV_interp1d, "-.g")
        # ax.legend(["raw", "SG", "spline", "interp1d"])
        # fig.savefig(
        #     os.path.join(
        #         "/data/Datasets/Flow/data_EF1/SAX_interpolation",
        #         "{}_RV.png".format(study_ID),
        #     )
        # )
        # plt.close("all")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    nifti_dir = "/data/Datasets/Flow/data_EF1/nifti/"
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    do_studies(study_IDs, nifti_dir)
