# imports - stdlib:
import os
from glob import glob

# imports 3rd party
import numpy as np
import nibabel as nib
import pylab as plt
import math
from math import factorial
from scipy.signal import savgol_filter
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
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
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
    save_dir = "/data/Datasets/Flow/data_EF1/flow_interpolation"
    save_area_dir = "/data/Datasets/Flow/data_EF1/flow_area_interp"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    tt_flow_dir = "/data/Datasets/Flow/data_EF1/nnUNet_raw_data/Task118_AscAoFlow/Results/tt_flow"

    for study_ID_counter, study_ID in enumerate(study_IDs):
        print(f"do_studies() [{study_ID_counter}]: {study_ID}")
        subject_dir = os.path.join(nifti_dir, study_ID)
        results_dir = os.path.join(subject_dir, "results_flow")
        flow_files = glob(f"{results_dir}/*ao_flow_rate.txt")

        for current_flow_file in flow_files:
            flow = np.loadtxt(current_flow_file)

            img_name = current_flow_file.replace("_ao_flow_rate.txt", "")
            img_name = img_name.replace(f"{results_dir}/", "")
            tt_flow = np.loadtxt(os.path.join(tt_flow_dir, f"{img_name}_tt_flow.txt"))
            T = len(tt_flow)
            area = np.loadtxt(os.path.join(results_dir, f"{img_name}_ao_area.txt"))

            # =============================================================================
            # Smooth volume curve - savitzky_golay
            # =============================================================================
            window_size, poly_order = 5, 3
            flow_SG = savitzky_golay(flow, window_size, poly_order)
            flow_SG_orig = savgol_filter(flow, 13, poly_order)

            # =============================================================================
            # Smooth volume curve - Spline
            # =============================================================================
            s1 = 1500

            x = np.linspace(0, T - 1, T)
            spl = splrep(x, flow, s=s1)
            volume_LV_spline = splev(x, spl)

            num_points = 100  # T*2
            xx = np.linspace(x.min(), x.max(), num_points)
            flow_interp = interp1d(x, flow, kind='linear')(xx)
            tt_flow_interp = interp1d(x, tt_flow, kind='linear')(xx)
            spl = splrep(xx, flow_interp, s=s1)
            volume_LV_spline_T2 = splev(xx, spl)

            # =============================================================================
            # Smooth volume curve - 1dinterp
            # =============================================================================
            x = np.linspace(0, T - 1, T)
            xx = np.linspace(np.min(x), np.max(x), T)
            itp = interp1d(x, flow)
            volume_LV_interp1d = itp(xx)

            # =============================================================================
            # Flow plots
            # ============================================================================
            fig, ax = plt.subplots()
            ax.plot(tt_flow, flow, "bo", label="raw")
            ax.plot(tt_flow, flow, "b")
            ax.plot(tt_flow, flow_SG_orig, "m--", label="SG scipy")
            # ax.plot(tt_flow, flow_SG, "k", label=f"SG new w{window_size}")
            # ax.plot(tt_flow, volume_LV_spline, "r", label=f"spline s{s1} T")
            ax.plot(tt_flow_interp, volume_LV_spline_T2, "g", label=f"spline 100 points")

            ax.legend()
            fig.savefig(os.path.join(save_dir, f"{img_name}.png"))
            plt.close("all")

            # =============================================================================
            # Area plots
            # ============================================================================
            num_points = 100  # T*2
            xx = np.linspace(x.min(), x.max(), num_points)
            area_interp = interp1d(x, area, kind='linear')(xx)
            tt_flow_interp = interp1d(x, tt_flow, kind='linear')(xx)
            spl = splrep(xx, area_interp, s=s1)
            area_spline = splev(xx, spl)

            fig, ax = plt.subplots()
            ax.plot(tt_flow, area, "bo", label="raw")
            ax.plot(tt_flow, area, "b")
            ax.plot(tt_flow_interp, area_spline, "g", label=f"spline 100")
            ax.legend()
            fig.savefig(os.path.join(save_area_dir, f"{img_name}.png"))
            plt.close("all")



# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    nifti_dir = "/data/Datasets/Flow/data_EF1/nifti/"
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    do_studies(study_IDs, nifti_dir)
