import numpy as np
from scipy.signal import resample
import pylab as plt
import cv2
import vtkmodules.all as vtk
from scipy import interpolate
from vtk.util import numpy_support
from scipy.signal import argrelextrema
from scipy import stats
from skimage import measure
import math
from math import factorial
import nibabel as nib
import pandas as pd
import os
import cc3d
from scipy.ndimage import binary_erosion, binary_fill_holes, label, sum as ndsum
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import splev, splrep, interp1d



def plot_volume_points(_volume, _global_volume, title, savefig_filename):
    _ED_frame = int(_global_volume[9])
    _ES_frame = int(_global_volume[10])
    _point_PER = int(_global_volume[15])

    fig, ax = plt.subplots()
    ax.plot(_volume)
    if _ED_frame != -1:
        ax.plot(_ED_frame, _volume[_ED_frame], 'g*')
        ax.annotate('ED', (_ED_frame, _volume[_ED_frame]))
    if _ES_frame != -1:
        ax.plot(_ES_frame, _volume[_ES_frame], 'g*')
        ax.annotate('ES', (_ES_frame, _volume[_ES_frame]))
    if _point_PER != -1:
        ax.plot(_point_PER, _volume[_point_PER], 'ro')
        ax.annotate('PER', (_point_PER, _volume[_point_PER]))

    plt.suptitle(title)
    plt.ylabel('Volume (mL)')
    plt.xlabel('Frames')
    plt.savefig(savefig_filename)
    plt.close('all')


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
    b = np.mat([[k ** i for i in order_range]
               for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)

    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def compute_complex_volumetric_params(logger, volume, time_per_frame):
    n_frames = volume.shape[0]
    _ES_frame = volume.argmin()
    _ED_frame = volume.argmax()
    _vol_first_deriv = np.gradient(volume)
    _global_volume = np.zeros(2)

    # =============================================================================
    # PER = peak_ejection_rate
    # =============================================================================
    # for local maxima
    indx_local_max = argrelextrema(_vol_first_deriv, np.greater)
    if len(indx_local_max[0]) > 1:
        indx_local_max = np.squeeze(np.asarray(indx_local_max))
    elif len(indx_local_max[0]) == 1:
        indx_local_max = indx_local_max[0]
    else:
        indx_local_max = [0]

    # for local minima
    indx_local_min = argrelextrema(_vol_first_deriv, np.less)
    if len(indx_local_min[0]) > 1:
        indx_local_min = np.squeeze(np.asarray(indx_local_min))
    elif len(indx_local_min[0]) == 1:
        indx_local_min = indx_local_min[0]

    if indx_local_max[0] != 0:
        max_ind = indx_local_max[_vol_first_deriv[indx_local_max].argsort()[
            ::-1]]
        max_ind = max_ind[max_ind > _ES_frame]
        if len(max_ind) > 2:
            max_ind = max_ind[0:2]
        elif len(max_ind) < 2:
            max_ind = np.hstack([max_ind, n_frames])
        max_ind = sorted(max_ind)
    else:
        max_ind = -1

    first_min = indx_local_min[_vol_first_deriv[indx_local_min].argmin()]
    vol_first_deriv_ES = _vol_first_deriv[_ES_frame:]

    try:
        volume_LV_ES = volume[0:_ES_frame]
        p = np.poly1d(np.polyfit(np.linspace(
            0, _ES_frame, _ES_frame), volume_LV_ES, 3))
        xp = np.linspace(0, _ES_frame, _ES_frame)
        volume_LV_ES_fit = p(xp)
        first_dervolume_ES_fit = np.gradient(volume_LV_ES_fit)
        point_PER2 = first_dervolume_ES_fit.argmin()
        PER2 = np.abs(
            first_dervolume_ES_fit[point_PER2] / time_per_frame * 1000)
    except:
        point_PER2 = first_min
        PER2 = np.abs(vol_first_deriv_ES[point_PER2] / time_per_frame * 1000)

    # PER = peak_ejection_rate
    _point_PER = _vol_first_deriv[0:_ES_frame].argmin()
    if _point_PER == first_min:
        if np.abs(_point_PER - _ES_frame / 2) <= 2:
            if point_PER2 == _point_PER:
                _PER = np.abs(
                    _vol_first_deriv[_point_PER] / time_per_frame * 1000)
            elif np.abs(point_PER2 - _point_PER) < 2:
                _point_PER = np.max([_point_PER, point_PER2])
                _PER = np.abs(
                    _vol_first_deriv[_point_PER] / time_per_frame * 1000)

            else:
                if point_PER2 > _point_PER:
                    _PER = PER2
                    _point_PER = point_PER2
                else:
                    _PER = np.abs(
                        _vol_first_deriv[_point_PER] / time_per_frame * 1000)
        else:
            if np.abs(point_PER2 - _ES_frame / 2) <= 2:
                _point_PER = point_PER2
                _PER = np.abs(
                    _vol_first_deriv[_point_PER] / time_per_frame * 1000)
            else:
                _point_PER = int(_ES_frame / 2)
                _PER = np.abs(
                    _vol_first_deriv[_point_PER] / time_per_frame * 1000)

    else:
        _point_PER = int(_ES_frame / 2)
        _PER = np.abs(_vol_first_deriv[_point_PER] / time_per_frame * 1000)
        logger.info('PER has a problem')
        _PER = -1

    # =============================================================================
    # Save
    # =============================================================================
    _global_volume[0] = _PER
    _global_volume[1] = _point_PER

    return _global_volume


def isCircularSegment(img, minPoolSize=10, minSegSize=100, emptySpaceFrac=0.15):
    """Returns True if the image `img' contains a circular segment with one hole."""
    binimg = (img == img.max()).astype(int)  # binary image

    if binimg.min() >= binimg.max():
        return False  # blank image

    # binary fill on isolated outer contour
    outer = binary_fill_holes(binimg).astype(int)
    inner = outer - binimg

    if label(inner)[1] != 1:  # multiple holes, or no hole at all, not just one for the pool
        return False

    # inner holes too small (or not present for non-circles)
    if ndsum(np.abs(inner)) < minPoolSize:
        return False

    region = np.argwhere(binimg)
    hull = ConvexHull(region)
    de = Delaunay(region[hull.vertices])

    simplexpts = de.find_simplex(np.argwhere(img == img))
    mask = (simplexpts.reshape(img.shape) != -1).astype(int)
    mask = binary_erosion(mask).astype(int)  # mask of whole segment area
    masksum = np.sum(mask)

    if masksum < minSegSize:  # masked area too small
        return False

    # space around mask in convex area too large (ie. weird shape)
    if np.sum(mask - outer) > masksum * emptySpaceFrac:
        return False

    return True


def evaluate_wall_thickness(logger, _seg_name, output_name_stem, part=None):
    """ Evaluate myocardial wall thickness. """
    # Read the segmentation image
    nim = nib.load(_seg_name)
    Z = nim.header['dim'][3]
    dx, dy, dz = nim.header['pixdim'][1:4]
    affine = nim.affine
    _seg = nim.get_fdata()
    _seg = _seg[:, :, :, 0]

    # Get largest 3D cc
    seg_single = np.where(_seg > 1, 1, _seg).astype(
        np.int32)  # combine all labels
    seg_cc = cc3d.connected_components(
        seg_single)  # select connected components
    largest = get_largest_cc(seg_cc)  # get largest connected component
    _seg = np.where(largest, _seg, 0)

    # Label class in the segmentation
    _label = {'BG': 0, 'LV': 1, 'Myo': 2, 'RV': 3}

    # Determine the AHA coordinate system using the mid-cavity slice
    aha_axis = determine_aha_coordinate_system(_seg, affine)

    # Determine the AHA part of each slice
    if not part:
        part_z = determine_aha_part(_seg, affine)
    else:
        part_z = {z: part for z in range(Z)}

    # Construct the points set to represent the endocardial contours
    endo_points = vtk.vtkPoints()
    thickness = vtk.vtkDoubleArray()
    thickness.SetName('Thickness')
    points_aha = vtk.vtkIntArray()
    points_aha.SetName('Segment ID')
    point_id = 0
    lines = vtk.vtkCellArray()

    # Save epicardial contour for debug and demonstration purposes
    save_epi_contour = False
    if save_epi_contour:
        epi_points = vtk.vtkPoints()
        points_epi_aha = vtk.vtkIntArray()
        points_epi_aha.SetName('Segment ID')
        point_epi_id = 0
        lines_epi = vtk.vtkCellArray()

    # For each slice
    for z in range(Z):
        # Check whether there is endocardial segmentation and it is not too small,
        # e.g. a single pixel, which either means the structure is missing or
        # causes problem in contour interpolation.
        seg_z = _seg[:, :, z]
        endo = (seg_z == _label['LV']).astype(np.uint8)
        endo = get_largest_cc(endo).astype(np.uint8)
        myo = (seg_z == _label['Myo']).astype(np.uint8)
        myo = remove_small_cc(myo).astype(np.uint8)
        epi = (endo | myo).astype(np.uint8)
        epi = get_largest_cc(epi).astype(np.uint8)
        pixel_thres = 10
        if (np.sum(endo) < pixel_thres) or (np.sum(myo) < pixel_thres):
            continue
        # Check if it's a full ring
        if not isCircularSegment(myo):
            # plt.imshow(myo)
            # plt.show()
            continue

        # Calculate the centre of the LV cavity
        # Get the largest component in case we have a bad segmentation
        cx, cy = [np.mean(x) for x in np.nonzero(endo)]
        lv_centre = np.dot(affine, np.array([cx, cy, z, 1]))[:3]

        # Extract endocardial contour
        # Note: cv2 considers an input image as a Y x X array, which is different
        # from nibabel which assumes a X x Y array.
        contours, _ = cv2.findContours(cv2.inRange(
            endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        endo_contour = contours[0][:, 0, :]

        # Extract epicardial contour
        contours, _ = cv2.findContours(cv2.inRange(
            epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        epi_contour = contours[0][:, 0, :]

        # Smooth the contours
        endo_contour = approximate_contour(endo_contour, periodic=True)
        epi_contour = approximate_contour(epi_contour, periodic=True)

        # A polydata representation of the epicardial contour
        epi_points_z = vtk.vtkPoints()
        for y, x in epi_contour:
            p = np.dot(affine, np.array([x, y, z, 1]))[:3]
            epi_points_z.InsertNextPoint(p)
        epi_poly_z = vtk.vtkPolyData()
        epi_poly_z.SetPoints(epi_points_z)

        # Point locator for the epicardial contour
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(epi_poly_z)
        locator.BuildLocator()

        # For each point on endocardium, find the closest point on epicardium
        N = endo_contour.shape[0]
        for i in range(N):
            y, x = endo_contour[i]

            # The world coordinate of this point
            p = np.dot(affine, np.array([x, y, z, 1]))[:3]
            endo_points.InsertNextPoint(p)

            # The closest epicardial point
            q = np.array(epi_points_z.GetPoint(locator.FindClosestPoint(p)))

            # The distance from endo to epi
            dist_pq = np.linalg.norm(q - p)

            # Add the point data
            thickness.InsertNextTuple1(dist_pq)
            seg_id = determine_aha_segment_id(
                logger, p, lv_centre, aha_axis, part_z[z])
            points_aha.InsertNextTuple1(seg_id)

            # Record the first point of the current contour
            if i == 0:
                contour_start_id = point_id

            # Add the line
            if i == (N - 1):
                lines.InsertNextCell(2, [point_id, contour_start_id])
            else:
                lines.InsertNextCell(2, [point_id, point_id + 1])

            # Increment the point index
            point_id += 1

        if save_epi_contour:
            # For each point on epicardium
            N = epi_contour.shape[0]
            for i in range(N):
                y, x = epi_contour[i]

                # The world coordinate of this point
                p = np.dot(affine, np.array([x, y, z, 1]))[:3]
                epi_points.InsertNextPoint(p)
                seg_id = determine_aha_segment_id(
                    p, lv_centre, aha_axis, part_z[z])
                points_epi_aha.InsertNextTuple1(seg_id)

                # Record the first point of the current contour
                if i == 0:
                    contour_start_id = point_epi_id

                # Add the line
                if i == (N - 1):
                    lines_epi.InsertNextCell(
                        2, [point_epi_id, contour_start_id])
                else:
                    lines_epi.InsertNextCell(
                        2, [point_epi_id, point_epi_id + 1])

                # Increment the point index
                point_epi_id += 1

    # Save to a vtk file
    endo_poly = vtk.vtkPolyData()
    endo_poly.SetPoints(endo_points)
    endo_poly.GetPointData().AddArray(thickness)
    endo_poly.GetPointData().AddArray(points_aha)
    endo_poly.SetLines(lines)

    writer = vtk.vtkPolyDataWriter()
    output_name = '{0}.vtk'.format(output_name_stem)
    writer.SetFileName(output_name)
    writer.SetInputData(endo_poly)
    writer.Write()

    if save_epi_contour:
        epi_poly = vtk.vtkPolyData()
        epi_poly.SetPoints(epi_points)
        epi_poly.GetPointData().AddArray(points_epi_aha)
        epi_poly.SetLines(lines_epi)

        writer = vtk.vtkPolyDataWriter()
        output_name = '{0}_epi.vtk'.format(output_name_stem)
        writer.SetFileName(output_name)
        writer.SetInputData(epi_poly)
        writer.Write()

    # Evaluate the wall thickness per AHA segment and save to a csv file
    table_thickness = np.zeros(17)
    np_thickness = numpy_support.vtk_to_numpy(thickness).astype(np.float32)
    np_points_aha = numpy_support.vtk_to_numpy(points_aha).astype(np.int8)

    for i in range(16):
        table_thickness[i] = np.mean(
            np_thickness[np_points_aha == (i + 1)]) * dx * dy
    table_thickness[-1] = np.mean(np_thickness) * dx * dy

    _index = [str(x) for x in np.arange(1, 17)] + ['Global']
    _df = pd.DataFrame(table_thickness, index=_index, columns=['Thickness'])
    _df.to_csv('{0}.csv'.format(output_name_stem))

    return table_thickness


def determine_aha_coordinate_system(seg_sa, affine_sa):
    """ Determine the AHA coordinate system using the mid-cavity slice
        of the short-axis image segmentation.
        """
    # Label class in the segmentation
    _label = {'BG': 0, 'LV': 1, 'Myo': 2, 'RV': 3}

    # Find the mid-cavity slice
    _, _, cz = [np.mean(x) for x in np.nonzero(seg_sa == _label['LV'])]
    z = int(round(cz))
    seg_z = seg_sa[:, :, z]

    endo = (seg_z == _label['LV']).astype(np.uint8)
    endo = get_largest_cc(endo).astype(np.uint8)
    myo = (seg_z == _label['Myo']).astype(np.uint8)
    myo = remove_small_cc(myo).astype(np.uint8)
    epi = (endo | myo).astype(np.uint8)
    epi = get_largest_cc(epi).astype(np.uint8)
    rv = (seg_z == _label['RV']).astype(np.uint8)
    rv = get_largest_cc(rv).astype(np.uint8)

    # Extract epicardial contour
    contours, _ = cv2.findContours(cv2.inRange(
        epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    epi_contour = contours[0][:, 0, :]

    # Find the septum, which is the intersection between LV and RV
    septum = []
    dilate_iter = 1
    max_iterations = 50
    while (len(septum) == 0 and dilate_iter < max_iterations):
        # Dilate the RV till it intersects with LV epicardium.
        # Normally, this is fulfilled after just one iteration.
        rv_dilate = cv2.dilate(rv, np.ones(
            (3, 3), dtype=np.uint8), iterations=dilate_iter)
        dilate_iter += 1
        for y, x in epi_contour:
            if rv_dilate[x, y] == 1:
                septum += [[x, y]]

    if dilate_iter >= max_iterations:
        raise Exception('Problem determine AHA coordinate systems')

    # The middle of the septum
    mx, my = septum[int(round(0.5 * len(septum)))]
    point_septum = np.dot(affine_sa, np.array([mx, my, z, 1]))[:3]

    # Find the centre of the LV cavity
    cx, cy = [np.mean(x) for x in np.nonzero(endo)]
    point_cavity = np.dot(affine_sa, np.array([cx, cy, z, 1]))[:3]

    # Determine the AHA coordinate system
    axis = {'lv_to_sep': point_septum - point_cavity}
    axis['lv_to_sep'] /= np.linalg.norm(axis['lv_to_sep'])
    axis['apex_to_base'] = np.copy(affine_sa[:3, 2])
    axis['apex_to_base'] /= np.linalg.norm(axis['apex_to_base'])
    if axis['apex_to_base'][2] < 0:
        axis['apex_to_base'] *= -1
    axis['inf_to_ant'] = np.cross(axis['apex_to_base'], axis['lv_to_sep'])
    return axis


def determine_aha_part(seg_sa, affine_sa, three_slices=False):
    """ Determine the AHA part for each slice. """
    # Label class in the segmentation
    _label = {'BG': 0, 'LV': 1, 'Myo': 2, 'RV': 3}

    # Sort the z-axis positions of the slices with both endo and epicardium
    # segmentations
    X, Y, Z = seg_sa.shape[:3]
    z_pos = []
    for z in range(Z):
        seg_z = seg_sa[:, :, z]
        endo = (seg_z == _label['LV']).astype(np.uint8)
        myo = (seg_z == _label['Myo']).astype(np.uint8)
        pixel_thres = 10
        if (np.sum(endo) < pixel_thres) or (np.sum(myo) < pixel_thres):
            continue
        z_pos += [(z, np.dot(affine_sa, np.array([X / 2.0, Y / 2.0, z, 1]))[2])]
    z_pos = sorted(z_pos, key=lambda x: -x[1])

    # Divide the slices into three parts: basal, mid-cavity and apical
    n_slice = len(z_pos)
    part_z = {}
    if three_slices:
        # Select three slices (basal, mid and apical) for strain analysis, inspired by:
        #
        # [1] Robin J. Taylor, et al. Myocardial strain measurement with
        # feature-tracking cardiovascular magnetic resonance: normal values.
        # European Heart Journal - Cardiovascular Imaging, (2015) 16, 871-881.
        #
        # [2] A. Schuster, et al. Cardiovascular magnetic resonance feature-
        # tracking assessment of myocardial mechanics: Intervendor agreement
        # and considerations regarding reproducibility. Clinical Radiology
        # 70 (2015), 989-998.

        # Use the slice at 25% location from base to apex.
        # Avoid using the first one or two basal slices, as the myocardium
        # will move out of plane at ES due to longitudinal motion, which will
        # be a problem for 2D in-plane motion tracking.
        z = int(round((n_slice - 1) * 0.25))
        part_z[z_pos[z][0]] = 'basal'

        # Use the central slice.
        z = int(round((n_slice - 1) * 0.5))
        part_z[z_pos[z][0]] = 'mid'

        # Use the slice at 75% location from base to apex.
        # In the most apical slices, the myocardium looks blurry and
        # may not be suitable for motion tracking.
        z = int(round((n_slice - 1) * 0.75))
        part_z[z_pos[z][0]] = 'apical'
    else:
        # Use all the slices
        i1 = int(math.ceil(n_slice / 3.0))
        i2 = int(math.ceil(2 * n_slice / 3.0))
        i3 = n_slice

        for i in range(0, i1):
            part_z[z_pos[i][0]] = 'basal'

        for i in range(i1, i2):
            part_z[z_pos[i][0]] = 'mid'

        for i in range(i2, i3):
            part_z[z_pos[i][0]] = 'apical'
    return part_z


def determine_aha_segment_id(logger, point, lv_centre, aha_axis, part):
    """ Determine the AHA segment ID given a point,
        the LV cavity center and the coordinate system.
        """
    d = point - lv_centre
    x = np.dot(d, aha_axis['inf_to_ant'])
    y = np.dot(d, aha_axis['lv_to_sep'])
    deg = math.degrees(math.atan2(y, x))
    seg_id = 0

    if part == 'basal':
        if (deg >= -30) and (deg < 30):
            seg_id = 1
        elif (deg >= 30) and (deg < 90):
            seg_id = 2
        elif (deg >= 90) and (deg < 150):
            seg_id = 3
        elif (deg >= 150) or (deg < -150):
            seg_id = 4
        elif (deg >= -150) and (deg < -90):
            seg_id = 5
        elif (deg >= -90) and (deg < -30):
            seg_id = 6
        else:
            logger.info('Error: wrong degree {0}!'.format(deg))
            exit(0)
    elif part == 'mid':
        if (deg >= -30) and (deg < 30):
            seg_id = 7
        elif (deg >= 30) and (deg < 90):
            seg_id = 8
        elif (deg >= 90) and (deg < 150):
            seg_id = 9
        elif (deg >= 150) or (deg < -150):
            seg_id = 10
        elif (deg >= -150) and (deg < -90):
            seg_id = 11
        elif (deg >= -90) and (deg < -30):
            seg_id = 12
        else:
            logger.info('Error: wrong degree {0}!'.format(deg))
            exit(0)
    elif part == 'apical':
        if (deg >= -45) and (deg < 45):
            seg_id = 13
        elif (deg >= 45) and (deg < 135):
            seg_id = 14
        elif (deg >= 135) or (deg < -135):
            seg_id = 15
        elif (deg >= -135) and (deg < -45):
            seg_id = 16
        else:
            logger.info('Error: wrong degree {0}!'.format(deg))
            exit(0)
    elif part == 'apex':
        seg_id = 17
    else:
        logger.info('Error: unknown part {0}!'.format(part))
        exit(0)
    return seg_id


def get_largest_cc(binary):
    """ Get the largest connected component in the foreground. """
    cc = measure.label(binary)
    n_cc = cc.max()
    max_n = -1
    max_area = 0
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area > max_area:
            max_area = area
            max_n = n
    largest_cc = (cc == max_n)
    return largest_cc


def remove_small_cc(binary, thres=10):
    """ Remove small connected component in the foreground. """
    cc = measure.label(binary)
    n_cc = cc.max()
    binary2 = np.copy(binary)
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area < thres:
            binary2[cc == n] = 0
    return binary2


def approximate_contour(contour, factor=4, smooth=0.05, periodic=False):
    """ Approximate a contour.
        contour: input contour
        factor: upsampling factor for the contour
        smooth: smoothing factor for controling the number of spline knots.
                Number of knots will be increased until the smoothing
                condition is satisfied:
                sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s
                which means the larger s is, the fewer knots will be used,
                thus the contour will be smoother but also deviating more
                from the input contour.
        periodic: set to True if this is a closed contour, otherwise False.
        return the upsampled and smoothed contour
    """
    # The input contour
    N = len(contour)
    dt = 1.0 / N
    t = np.arange(N) * dt
    x = contour[:, 0]
    y = contour[:, 1]

    # Pad the contour before approximation to avoid underestimating
    # the values at the end points
    r = int(0.5 * N)
    t_pad = np.concatenate(
        (np.arange(-r, 0) * dt, t, 1 + np.arange(0, r) * dt))
    if periodic:
        x_pad = np.concatenate((x[-r:], x, x[:r]))
        y_pad = np.concatenate((y[-r:], y, y[:r]))
    else:
        x_pad = np.concatenate(
            (np.repeat(x[0], repeats=r), x, np.repeat(x[-1], repeats=r)))
        y_pad = np.concatenate(
            (np.repeat(y[0], repeats=r), y, np.repeat(y[-1], repeats=r)))

    # Fit the contour with splines with a smoothness constraint
    fx = interpolate.UnivariateSpline(t_pad, x_pad, s=smooth * len(t_pad))
    fy = interpolate.UnivariateSpline(t_pad, y_pad, s=smooth * len(t_pad))

    # Evaluate the new contour
    N2 = N * factor
    dt2 = 1.0 / N2
    t2 = np.arange(N2) * dt2
    x2, y2 = fx(t2), fy(t2)
    contour2 = np.stack((x2, y2), axis=1)
    return contour2


def compute_volumes(logger, _subject_dir, _filename_seg, _results_dir, _subject_id):
    if os.path.exists(_filename_seg):
        nim = nib.load(_filename_seg)
        seg = nim.get_fdata()
        X, Y, Z, T = seg.shape
        dx, dy, dz = nim.header['pixdim'][1:4]
        volume_per_voxel = dx * dy * dz * 1e-3
        density = 1.05
        time_per_frame = nim.header['pixdim'][4]
        HR = 60000.0 / (time_per_frame * T)
        if os.path.exists(os.path.join(_subject_dir, 'sa_HR.txt')):
            HR2 = float(np.loadtxt(os.path.join(_subject_dir, 'sa_HR.txt')))
            if np.abs(HR - HR2) > 5:
                HR = HR2

        # =============================================================================
        # Compute volumes over time
        # =============================================================================
        volume_LV = np.zeros(T)
        volume_LVM = np.zeros(T)
        volume_RV = np.zeros(T)
        for fr in range(T):
            volume_LV[fr] = np.sum(seg[:, :, :, fr] == 1) * volume_per_voxel
            volume_LVM[fr] = np.sum(seg[:, :, :, fr] == 2) * \
                volume_per_voxel * density
            volume_RV[fr] = np.sum(seg[:, :, :, fr] == 3) * volume_per_voxel

        # =============================================================================
        # Smooth volume curve per slices
        # =============================================================================
        # window_size, poly_order = 13, 3
        # volume_LV = savitzky_golay(volume_LV, window_size, poly_order)
        # volume_LVM = savitzky_golay(volume_LVM, window_size, poly_order)
        # volume_RV = savitzky_golay(volume_RV, window_size, poly_order)
        tt_sa = np.loadtxt(os.path.join(_subject_dir, "sa_tt.txt"))

        # Save uninterpolated versions of the curves
        np.savetxt(os.path.join(_results_dir, 'sa_tt_orig.txt'), tt_sa)
        np.savetxt(os.path.join(_results_dir, 'LVV_orig.txt'), volume_LV)
        np.savetxt(os.path.join(_results_dir, 'RVV_orig.txt'), volume_RV)
        np.savetxt(os.path.join(_results_dir, 'LVMV_orig.txt'), volume_LVM)

        num_points = 75
        s = 50
        x = np.linspace(0, T - 1, T)

        if num_points is not None:  # Increase number of points in curve
            xx = np.linspace(x.min(), x.max(), num_points)
            tt_sa = interp1d(x, tt_sa, kind='linear')(xx)

            volume_LV_interp = interp1d(x, volume_LV, kind='linear')(xx)
            spl = splrep(xx, volume_LV_interp, s=s)
            volume_LV = splev(xx, spl)
        
            volume_RV_interp = interp1d(x, volume_RV, kind='linear')(xx)
            spl = splrep(xx, volume_RV_interp, s=s)
            volume_RV = splev(xx, spl)

            volume_LVM_interp = interp1d(x, volume_LVM, kind='linear')(xx)
            spl = splrep(xx, volume_LVM_interp, s=s)
            volume_LVM = splev(xx, spl)
        else:  # Keep original number of points
            spl = splrep(x, volume_LV, s=s)
            volume_LV = splev(x, spl)

            spl = splrep(x, volume_RV, s=s)
            volume_RV = splev(x, spl)

            spl = splrep(x, volume_LVM, s=s)
            volume_LVM = splev(x, spl)

        np.savetxt(os.path.join(_results_dir, 'sa_tt.txt'), tt_sa)
        np.savetxt(os.path.join(_results_dir, 'LVV.txt'), volume_LV)
        np.savetxt(os.path.join(_results_dir, 'RVV.txt'), volume_RV)
        np.savetxt(os.path.join(_results_dir, 'LVMV.txt'), volume_LVM)

        # =============================================================================
        # Generate report simple: EDV, ESV
        # =============================================================================
        LVEDV = np.max(volume_LV)
        LVESV = np.min(volume_LV)
        LVSV = LVEDV - LVESV
        LVEF = LVSV / LVEDV * 100
        LVM = volume_LVM[0]
        RVEDV = np.max(volume_RV)
        RVESV = np.min(volume_RV)
        RVSV = RVEDV - RVESV
        RVEF = RVSV / RVEDV * 100

        report_volumes_simple = np.zeros((16, 1))
        report_volumes_simple[0] = LVEDV
        report_volumes_simple[1] = LVESV
        report_volumes_simple[2] = LVSV
        report_volumes_simple[3] = LVEF
        report_volumes_simple[4] = LVM
        report_volumes_simple[5] = RVEDV
        report_volumes_simple[6] = RVESV
        report_volumes_simple[7] = RVSV
        report_volumes_simple[8] = RVEF
        report_volumes_simple[9] = np.argmax(volume_LV)
        report_volumes_simple[10] = np.argmin(volume_LV)
        report_volumes_simple[11] = np.argmax(volume_RV)
        report_volumes_simple[12] = np.argmin(volume_RV)
        report_volumes_simple[13] = HR

        try:
            vec_LV = compute_complex_volumetric_params(
                logger, volume_LV, time_per_frame)
        except:
            logger.exception('Problem computing full report')
            vec_LV = -np.ones(4)
        report_volumes_simple[14] = vec_LV[0]  # LVPER
        report_volumes_simple[15] = vec_LV[1]  # point LVPER

        df = pd.DataFrame(report_volumes_simple.T)
        df.to_csv('{0}/report_volumes.csv'.format(_results_dir),
                  header=['LVEDV', 'LVESV', 'LVSV', 'LVEF', 'LVM', 'RVEDV', 'RVESV', 'RVSV',
                          'RVEF', 'LV ED frame', 'LV ES frame', 'RV ED frame', 'RV ES frame', 'HR', 'PER',
                          'point_PER'],
                  index=False)

        # =============================================================================
        # Plot
        # =============================================================================
        plot_volume_points(volume_LV, report_volumes_simple, '{}: LVV'.format(_subject_id),
                           os.path.join(_results_dir, 'LVV_params.png'))

        return volume_LV, volume_RV


def sa_pass_quality_control_images(logger, seg_sa_name):
    """ Quality control for short-axis image segmentation """
    nim = nib.load(seg_sa_name)
    seg_sa = nim.get_fdata()[:, :, :, 0]  # Only check in ED frame
    X, Y, Z = seg_sa.shape[:3]

    # Label class in the segmentation
    _label = {'LV': 1, 'Myo': 2, 'RV': 3}

    # Criterion 1: every class exists and the area is above a threshold
    # Count number of pixels in 3D
    for l_name, l in _label.items():
        pixel_thres = 10
        if np.sum(seg_sa == l) < pixel_thres:
            logger.info('{0}: The segmentation for class {1} is smaller than {2} pixels. '
                        'It does not pass the quality control.'.format(seg_sa_name, l_name, pixel_thres))
            return False

    # Criterion 2: number of slices with LV segmentations is above a threshold
    # and there is no missing segmentation in between the slices
    z_pos = []
    for z in range(Z):
        seg_z = seg_sa[:, :, z]
        endo = (seg_z == _label['LV']).astype(np.uint8)
        myo = (seg_z == _label['Myo']).astype(np.uint8)
        pixel_thres = 10
        if (np.sum(endo) < pixel_thres) or (np.sum(myo) < pixel_thres):
            continue
        z_pos += [z]
    n_slice = len(z_pos)
    slice_thres = 6
    if n_slice < slice_thres:
        logger.info('{0}: The segmentation has less than {1} slices. '
                    'It does not pass the quality control.'.format(seg_sa_name, slice_thres))
        return False

    if n_slice != (np.max(z_pos) - np.min(z_pos) + 1):
        logger.info('{0}: There is missing segmentation between the slices. '
                    'It does not pass the quality control.'.format(seg_sa_name))
        return False

    # Criterion 3: LV and RV exists on the mid-cavity slice
    _, _, cz = [np.mean(x) for x in np.nonzero(seg_sa == _label['LV'])]
    z = int(round(cz))
    seg_z = seg_sa[:, :, z]

    endo = (seg_z == _label['LV']).astype(np.uint8)
    endo = get_largest_cc(endo).astype(np.uint8)
    myo = (seg_z == _label['Myo']).astype(np.uint8)
    myo = remove_small_cc(myo).astype(np.uint8)
    epi = (endo | myo).astype(np.uint8)
    epi = get_largest_cc(epi).astype(np.uint8)
    rv = (seg_z == _label['RV']).astype(np.uint8)
    rv = get_largest_cc(rv).astype(np.uint8)
    pixel_thres = 10
    if np.sum(epi) < pixel_thres or np.sum(rv) < pixel_thres:
        logger.info('{0}: Can not find LV epi or RV to determine the AHA '
                    'coordinate system.'.format(seg_sa_name))
        return False
    return True


def sa_pass_quality_volumes(volume_LV, volume_RV):
    # Criterion 1: Volume at the beginning and end cardiac cycle is similar
    if (volume_LV[0] < volume_LV[-1] and np.abs(volume_LV[0] - volume_LV[-1]) > 0.15 * volume_LV[0]) \
            or volume_RV[0] < volume_RV[-1] and np.abs(volume_RV[0] - volume_RV[-1]) > 0.15 * volume_RV[0]:
        return False
    LVSV = np.max(volume_LV) - np.min(volume_LV)
    RVSV = np.max(volume_RV) - np.min(volume_RV)
    # Criterion 2: Stroke volume difference bigger than 10mL
    if LVSV - RVSV < 10:
        return False
    return True


def sa_pass_quality_LSTM(logger, model, volume, ttype):
    Mean_SAX = 102.15955715897826
    Std_SAX = 40.06083828498268
    optimal_threshold_w = 0.7857013

    # load model
    N_frames = 50
    X = resample(volume, N_frames)
    X_subj_norm = (X - Mean_SAX) / Std_SAX
    X_subj_resh = X_subj_norm.reshape((1, 1, N_frames))
    y_prob = model.predict(X_subj_resh)
    y_prob = y_prob[:, 0]
    label_QC = np.zeros(1)
    label_QC[y_prob > optimal_threshold_w] = 1
    if label_QC[0] == 1:
        logger.info('{} Rejected due to unphysiological volume'.format(ttype))
        return False
    else:
        return True
