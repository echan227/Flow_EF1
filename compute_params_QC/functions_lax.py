import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import math
import warnings
from scipy.spatial import distance
from skimage import measure
import cv2
from collections import defaultdict
from numpy.linalg import norm
from scipy.interpolate import splprep, splev
from scipy.ndimage.morphology import binary_closing as closing
from skimage.morphology import skeletonize
from collections import Counter
from skimage.measure import label
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from scipy.signal import resample

debug = False
Nsegments_length = 15


# =============================================================================
# Function
# =============================================================================
def getLargestCC(segmentation):
    nb_labels = np.unique(segmentation)[1:]
    out_image = np.zeros_like(segmentation)
    for ncc in nb_labels:
        _aux = np.squeeze(segmentation == ncc).astype(float)  # get myocardial labe
        labels = label(_aux)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        out_image += largestCC * ncc
    return out_image


def binarymatrix(A):
    A_aux = np.copy(A)
    A = map(tuple, A)
    dic = Counter(A)
    for (i, j) in dic.items():
        if j > 1:
            ind = np.where(((A_aux[:, 0] == i[0]) & (A_aux[:, 1] == i[1])))[0]
            A_aux = np.delete(A_aux, ind[1:], axis=0)
    if np.linalg.norm(A_aux[:, 0] - A_aux[:, -1]) < 0.01:
        A_aux = A_aux[:-1, :]
    return A_aux


def get_right_atrial_volumes(seg, _fr, _pointsRV, logger):
    """
    This function gets the centre line (height) of the atrium and atrial dimension at 15 points along this line.
    """
    _apex_RV, _rvlv_point, _free_rv_point = _pointsRV
    if debug:
        plt.figure()
        plt.imshow(seg)
        plt.plot(_apex_RV[1], _apex_RV[0], 'mo')
        plt.plot(_rvlv_point[1], _rvlv_point[0], 'c*')
        plt.plot(_free_rv_point[1], _free_rv_point[0], 'y*')

    mid_valve_RV = np.mean([_rvlv_point, _free_rv_point], axis=0)
    _atria_seg = np.squeeze(seg == 5).astype(float)  # get atria label
    rv_seg = np.squeeze(seg == 3).astype(float)  # get atria label

    # Generate contours from the atria
    _contours_RA = measure.find_contours(_atria_seg, 0.8)
    _contours_RA = _contours_RA[0]

    contours_RV = measure.find_contours(rv_seg, 0.8)
    contours_RV = contours_RV[0]

    # Compute distance between mid_valve and every point in contours
    dist = distance.cdist(_contours_RA, [mid_valve_RV])
    ind_mitral_valve = dist.argmin()
    mid_valve_RA = _contours_RA[ind_mitral_valve, :]
    dist = distance.cdist(_contours_RA, [mid_valve_RA])
    ind_top_atria = dist.argmax()
    top_atria = _contours_RA[ind_top_atria, :]
    ind_base1 = distance.cdist(_contours_RA, [_rvlv_point]).argmin()
    ind_base2 = distance.cdist(_contours_RA, [_free_rv_point]).argmin()
    atria_edge1 = _contours_RA[ind_base1, :]
    atria_edge2 = _contours_RA[ind_base2, :]

    if debug:
        plt.figure()
        plt.imshow(seg)
        plt.plot(_contours_RA[:, 1], _contours_RA[:, 0], 'r-')
        plt.plot(contours_RV[:, 1], contours_RV[:, 0], 'k-')
        plt.plot(top_atria[1], top_atria[0], 'mo')
        plt.plot(mid_valve_RA[1], mid_valve_RA[0], 'co')
        plt.plot(atria_edge1[1], atria_edge1[0], 'go')
        plt.plot(atria_edge2[1], atria_edge2[0], 'bo')
        plt.plot(_rvlv_point[1], _rvlv_point[0], 'k*')
        plt.plot(_free_rv_point[1], _free_rv_point[0], 'b*')

    # Rotate contours by theta degrees
    radians = np.arctan2(np.array((atria_edge1[0] - atria_edge2[0]) / 2),
                         np.array((atria_edge1[1] - atria_edge2[1]) / 2))

    # Rotate contours
    _x = _contours_RA[:, 1]
    y = _contours_RA[:, 0]
    xx_B = _x * math.cos(radians) + y * math.sin(radians)
    yy_B = -_x * math.sin(radians) + y * math.cos(radians)

    # Rotate points
    x_1 = atria_edge1[1]
    y_1 = atria_edge1[0]
    x_2 = atria_edge2[1]
    y_2 = atria_edge2[0]
    x_4 = top_atria[1]
    y_4 = top_atria[0]
    x_5 = mid_valve_RA[1]
    y_5 = mid_valve_RA[0]

    xx_1 = x_1 * math.cos(radians) + y_1 * math.sin(radians)
    yy_1 = -x_1 * math.sin(radians) + y_1 * math.cos(radians)
    xx_2 = x_2 * math.cos(radians) + y_2 * math.sin(radians)
    yy_2 = -x_2 * math.sin(radians) + y_2 * math.cos(radians)
    xx_4 = x_4 * math.cos(radians) + y_4 * math.sin(radians)
    yy_4 = -x_4 * math.sin(radians) + y_4 * math.cos(radians)
    xx_5 = x_5 * math.cos(radians) + y_5 * math.sin(radians)
    yy_5 = -x_5 * math.sin(radians) + y_5 * math.cos(radians)

    # make vertical line through mid_valve_from_atriumcontours_rot
    contours_RA_rot = np.asarray([xx_B, yy_B]).T
    top_atria_rot = np.asarray([xx_4, yy_4])

    # Make more points for the contours.
    intpl_XX = []
    intpl_YY = []
    for ind, coords in enumerate(contours_RA_rot):
        coords1 = coords
        if ind < (len(contours_RA_rot) - 1):
            coords2 = contours_RA_rot[ind + 1]

        else:
            coords2 = contours_RA_rot[0]
        warnings.simplefilter('ignore', np.RankWarning)
        coeff = np.polyfit([coords1[0], coords2[0]], [coords1[1], coords2[1]], 1)
        xx_es = np.linspace(coords1[0], coords2[0], 10)
        intp_val = np.polyval(coeff, xx_es)
        intpl_XX = np.hstack([intpl_XX, xx_es])
        intpl_YY = np.hstack([intpl_YY, intp_val])

    contour_smth = np.vstack([intpl_XX, intpl_YY]).T

    # find the crossing between vert_line and contours_RA_rot.
    dist2 = distance.cdist(contour_smth, [top_atria_rot])
    min_dist2 = np.min(dist2)
    # # step_closer
    newy_atra = top_atria_rot[1] + min_dist2
    new_top_atria = [top_atria_rot[0], newy_atra]
    dist3 = distance.cdist(contour_smth, [new_top_atria])
    ind_min_dist3 = dist3.argmin()

    ind_alt_atria_top = contours_RA_rot[:, 1].argmin()
    final_mid_avalve = np.asarray([xx_5, yy_5])
    final_top_atria = np.asarray([contours_RA_rot[ind_alt_atria_top, 0], contours_RA_rot[ind_alt_atria_top, 1]])
    final_perp_top_atria = contour_smth[ind_min_dist3, :]
    final_atrial_edge1 = np.asarray([xx_1, yy_1])
    final_atrial_edge2 = np.asarray([xx_2, yy_2])

    if debug:
        plt.figure()
        plt.plot(contour_smth[:, 0], contour_smth[:, 1], 'r-')
        plt.plot(final_atrial_edge2[0], final_atrial_edge2[1], 'y*')
        plt.plot(final_atrial_edge1[0], final_atrial_edge1[1], 'm*')
        plt.plot(final_top_atria[0], final_top_atria[1], 'c*')
        plt.plot(final_mid_avalve[0], final_mid_avalve[1], 'b*')
        plt.title('RA 4Ch frame {}'.format(_fr))

    alength_top = distance.pdist([final_mid_avalve, final_top_atria])[0]
    alength_perp = distance.pdist([final_mid_avalve, final_perp_top_atria])[0]
    a_segmts = (final_mid_avalve[1] - final_top_atria[1]) / Nsegments_length

    # get length dimension (width) of atrial seg at each place.
    a_diams = np.zeros(Nsegments_length)
    diam1 = abs(np.diff([xx_1, xx_2]))
    points_aux = np.zeros(((Nsegments_length - 1) * 2, 2))
    k = 0
    for ib in range(Nsegments_length):
        if ib == 0:
            a_diams[ib] = diam1
        else:
            vert_y = final_mid_avalve[1] - a_segmts * ib
            rgne_vertY = a_segmts / 6
            min_Y = vert_y - rgne_vertY
            max_Y = vert_y + rgne_vertY
            ind_sel_conts = np.where(np.logical_and(intpl_YY >= min_Y, intpl_YY <= max_Y))[0]

            if len(ind_sel_conts) == 0:
                logger.error('Problem in disk {}'.format(ib))
                continue

            y_sel_conts = contour_smth[ind_sel_conts, 1]
            x_sel_conts = contour_smth[ind_sel_conts, 0]
            min_ys = np.argmin(np.abs(y_sel_conts - vert_y))

            p1 = ind_sel_conts[min_ys]
            point1 = contour_smth[p1]

            mean_x = np.mean([np.min(x_sel_conts), np.max(x_sel_conts)])
            if mean_x < point1[0]:
                ind_xs = np.where(contour_smth[ind_sel_conts, 0] < mean_x)[0]
                pts = contour_smth[ind_sel_conts[ind_xs], :]
                min_ys = np.argmin(np.abs(pts[:, 1] - vert_y))
                point2 = pts[min_ys]
                a_diam = distance.pdist([point1, point2])[0]
            elif np.min(x_sel_conts) == np.max(x_sel_conts):
                logger.info('Frame {}, disk {} diameter is zero'.format(_fr, ib))
                a_diam = 0
                point2 = np.zeros(2)
                point1 = np.zeros(2)
            else:
                ind_xs = np.where(contour_smth[ind_sel_conts, 0] > mean_x)[0]
                if len(ind_xs) > 0:
                    pts = contour_smth[ind_sel_conts[ind_xs], :]
                    min_ys = np.argmin(np.abs(pts[:, 1] - vert_y))
                    point2 = pts[min_ys]
                    a_diam = distance.pdist([point1, point2])[0]
                else:
                    a_diam = 0
                    point2 = np.zeros(2)
                    point1 = np.zeros(2)
                    logger.info('la_4Ch: Frame {}, disk {} diameter is zero'.format(_fr, ib))

            a_diams[ib] = a_diam
            points_aux[k, :] = point1
            points_aux[k + 1, :] = point2

            k += 2

    points_rotate = np.zeros(((Nsegments_length - 1) * 2 + 5, 2))
    points_rotate[0, :] = final_mid_avalve
    points_rotate[1, :] = final_top_atria
    points_rotate[2, :] = final_perp_top_atria
    points_rotate[3, :] = final_atrial_edge1
    points_rotate[4, :] = final_atrial_edge2
    points_rotate[5:, :] = points_aux

    radians2 = 2 * np.pi - radians
    points_non_roatate_ = np.zeros_like(points_rotate)
    for _jj, p in enumerate(points_non_roatate_):
        points_non_roatate_[_jj, 0] = points_rotate[_jj, 0] * math.cos(radians2) + points_rotate[_jj, 1] * math.sin(
            radians2)
        points_non_roatate_[_jj, 1] = -points_rotate[_jj, 0] * math.sin(radians2) + points_rotate[_jj, 1] * math.cos(
            radians2)

    length_apex = distance.pdist([_apex_RV, _free_rv_point])
    if debug:
        plt.close('all')
    return a_diams, alength_top, alength_perp, points_non_roatate_, _contours_RA, length_apex


def get_left_atrial_volumes(seg, _seq, _fr, _points, logger):
    """
    This function gets the centre line (height) of the atrium and atrial dimension at 15 points along this line.
    """
    _apex, _mid_valve, anterior_2Ch, inferior_2Ch = _points
    if debug:
        plt.figure()
        plt.imshow(seg)
        plt.plot(_apex[1], _apex[0], 'mo')
        plt.plot(_mid_valve[1], _mid_valve[0], 'c*')
        plt.plot(anterior_2Ch[1], anterior_2Ch[0], 'y*')
        plt.plot(inferior_2Ch[1], inferior_2Ch[0], 'r*')

    if _seq == 'la_2Ch':
        _atria_seg = np.squeeze(seg == 3).astype(float)  # get atria label
    else:
        _atria_seg = np.squeeze(seg == 4).astype(float)  # get atria label

    # Generate contours from the atria
    contours = measure.find_contours(_atria_seg, 0.8)
    contours = contours[0]

    # Compute distance between mid_valve and every point in contours
    dist = distance.cdist(contours, [_mid_valve])
    ind_mitral_valve = dist.argmin()
    _mid_valve = contours[ind_mitral_valve, :]
    dist = distance.cdist(contours, [contours[ind_mitral_valve, :]])
    ind_top_atria = dist.argmax()
    top_atria = contours[ind_top_atria, :]
    length_apex_mid_valve = distance.pdist([_apex, _mid_valve])
    length_apex_inferior_2Ch = distance.pdist([_apex, inferior_2Ch])
    length_apex_anterior_2Ch = distance.pdist([_apex, anterior_2Ch])
    lines_LV_ = np.concatenate([length_apex_mid_valve, length_apex_inferior_2Ch, length_apex_anterior_2Ch])
    points_LV_ = np.vstack([_apex, _mid_valve, inferior_2Ch, anterior_2Ch])

    ind_base1 = distance.cdist(contours, [inferior_2Ch]).argmin()
    ind_base2 = distance.cdist(contours, [anterior_2Ch]).argmin()
    atria_edge1 = contours[ind_base1, :]
    atria_edge2 = contours[ind_base2, :]
    # mid valve based on atria
    x_mid_valve_atria = atria_edge1[0] + ((atria_edge2[0] - atria_edge1[0]) / 2)
    y_mid_valve_atria = atria_edge1[1] + ((atria_edge2[1] - atria_edge1[1]) / 2)
    mid_valve_atria = np.array([x_mid_valve_atria, y_mid_valve_atria])
    ind_mid_valve = distance.cdist(contours, [mid_valve_atria]).argmin()
    mid_valve_atria = contours[ind_mid_valve, :]

    if debug:
        plt.figure()
        plt.imshow(seg)
        plt.plot(top_atria[1], top_atria[0], 'mo')
        plt.plot(mid_valve_atria[1], mid_valve_atria[0], 'c*')
        plt.plot(atria_edge1[1], atria_edge1[0], 'y*')
        plt.plot(atria_edge2[1], atria_edge2[0], 'r*')

    # Rotate contours by theta degrees
    radians = np.arctan2(np.array((atria_edge2[0] - atria_edge1[0]) / 2),
                         np.array((atria_edge2[1] - atria_edge1[1]) / 2))

    # Rotate contours
    _x = contours[:, 1]
    y = contours[:, 0]
    xx_B = _x * math.cos(radians) + y * math.sin(radians)
    yy_B = -_x * math.sin(radians) + y * math.cos(radians)

    # Rotate points
    x_1 = atria_edge1[1]
    y_1 = atria_edge1[0]
    x_2 = atria_edge2[1]
    y_2 = atria_edge2[0]
    x_4 = top_atria[1]
    y_4 = top_atria[0]
    x_5 = mid_valve_atria[1]
    y_5 = mid_valve_atria[0]

    xx_1 = x_1 * math.cos(radians) + y_1 * math.sin(radians)
    yy_1 = -x_1 * math.sin(radians) + y_1 * math.cos(radians)
    xx_2 = x_2 * math.cos(radians) + y_2 * math.sin(radians)
    yy_2 = -x_2 * math.sin(radians) + y_2 * math.cos(radians)
    xx_4 = x_4 * math.cos(radians) + y_4 * math.sin(radians)
    yy_4 = -x_4 * math.sin(radians) + y_4 * math.cos(radians)
    xx_5 = x_5 * math.cos(radians) + y_5 * math.sin(radians)
    yy_5 = -x_5 * math.sin(radians) + y_5 * math.cos(radians)

    # make vertical line through mid_valve_from_atrium
    contours_rot = np.asarray([xx_B, yy_B]).T
    top_atria_rot = np.asarray([xx_4, yy_4])

    # Make more points for the contours.
    intpl_XX = []
    intpl_YY = []
    for ind, coords in enumerate(contours_rot):
        coords1 = coords
        if ind < (len(contours_rot) - 1):
            coords2 = contours_rot[ind + 1]
        else:
            coords2 = contours_rot[0]
        warnings.simplefilter('ignore', np.RankWarning)
        coeff = np.polyfit([coords1[0], coords2[0]], [coords1[1], coords2[1]], 1)
        xx_es = np.linspace(coords1[0], coords2[0], 10)
        intp_val = np.polyval(coeff, xx_es)
        intpl_XX = np.hstack([intpl_XX, xx_es])
        intpl_YY = np.hstack([intpl_YY, intp_val])

    contour_smth = np.vstack([intpl_XX, intpl_YY]).T

    # find the crossing between vert_line and contours_rot.
    dist2 = distance.cdist(contour_smth, [top_atria_rot])
    min_dist2 = np.min(dist2)
    newy_atra = top_atria_rot[1] + min_dist2
    new_top_atria = [top_atria_rot[0], newy_atra]
    dist3 = distance.cdist(contour_smth, [new_top_atria])
    ind_min_dist3 = dist3.argmin()

    ind_alt_atria_top = contours_rot[:, 1].argmin()
    final_top_atria = np.asarray([contours_rot[ind_alt_atria_top, 0], contours_rot[ind_alt_atria_top, 1]])
    final_perp_top_atria = contour_smth[ind_min_dist3, :]
    final_atrial_edge1 = np.asarray([xx_1, yy_1])
    final_atrial_edge2 = np.asarray([xx_2, yy_2])
    final_mid_avalve = np.asarray([xx_5, yy_5])

    if debug:
        plt.figure()
        plt.plot(contour_smth[:, 0], contour_smth[:, 1], 'r-')
        plt.plot(final_atrial_edge2[0], final_atrial_edge2[1], 'y*')
        plt.plot(final_atrial_edge1[0], final_atrial_edge1[1], 'm*')
        plt.plot(final_perp_top_atria[0], final_perp_top_atria[1], 'ko')
        plt.plot(final_top_atria[0], final_top_atria[1], 'c*')
        plt.plot(new_top_atria[0], new_top_atria[1], 'g*')
        plt.plot(final_mid_avalve[0], final_mid_avalve[1], 'b*')
        plt.title('LA {}  frame {}'.format(_seq, _fr))

    # now find length of atrium divide in the  15 segments
    alength_top = distance.pdist([final_mid_avalve, final_top_atria])[0]
    alength_perp = distance.pdist([final_mid_avalve, final_perp_top_atria])[0]
    a_segmts = (final_mid_avalve[1] - final_top_atria[1]) / Nsegments_length

    a_diams = np.zeros(Nsegments_length)
    diam1 = abs(np.diff([xx_1, xx_2]))
    points_aux = np.zeros(((Nsegments_length - 1) * 2, 2))
    k = 0
    for ib in range(Nsegments_length):
        if ib == 0:
            a_diams[ib] = diam1
        else:
            vert_y = final_mid_avalve[1] - a_segmts * ib
            rgne_vertY = a_segmts / 6
            min_Y = vert_y - rgne_vertY
            max_Y = vert_y + rgne_vertY
            ind_sel_conts = np.where(np.logical_and(intpl_YY >= min_Y, intpl_YY <= max_Y))[0]

            if len(ind_sel_conts) == 0:
                logger.info('Problem in disk {}'.format(ib))
                continue

            y_sel_conts = contour_smth[ind_sel_conts, 1]
            x_sel_conts = contour_smth[ind_sel_conts, 0]
            min_ys = np.argmin(np.abs(y_sel_conts - vert_y))

            p1 = ind_sel_conts[min_ys]
            point1 = contour_smth[p1]

            mean_x = np.mean([np.min(x_sel_conts), np.max(x_sel_conts)])
            if mean_x < point1[0]:
                ind_xs = np.where(contour_smth[ind_sel_conts, 0] < mean_x)[0]
                pts = contour_smth[ind_sel_conts[ind_xs], :]
                min_ys = np.argmin(np.abs(pts[:, 1] - vert_y))
                point2 = pts[min_ys]
                a_diam = distance.pdist([point1, point2])[0]

            elif np.min(x_sel_conts) == np.max(x_sel_conts):
                logger.info('Frame {}, disk {} diameter is zero'.format(_fr, ib))
                a_diam = 0
                point2 = np.zeros(2)
                point1 = np.zeros(2)
            else:
                ind_xs = np.where(contour_smth[ind_sel_conts, 0] > mean_x)[0]
                if len(ind_xs) > 0:
                    pts = contour_smth[ind_sel_conts[ind_xs], :]
                    min_ys = np.argmin(np.abs(pts[:, 1] - vert_y))
                    point2 = pts[min_ys]
                    a_diam = distance.pdist([point1, point2])[0]

                else:
                    a_diam = 0
                    point2 = np.zeros(2)
                    point1 = np.zeros(2)
                    logger.info('la_4Ch - Frame {}, disk {} diameter is zero'.format(_fr, ib))

            a_diams[ib] = a_diam
            points_aux[k, :] = point1
            points_aux[k + 1, :] = point2

            k += 2

    points_rotate = np.zeros(((Nsegments_length - 1) * 2 + 5, 2))
    points_rotate[0, :] = final_mid_avalve
    points_rotate[1, :] = final_top_atria
    points_rotate[2, :] = final_perp_top_atria
    points_rotate[3, :] = final_atrial_edge1
    points_rotate[4, :] = final_atrial_edge2
    points_rotate[5:, :] = points_aux

    radians2 = 2 * np.pi - radians
    points_non_roatate_ = np.zeros_like(points_rotate)
    for _jj, p in enumerate(points_non_roatate_):
        points_non_roatate_[_jj, 0] = points_rotate[_jj, 0] * math.cos(radians2) + points_rotate[_jj, 1] * math.sin(
            radians2)
        points_non_roatate_[_jj, 1] = -points_rotate[_jj, 0] * math.sin(radians2) + points_rotate[_jj, 1] * math.cos(
            radians2)
    if debug:
        plt.close('all')
    return a_diams, alength_top, alength_perp, points_non_roatate_, contours, lines_LV_, points_LV_


def detect_LV_points(seg, logger):
    myo_seg = np.squeeze(seg == 2).astype(float)
    kernel = np.ones((2, 2), np.uint8)
    myo_seg_dil = cv2.dilate(myo_seg, kernel, iterations=2)
    myo2 = get_processed_myocardium(myo_seg_dil, _label=1)
    cl_pts, _mid_valve = get_sorted_sk_pts(myo2, logger)
    dist_myo = distance.cdist(cl_pts, [_mid_valve])
    ind_apex = dist_myo.argmax()
    _apex = cl_pts[ind_apex, :]
    _septal_mv = cl_pts[0, 0], cl_pts[0, 1]
    _ant_mv = cl_pts[-1, 0], cl_pts[-1, 1]

    return np.asarray(_apex), np.asarray(_mid_valve), np.asarray(_septal_mv), np.asarray(_ant_mv)


def get_processed_myocardium(seg, _label=2):
    """
    This function tidies the LV myocardial segmentation, taking only the single
    largest connected component, and performing an opening (erosion+dilation)
    """

    myo_aux = np.squeeze(seg == _label).astype(float)  # get myocardial label
    myo_aux = closing(myo_aux, structure=np.ones((2, 2))).astype(float)
    cc_aux = measure.label(myo_aux, connectivity=1)
    ncc_aux = len(np.unique(cc_aux))

    if not ncc_aux <= 1:
        cc_counts, cc_inds = np.histogram(cc_aux, range(ncc_aux + 1))
        cc_inds = cc_inds[:-1]
        cc_inds_sorted = [_x for (y, _x) in sorted(zip(cc_counts, cc_inds))]
        biggest_cc_ind = cc_inds_sorted[-2]  # Take second largest CC (after background)
        myo_aux = closing(myo_aux, structure=np.ones((2, 2))).astype(float)

        # Take largest connected component
        if not (len(np.where(cc_aux > 0)[0]) == len(np.where(cc_aux == biggest_cc_ind)[0])):
            mask = cc_aux == biggest_cc_ind
            myo_aux *= mask
            myo_aux = closing(myo_aux).astype(float)

    return myo_aux


def get_sorted_sk_pts(myo, logger, n_samples=48, centroid=np.array([0, 0])):
    #   ref -       reference start point for spline point ordering
    #   n_samples  output number of points for sampling spline

    # check for side branches? need connectivity check
    sk_im = skeletonize(myo)

    myo_pts = np.asarray(np.nonzero(myo)).transpose()
    sk_pts = np.asarray(np.nonzero(sk_im)).transpose()

    # convert to radial coordinates and sort circumferential
    if centroid[0] == 0 and centroid[1] == 0:
        centroid = np.mean(sk_pts, axis=0)

    # get skeleton consisting only of longest path
    sk_im = get_longest_path(sk_im, logger)

    # sort centreline points based from boundary points at valves as start
    # and end point. Make ref point out of LV through valve
    out = skeleton_endpoints(sk_im.astype(int))
    end_pts = np.asarray(np.nonzero(out)).transpose()
    sk_pts = np.asarray(np.nonzero(sk_im)).transpose()

    if len(end_pts) > 2:
        logger.info('Error! More than 2 end-points in LA myocardial skeleton.')
        cl_pts = []
        _mid_valve = []
        return cl_pts, _mid_valve
    else:
        # set reference to vector pointing from centroid to mid-valve
        _mid_valve = np.mean(end_pts, axis=0)
        ref = (_mid_valve - centroid) / norm(_mid_valve - centroid)
        sk_pts2 = sk_pts - centroid  # centre around centroid
        myo_pts2 = myo_pts - centroid
        theta = np.zeros([len(sk_pts2), ])
        theta_myo = np.zeros([len(myo_pts2), ])

        eps = 0.0001
        if len(sk_pts2) <= 5:
            logger.info('Skeleton failed! Only of length {}'.format(len(sk_pts2)))
            cl_pts = []
            _mid_valve = []
            return cl_pts, _mid_valve
        else:
            # compute angle theta for skeleton points
            for k, ss in enumerate(sk_pts2):
                if (np.dot(ref, ss) / norm(ss) < 1.0 + eps) and (np.dot(ref, ss) / norm(ss) > 1.0 - eps):
                    theta[k] = 0
                elif (np.dot(ref, ss) / norm(ss) < -1.0 + eps) and (np.dot(ref, ss) / norm(ss) > -1.0 - eps):
                    theta[k] = 180
                else:
                    theta[k] = math.acos(np.dot(ref, ss) / norm(ss)) * 180 / np.pi
                detp = ref[0] * ss[1] - ref[1] * ss[0]
                if detp > 0:
                    theta[k] = 360 - theta[k]
            thinds = theta.argsort()
            sk_pts = sk_pts[thinds, :].astype(float)  # ordered centreline points

            # # compute angle theta for myo points
            for k, ss in enumerate(myo_pts2):
                # compute angle theta
                eps = 0.0001
                if (np.dot(ref, ss) / norm(ss) < 1.0 + eps) and (np.dot(ref, ss) / norm(ss) > 1.0 - eps):
                    theta_myo[k] = 0
                elif (np.dot(ref, ss) / norm(ss) < -1.0 + eps) and (np.dot(ref, ss) / norm(ss) > -1.0 - eps):
                    theta_myo[k] = 180
                else:
                    theta_myo[k] = math.acos(np.dot(ref, ss) / norm(ss)) * 180 / np.pi
                detp = ref[0] * ss[1] - ref[1] * ss[0]
                if detp > 0:
                    theta_myo[k] = 360 - theta_myo[k]
            # sub-sample and order myo points circumferential
            theta_myo.sort()

            # Remove duplicates
            sk_pts = binarymatrix(sk_pts)
            # fit b-spline curve to skeleton, sample fixed number of points
            tck, u = splprep(sk_pts.T, s=10.0, nest=-1, quiet=2)
            u_new = np.linspace(u.min(), u.max(), n_samples)
            cl_pts = np.zeros([n_samples, 2])
            cl_pts[:, 0], cl_pts[:, 1] = splev(u_new, tck)

            # get centreline theta
            cl_theta = np.zeros([len(cl_pts), ])
            cl_pts2 = cl_pts - centroid  # centre around centroid
            for k, ss in enumerate(cl_pts2):
                # compute angle theta
                if (np.dot(ref, ss) / norm(ss) < 1.0 + eps) and (np.dot(ref, ss) / norm(ss) > 1.0 - eps):
                    cl_theta[k] = 0
                else:
                    cl_theta[k] = math.acos(np.dot(ref, ss) / norm(ss)) * 180 / np.pi
                detp = ref[0] * ss[1] - ref[1] * ss[0]
                if detp > 0:
                    cl_theta[k] = 360 - cl_theta[k]
            cl_theta.sort()
            return cl_pts, _mid_valve


def get_longest_path(skel, logger):
    # first create edges from skeleton
    sk_im = skel.copy()
    # remove bad (L-shaped) junctions
    sk_im = remove_bad_junctions(sk_im, logger)

    # get seeds for longest path from existing end-points
    out = skeleton_endpoints(sk_im.astype(int))
    end_pts = np.asarray(np.nonzero(out)).transpose()
    if len(end_pts) == 0:
        logger.info('ERROR! No end-points detected! Exiting.')
    # break
    elif len(end_pts) == 1:
        logger.info('Warning! Only 1 end-point detected!')
    elif len(end_pts) > 2:
        logger.info('Warning! {} end-points detected!'.format(len(end_pts)))

    sk_pts = np.asarray(np.nonzero(sk_im)).transpose()
    # search indices of sk_pts for end points
    tmp_inds = np.ravel_multi_index(sk_pts.T, (np.max(sk_pts[:, 0]) + 1, np.max(sk_pts[:, 1]) + 1))
    seed_inds = np.zeros((len(end_pts), 1))
    for i, e in enumerate(end_pts):
        seed_inds[i] = int(
            np.where(tmp_inds == np.ravel_multi_index(e.T, (np.max(sk_pts[:, 0]) + 1, np.max(sk_pts[:, 1]) + 1)))[0])
    sk_im_inds = np.zeros_like(sk_im, dtype=int)

    for i, p in enumerate(sk_pts):
        sk_im_inds[p[0], p[1]] = i

    kernel1 = np.uint8([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])
    edges = []
    for i, p in enumerate(sk_pts):
        mask = sk_im_inds[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2]
        o = np.multiply(kernel1, mask)
        for c in o[o > 0]:
            edges.append(['{}'.format(i), '{}'.format(c)])
    # create graph
    G = defaultdict(list)
    for (ss, t) in edges:
        if t not in G[ss]:
            G[ss].append(t)
        if ss not in G[t]:
            G[t].append(ss)
    # find max path
    max_path = []
    for j in range(len(seed_inds)):
        all_paths = depth_first_search(G, str(int(seed_inds[j][0])))
        max_path2 = max(all_paths, key=lambda l: len(l))
        if len(max_path2) > len(max_path):
            max_path = max_path2
    # create new image only with max path
    sk_im_maxp = np.zeros_like(sk_im, dtype=int)
    for j in max_path:
        p = sk_pts[int(j)]
        sk_im_maxp[p[0], p[1]] = 1
    return sk_im_maxp


def skeleton_endpoints(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)

    # now look through to find the value of 11
    out = np.zeros_like(skel)
    out[np.where(filtered == 11)] = 1

    return out


def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def detect_RV_points(_seg, septal_mv, logger):
    rv_seg = np.squeeze(_seg == 3).astype(float)

    sk_pts = measure.find_contours(rv_seg, 0.8)
    if len(sk_pts) > 1:
        nb_pts = []
        for ll in range(len(sk_pts)):
            nb_pts.append(len(sk_pts[ll]))
        sk_pts = sk_pts[np.argmax(nb_pts)]
    sk_pts = np.squeeze(sk_pts)
    sk_pts = np.unique(sk_pts, axis=0)
    centroid = np.mean(sk_pts, axis=0)

    _lv_valve = closest_node(np.squeeze(septal_mv), sk_pts)
    ref = (_lv_valve - centroid) / norm(_lv_valve - centroid)

    sk_pts2 = sk_pts - centroid  # centre around centroid
    theta = np.zeros([len(sk_pts2), ])

    eps = 0.0001
    if len(sk_pts2) <= 5:
        logger.info('Skeleton failed! Only of length {}'.format(len(sk_pts2)))
        _cl_pts = []
    else:
        # compute angle theta for skeleton points
        for k, ss in enumerate(sk_pts2):
            if (np.dot(ref, ss) / norm(ss) < 1.0 + eps) and (np.dot(ref, ss) / norm(ss) > 1.0 - eps):
                theta[k] = 0
            elif (np.dot(ref, ss) / norm(ss) < -1.0 + eps) and (np.dot(ref, ss) / norm(ss) > -1.0 - eps):
                theta[k] = 180
            else:
                theta[k] = math.acos(np.dot(ref, ss) / norm(ss)) * 180 / np.pi
            detp = ref[0] * ss[1] - ref[1] * ss[0]
            if detp > 0:
                theta[k] = 360 - theta[k]
        thinds = theta.argsort()
        sk_pts = sk_pts[thinds, :].astype(float)  # ordered centreline points

        # Remove duplicates
        sk_pts = binarymatrix(sk_pts)
        # fit b-spline curve to skeleton, sample fixed number of points
        tck, u = splprep(sk_pts.T, s=10.0, per=1, quiet=2)

        u_new = np.linspace(u.min(), u.max(), 80)
        _cl_pts = np.zeros([80, 2])
        _cl_pts[:, 0], _cl_pts[:, 1] = splev(u_new, tck)

    dist_rv = distance.cdist(_cl_pts, [_lv_valve])
    _ind_apex = dist_rv.argmax()
    _apex_RV = _cl_pts[_ind_apex, :]

    m = np.diff(_cl_pts[:, 0]) / np.diff(_cl_pts[:, 1])
    angle = np.arctan(m) * 180 / np.pi
    idx = np.sign(angle)
    _ind_free_wall = np.where(idx == -1)[0]

    _area = 10000 * np.ones(len(_ind_free_wall))
    for ai, ind in enumerate(_ind_free_wall):
        AB = np.linalg.norm(_lv_valve - _apex_RV)
        BC = np.linalg.norm(_lv_valve - _cl_pts[ind, :])
        AC = np.linalg.norm(_cl_pts[ind, :] - _apex_RV)
        if AC > 10 and BC > 10:
            _area[ai] = np.abs(AB ** 2 + BC ** 2 - AC ** 2)
    _free_rv_point = _cl_pts[_ind_free_wall[_area.argmin()], :]

    return np.asarray(_apex_RV), np.asarray(_lv_valve), np.asarray(_free_rv_point)


def remove_bad_junctions(skel, logger):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    # kernel_A used for unnecessary nodes in L-shaped junctions (retain diags)
    kernels_A = [np.uint8([[0, 1, 0],
                           [1, 10, 1],
                           [0, 1, 0]])]
    src_depth = -1
    for k in kernels_A:
        filtered = cv2.filter2D(skel, src_depth, k)
        skel[filtered >= 13] = 0
        if len(np.where(filtered == 14)[0]) > 0:
            logger.info('Warning! You have a 3x3 loop!')

    return skel


def depth_first_search(G, v, seen=None, path=None):
    if seen is None:
        seen = []
    if path is None:
        path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(depth_first_search(G, t, seen, t_path))
    return paths



def compute_atria_MAPSE_TAPSE_params(study_ID, subject_dir, results_dir, logger):
    window_size, poly_order = 7, 3
    QC_atria_2Ch, QC_atria_4Ch = 0, 0

    # la_2Ch
    filename_la_seg_2Ch = os.path.join(subject_dir, 'la_2Ch_seg_nnUnet.nii.gz')
    if os.path.exists(filename_la_seg_2Ch):
        nim = nib.load(filename_la_seg_2Ch)
        la_seg_2Ch = nim.get_fdata()
        dx, dy, dz = nim.header['pixdim'][1:4]
        area_per_voxel = dx * dy
        if len(la_seg_2Ch.shape) == 4:
            la_seg_2Ch = la_seg_2Ch[:, :, 0, :]
        X, Y, N_frames_2Ch = la_seg_2Ch.shape
        # =============================================================================
        # Params
        # =============================================================================
        area_LV_2Ch = np.zeros(N_frames_2Ch)
        length_LV_2Ch = np.zeros(N_frames_2Ch)
        la_diams_2Ch = np.zeros((N_frames_2Ch, Nsegments_length))
        length_top_2Ch = np.zeros(N_frames_2Ch)
        length_perp_2Ch = np.zeros(N_frames_2Ch)
        LV_mid_mapse_2Ch = np.zeros(N_frames_2Ch)
        LV_sept_mapse_2Ch = np.zeros(N_frames_2Ch)
        LV_anterior_mapse_2Ch = np.zeros(N_frames_2Ch)
        points_LV_2Ch = np.zeros((N_frames_2Ch, 4, 2))
        # =============================================================================
        # Get largest connected components
        # =============================================================================
        for fr in range(N_frames_2Ch):
            la_seg_2Ch[:, :, fr] = getLargestCC(la_seg_2Ch[:, :, fr])

        # =============================================================================
        # Compute area
        # =============================================================================
        for fr in range(N_frames_2Ch):
            area_LV_2Ch[fr] = np.sum(
                np.squeeze(la_seg_2Ch[:, :, fr] == 3).astype(float)) * area_per_voxel  # get atria label
        # =============================================================================
        # Compute simpson's rule
        # =============================================================================
        for fr in range(N_frames_2Ch):
            try:
                apex, mid_valve, anterior, inferior = detect_LV_points(la_seg_2Ch[:, :, fr],logger)
                points = np.vstack([apex, mid_valve, anterior, inferior])
                points_LV_2Ch[fr, :] = points
            except Exception:
                logger.error('Problem detecting LV points {} in la_2Ch fr {}'.format(study_ID, fr))
                QC_atria_2Ch = 1

            if QC_atria_2Ch == 0:
                # =============================================================================
                # 2Ch
                # =============================================================================
                try:
                    la_dia, lentop, lenperp, points_non_roatate, contours_LA, lines_LV, points_LV = \
                        get_left_atrial_volumes(la_seg_2Ch[:, :, fr], 'la_2Ch', fr, points,logger)
                    la_diams_2Ch[fr, :] = la_dia * dx
                    length_top_2Ch[fr] = lentop * dx
                    length_perp_2Ch[fr] = lenperp * dx

                    LV_mid_mapse_2Ch[fr] = lines_LV[0] * dx  # length_apex_mid_valve
                    LV_sept_mapse_2Ch[fr] = lines_LV[1] * dx  # length_apex_inferior_2Ch
                    LV_anterior_mapse_2Ch[fr] = lines_LV[2]* dx   # length_apex_anterior_2Ch
                    LV_atria_points_2Ch = np.zeros((9, 2))
                    LV_atria_points_2Ch[0, :] = points_non_roatate[0, :]  # final_mid_avalve
                    LV_atria_points_2Ch[1, :] = points_non_roatate[1, :]  # final_top_atria
                    LV_atria_points_2Ch[2, :] = points_non_roatate[2, :]  # final_perp_top_atria
                    LV_atria_points_2Ch[3, :] = points_non_roatate[3, :]  # final_atrial_edge1
                    LV_atria_points_2Ch[4, :] = points_non_roatate[4, :]  # final_atrial_edge2
                    LV_atria_points_2Ch[5, :] = points_LV[0, :]  # apex
                    LV_atria_points_2Ch[6, :] = points_LV[1, :]  # mid_valve
                    LV_atria_points_2Ch[7, :] = points_LV[2, :]  # inferior_2Ch
                    LV_atria_points_2Ch[8, :] = points_LV[3, :]  # anterior_2Ch

                except Exception:
                    logger.error('Problem in disk-making with subject {} in la_2Ch fr {}'.
                                        format(study_ID, fr))
                    QC_atria_2Ch = 1
        # =============================================================================
        # MPASE/TAPSE
        # =============================================================================
        LV_mid_mapse_2Ch = LV_mid_mapse_2Ch[0] - LV_mid_mapse_2Ch
        LV_sept_mapse_2Ch = LV_sept_mapse_2Ch[0] - LV_sept_mapse_2Ch  # inferior_2Ch/septal_2Ch
        LV_anterior_mapse_2Ch = LV_anterior_mapse_2Ch[0] - LV_anterior_mapse_2Ch  # anterior_2Ch/lateral_2Ch

        x = np.linspace(0, N_frames_2Ch - 1, N_frames_2Ch)
        xx = np.linspace(np.min(x), np.max(x), N_frames_2Ch)
        itp = interp1d(x, LV_mid_mapse_2Ch)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LV_mid_mapse_2Ch_smooth = yy_sg

        itp = interp1d(x, LV_sept_mapse_2Ch)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LV_sept_mapse_2Ch_smooth = yy_sg

        itp = interp1d(x, LV_anterior_mapse_2Ch)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LV_anterior_mapse_2Ch_smooth = yy_sg

        np.savetxt(os.path.join(results_dir, 'LV_ant_mapse_2Ch.txt'), LV_anterior_mapse_2Ch)
        np.savetxt(os.path.join(results_dir, 'LV_sept_mapse_2Ch.txt'), LV_sept_mapse_2Ch)
        np.savetxt(os.path.join(results_dir, 'LV_mid_mapse_2Ch.txt'), LV_mid_mapse_2Ch)

        np.savetxt(os.path.join(results_dir, 'LV_ant_mapse_smooth_2Ch.txt'), LV_anterior_mapse_2Ch_smooth)
        np.savetxt(os.path.join(results_dir, 'LV_sept_mapse_smooth_2Ch.txt'), LV_sept_mapse_2Ch_smooth)
        np.savetxt(os.path.join(results_dir, 'LV_mid_mapse_smooth_2Ch.txt'), LV_mid_mapse_2Ch_smooth)

    else:
        QC_atria_2Ch = 1
        LV_anterior_mapse_2Ch = np.zeros(20)
        LV_sept_mapse_2Ch = np.zeros(20)
        LV_mid_mapse_2Ch = np.zeros(20)
        LV_anterior_mapse_2Ch_smooth = np.zeros(20)
        LV_sept_mapse_2Ch_smooth = np.zeros(20)
        LV_mid_mapse_2Ch_smooth = np.zeros(20)
        np.savetxt(os.path.join(results_dir, 'LV_ant_mapse_2Ch.txt'), LV_anterior_mapse_2Ch)
        np.savetxt(os.path.join(results_dir, 'LV_sept_mapse_2Ch.txt'), LV_sept_mapse_2Ch)
        np.savetxt(os.path.join(results_dir, 'LV_mid_mapse_2Ch.txt'), LV_mid_mapse_2Ch)
        np.savetxt(os.path.join(results_dir, 'LV_ant_mapse_smooth_2Ch.txt'), LV_anterior_mapse_2Ch_smooth)
        np.savetxt(os.path.join(results_dir, 'LV_sept_mapse_smooth_2Ch.txt'), LV_sept_mapse_2Ch_smooth)
        np.savetxt(os.path.join(results_dir, 'LV_mid_mapse_smooth_2Ch.txt'), LV_mid_mapse_2Ch_smooth)
        
     # la_4Ch
    filename_la_seg_4Ch = os.path.join(subject_dir, 'la_4Ch_seg_nnUnet.nii.gz')
    if os.path.exists(filename_la_seg_4Ch):
        nim = nib.load(filename_la_seg_4Ch)
        la_seg_4Ch = nim.get_fdata()
        dx, dy, dz = nim.header['pixdim'][1:4]
        area_per_voxel = dx * dy
        if len(la_seg_4Ch.shape) == 4:
            la_seg_4Ch = la_seg_4Ch[:, :, 0, :]
        la_seg_4Ch = np.transpose(la_seg_4Ch, [1, 0, 2])
        X, Y, N_frames_4Ch = la_seg_4Ch.shape
        # =============================================================================
        # Params
        # =============================================================================
        area_LV_4Ch = np.zeros(N_frames_4Ch)
        length_LV_4Ch = np.zeros(N_frames_4Ch)
        la_diams_4Ch = np.zeros((N_frames_4Ch, Nsegments_length)) 
        length_top_4Ch = np.zeros(N_frames_4Ch)
        length_perp_4Ch = np.zeros(N_frames_4Ch)
        LV_mid_mapse_4Ch = np.zeros(N_frames_4Ch)
        LV_sept_mapse_4Ch = np.zeros(N_frames_4Ch)
        LV_anterior_mapse_4Ch = np.zeros(N_frames_4Ch)
        RA_tapse_seq = np.zeros(N_frames_4Ch)

        points_LV_4Ch = np.zeros((N_frames_4Ch, 4, 2))
        points_RV_4Ch = np.zeros((N_frames_4Ch, 3, 2))

        # RA params
        la_diams_RV = np.zeros((N_frames_4Ch, Nsegments_length))  # LA_4Ch
        length_top_RV = np.zeros(N_frames_4Ch)  # LA_4Ch
        length_perp_RV = np.zeros(N_frames_4Ch)  # LA_4Ch
        area_RV = np.zeros(N_frames_4Ch)  # LA_4Ch

        # =============================================================================
        # Get largest connected components
        # =============================================================================
        for fr in range(N_frames_4Ch):
            la_seg_4Ch[:, :, fr] = getLargestCC(la_seg_4Ch[:, :, fr])

        # =============================================================================
        # Compute area
        # =============================================================================
        for fr in range(N_frames_4Ch):
            area_LV_4Ch[fr] = np.sum(
                np.squeeze(la_seg_4Ch[:, :, fr] == 4).astype(float)) * area_per_voxel  # get atria label
            area_RV[fr] = np.sum(np.squeeze(la_seg_4Ch[:, :, fr] == 5).astype(float)) * area_per_voxel  # in mm2
        # =============================================================================
        # Compute simpson's rule
        # =============================================================================
        for fr in range(N_frames_4Ch):
            try:
                apex, mid_valve, anterior, inferior = detect_LV_points(la_seg_4Ch[:, :, fr],logger)
                points = np.vstack([apex, mid_valve, anterior, inferior])
                apex_RV, rvlv_point, free_rv_point = detect_RV_points(la_seg_4Ch[:, :, fr], anterior, logger)
                pointsRV = np.vstack([apex_RV, rvlv_point, free_rv_point])
                points_LV_4Ch[fr, :] = points
                points_RV_4Ch[fr, :] = pointsRV
            except Exception:
                logger.error('Problem detecting LV or RV points {} in la_4Ch fr {}'.format(study_ID, fr))
                QC_atria_4Ch = 1

            if QC_atria_4Ch == 0:
                try:
                    la_dia, lentop, lenperp, points_non_roatate, contours_LA, lines_LV, points_LV = \
                        get_left_atrial_volumes(la_seg_4Ch[:, :, fr], 'la_4Ch', fr, points,logger)
                    la_diams_4Ch[fr, :] = la_dia * dx
                    length_top_4Ch[fr] = lentop * dx
                    length_perp_4Ch[fr] = lenperp * dx

                    LV_mid_mapse_4Ch[fr] = lines_LV[0]  # length_apex_mid_valve
                    LV_sept_mapse_4Ch[fr] = lines_LV[1]  # length_apex_inferior_4Ch
                    LV_anterior_mapse_4Ch[fr] = lines_LV[2]  # length_apex_anterior_4Ch
                    LV_atria_points_4Ch = np.zeros((9, 2))
                    LV_atria_points_4Ch[0, :] = points_non_roatate[0, :]  # final_mid_avalve
                    LV_atria_points_4Ch[1, :] = points_non_roatate[1, :]  # final_top_atria
                    LV_atria_points_4Ch[2, :] = points_non_roatate[2, :]  # final_perp_top_atria
                    LV_atria_points_4Ch[3, :] = points_non_roatate[3, :]  # final_atrial_edge1
                    LV_atria_points_4Ch[4, :] = points_non_roatate[4, :]  # final_atrial_edge2
                    LV_atria_points_4Ch[5, :] = points_LV[0, :]  # apex
                    LV_atria_points_4Ch[6, :] = points_LV[1, :]  # mid_valve
                    LV_atria_points_4Ch[7, :] = points_LV[2, :]  # lateral_4Ch
                    LV_atria_points_4Ch[8, :] = points_LV[3, :]  # septal_4Ch

                except Exception:
                    logger.error('Problem in disk-making with subject {} in la_4Ch fr {}'.
                                        format(study_ID, fr))
                    QC_atria_4Ch = 1

                try:
                    la_dia, lentop, lenperp, points_non_roatate, contours_RA, RA_tapse_seq[fr] = \
                        get_right_atrial_volumes(la_seg_4Ch[:, :, fr], fr, pointsRV, logger)

                    la_diams_RV[fr, :] = la_dia * dx
                    length_top_RV[fr] = lentop * dx
                    length_perp_RV[fr] = lenperp * dx

                    RV_atria_points_4Ch = np.zeros((8, 2))
                    RV_atria_points_4Ch[0, :] = points_non_roatate[0, :]  # final_mid_avalve
                    RV_atria_points_4Ch[1, :] = points_non_roatate[1, :]  # final_top_atria
                    RV_atria_points_4Ch[2, :] = points_non_roatate[2, :]  # final_perp_top_atria
                    RV_atria_points_4Ch[3, :] = points_non_roatate[3, :]  # final_atrial_edge1
                    RV_atria_points_4Ch[4, :] = points_non_roatate[4, :]  # final_atrial_edge2
                    RV_atria_points_4Ch[5, :] = pointsRV[0, :]  # apex_RV
                    RV_atria_points_4Ch[6, :] = pointsRV[1, :]  # rvlv_point
                    RV_atria_points_4Ch[7, :] = pointsRV[2, :]  # free_rv_point
                except Exception:
                    logger.error(
                        'RV Problem in disk-making with subject {} in la_4Ch fr {}'.format(study_ID, fr))
                    QC_atria_4Ch = 1

        # =============================================================================
        # MPASE/TAPSE
        # ============================================================================
        LV_mid_mapse_4Ch = LV_mid_mapse_4Ch[0] - LV_mid_mapse_4Ch
        LV_sept_mapse_4Ch = LV_sept_mapse_4Ch[0] - LV_sept_mapse_4Ch  # septal_4Ch
        LV_anterior_mapse_4Ch = LV_anterior_mapse_4Ch[0] - LV_anterior_mapse_4Ch  # lateral_4Ch
        RA_tapse = RA_tapse_seq[0] - RA_tapse_seq

        x = np.linspace(0, N_frames_4Ch - 1, N_frames_4Ch)
        xx = np.linspace(np.min(x), np.max(x), N_frames_4Ch)
        itp = interp1d(x, LV_mid_mapse_4Ch)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LV_mid_mapse_4Ch_smooth = yy_sg

        itp = interp1d(x, LV_sept_mapse_4Ch)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LV_sept_mapse_4Ch_smooth = yy_sg

        itp = interp1d(x, LV_anterior_mapse_4Ch)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LV_anterior_mapse_4Ch_smooth = yy_sg

        np.savetxt(os.path.join(results_dir, 'LV_ant_mapse_4Ch.txt'), LV_anterior_mapse_4Ch)
        np.savetxt(os.path.join(results_dir, 'LV_sept_mapse_4Ch.txt'), LV_sept_mapse_4Ch)
        np.savetxt(os.path.join(results_dir, 'LV_mid_mapse_4Ch.txt'), LV_mid_mapse_4Ch)

        np.savetxt(os.path.join(results_dir, 'LV_ant_mapse_smooth_4Ch.txt'), LV_anterior_mapse_4Ch_smooth)
        np.savetxt(os.path.join(results_dir, 'LV_sept_mapse_smooth_4Ch.txt'), LV_sept_mapse_4Ch_smooth)
        np.savetxt(os.path.join(results_dir, 'LV_mid_mapse_smooth_4Ch.txt'), LV_mid_mapse_4Ch_smooth)

        itp = interp1d(x, RA_tapse)
        RA_tapse_smooth = savgol_filter(itp(xx), window_size, poly_order)
        np.savetxt(os.path.join(results_dir, 'RA_tapse_smooth_la4Ch.txt'), RA_tapse_smooth)
        np.savetxt(os.path.join(results_dir, 'RA_tapse_la4Ch.txt'), RA_tapse)
        
    else:
        QC_atria_4Ch = 1
        LV_anterior_mapse_4Ch = -1*np.ones(20)
        LV_sept_mapse_4Ch =  -1*np.ones(20)
        LV_mid_mapse_4Ch =  -1*np.ones(20)
        RA_tapse =  -1*np.ones(20)
        LV_anterior_mapse_4Ch_smooth =  -1*np.ones(20)
        LV_sept_mapse_4Ch_smooth =  -1*np.ones(20)
        LV_mid_mapse_4Ch_smooth =  -1*np.ones(20)
        RA_tapse_smooth = -1*np.ones(20)
        np.savetxt(os.path.join(results_dir, 'LV_ant_mapse_4Ch.txt'), LV_anterior_mapse_4Ch)
        np.savetxt(os.path.join(results_dir, 'LV_sept_mapse_4Ch.txt'), LV_sept_mapse_4Ch)
        np.savetxt(os.path.join(results_dir, 'LV_mid_mapse_4Ch.txt'), LV_mid_mapse_4Ch)
        np.savetxt(os.path.join(results_dir, 'LV_ant_mapse_smooth_4Ch.txt'), LV_anterior_mapse_4Ch_smooth)
        np.savetxt(os.path.join(results_dir, 'LV_sept_mapse_smooth_4Ch.txt'), LV_sept_mapse_4Ch_smooth)
        np.savetxt(os.path.join(results_dir, 'LV_mid_mapse_smooth_4Ch.txt'), LV_mid_mapse_4Ch_smooth)
        np.savetxt(os.path.join(results_dir, 'RA_tapse_smooth_la4Ch.txt'), RA_tapse_smooth)
        np.savetxt(os.path.join(results_dir, 'RA_tapse_la4Ch.txt'), RA_tapse)


    if QC_atria_2Ch == 0:
        # =============================================================================
        # Save points
        # =============================================================================
        np.save(os.path.join(results_dir, '{}_LV_atria_points_2Ch'.format(study_ID)), LV_atria_points_2Ch)
        np.save(os.path.join(results_dir, 'points_LV_2Ch'), points_LV_2Ch)
        
        # =============================================================================
        # Compute volumes
        # =============================================================================
        LA_volumes_2Ch = np.zeros(N_frames_2Ch)
        for fr in range(N_frames_2Ch):
            d1d2 = la_diams_2Ch[fr, :] * la_diams_2Ch[fr, :]
            length = np.min([length_top_2Ch[fr], length_top_2Ch[fr]])
            LA_volumes_2Ch[fr] = math.pi / 4 * length * np.sum(d1d2) / Nsegments_length / 1000

        x = np.linspace(0, N_frames_2Ch - 1, N_frames_2Ch)
        xx = np.linspace(np.min(x), np.max(x), N_frames_2Ch)
        itp = interp1d(x, LA_volumes_2Ch)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LA_volumes_2Ch_smooth = yy_sg

        np.savetxt(os.path.join(results_dir, 'LA_volumes_2Ch.txt'), LA_volumes_2Ch)
        np.savetxt(os.path.join(results_dir, 'LA_volumes_2Ch_smooth.txt'), LA_volumes_2Ch_smooth)
    else:
        LA_volumes_2Ch_smooth = np.zeros(20)
    
    if QC_atria_4Ch == 0:
        # =============================================================================
        # Save points
        # =============================================================================
        np.save(os.path.join(results_dir, '{}_LV_atria_points_4Ch'.format(study_ID)), LV_atria_points_4Ch)
        np.save(os.path.join(results_dir, '{}_RV_atria_points_4Ch'.format(study_ID)), RV_atria_points_4Ch)
        np.save(os.path.join(results_dir, 'points_LV_4Ch'), points_LV_4Ch)
        np.save(os.path.join(results_dir, 'points_RV_4Ch'), points_RV_4Ch)

        # =============================================================================
        # Compute volumes
        # =============================================================================
        # LA volumes
        LA_volumes_4Ch = np.zeros(N_frames_4Ch)
        for fr in range(N_frames_4Ch):
            d1d2 = la_diams_4Ch[fr, :] * la_diams_4Ch[fr, :]
            length = np.min([length_top_4Ch[fr], length_top_4Ch[fr]])
            LA_volumes_4Ch[fr] = math.pi / 4 * length * np.sum(d1d2) / Nsegments_length / 1000
        
        # RA volumes
        RA_volumes_SR = np.zeros(N_frames_4Ch)
        RA_volumes_area = np.zeros(N_frames_4Ch)

        for fr in range(N_frames_4Ch):
            d1d2 = la_diams_RV[fr, :] * la_diams_RV[fr, :]
            length = length_top_RV[fr]
            RA_volumes_SR[fr] = math.pi / 4 * length * np.sum(d1d2) / Nsegments_length / 1000
            RA_volumes_area[fr] = 0.85 * area_RV[fr] * area_RV[fr] / length / 1000

        x = np.linspace(0, N_frames_4Ch - 1, N_frames_4Ch)
        xx = np.linspace(np.min(x), np.max(x), N_frames_4Ch)
        itp = interp1d(x, LA_volumes_4Ch)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LA_volumes_4Ch_smooth = yy_sg

        np.savetxt(os.path.join(results_dir, 'LA_volumes_4Ch.txt'), LA_volumes_4Ch)
        np.savetxt(os.path.join(results_dir, 'LA_volumes_4Ch_smooth.txt'), LA_volumes_4Ch_smooth)

        x = np.linspace(0, N_frames_4Ch - 1, N_frames_4Ch)
        xx = np.linspace(np.min(x), np.max(x), N_frames_4Ch)
        itp = interp1d(x, RA_volumes_SR)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        RA_volumes_SR_smooth = yy_sg
        itp = interp1d(x, RA_volumes_area)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        RA_volumes_area_smooth = yy_sg

        np.savetxt(os.path.join(results_dir, 'RA_volumes_SR.txt'), RA_volumes_SR)
        np.savetxt(os.path.join(results_dir, 'RA_volumes_area.txt'), RA_volumes_area)
        np.savetxt(os.path.join(results_dir, 'RA_volumes_SR_smooth.txt'), RA_volumes_SR_smooth)
        np.savetxt(os.path.join(results_dir, 'RA_volumes_area_smooth.txt'), RA_volumes_area_smooth)
    else:
        LA_volumes_4Ch_smooth = np.zeros(20)
        RA_volumes_area_smooth = np.zeros(20)
        RA_volumes_SR_smooth = np.zeros(20)

    if QC_atria_4Ch == 0 and QC_atria_2Ch == 0 and N_frames_2Ch == N_frames_4Ch:
        # =============================================================================
        # Compute volumes
        # =============================================================================
        LA_volumes_SR = np.zeros(N_frames_4Ch)
        LA_volumes_area = np.zeros(N_frames_4Ch)
        
        for fr in range(N_frames_4Ch):
            d1d2 = la_diams_2Ch[fr, :] * la_diams_4Ch[fr, :]
            length = np.min([length_top_2Ch[fr], length_top_4Ch[fr]])
            LA_volumes_SR[fr] = math.pi / 4 * length * np.sum(d1d2) / Nsegments_length / 1000

        # Area
        if N_frames_2Ch == N_frames_4Ch:
            LA_volumes_area[fr] = 0.85 * area_LV_2Ch[fr] * area_LV_4Ch[fr] / length / 1000

        x = np.linspace(0, N_frames_4Ch - 1, N_frames_4Ch)
        xx = np.linspace(np.min(x), np.max(x), N_frames_4Ch)
        itp = interp1d(x, LA_volumes_SR)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LA_volumes_SR_smooth = yy_sg

        itp = interp1d(x, LA_volumes_area)
        yy_sg = savgol_filter(itp(xx), window_size, poly_order)
        LA_volumes_area_smooth = yy_sg

        np.savetxt(os.path.join(results_dir, 'LA_volumes_SR.txt'), LA_volumes_SR)
        np.savetxt(os.path.join(results_dir, 'LA_volumes_area.txt'), LA_volumes_area)
        np.savetxt(os.path.join(results_dir, 'LA_volumes_SR_smooth.txt'), LA_volumes_SR_smooth)
        np.savetxt(os.path.join(results_dir, 'LA_volumes_area_smooth.txt'), LA_volumes_area_smooth)

    # Simpson and Area method if not the same number of slices between methods
    elif QC_atria_4Ch == 0 and QC_atria_2Ch == 0 and N_frames_2Ch != N_frames_4Ch:
            max_frames = max(N_frames_2Ch, N_frames_4Ch)
            length_top_2Ch_itp = resample(length_top_2Ch, max_frames)
            length_top_4Ch_itp = resample(length_top_4Ch, max_frames)
            area_LV_2Ch_itp = resample(area_LV_2Ch, max_frames)
            area_LV_4Ch_itp = resample(area_LV_4Ch, max_frames)
            la_diams_2Ch_itp = resample(la_diams_2Ch, max_frames)
            la_diams_4Ch_itp = resample(la_diams_4Ch, max_frames)

            LA_volumes_SR = np.zeros(max_frames)
            LA_volumes_area = np.zeros(max_frames)

            # calculate simpson and area
            for fr in range(max_frames):
                d1d2 = la_diams_2Ch_itp[fr, :] * la_diams_4Ch_itp[fr, :]
                length = np.min([length_top_2Ch_itp[fr], length_top_4Ch_itp[fr]])
                LA_volumes_SR[fr] = math.pi / 4 * length * np.sum(d1d2) / Nsegments_length / 1000
            
            if N_frames_2Ch == N_frames_4Ch:
                LA_volumes_area[fr] = 0.85 * area_LV_2Ch_itp[fr] * area_LV_4Ch_itp[fr] / length / 1000

            x = np.linspace(0, max_frames - 1, max_frames)
            xx = np.linspace(np.min(x), np.max(x), max_frames)
            itp = interp1d(x, LA_volumes_SR)
            yy_sg = savgol_filter(itp(xx), window_size, poly_order)
            LA_volumes_SR_smooth = yy_sg

            itp = interp1d(x, LA_volumes_area)
            yy_sg = savgol_filter(itp(xx), window_size, poly_order)
            LA_volumes_area_smooth = yy_sg

            np.savetxt(os.path.join(results_dir, 'LA_volumes_SR.txt'), LA_volumes_SR)
            np.savetxt(os.path.join(results_dir, 'LA_volumes_area.txt'), LA_volumes_area)
            np.savetxt(os.path.join(results_dir, 'LA_volumes_SR_smooth.txt'), LA_volumes_SR_smooth)
            np.savetxt(os.path.join(results_dir, 'LA_volumes_area_smooth.txt'), LA_volumes_area_smooth)
    else:
        LA_volumes_SR_smooth = np.zeros(20)
        LA_volumes_area_smooth = np.zeros(20)

    # =============================================================================
    # PLOTS
    # =============================================================================
    plt.figure()
    plt.plot(LA_volumes_2Ch_smooth, label='LA volumes 2Ch')
    plt.plot(LA_volumes_4Ch_smooth, label='LA volumes 4Ch')
    plt.plot(LA_volumes_SR_smooth, label='Simpson method')
    plt.plot(LA_volumes_area_smooth, label='Area method')
    plt.legend()
    plt.title('Left Atrial Volume')
    plt.savefig(os.path.join(results_dir, 'LA_volume_area.png'))
    plt.close('all')

    plt.figure()
    plt.plot(RA_volumes_SR_smooth, label='Simpson method')
    plt.plot(RA_volumes_area_smooth, label='Area method')
    plt.legend()
    plt.title('Right Atrial Volume')
    plt.savefig(os.path.join(results_dir, 'RA_volume_area.png'))
    plt.close('all')

    # =============================================================================
    # Compute condsuit and reservoir
    # =============================================================================
    if QC_atria_4Ch == 0 and QC_atria_2Ch == 0:
        try:
            LAmax = np.max(LA_volumes_SR_smooth)
            ES_frame_LA = LA_volumes_SR_smooth.argmax()
            LAmin = np.min(LA_volumes_SR_smooth)
            vol_first_deriv = np.gradient(LA_volumes_SR_smooth[::2])
            indx_local_max = argrelextrema(vol_first_deriv, np.greater)
            if len(indx_local_max[0]) > 1:
                indx_local_max = np.squeeze(np.asarray(indx_local_max))
            elif len(indx_local_max[0]) == 1:
                indx_local_max = indx_local_max[0]

            indx_local_max = np.squeeze(np.asarray(indx_local_max[indx_local_max > int(ES_frame_LA / 2)])) * 2
            if indx_local_max.size > 0:
                LA_reservoir = np.mean(LAmax - LA_volumes_SR_smooth[indx_local_max])
                LA_reservoir_point = int(np.mean(indx_local_max))
                LA_pump_point = np.argmin(LA_volumes_SR_smooth[LA_reservoir_point:]) + LA_reservoir_point
                LA_pump = LA_volumes_SR_smooth[LA_reservoir_point] - LA_volumes_SR_smooth[LA_pump_point]

                fig, ax = plt.subplots()
                ax.plot(LA_volumes_SR_smooth)
                ax.plot(LA_reservoir_point, LA_volumes_SR_smooth[LA_reservoir_point], 'ro')
                ax.annotate('LA_reservoir', (LA_reservoir_point, LA_volumes_SR_smooth[LA_reservoir_point]))
                ax.plot(ES_frame_LA, LAmax, 'ro')
                ax.annotate('LA max', (ES_frame_LA, LAmax))
                ax.plot(LA_pump_point, LA_volumes_SR_smooth[LA_pump_point], 'ro')
                ax.annotate('LA pump', (LA_pump_point, LA_volumes_SR_smooth[LA_pump_point]))
                ax.set_title('{}: LAV'.format(study_ID))
                plt.savefig(os.path.join(results_dir, 'LA_volume_points.png'))
                plt.close('all')
            else:
                QC_atria_4Ch = 1
                LAmax = -1
                LAmin =  -1
                LA_reservoir  =  -1
                LA_pump =  -1
                LA_reservoir_point =  -1
                LA_pump_point =  -1
        except Exception:
            logger.error('Problem in calculating LA conduit a with subject {}'.format(study_ID))
            QC_atria_4Ch = 1
            LAmax = -1
            LAmin =  -1
            LA_reservoir  =  -1
            LA_pump =  -1
            LA_reservoir_point =  -1
            LA_pump_point =  -1
    
    else:
        LAmax = -1
        LAmin =  -1
        LA_reservoir  =  -1
        LA_pump =  -1
        LA_reservoir_point =  -1
        LA_pump_point =  -1
    
    # =============================================================================
    # Compute condsuit and reservoir RV
    # =============================================================================
    if QC_atria_4Ch == 0:
        try:
            RAmax = np.max(RA_volumes_SR_smooth)
            ES_frame = RA_volumes_SR_smooth.argmax()
            RAmin = np.min(RA_volumes_SR_smooth)
            vol_first_deriv = np.gradient(RA_volumes_SR_smooth[::2])
            indx_local_max = argrelextrema(vol_first_deriv, np.greater)
            if len(indx_local_max[0]) > 1:
                indx_local_max = np.squeeze(np.asarray(indx_local_max))
            elif len(indx_local_max[0]) == 1:
                indx_local_max = indx_local_max[0]

            indx_local_max = np.squeeze(np.asarray(indx_local_max[indx_local_max > int(ES_frame / 2)])) * 2
            if indx_local_max.size > 0:
                RA_reservoir = np.mean(RAmax - RA_volumes_SR_smooth[indx_local_max])
                RA_reservoir_point = int(np.mean(indx_local_max))
                RA_pump_point = np.argmin(RA_volumes_SR_smooth[RA_reservoir_point:]) + RA_reservoir_point

                RA_pump = RA_volumes_SR_smooth[RA_reservoir_point] - RA_volumes_SR_smooth[RA_pump_point]

                fig, ax = plt.subplots()
                ax.plot(RA_volumes_SR_smooth)
                ax.plot(RA_reservoir_point, RA_volumes_SR_smooth[RA_reservoir_point], 'ro')
                ax.annotate('RA_reservoir', (RA_reservoir_point, RA_volumes_SR_smooth[RA_reservoir_point]))
                ax.plot(ES_frame, RAmax, 'ro')
                ax.annotate('RA max', (ES_frame, RAmax))
                ax.plot(RA_pump_point, RA_volumes_SR_smooth[RA_pump_point], 'ro')
                ax.annotate('RA pump', (RA_pump_point, RA_volumes_SR_smooth[RA_pump_point]))
                ax.set_title('{}: RAV'.format(study_ID))
                plt.savefig(os.path.join(results_dir, 'RA_volume_points.png'))
                plt.close('all')
            else:
                QC_atria_4Ch = 1
                RAmax = -1
                RAmin = -1
                RA_reservoir = -1
                RA_pump  = -1
                RA_reservoir_point = -1
                RA_pump_point = -1
        except Exception:
            logger.error('Problem in calculating RA conduit a with subject {}'.format(study_ID))
            QC_atria_4Ch = 1
            RAmax = -1
            RAmin = -1
            RA_reservoir = -1
            RA_pump  = -1
            RA_reservoir_point = -1
            RA_pump_point = -1
    else:
        RAmax = -1
        RAmin = -1
        RA_reservoir = -1
        RA_pump  = -1
        RA_reservoir_point = -1
        RA_pump_point = -1
    # =============================================================================
    # MAPSE
    # =============================================================================
    try:
        f, ax = plt.subplots()
        f2, ax2 = plt.subplots()
        ax.plot(LV_sept_mapse_2Ch, label='Septal 2Ch MAPSE')
        ax.plot(LV_anterior_mapse_2Ch, label='Ant 2Ch MAPSE')
        ax.plot(LV_mid_mapse_2Ch, label='Mid 2Ch MAPSE')
        ax.plot(LV_sept_mapse_4Ch, label='Septal 4Ch MAPSE')
        ax.plot(LV_anterior_mapse_4Ch, label='Ant 4Ch MAPSE')
        ax.plot(LV_mid_mapse_4Ch, label='Mid 4Ch MAPSE')

        ax2.plot(LV_sept_mapse_2Ch_smooth, label='Septal 2Ch MAPSE')
        ax2.plot(LV_anterior_mapse_2Ch_smooth, label='Ant 2Ch MAPSE')
        ax2.plot(LV_mid_mapse_2Ch_smooth, label='Mid 2Ch MAPSE')
        ax2.plot(LV_sept_mapse_4Ch_smooth, label='Septal 4Ch MAPSE')
        ax2.plot(LV_anterior_mapse_4Ch_smooth, label='Ant 4Ch MAPSE')
        ax2.plot(LV_mid_mapse_4Ch_smooth, label='Mid 4Ch MAPSE')

        ax.legend()
        ax.set_title('MAPSE')
        f.savefig(os.path.join(results_dir, 'MAPSE.png'))

        ax2.legend()
        ax2.set_title('MAPSE smooth')
        f2.savefig(os.path.join(results_dir, 'MAPSE_smooth.png'))
        plt.close('all')

        f, ax = plt.subplots()
        ax.plot(RA_tapse_smooth)
        ax.set_title('TAPSE_smooth.png')
        f.savefig(os.path.join(results_dir, 'TAPSE.png'))
        plt.close('all')

        f, ax = plt.subplots()
        ax.plot(RA_tapse_smooth)
        ax.set_title('{}: TAPSE'.format(study_ID))
        ax.plot(RA_tapse_smooth.argmax(), RA_tapse_smooth[RA_tapse_smooth.argmax()], 'ro')
        ax.annotate('TAPSE', (RA_tapse_smooth.argmax(), RA_tapse_smooth[RA_tapse_smooth.argmax()]))
        f.savefig(os.path.join(results_dir, 'TAPSE_final.png'))
        plt.close('all')

        f, ax = plt.subplots()
        ax.plot(LV_mid_mapse_2Ch_smooth)
        ax.plot(LV_mid_mapse_2Ch_smooth.argmax(), LV_mid_mapse_2Ch_smooth[LV_mid_mapse_2Ch_smooth.argmax()], 'ro')
        ax.annotate('MAPSE', (LV_mid_mapse_2Ch_smooth.argmax(), LV_mid_mapse_2Ch_smooth[LV_mid_mapse_2Ch_smooth.argmax()]))
        ax.set_title('{}: MAPSE'.format(study_ID))
        f.savefig(os.path.join(results_dir, 'MAPSE_final.png'))
        plt.close('all')

    except Exception:
        logger.error('Problem in calculating MAPSE/TAPSE with subject {}'.format(study_ID))
        QC_atria_4Ch = 1
   
   # =============================================================================
    # SAVE RESULTS
    # =============================================================================
    vols = -1*np.ones(39, dtype=object)
    vols[0] = study_ID
    vols[1] = LAmin  # LA min simpsons
    vols[2] = LAmax  # LA max simpsons
    vols[3] = np.min(LA_volumes_area_smooth)  # LA min area
    vols[4] = np.max(LA_volumes_area_smooth)  # LA max area
    vols[9] = LA_reservoir  # LA reservoir
    vols[10] = LA_pump  # LA pump
    vols[11] = LA_volumes_SR_smooth.argmin()  # LA reservoir
    vols[12] = LA_volumes_SR_smooth.argmax()  # LA pump
    vols[13] = LA_reservoir_point  # LA reservoir
    vols[14] = LA_pump_point  # LA pump

    vols[5] = np.min(LA_volumes_2Ch_smooth)  # LA min 2Ch
    vols[6] = np.max(LA_volumes_2Ch_smooth)  # LA max 2Ch
    vols[7] = np.min(LA_volumes_4Ch_smooth)  # LA min 4Ch
    vols[8] = np.max(LA_volumes_4Ch_smooth)  # LA max 4Ch

    vols[15] = RAmin  # RA min simpsons
    vols[16] = RAmax  # RA max simpsons
    vols[17] = np.min(RA_volumes_area_smooth)  # RA min area
    vols[18] = np.max(RA_volumes_area_smooth)  # RA max area
    vols[19] = RA_reservoir  # RA reservoir
    vols[20] = RA_pump  # RA pump

    vols[21] = RA_volumes_SR_smooth.argmin()  # LA reservoir
    vols[22] = RA_volumes_SR_smooth.argmax()  # LA pump
    vols[23] = RA_reservoir_point  # LA reservoir
    vols[24] = RA_pump_point  # LA pump

    vols[25] = LV_sept_mapse_2Ch[0]  # LV 2Ch EDV sept_mapse
    vols[26] = np.max(LV_sept_mapse_2Ch_smooth)  # LV 2Ch ESV sept_mapse
    vols[27] = LV_mid_mapse_2Ch[0]  # LV 2Ch EDV mid_mapse
    vols[28] = np.max(LV_mid_mapse_2Ch_smooth)  # LV 2Ch ESV mid_mapse
    vols[29] = LV_anterior_mapse_2Ch[0]  # LV 2Ch EDV ant_mapse
    vols[30] = np.max(LV_anterior_mapse_2Ch_smooth)  # LV 2Ch ESV ant_mapse

    vols[31] = LV_sept_mapse_4Ch[0]  # LV 4Ch EDV sept_mapse
    vols[32] = np.max(LV_sept_mapse_4Ch_smooth)  # LV 4Ch ESV sept_mapse
    vols[33] = LV_mid_mapse_4Ch[0]  # LV 4Ch EDV mid_mapse
    vols[34] = np.max(LV_mid_mapse_4Ch_smooth)  # LV 4Ch ESV mid_mapse
    vols[35] = LV_anterior_mapse_4Ch[0]  # LV 4Ch EDV ant_mapse
    vols[36] = np.max(LV_anterior_mapse_4Ch_smooth)  # LV 4Ch ESV ant_mapse

    vols[37] = RA_tapse[0]  # RA 4Ch EDV ant_mapse
    vols[38] = np.max(RA_tapse_smooth)  # RA 4Ch ESV ant_mapse

    vols = np.reshape(vols, [1, 39])
    df = pd.DataFrame(vols)
    df.to_csv(os.path.join(results_dir, 'clinical_measure_atria.csv'),
            header=['eid', 'LAEDVsimp', 'LAESVsimp', 'LAEDVarea', 'LAESVarea', 'LAEDV 2Ch', 'LAESV 2Ch',
                    'LAEDV 4Ch', 'LAESV 4Ch', 'LARes', 'LAPump', 'point min LA', 'point max LA',
                    'point reservoir LA', 'point pump LA', 'RAEDVsimp',
                    'RAESVsimp', 'RAEDVarea', 'RAESVarea', 'RARes', 'RAPump', 'point min RA', 'point max RA',
                    'point reservoir RA', 'point pump RA',
                    'LA_EDVsept_mapse2Ch', 'LA_ESVsept_mapse2Ch',
                    'LA_EDVmid_mapse2Ch', 'LA_ESVmid_mapse2Ch', 'LA_EDVant_mapse2Ch', 'LA_ESVant_mapse2Ch',
                    'LA_EDVsept_mapse4Ch', 'LA_ESVsept_mapse4Ch', 'LA_EDVmid_mapse4Ch',
                    'LA_ESVmid_mapse4Ch',
                    'LA_EDVant_mapse4Ch', 'LA_ESVant_mapse4Ch', 'RA_EDV_tapse4Ch',
                    'RA_ESV_tapse4Ch'], index=False)
