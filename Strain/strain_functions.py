
import os
import nibabel as nib
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage.morphology import binary_closing as closing
from skimage.morphology import binary_dilation
from scipy.spatial import distance
import cv2
from numpy.linalg import norm
from scipy.spatial import distance as dd
from skimage import measure
from skimage.morphology import skeletonize
import math
from scipy.ndimage import binary_fill_holes, label, sum as ndsum
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from collections import Counter
from scipy import interpolate
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from collections import defaultdict



def compute_RV_cric_strain(strain_circ_RV,strain_circ_RV_raw, results_dir, dt):
    flag_circ = 0
    ################### FIND PEAK STRAIN ####################################
    SA_circ_str_sg = np.mean(strain_circ_RV, axis=0)
    SA_circ_str_raw = np.mean(strain_circ_RV_raw, axis=0)
    peaks_savgol, _ = signal.find_peaks(-SA_circ_str_sg)
    max_peak_ind_savgol = np.argmax(-SA_circ_str_sg[peaks_savgol])
    max_peak_savgol = peaks_savgol[max_peak_ind_savgol]

    ########## now find the right max
    if max_peak_savgol <= 0.7 * SA_circ_str_sg.shape[0]:
        def_max_point = [max_peak_savgol, SA_circ_str_sg[max_peak_savgol]]
    ############### ###############################
    elif max_peak_savgol > 0.7 * SA_circ_str_sg.shape[0] and len(peaks_savgol) > 1:
        # From the remaining peaks take the highest one
        peaks_savgol_2 = np.delete(peaks_savgol, max_peak_ind_savgol)
        max_peak_ind_2 = np.argmax(-SA_circ_str_sg[peaks_savgol_2])
        max_peak_savgol_2 = peaks_savgol_2[max_peak_ind_2]
        if max_peak_savgol_2 <= 0.7 * SA_circ_str_sg.shape[0]:
            max_peak_savgol = max_peak_savgol_2
            max_peak_ind = max_peak_ind_2
            def_max_point = [max_peak_savgol, SA_circ_str_sg[max_peak_savgol]]
        else:
            peaks_savgol_2 = np.delete(peaks_savgol_2, max_peak_ind_2)
            max_peak_ind = np.argmax(-SA_circ_str_sg[peaks_savgol_2])
            max_peak_savgol = peaks_savgol_2[max_peak_ind_2]
            def_max_point = [max_peak_savgol, SA_circ_str_sg[max_peak_savgol]]

    else:  # take the value, but flag it
        def_max_point = [max_peak_savgol, SA_circ_str_sg[max_peak_savgol]]
        flag_circ == 1
    
    np.savetxt('{0}/RV_circ_strain.txt'.format(results_dir), SA_circ_str_sg)

    # Find diastolic strain
    diast_strain_circ, TPK_diast_strain_circ, Ys_slope_circ, range_slope_circ, Y_on_curve_circ, min_idx_circ = get_diastolic(SA_circ_str_sg, def_max_point,dt)

    fig, ax = plt.subplots()
    ax.plot(SA_circ_str_sg, 'g', label='circ RV strain smooth')
    ax.plot(max_peak_savgol, SA_circ_str_sg[max_peak_savgol], "rs", label='SG_max')
    ax.plot(range_slope_circ, Ys_slope_circ, 'r')
    ax.plot(min_idx_circ, Y_on_curve_circ, 'rx', label='diastolic SR')
    ax.legend()
    plt.savefig(os.path.join(results_dir, 'RV_circ_strain.png'))
    plt.close()

    vols = -1*np.ones(5)
    vols[0] = SA_circ_str_sg[max_peak_savgol]
    vols[1] = max_peak_savgol
    vols[2] = diast_strain_circ
    vols[3] = TPK_diast_strain_circ
    vols[4] = flag_circ

    return vols



def get_points_sax_RV_circ_strain (img_dir, seg_dir, results_dir, N_points = 7):
    seg_sa = nib.load(seg_dir).get_fdata()
    img_sa = nib.load(img_dir).get_fdata()
    X, Y, Z, N_frames = img_sa.shape
    dt = nib.load(img_dir).header['pixdim'][4]
    _, _, _, N_frames = img_sa.shape
    strain_circ_RV = np.zeros((2, N_frames))
    strain_circ_RV_raw = np.zeros((2, N_frames))
    for idx2, sl in enumerate([int(Z/2)-1,int(Z/2)]):# Only 3 middle slices
        total_length_sa = np.zeros(N_frames)
        for fr in range(N_frames):
            img = img_sa[:, :, sl, fr]
            seg = getLargestCC(seg_sa[:, :, sl, fr])
            rv_seg = np.squeeze(seg == 3).astype(float)  # get atria label
            contours_RV = measure.find_contours(rv_seg, 0.8)
            contours_RV = measure.find_contours(rv_seg, 0.8)
            contours_RV = max(contours_RV, key = len)
            landmark_points = detect_intersection_pts(seg)

            dist = distance.cdist(contours_RV, [landmark_points[0,:]], 'euclidean')
            ind_P1 = dist.argmin()
            P1 = contours_RV[ind_P1, :] # free wall
            dist = distance.cdist(contours_RV, [landmark_points[1,:]], 'euclidean')
            ind_P2 = dist.argmin()
            P2= contours_RV[ind_P2, :] #close to the lungs

            # Detect points free wall
            if ind_P2 > ind_P1:
                points_free_wall = np.concatenate([contours_RV[ind_P2:],contours_RV[0:ind_P1]])
            else:
                # TO CHECK!
                points_free_wall = np.concatenate([contours_RV[ind_P1:],contours_RV[0:ind_P2]])

            # Sample to get N_points
            points_free_wall = binarymatrix(points_free_wall)
            # fit b-spline curve to skeleton, sample fixed number of points
            tck, u = splprep(points_free_wall.T, u=None, s=10.0, per=0, nest=-1, quiet=2)
            u_new = np.linspace(u.min(), u.max(), N_points)
            cl_pts = np.zeros([N_points, 2])
            cl_pts[:, 0], cl_pts[:, 1] = splev(u_new, tck, der=0)

            if fr == 0:
                f, ax = plt.subplots()
                ax.imshow(img, cmap='gray')
                ax.imshow(seg, alpha=0.3)
                ax.axis('off')
                # Hide axes ticks
                ax.set_xticks([])
                ax.set_yticks([])
                ax.plot(contours_RV[:, 1], contours_RV[:, 0], 'w.')
                ax.plot(points_free_wall[:, 1], points_free_wall[:, 0], 'ko')
                ax.plot(cl_pts[:, 1], cl_pts[:, 0], 'm.')
                ax.plot(P1[1],P1[0], 'ro')
                ax.plot(P2[1],P2[0], 'co')
                plt.savefig(os.path.join(results_dir, 'RV_circ_strain_seg_sl{0}.png'.format(sl)))
                plt.close()

            total_dist_fr = 0
            for xi in range(len(cl_pts) - 1):
                points_1 = cl_pts[xi]
                points_2 = cl_pts[xi + 1]
                ind_dist = distance.pdist([points_1, points_2])
                total_dist_fr = total_dist_fr + ind_dist
                total_length_sa[fr] = total_dist_fr

        # calculate strain
        SA_circ_str = np.zeros(N_frames)

        for fr in range(N_frames):
            SA_circ_str[fr] = (total_length_sa[fr] - total_length_sa[0]) / total_length_sa[0]
        SA_circ_str_psl = signal.savgol_filter(SA_circ_str, 11, 3)
        strain_circ_RV[idx2,:] = SA_circ_str_psl
        strain_circ_RV_raw[idx2,:] = SA_circ_str

    return strain_circ_RV, strain_circ_RV_raw, dt

def compute_RV_longit_strain(total_length_4Ch,  total_apex_free, total_apex_rvlv, N_frames_4Ch, results_dir, dt):
    flag_long = 0
    RV_long_str = np.zeros(N_frames_4Ch)
    RV_lin_str_free =np.zeros(N_frames_4Ch)
    RV_lin_str_rvlv =np.zeros(N_frames_4Ch)
    for fr in range(N_frames_4Ch):
        RV_long_str[fr] = (total_length_4Ch[fr] - total_length_4Ch[0]) / total_length_4Ch[0]
        RV_lin_str_free[fr] = (total_apex_free[fr] - total_apex_free[0]) / total_apex_free[0]
        RV_lin_str_rvlv[fr] = (total_apex_rvlv[fr] - total_apex_rvlv[0]) / total_apex_rvlv[0]

    average_len_strain = (np.array(RV_lin_str_free) + np.array(RV_lin_str_rvlv))/ 2.0

    RV_long_str_sg = signal.savgol_filter(RV_long_str, 11, 3)
    RV_lin_str_free_sg = signal.savgol_filter(RV_lin_str_free, 11, 3)
    RV_lin_str_rvlv_sg = signal.savgol_filter(RV_lin_str_rvlv, 11, 3)
    average_len_strain_sg = signal.savgol_filter(average_len_strain, 11, 3)

    # =============================================================================
    # FIND PEAK STRAIN
    # =============================================================================
    # Total average strain
    peaks_savgol, _ = signal.find_peaks(-average_len_strain_sg)
    max_peak_ind_savgol = np.argmax(-average_len_strain_sg[peaks_savgol])
    max_peak_savgol = peaks_savgol[max_peak_ind_savgol]

    # now find the right max
    if max_peak_savgol <= 0.7 * average_len_strain_sg.shape[0]:
        def_max_point = [max_peak_savgol, average_len_strain_sg[max_peak_savgol]]
    elif max_peak_savgol > 0.7 * average_len_strain_sg.shape[0] and len(peaks_savgol) > 1:
        # From the remaining peaks take the highest one
        peaks_savgol_2 = np.delete(peaks_savgol, max_peak_ind_savgol)
        max_peak_ind_2 = np.argmax(-average_len_strain_sg[peaks_savgol_2])
        max_peak_savgol_2 = peaks_savgol_2[max_peak_ind_2]
        if max_peak_savgol_2 <= 0.7 * average_len_strain_sg.shape[0]:
            max_peak_savgol = max_peak_savgol_2
            max_peak_ind = max_peak_ind_2
            def_max_point = [max_peak_savgol, average_len_strain_sg[max_peak_savgol]]
        else:
            peaks_savgol_2 = np.delete(peaks_savgol_2, max_peak_ind_2)
            max_peak_ind = np.argmax(-average_len_strain_sg[peaks_savgol_2])
            max_peak_savgol = peaks_savgol_2[max_peak_ind_2]
            def_max_point = [max_peak_savgol, average_len_strain_sg[max_peak_savgol]]

    else:  # take the value, but flag it
        def_max_point = [max_peak_savgol, average_len_strain_sg[max_peak_savgol]]
        flag_long = 1

    # get diastolic strainrate
    diast_strain_av, TPK_diast_strain_av, Ys_slope_av, range_slope_av, Y_on_curve_av, min_idx_av = get_diastolic(average_len_strain_sg, def_max_point,dt)

    ########## free wall#########
    peaks_savgol_fw, _ = signal.find_peaks(-RV_lin_str_free_sg)
    max_peak_ind_savgol_fw = np.argmax(-RV_lin_str_free_sg[peaks_savgol_fw])
    max_peak_savgol_fw = peaks_savgol[max_peak_ind_savgol_fw]

    ## now find the right max
    if max_peak_savgol_fw <= 0.7 * RV_lin_str_free_sg.shape[0]:
        def_max_point_fw = [max_peak_savgol_fw, RV_lin_str_free_sg[max_peak_savgol_fw]]
    elif max_peak_savgol_fw > 0.7 * RV_lin_str_free_sg.shape[0] and len(peaks_savgol_fw) > 1:
        # From the remaining peaks take the highest one
        peaks_savgol_2 = np.delete(peaks_savgol_fw, max_peak_ind_savgol_fw)
        max_peak_ind_2 = np.argmax(-RV_lin_str_free_sg[peaks_savgol_2])
        max_peak_savgol_2 = peaks_savgol_2[max_peak_ind_2]
        if max_peak_savgol_2 <= 0.7 * RV_lin_str_free_sg.shape[0]:
            max_peak_savgol_fw = max_peak_savgol_2
            max_peak_ind_fw = max_peak_ind_2
            def_max_point_fw = [max_peak_savgol_fw, RV_lin_str_free_sg[max_peak_savgol_fw]]
        else:
            peaks_savgol_2 = np.delete(peaks_savgol_2, max_peak_ind_2)
            max_peak_ind_fw = np.argmax(-RV_lin_str_free_sg[peaks_savgol_2])
            max_peak_savgol_fw = peaks_savgol_2[max_peak_ind_2]
            def_max_point_fw = [max_peak_savgol_fw, RV_lin_str_free_sg[max_peak_savgol_fw]]

    else:  # take the value, but flag it
        def_max_point_fw = [max_peak_savgol_fw, RV_lin_str_free_sg[max_peak_savgol_fw]]
        flag_long == 1

    # get diastolic strainrate
    diast_strain_fw, TPK_diast_strain_fw, Ys_slope_fw, range_slope_fw, Y_on_curve_fw, min_idx_fw = \
            get_diastolic(RV_lin_str_free_sg, def_max_point_fw,dt)

    # Create plots with pre-defined labels.
    fig, ax = plt.subplots()
    ax.plot(RV_lin_str_free_sg, 'm', label='free-wall RV l-strain')
    ax.plot(average_len_strain_sg, 'y', label='average RV l-strain')
    ax.plot(max_peak_savgol, average_len_strain_sg[max_peak_savgol], "rs", label='peak strain')
    ax.plot(max_peak_savgol_fw, RV_lin_str_free_sg[max_peak_savgol_fw], "rs", label='peak strain')
    ax.plot(range_slope_fw, Ys_slope_fw, 'r')
    ax.plot(range_slope_av, Ys_slope_av, 'r')
    ax.plot(min_idx_fw, Y_on_curve_fw, 'rx', label='diastolic SR fw')
    ax.plot(min_idx_av, Y_on_curve_av, 'rx', label='diastolic SR average')
    ax.legend()
    plt.savefig(os.path.join(results_dir, 'RV_long_strain.png'))
    plt.close()

    np.savetxt('{0}/RV_long_strain_average.txt'.format(results_dir), average_len_strain_sg)
    np.savetxt('{0}/RV_long_strain_free-wall.txt'.format(results_dir), RV_lin_str_free_sg)

    vols = -np.ones(11)
    vols[0] = average_len_strain_sg[max_peak_savgol]
    vols[1] = max_peak_savgol * dt
    vols[2] = RV_lin_str_free_sg[max_peak_savgol_fw]
    vols[3] = max_peak_savgol_fw * dt
    vols[4] = diast_strain_av
    vols[5] = TPK_diast_strain_av
    vols[6] = diast_strain_fw
    vols[7] = TPK_diast_strain_fw
    vols[10] = flag_long

    return vols


def get_diastolic(curve_orig, def_max_point,dt):
    try:
        max_peak_idx = def_max_point[0]
        max = def_max_point[1]
        curve = curve_orig * -1
        ## FIND Peak descend using  smoothened curves.

        # get first derivatives and a second more smooth line to check peak point
        # curve_savg = signal.savgol_filter(curve, 15, 3)
        first_derived = np.gradient(curve)
        # Find mapse to check only minima after peak, but before atrial kick, so taking out last 25% of data (idx_at)
        idx_at = int(len(curve) / 4)
        tmp = np.argmin(first_derived[max_peak_idx:-idx_at])
        min_idx = max_peak_idx+tmp
        min = np.min(first_derived[:-idx_at])

        # get e'
        if len(curve) < 45:
            mean_min = first_derived[min_idx]
        else:
            mean_min = np.average(
                [first_derived[min_idx - 1], first_derived[min_idx], first_derived[min_idx + 1]])

        # make slopeline for plot
        num = int(len(curve) / 6)
        if num > 3:
            num = 3
        if num < 1:
            num = 1
        Y_on_curve = curve_orig[min_idx]
        min_X_range_slope = min_idx - num
        max_X_range_slope = min_idx + num
        range_slope = np.arange(min_X_range_slope, max_X_range_slope, 0.4)
        Ys_slope = ((range_slope - min_idx) * (mean_min * -1)) + Y_on_curve

        # diast_LV_mid_mapse_la_4Ch[d] = mean_min
        diast_strain = (mean_min) / (dt / 1000)
        TPK_diast_strain = (min_idx * (dt / 1000))

    except:
        # TODO how to handle exception (similar to other data, but not sure how to do this.
        diast_strain = -1
        TPK_diast_strain =  -1
        Ys_slope = -1
        Y_on_curve = -1

    return diast_strain, TPK_diast_strain, Ys_slope, range_slope, Y_on_curve, min_idx


def get_points_4Ch_RV_longit_strain(la_4Ch_img_dir, la_4Ch_seg_dir, results_dir, N_points = 7):
    nim = nib.load(la_4Ch_img_dir)
    seg_la_4Ch = nib.load(la_4Ch_seg_dir).get_fdata()[:, :, 0, :]
    img_la_4Ch = nim.get_fdata()[:, :, 0, :]
    _, _, N_frames_4Ch = img_la_4Ch.shape
    dt = nib.load(la_4Ch_img_dir).header['pixdim'][4]

    total_length_4Ch = np.zeros(N_frames_4Ch)
    total_apex_rvlv = np.zeros(N_frames_4Ch)
    total_apex_free = np.zeros(N_frames_4Ch)

    for fr in range(N_frames_4Ch):
        la_seg_fr = getLargestCC(seg_la_4Ch[:,:,fr])

        apex, mid_valve, anterior, inferior = detect_LV_points(la_seg_fr)
        apex_RV, rvlv_point, free_rv_point = detect_RV_points(la_seg_fr, anterior)

        # Get contours
        rv_seg = np.squeeze(la_seg_fr == 3).astype(float)  # get atria label
        contours_RV = measure.find_contours(rv_seg, 0.8)
        contours_RV = max(contours_RV, key = len)

        # Get closest points to the contours
        dist = distance.cdist(contours_RV, [apex_RV], 'euclidean')
        ind_apex = dist.argmin()
        apex_RV = contours_RV[ind_apex, :]
        dist = distance.cdist(contours_RV, [rvlv_point], 'euclidean')
        ind_rvlv = dist.argmin()
        rvlv_point = contours_RV[ind_rvlv, :]
        dist = distance.cdist(contours_RV, [free_rv_point], 'euclidean')
        ind_free_rv = dist.argmin()
        free_rv_point = contours_RV[ind_free_rv, :]

        # Detect points free wall
        if ind_rvlv < ind_apex < ind_free_rv:
            points_free_wall = contours_RV[ind_rvlv:ind_free_rv]
        elif ind_rvlv < ind_apex  and  ind_apex > ind_free_rv:
            points_free_wall = np.vstack([contours_RV[ind_free_rv:ind_apex], contours_RV[ind_apex+1:], contours_RV[0:ind_rvlv]])
        else:
            # TO CHECK!
            points_free_wall = contours_RV[ind_free_rv:ind_rvlv]

        # Sample to get N_points
        points_free_wall = binarymatrix(points_free_wall)
        # fit b-spline curve to skeleton, sample fixed number of points
        tck, u = splprep(points_free_wall.T, u=None, s=10.0, per=0, nest=-1, quiet=2)
        u_new = np.linspace(u.min(), u.max(), N_points)
        cl_pts = np.zeros([N_points, 2])
        cl_pts[:, 0], cl_pts[:, 1] = splev(u_new, tck, der=0)

        if fr == 0:
            f, ax = plt.subplots()
            ax.imshow(img_la_4Ch[:, :, fr], cmap='gray')
            ax.imshow(la_seg_fr, alpha=0.3)
            ax.axis('off')
            # Hide axes ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.plot(contours_RV[:, 1], contours_RV[:, 0], 'w.')
            ax.plot(points_free_wall[:, 1], points_free_wall[:, 0], 'k.')
            ax.plot(cl_pts[:, 1], cl_pts[:, 0], 'm.')
            plt.plot(apex_RV[1],apex_RV[0], 'ro')
            plt.plot(rvlv_point[1],rvlv_point[0], 'bo')
            plt.plot(free_rv_point[1],free_rv_point[0], 'yo')
            plt.savefig(os.path.join(results_dir, 'RV_long_strain_seg.png'))
            plt.close()

        # calculate total length
        total_dist_fr = 0
        for xi in range(len(cl_pts) - 1):
            points_1 = cl_pts[xi]
            points_2 = cl_pts[xi + 1]
            ind_dist = distance.pdist([points_1, points_2])
            total_dist_fr = total_dist_fr + ind_dist
            # total_dist_fr[xi] = ind_dist
        total_length_4Ch[fr] = total_dist_fr
        total_apex_free[fr] = distance.pdist([apex_RV, free_rv_point])
        total_apex_rvlv[fr] = distance.pdist([apex_RV, rvlv_point])

    return total_length_4Ch, total_apex_free, total_apex_rvlv, N_frames_4Ch, dt

def strain_Tshape(distance_apex_mid_valve, results_dir, N_frames, chamber = '2Ch',  window_size = 7, poly_order = 3):
    if len(np.unique((np.where(distance_apex_mid_valve == 0)[0]))) < 5:
        longit_strain_T_la = (distance_apex_mid_valve - distance_apex_mid_valve[0]) / \
                                    distance_apex_mid_valve[0]
        longit_strain_T_la *= 100
        if np.sum(longit_strain_T_la < -60) > 0:
            indx = np.where(longit_strain_T_la < -60)[0]
            longit_strain_T_la[indx] = np.NaN
            s = pd.Series(longit_strain_T_la)
            longit_strain_T_la = s.interpolate(method='polynomial', order=2).to_numpy()
            if (np.isnan(longit_strain_T_la) == True).any():
                longit_strain_T_la[np.isnan(longit_strain_T_la) == True] = 0
        np.savetxt(os.path.join(results_dir, 'long_la_{}_strain_T.txt'.format(chamber)), longit_strain_T_la)
        x = np.linspace(0, N_frames - 1, N_frames)
        xx = np.linspace(np.min(x), np.max(x), N_frames)
        itp = interp1d(x, longit_strain_T_la)
        longit_strain_T_la_smooth = savgol_filter(itp(xx), window_size, poly_order)
        np.savetxt(os.path.join(results_dir, 'long_la_{}_strain_T_smooth.txt'.format(chamber)),
                    longit_strain_T_la_smooth)
    else:
        longit_strain_T_la = -1 * np.ones(N_frames)
        longit_strain_T_la_smooth = -1 * np.ones(N_frames)
        np.savetxt(os.path.join(results_dir, 'long_la_{}_strain_T.txt'.format(chamber)), longit_strain_T_la)
        np.savetxt(os.path.join(results_dir, 'long_la_{}_strain_T_smooth.txt'.format(chamber)),
                    longit_strain_T_la_smooth)
    
    return longit_strain_T_la_smooth

def compute_MAPSE(N_frames, valve_points, dx, results_dir, chamber = '2Ch', window_size = 7, poly_order = 3):
    dist_mapse = np.zeros((N_frames, 2))
    for fr in range(N_frames):
        dist_mapse[fr, 0] = -(math.dist(valve_points[0, 0, :], valve_points[fr, 0, :])) * dx
        dist_mapse[fr, 1] = -(math.dist(valve_points[0, 1, :], valve_points[fr, 1, :])) * dx

    if len(np.unique((np.where(dist_mapse == 0)[0]))) < 6:
        mapse_la = np.mean(dist_mapse, axis=1)
        if np.sum(mapse_la < -60) > 0:
            indx = np.where(mapse_la < -60)[0]
            mapse_la[indx] = np.NaN
            s = pd.Series(mapse_la)
            mapse_la = s.interpolate(method='polynomial', order=2).to_numpy()
            if (np.isnan(mapse_la) == True).any():
                mapse_la[np.isnan(mapse_la) == True] = 0
        np.savetxt(os.path.join(results_dir, 'mapse_{}.txt'.format(chamber)), mapse_la)
        x = np.linspace(0, N_frames - 1, N_frames)
        xx = np.linspace(np.min(x), np.max(x), N_frames)
        itp = interp1d(x, mapse_la)
        mapse_la_smooth = savgol_filter(itp(xx), window_size, poly_order)
        np.savetxt(os.path.join(results_dir, 'mapse_{}_smooth.txt'.format(chamber)),
                    mapse_la_smooth)
    else:
        mapse_la = -1 * np.ones(N_frames)
        mapse_la_smooth = -1 * np.ones(N_frames)
        np.savetxt(os.path.join(results_dir, 'mapse_{}.txt'.format(chamber)), mapse_la)
        np.savetxt(os.path.join(results_dir, 'mapse_{}_smooth.txt'.format(chamber)),
                    mapse_la_smooth)
    return mapse_la_smooth


def compute_line_strain(N_frames, points_myo_la, results_dir, chamber = '2Ch', n_samples = 24, nb_layers_LA = 2, window_size = 7, poly_order = 3):
    dist_la = np.zeros((N_frames, nb_layers_LA))
    for fr in range(N_frames):
        for ly in range(nb_layers_LA):
            dd = 0
            for ss in range(n_samples - 1):
                dd += np.linalg.norm(
                    points_myo_la[fr, ly, ss, :] - points_myo_la[fr, ly, ss + 1, :])

            dist_la[fr, ly] = dd

    if len(np.unique((np.where(dist_la == 0)[0]))) < 5:
        longit_strain_la = np.mean((dist_la - dist_la[0, :]) / dist_la[0, :], axis=1)
        longit_strain_la *= 100
        np.savetxt(os.path.join(results_dir, 'long_{}_strain.txt'.format(chamber)), longit_strain_la)
        x = np.linspace(0, N_frames - 1, N_frames)
        xx = np.linspace(np.min(x), np.max(x), N_frames)
        itp = interp1d(x, longit_strain_la)
        longit_strain_la_smooth = savgol_filter(itp(xx), window_size, poly_order)
        np.savetxt(os.path.join(results_dir, 'long_{}_smooth.txt'.format(chamber)),
                    longit_strain_la_smooth)
    else:
        longit_strain_la = -1 * np.ones(N_frames)
        longit_strain_la_smooth = -1 * np.ones(N_frames)
        np.savetxt(os.path.join(results_dir, 'long_{}_strain.txt'.format(chamber)), longit_strain_la)
        np.savetxt(os.path.join(results_dir, 'long_{}_smooth.txt'.format(chamber)),
                    longit_strain_la_smooth)
    return longit_strain_la_smooth


def get_points_4Ch_longit_strain(la_4Ch_img_dir, la_4Ch_seg_dir, results_dir, n_samples = 24, nb_layers_LA = 2):
    points_per_segment = int(n_samples / 6)
    nim = nib.load(la_4Ch_img_dir)
    seg_la_4Ch = nib.load(la_4Ch_seg_dir).get_fdata()[:, :, 0, :]
    img_la_4Ch = nim.get_fdata()[:, :, 0, :]
    _, _, N_frames_4Ch = img_la_4Ch.shape
    dx, dy, dz = nim.header['pixdim'][1:4]
    points_myo_la_4Ch = np.zeros((N_frames_4Ch, nb_layers_LA, n_samples, 2))
    distance_apex_mid_valve_4Ch = np.zeros(N_frames_4Ch)
    valve_points_4Ch = np.zeros((N_frames_4Ch, 2, 2))
    dt = nib.load(la_4Ch_img_dir).header['pixdim'][4]
    tt_4Ch = np.arange(0, N_frames_4Ch) * dt

    for fr in range(N_frames_4Ch):
        data_seg = seg_la_4Ch[:, :, fr]
        data_img = img_la_4Ch[:, :, fr]
        endo_contour, epi_contour = get_layers_myocardium_LA(data_seg, n_samples)
        points_myo_la_4Ch[fr, 0, :, :] = np.asarray(endo_contour)
        points_myo_la_4Ch[fr, 1, :, :] = np.asarray(epi_contour)
        # Compute valve strain
        valve1 = endo_contour[0, :]
        valve2 = endo_contour[-1, :]
        _mid_valve = np.mean([valve1, valve2], axis=0)
        dist_myo = distance.cdist(endo_contour, [_mid_valve])
        ind_apex = dist_myo.argmax()
        _apex = endo_contour[ind_apex, :]
        valve_points_4Ch[fr, 0, :] = valve1
        valve_points_4Ch[fr, 1, :] = valve2
        distance_apex_mid_valve_4Ch[fr] = math.dist(_apex, _mid_valve)
        if fr == 0:
            plt.figure()
            plt.imshow(data_img + data_seg, 'gray')
            colors = ('b', 'g', 'r', 'c', 'm', 'y')
            for p in range(6):
                plt.plot(endo_contour[0, 1], endo_contour[0, 0], 'o', color='orange')
                plt.plot(epi_contour[0, 1], epi_contour[0, 0], 'o', color='orange')
                plt.plot(endo_contour[p * points_per_segment:(p + 1) * points_per_segment + 1, 1],
                            endo_contour[p * points_per_segment:(p + 1) * points_per_segment + 1, 0],
                            color=colors[p], label='endo', marker='.')
                plt.plot(epi_contour[p * points_per_segment:(p + 1) * points_per_segment + 1, 1],
                            epi_contour[p * points_per_segment:(p + 1) * points_per_segment + 1, 0],
                            color=colors[p], label='epi', marker='.')
            plt.savefig(os.path.join(results_dir, 'LV_strain_layers_la_4Ch.png'))
            plt.close('all')

            plt.figure()
            plt.imshow(data_img + data_seg, 'gray')
            plt.plot(_mid_valve[1], _mid_valve[0], 'y.')
            plt.plot(_apex[1], _apex[0], 'y.')
            x = [_mid_valve[1], _apex[1]]
            y = [_mid_valve[0], _apex[0]]
            plt.plot(x, y, 'y-o')
            plt.plot(valve2[1], valve2[0], 'ro')
            x = [valve1[1], valve2[1]]
            y = [valve1[0], valve2[0]]
            plt.plot(x, y, 'r-o')
            plt.axis('equal')
            plt.savefig(os.path.join(results_dir, 'LV_strain_la_4Ch_T_fr_{}.png'.format(fr)))
            plt.close('all')

    return N_frames_4Ch, points_myo_la_4Ch, valve_points_4Ch, dx, dt, distance_apex_mid_valve_4Ch, tt_4Ch

def get_points_2Ch_longit_strain(la_2Ch_img_dir, la_2Ch_seg_dir, results_dir, n_samples = 24, nb_layers_LA = 2):
    points_per_segment = int(n_samples / 6)
    nim = nib.load(la_2Ch_img_dir)
    seg_la_2Ch = nib.load(la_2Ch_seg_dir).get_fdata()[:, :, 0, :]
    img_la_2Ch = nim.get_fdata()[:, :, 0, :]
    _, _, N_frames_2Ch = img_la_2Ch.shape
    dx, dy, dz = nim.header['pixdim'][1:4]
    points_myo_la_2Ch = np.zeros((N_frames_2Ch, nb_layers_LA, n_samples, 2))
    distance_apex_mid_valve_2Ch = np.zeros(N_frames_2Ch)
    valve_points_2Ch = np.zeros((N_frames_2Ch, 2, 2))
    dt = nib.load(la_2Ch_img_dir).header['pixdim'][4]
    tt_2Ch = np.arange(0, N_frames_2Ch) * dt
    for fr in range(N_frames_2Ch):
        data_seg = seg_la_2Ch[:, :, fr]
        data_img = img_la_2Ch[:, :, fr]
        endo_contour, epi_contour = get_layers_myocardium_LA(data_seg, n_samples)
        points_myo_la_2Ch[fr, 0, :, :] = np.asarray(endo_contour)
        points_myo_la_2Ch[fr, 1, :, :] = np.asarray(epi_contour)
        # Compute valve strain
        valve1 = endo_contour[0, :]
        valve2 = endo_contour[-1, :]
        _mid_valve = np.mean([valve1, valve2], axis=0)
        dist_myo = distance.cdist(endo_contour, [_mid_valve])
        ind_apex = dist_myo.argmax()
        _apex = endo_contour[ind_apex, :]
        valve_points_2Ch[fr, 0, :] = valve1
        valve_points_2Ch[fr, 1, :] = valve2
        distance_apex_mid_valve_2Ch[fr] = math.dist(_apex, _mid_valve)
        if fr == 0:
            plt.figure()
            plt.imshow(data_img + data_seg, 'gray')
            colors = ('b', 'g', 'r', 'c', 'm', 'y')
            for p in range(6):
                plt.plot(endo_contour[0, 1], endo_contour[0, 0], 'o', color='orange')
                plt.plot(epi_contour[0, 1], epi_contour[0, 0], 'o', color='orange')
                plt.plot(endo_contour[p * points_per_segment:(p + 1) * points_per_segment + 1, 1],
                            endo_contour[p * points_per_segment:(p + 1) * points_per_segment + 1, 0],
                            color=colors[p], label='endo', marker='.')
                plt.plot(epi_contour[p * points_per_segment:(p + 1) * points_per_segment + 1, 1],
                            epi_contour[p * points_per_segment:(p + 1) * points_per_segment + 1, 0],
                            color=colors[p], label='epi', marker='.')
            plt.savefig(os.path.join(results_dir, 'LV_strain_layers_la_2Ch.png'))
            plt.close('all')
        
            plt.figure()
            plt.imshow(data_img + data_seg, 'gray')
            plt.plot(_mid_valve[1], _mid_valve[0], 'y.')
            plt.plot(_apex[1], _apex[0], 'y.')
            x = [_mid_valve[1], _apex[1]]
            y = [_mid_valve[0], _apex[0]]
            plt.plot(x, y, 'y-o')
            plt.plot(valve2[1], valve2[0], 'ro')
            x = [valve1[1], valve2[1]]
            y = [valve1[0], valve2[0]]
            plt.plot(x, y, 'r-o')
            plt.axis('equal')
            plt.savefig(os.path.join(results_dir, 'LV_strain_la_2Ch_T_fr_{}.png'.format(fr)))
            plt.close('all')

    return N_frames_2Ch, points_myo_la_2Ch, valve_points_2Ch, dx, dt, distance_apex_mid_valve_2Ch, tt_2Ch

def compute_myo_points(img_dir, seg_dir, results_dir, n_samples = 24, nb_layers_SA = 3):
    seg = nib.load(seg_dir).get_fdata()
    img = nib.load(img_dir).get_fdata()
    X, Y, Z, N_frames = img.shape
    ES_frame = np.sum(np.sum(np.sum(seg == 1, axis=0), axis=0), axis=0).argmin()
    slices_with_seg = np.where(np.sum(np.sum(seg[:, :, :, ES_frame], axis=0), axis=0) != 0)[0]
    points_per_segment = int(n_samples / 6)
    if len(slices_with_seg) > 5:
        slices_selected = slices_with_seg[2:-2]
    elif len(slices_with_seg) > 3:
        slices_selected = slices_with_seg[1:-1]
    else:
        slices_selected = slices_with_seg

    dt = nib.load(img_dir).header['pixdim'][4]
    tt = np.arange(0, N_frames) * dt
    points_myo = np.zeros((N_frames, len(slices_selected), nb_layers_SA, n_samples, 2))
    slice_to_reject = []
    for s, sl in enumerate(slices_selected):
        for fr in range(N_frames):
            data_seg = seg[:, :, sl, fr]
            data_img = img[:, :, sl, fr]
            try:
                points_myo_old, centroid = get_layers_myocardium_SA(data_seg, n_samples)
                if len(points_myo_old) > 0:
                    if fr == 0:
                        plt.figure()
                        plt.imshow(data_img, 'gray')
                        colors = ('b', 'g', 'r', 'c', 'm', 'y')
                        for p in range(6):
                            plt.plot(points_myo_old[2][0, 1], points_myo_old[2][0, 0], 'o', color='orange')
                            plt.plot(points_myo_old[1][0, 1], points_myo_old[1][0, 0], 'o', color='orange')
                            plt.plot(points_myo_old[0][0, 1], points_myo_old[0][0, 0], 'o', color='orange')
                            plt.plot(points_myo_old[2][p * points_per_segment:(p + 1) * points_per_segment + 1, 1],
                                    points_myo_old[2][p * points_per_segment:(p + 1) * points_per_segment + 1, 0],
                                    color=colors[p], label='epi', marker='.')
                            plt.plot(points_myo_old[1][p * points_per_segment:(p + 1) * points_per_segment + 1, 1],
                                    points_myo_old[1][p * points_per_segment:(p + 1) * points_per_segment + 1, 0],
                                    color=colors[p], label='cl', marker='.')
                            plt.plot(points_myo_old[0][p * points_per_segment:(p + 1) * points_per_segment + 1, 1],
                                    points_myo_old[0][p * points_per_segment:(p + 1) * points_per_segment + 1, 0],
                                    color=colors[p], label='endo', marker='.')
                        plt.plot(centroid[1], centroid[0], 'm.', label='centroid')
                        plt.savefig(os.path.join(results_dir, 'LV_strain_layers_SAX_sl_{}.png'.format(sl)))
                        plt.close('all')
                    for i in range(3):
                        points_myo[fr, s, i, :, :] = np.asarray(points_myo_old[i])
            except Exception:
                print('Error strain SAX:frame {} and slice {}'.format(fr, sl))
                slice_to_reject.append(s)

            if np.sum(points_myo[:, s, :, :, :]) == 0:
                slice_to_reject.append(s)
    if len(slice_to_reject) >= 1:
        slice_to_reject = np.unique(np.array(slice_to_reject))
        slices_selected = np.delete(slices_selected, slice_to_reject)
        points_myo = np.delete(points_myo, slice_to_reject, axis=1)

    return N_frames,slices_selected, points_myo, tt, dt

def get_processed_myocardium(_seg, _label=2):
    """
    This function tidies the LV myocardial segmentation, taking only the single
    largest connected component, and performing an opening (erosion+dilation)
    """
    myo_aux = np.squeeze(_seg == _label).astype(float)  # get myocardial label
    myo_aux = closing(myo_aux).astype(float)

    if _label == 2:
        # Take largest cc
        myo_aux = isolateLargestMask(myo_aux)
        if not isAnnulus(myo_aux):
            closeMask(myo_aux)
            if not isAnnulus(myo_aux):
                return []

    cc_aux = measure.label(myo_aux)
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


def calculateConvexHull(mask):
    """Returns a binary mask convex hull of the given binary mask."""
    m = mask > 0

    region = np.argwhere(m)
    hull = ConvexHull(region)
    de = Delaunay(region[hull.vertices])

    simplexpts = de.find_simplex(np.argwhere(m == m))
    return simplexpts.reshape(m.shape) != -1


def closeMask(mask):
    """Returns a mask with a 1 pixel rim added if need to ensure the mask is an annulus."""
    if not isAnnulus(mask):
        hullmask = calculateConvexHull(mask)
        largemask = binary_dilation(hullmask)
        rim = largemask.astype(mask.dtype) - hullmask.astype(mask.dtype)
    else:
        rim = 0

    return mask + rim


def get_longest_path_SA(skel):
    # first create edges from skeleton
    sk_im = skel.copy()
    # remove bad (L-shaped) junctions
    sk_im = remove_bad_junctions(sk_im)

    # get seeds for longest path from existing end-points
    end_pts = np.zeros([2, 2])
    count = 0
    while len(end_pts) > 0:
        if count > 0:
            for j in end_pts:
                sk_im[j[0], j[1]] = 0

        count += 1
        out = skeleton_endpoints(sk_im.astype(int))
        end_pts = np.asarray(np.nonzero(out)).transpose()

    return sk_im


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


def remove_bad_junctions(skel):
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
            print('Warning! You have a 3x3 loop!')

    return skel

def binarymatrix(A): 
    A_aux = np.copy(A)
    A = map(tuple,A)
    dic = Counter(A)
    for (i,j) in dic.items(): 
       if j>1: 
          ind = np.where( ((A_aux[:,0] == i[0]) & (A_aux[:,1] == i[1])))[0]
          A_aux = np.delete(A_aux,ind[1:],axis=0)
    if np.linalg.norm(A_aux[:,0]-A_aux[:,-1]) < 0.01:
        A_aux=A_aux[:-1,:]
    return A_aux

def get_sorted_sk_pts(myo, ref=np.array([1, 0]), n_samples=48, centroid=np.array([0, 0])):
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
    sk_im = get_longest_path(sk_im)

    # sort centreline points based from boundary points at valves as start
    # and end point. Make ref point out of LV through valve
    out = skeleton_endpoints(sk_im.astype(int))
    end_pts = np.asarray(np.nonzero(out)).transpose()
    sk_pts = np.asarray(np.nonzero(sk_im)).transpose()

    if len(end_pts) > 2:
        print('Error! More than 2 end-points in LA myocardial skeleton.')
    else:
        # set reference to vector pointing from centroid to mid-valve
        mid_valve = np.mean(end_pts, axis=0)
        ref = (mid_valve - centroid) / norm(mid_valve - centroid)
    sk_pts2 = sk_pts - centroid  # centre around centroid
    myo_pts2 = myo_pts - centroid
    theta = np.zeros([len(sk_pts2), ])
    theta_myo = np.zeros([len(myo_pts2), ])

    eps = 0.0001
    if len(sk_pts2) <= 5:
        print('Skeleton failed! Only of length {}'.format(len(sk_pts2)))
        cl_pts = []
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
        tck, u = splprep(sk_pts.T, u=None, s=10.0, per=0, nest=-1, quiet=2)
        u_new = np.linspace(u.min(), u.max(), n_samples)
        cl_pts = np.zeros([n_samples, 2])
        cl_pts[:, 0], cl_pts[:, 1] = splev(u_new, tck, der=0)

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
    return cl_pts, mid_valve

def detect_LV_points(seg):
    myo_seg = np.squeeze(seg == 2).astype(float)
    kernel = np.ones((2, 2), np.uint8)
    myo_seg_dil = cv2.dilate(myo_seg, kernel, iterations=2)
    myo2 = get_processed_myocardium(myo_seg_dil, _label=1)
    cl_pts, mid_valve = get_sorted_sk_pts(myo2)
    dist_myo = distance.cdist(cl_pts, [mid_valve], 'euclidean')
    ind_apex = dist_myo.argmax()
    _apex = cl_pts[ind_apex, :]
    _septal_mv = cl_pts[0, 0], cl_pts[0, 1]
    _ant_mv = cl_pts[-1, 0], cl_pts[-1, 1]

    return np.asarray(_apex), np.asarray(mid_valve), np.asarray(_septal_mv), np.asarray(_ant_mv)


def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def detect_RV_points(_seg, septal_mv):
    rv_seg = np.squeeze(_seg == 3).astype(float)

    sk_pts = measure.find_contours(rv_seg, 0.8)
    if len(sk_pts) > 1:
        nb_pts = []
        for l in range(len(sk_pts)):
            nb_pts.append(len(sk_pts[l]))
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
        print('Skeleton failed! Only of length {}'.format(len(sk_pts2)))
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
        tck, u = splprep(sk_pts.T, u=None, s=10.0, per=1, quiet=2)

        u_new = np.linspace(u.min(), u.max(), 80)
        _cl_pts = np.zeros([80, 2])
        _cl_pts[:, 0], _cl_pts[:, 1] = splev(u_new, tck, der=0)

    dist_rv = distance.cdist(_cl_pts, [_lv_valve], 'euclidean')
    _ind_apex = dist_rv.argmax()
    _apex_RV = _cl_pts[_ind_apex, :]

    m = np.diff(_cl_pts[:, 0]) / np.diff(_cl_pts[:, 1])
    angle = np.arctan(m) * 180 / np.pi
    idx = np.sign(angle)
    _ind_free_wall = np.where(idx == -1)[0]

    area = 10000*np.ones(len(_ind_free_wall))
    for ai,ind in enumerate(_ind_free_wall):
        AB = np.linalg.norm(_lv_valve-_apex_RV)
        BC = np.linalg.norm(_lv_valve-_cl_pts[ind, :])
        AC = np.linalg.norm(_cl_pts[ind, :]-_apex_RV)
        if AC > 10 and BC >10:
            area[ai] = np.abs(AB**2+BC**2-AC**2)
    # plt.imshow(rv_seg)
    # plt.plot(_apex_RV[1],_apex_RV[0], 'ro')
    # plt.plot(_lv_valve[1],_lv_valve[0], 'ro')
    # plt.plot(_cl_pts[_ind_free_wall[area.argmin()], 1],_cl_pts[_ind_free_wall[area.argmin()], 0], 'bo')
    # plt.show()

    _free_rv_point = _cl_pts[_ind_free_wall[area.argmin()], :]

    return np.asarray(_apex_RV), np.asarray(_lv_valve), np.asarray(_free_rv_point)

def detect_intersection_pts(data):
    label_lv = np.copy(data)
    label_lv[label_lv == 3] = 0
    label_lv[label_lv == 2] = 1

    label_rv = np.copy(data)
    label_rv[label_rv == 1] = 0
    label_rv[label_rv == 2] = 0

    label_rv_dil = binary_dilation(label_rv).astype(float)
    label_rv_dil2 = binary_dilation(label_rv_dil).astype(float)
    junctions_score = np.logical_and(label_rv_dil2, label_lv)
    aux_pts = np.asarray(np.nonzero(junctions_score)).transpose()
    dist_pts = distance.cdist(aux_pts, aux_pts)
    # added as it often is a little too far away and fails #TODO when RV is not there (in some frames) use other frames #NOTE
    if dist_pts.size == 0:
        label_rv_dil3 = binary_dilation(label_rv_dil2).astype(float)
        label_rv_dil4 = binary_dilation(label_rv_dil3).astype(float)
        junctions_score = np.logical_and(label_rv_dil4, label_lv)
        aux_pts = np.asarray(np.nonzero(junctions_score)).transpose()
        dist_pts = distance.cdist(aux_pts, aux_pts)

    ind = np.where(dist_pts == np.max(dist_pts))[0]
    junctions_pts = np.asarray([aux_pts[ind[0], :], aux_pts[ind[-1], :]])

    return junctions_pts

def get_longest_path(skel):
    # first create edges from skeleton
    sk_im = skel.copy()
    # remove bad (L-shaped) junctions
    sk_im = remove_bad_junctions(sk_im)

    # get seeds for longest path from existing end-points
    out = skeleton_endpoints(sk_im.astype(int))
    end_pts = np.asarray(np.nonzero(out)).transpose()
    if len(end_pts) == 0:
        print('ERROR! No end-points detected! Exiting.')
    # break
    elif len(end_pts) == 1:
        print('Warning! Only 1 end-point detected!')
    elif len(end_pts) > 2:
        print('Warning! {0} end-points detected!'.format(len(end_pts)))

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
            edges.append(['{0}'.format(i), '{0}'.format(c)])
    # create graph
    G = defaultdict(list)
    for (ss, t) in edges:
        if t not in G[ss]:
            G[ss].append(t)
        if ss not in G[t]:
            G[t].append(ss)
    # print G.items()
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
    _x = contour[:, 0]
    y = contour[:, 1]

    # Pad the contour before approximation to avoid underestimating
    # the values at the end points
    r = int(0.5 * N)
    t_pad = np.concatenate((np.arange(-r, 0) * dt, t, 1 + np.arange(0, r) * dt))
    if periodic:
        x_pad = np.concatenate((_x[-r:], _x, _x[:r]))
        y_pad = np.concatenate((y[-r:], y, y[:r]))
    else:
        x_pad = np.concatenate((np.repeat(_x[0], repeats=r), _x, np.repeat(_x[-1], repeats=r)))
        y_pad = np.concatenate((np.repeat(y[0], repeats=r), y, np.repeat(y[-1], repeats=r)))

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


def get_layers_myocardium_SA(data, _n_samples = 24):
    #   data -       segmentation
    #   n_samples  initial number of points for fitting smooth spline
    #   n_samples  output number of points for sampling spline

    # Delete outliers
    data_bin = np.copy(data)
    data_bin[data_bin > 0] = 1
    data_bin = get_largest_cc(data_bin).astype(np.uint8)

    if np.linalg.norm(data_bin * data - data) > 0.1:
        data = data_bin * data
    try:
        myo = get_processed_myocardium(data)
    except Exception:
        _points_myo = []
        _centroid = []
        return _points_myo, _centroid
    if len(myo) == 0:
        _points_myo = []
        _centroid = []
        return _points_myo, _centroid
    landmark_points = detect_intersection_pts(data)

    # get skeleton consisting only of longest path
    sk_im = skeletonize(myo)
    sk_im = get_longest_path_SA(sk_im)
    sk_pts = np.asarray(np.nonzero(sk_im)).transpose()
    # Get centoid
    _centroid = np.mean(sk_pts, axis=0)
    # sort centreline points based from boundary points at valves as start
    # and end point. Make ref point out of LV through valve
    P1 = landmark_points[0, :]  # free wall
    ref = (P1 - _centroid) / norm(P1 - _centroid)

    # plt.figure()
    # plt.imshow(data/np.max(data)+sk_im,'gray')
    # plt.plot(landmark_points[0,1], landmark_points[0,0],'.r')
    # plt.plot(landmark_points[1,1], landmark_points[1,0],'.c')

    # Label class in the segmentation
    _label = {'BG': 0, 'LV': 1, 'Myo': 2, 'RV': 3}

    endo = (data == _label['LV']).astype(np.uint8)
    endo = get_largest_cc(endo).astype(np.uint8)
    myo = (data == _label['Myo']).astype(np.uint8)
    myo = remove_small_cc(myo).astype(np.uint8)
    epi = (endo | myo).astype(np.uint8)
    epi = get_largest_cc(epi).astype(np.uint8)
    pixel_thres = 10
    if (np.sum(endo) < pixel_thres) or (np.sum(myo) < pixel_thres):
        _points_myo = []
        _centroid = []
        return _points_myo, _centroid

    contours, hierarchy = cv2.findContours(cv2.inRange(endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    endo_pts = contours[0][:, 0, :]

    # Extract epicardial contour
    contours, hierarchy = cv2.findContours(cv2.inRange(epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    epi_pts = contours[0][:, 0, :]

    # Smooth the contours
    endo_pts = approximate_contour(endo_pts, periodic=True)
    epi_pts = approximate_contour(epi_pts, periodic=True)

    endo_pts[:, [0, 1]] = endo_pts[:, [1, 0]]
    epi_pts[:, [0, 1]] = epi_pts[:, [1, 0]]
    # plt.figure()
    # plt.imshow(data)
    # plt.plot(endo_pts[:,1],endo_pts[:,0])
    # plt.plot(epi_pts[:,1],epi_pts[:,0])

    endo_pts2 = endo_pts - _centroid
    epi_pts2 = epi_pts - _centroid
    theta_endo = np.zeros([len(endo_pts2), ])
    theta_epi = np.zeros([len(epi_pts2), ])

    # =============================================================================
    # Compute theta for each layer
    # =============================================================================
    if len(endo_pts2) <= 5 or len(epi_pts2) <= 5:
        print('Edges failed! Only of length {}'.format(len(endo_pts2)))
    else:
        # compute angle theta for endo points
        for k, _ss in enumerate(endo_pts2):
            eps = 0.0001
            if (np.dot(ref, _ss) / norm(_ss) < 1.0 + eps) and (np.dot(ref, _ss) / norm(_ss) > 1.0 - eps):
                theta_endo[k] = 0
            elif (np.dot(ref, _ss) / norm(_ss) < -1.0 + eps) and (np.dot(ref, _ss) / norm(_ss) > -1.0 - eps):
                theta_endo[k] = 180
            else:
                theta_endo[k] = math.acos(np.dot(ref, _ss) / norm(_ss)) * 180 / np.pi
            detp = ref[0] * _ss[1] - ref[1] * _ss[0]
            if detp > 0:
                theta_endo[k] = 360 - theta_endo[k]
        thinds = theta_endo.argsort()
        endo_pts = endo_pts[thinds, :].astype(float)  # ordered centreline points
        theta_endo.sort()

        # compute angle theta for epi points
        for k, _ss in enumerate(epi_pts2):
            eps = 0.0001
            if (np.dot(ref, _ss) / norm(_ss) < 1.0 + eps) and (np.dot(ref, _ss) / norm(_ss) > 1.0 - eps):
                theta_epi[k] = 0
            elif (np.dot(ref, _ss) / norm(_ss) < -1.0 + eps) and (np.dot(ref, _ss) / norm(_ss) > -1.0 - eps):
                theta_epi[k] = 180
            else:
                theta_epi[k] = math.acos(np.dot(ref, _ss) / norm(_ss)) * 180 / np.pi
            detp = ref[0] * _ss[1] - ref[1] * _ss[0]
            if detp > 0:
                theta_epi[k] = 360 - theta_epi[k]
        thinds = theta_epi.argsort()
        epi_pts = epi_pts[thinds, :].astype(float)  # ordered centreline points
        theta_epi.sort()

        # =============================================================================
        # Sample points for each layer
        # =============================================================================
        epi_pts_tmp = np.zeros([_n_samples, 2])
        endo_pts_tmp = np.zeros([_n_samples, 2])

        if len(epi_pts) < _n_samples or len(endo_pts) < _n_samples / 2:
            print('Error: very low number of segmentation points')
            _points_myo = []
            return _points_myo, _centroid

        # epi points
        fail_cs = []
        for cs in range(_n_samples):
            theta_frac = 360.0 / _n_samples
            cur_inds = np.where(np.logical_and(theta_epi >= cs * theta_frac, theta_epi <= (cs + 1) * theta_frac))[0]
            if len(cur_inds) == 0:
                fail_cs.append(cs)
            else:
                epi_pts_tmp[cs] = np.mean(epi_pts[cur_inds, :], axis=0)
        epi_pts_tmp = np.delete(epi_pts_tmp, fail_cs, axis=0)

        # Remove duplicates
        epi_pts_tmp = binarymatrix(epi_pts_tmp)
        # fit b-spline loop to skeleton, sample fixed number of points
        try:
            tck, u = splprep(epi_pts_tmp.T, s=0.0, per=1, nest=-1, quiet=2)
        except Exception:
            print('Problem slprep')
            return _points_myo, _centroid
        u_new = np.linspace(u.min(), u.max(), _n_samples)
        epi_pts_final = np.zeros([_n_samples, 2])
        epi_pts_final[:, 0], epi_pts_final[:, 1] = splev(u_new, tck)

        # endo points
        fail_cs = []
        for cs in range(_n_samples):
            theta_frac = 360.0 / _n_samples
            cur_inds = np.where(np.logical_and(theta_endo >= cs * theta_frac, theta_endo <= (cs + 1) * theta_frac))[
                0]
            if len(cur_inds) == 0:
                fail_cs.append(cs)
            else:
                endo_pts_tmp[cs] = np.mean(endo_pts[cur_inds, :], axis=0)
        endo_pts_tmp = np.delete(endo_pts_tmp, fail_cs, axis=0)
        # Remove duplicates
        endo_pts_tmp = binarymatrix(endo_pts_tmp)
        # fit b-spline loop to skeleton, sample fixed number of points
        try:
            tck, u = splprep(endo_pts_tmp.T, s=0.0, per=1, nest=-1, quiet=2)
        except Exception:
            print('Problem splprep')
            return _points_myo, _centroid
        u_new = np.linspace(u.min(), u.max(), _n_samples)
        endo_pts_final = np.zeros([_n_samples, 2])
        endo_pts_final[:, 0], endo_pts_final[:, 1] = splev(u_new, tck)
        sk_pts_final = (epi_pts_final + endo_pts_final) / 2

        _points_myo = [endo_pts_final, sk_pts_final, epi_pts_final]

    return _points_myo, _centroid


def remove_mitral_valve_points(_endo_contour, _epi_contour, mitral_plane):
    """ Remove the mitral valve points from the contours and
        start the contours from the point next to the mitral valve plane.
        So connecting the lines will be easier in the next step.
        """
    N = _endo_contour.shape[0]
    start_i = 0
    for ii in range(N):
        y, _x = _endo_contour[ii]
        prev_y, prev_x = _endo_contour[(ii - 1) % N]
        if not mitral_plane[_x, y] and mitral_plane[prev_x, prev_y]:
            start_i = ii
            break
    _endo_contour = np.concatenate((_endo_contour[start_i:], _endo_contour[:start_i]))

    N = _endo_contour.shape[0]
    end_i = N
    for ii in range(N):
        y, _x = _endo_contour[ii]
        if mitral_plane[_x, y]:
            end_i = ii
            break
    _endo_contour = _endo_contour[:end_i]

    N = _epi_contour.shape[0]
    start_i = 0
    for ii in range(N):
        y, _x = _epi_contour[ii]
        y2, x2 = _epi_contour[(ii - 1) % N]
        if not mitral_plane[_x, y] and mitral_plane[x2, y2]:
            start_i = ii
            break
    _epi_contour = np.concatenate((_epi_contour[start_i:], _epi_contour[:start_i]))

    N = _epi_contour.shape[0]
    end_i = N
    for ii in range(N):
        y, _x = _epi_contour[ii]
        if mitral_plane[_x, y]:
            end_i = ii
            break
    _epi_contour = _epi_contour[:end_i]
    return _endo_contour, _epi_contour


def get_largest_cc(binary):
    """ Get the largest connected component in the foreground. """
    cc, n_cc = measure.label(binary, return_num=True)
    max_n = -1
    max_area = 0
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area > max_area:
            max_area = area
            max_n = n
    largest_cc = (cc == max_n)
    return largest_cc

def getLargestCC(segmentation):
    nb_labels = np.unique(segmentation)[1:]
    out_image = np.zeros_like(segmentation)
    for ncc in nb_labels:
        aux = np.squeeze(segmentation == ncc).astype(float)  # get myocardial labe
        labels = measure.label(aux)
        assert( labels.max() != 0 ) # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        out_image += largestCC*ncc
    return out_image


def remove_small_cc(binary, thres=10):
    """ Remove small connected component in the foreground. """
    cc, n_cc = measure.label(binary, return_num=True)
    binary2 = np.copy(binary)
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area < thres:
            binary2[cc == n] = 0
    return binary2



def get_layers_myocardium_LA(seg_z, _n_samples = 24):
    #   data -       segmentation
    #   n_samples  output number of points for sampling spline

    # Label class in the segmentation
    _label = {'BG': 0, 'LV': 1, 'Myo': 2, 'RV': 3, 'LA': 4, 'RA': 5}
    # Check whether there is the endocardial segmentation
    # Only keep the largest connected component
    endo = (seg_z == _label['LV']).astype(np.uint8)
    endo = get_largest_cc(endo).astype(np.uint8)
    # The myocardium may be split to two parts due to the very thin apex.
    # So we do not apply get_largest_cc() to it. However, we remove small pieces, which
    # may cause problems in determining the contours.
    myo = (seg_z == _label['Myo']).astype(np.uint8)
    myo = remove_small_cc(myo).astype(np.uint8)
    epi = (endo | myo).astype(np.uint8)
    epi = get_largest_cc(epi).astype(np.uint8)

    # Extract endocardial contour
    # Note: cv2 considers an input image as a Y x X array, which is different
    # from nibabel which assumes a X x Y array.
    try:
        contours, hierarchy = cv2.findContours(cv2.inRange(endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        _endo_contour = contours[0][:, 0, :]
    except Exception:
        print('_endo_contour')
        return [], []

    try:
        # Extract epicardial contour
        contours, hierarchy = cv2.findContours(cv2.inRange(epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        _epi_contour = contours[0][:, 0, :]
    except Exception:
        print('_epi_contour')
        return [], []

    # Record the points located on the mitral valve plane.
    mitral_plane = np.zeros(seg_z.shape)
    N = _epi_contour.shape[0]
    for ii in range(N):
        yy, _x = _epi_contour[ii]
        if endo[_x, yy]:
            mitral_plane[_x, yy] = 1
    # Remove the mitral valve points from the contours and
    # start the contours from the point next to the mitral valve plane.
    # So connecting the lines will be easier in the next step.
    if np.sum(mitral_plane) >= 1:
        _endo_contour, _epi_contour = remove_mitral_valve_points(_endo_contour, _epi_contour, mitral_plane)

    # Smooth the contours
    if len(_endo_contour) >= 2:
        _endo_contour = approximate_contour(_endo_contour)
    if len(_epi_contour) >= 2:
        _epi_contour = approximate_contour(_epi_contour)

    _endo_contour[:, [0, 1]] = _endo_contour[:, [1, 0]]
    _epi_contour[:, [0, 1]] = _epi_contour[:, [1, 0]]

    if len(_endo_contour) >= 2:
        tck, u = splprep(_endo_contour.T, s=10.0, nest=-1)
        u_new = np.linspace(u.min(), u.max(), _n_samples)
        endo_contour2 = np.zeros([_n_samples, 2])
        endo_contour2[:, 0], endo_contour2[:, 1] = splev(u_new, tck)

    if len(_epi_contour) >= 2:
        tck, u = splprep(_epi_contour.T, s=10.0, nest=-1)
        u_new = np.linspace(u.min(), u.max(), _n_samples)
        epi_contour2 = np.zeros([_n_samples, 2])
        epi_contour2[:, 0], epi_contour2[:, 1] = splev(u_new, tck)

    return endo_contour2, epi_contour2


def isAnnulus(mask):
    """Return True if `mask' defines an annular mask image."""
    _, numfeatures = label(mask)

    if numfeatures != 1:  # multiple features
        return False

    cavity = binary_fill_holes(mask > 0).astype(mask.dtype) - mask

    if cavity.sum() == 0:  # no enclosed area
        return False

    _, numfeatures = label(cavity)

    return numfeatures == 1  # exactly 1 enclosed area


def isolateLargestMask(mask):
    """Label the binary images in `mask' and return an image retaining only the largest."""
    labeled, numfeatures = label(mask)  # label each separate object with a different number

    if numfeatures > 1:  # if there's more than one object in the segmentation, keep only the largest as the best guess
        sums = ndsum(mask, labeled, range(numfeatures + 1))  # sum the pixels under each label
        maxfeature = np.where(sums == max(sums))  # choose the maximum sum whose index will be the label number
        mask = mask * (labeled == maxfeature)  # mask out the prediction under the largest label

    return mask


def isolateCavity(mask):
    """Returns the cavity mask from given annulus mask."""
    assert mask.sum() > 0
    cavity = binary_fill_holes(mask > 0).astype(mask.dtype) - mask

    if cavity.sum() == 0:
        return isolateCavity(closeMask(mask))  # try again with closed mask
    else:
        return cavity


def calc_peak_diast_strain(curve_orig, dt, results_dir, Name_curve):
    # TODO: see below: how to deal with error/exception, not '-1' but maybe like in the other def using []?
    curve = curve_orig * -1
    try:
        if np.size(np.unique(curve)) > 2:
            # get first derivatives and a second more smooth line to check peak point
            # curve_savg = signal.savgol_filter(curve, 15, 3)
            first_derived = np.gradient(curve)

            ## FIND Peak descend using different smoothened curves.
            # Find mapse to check only minima after, but not atrial kick, so taking out last 25% of data (idx_at)
            idx_at = int(len(curve) / 4)
            peak_idx = np.argmax(curve)
            min_idx = np.argmin(first_derived[:-idx_at])
            min = np.min(first_derived[:-idx_at])
            max_idx = np.argmax(first_derived[:-idx_at])
            max = np.max(curve)
            peak_strain = max * -1
            TPK_strain = max_idx * dt

            # get e'
            if min_idx > peak_idx:
                if len(curve) < 45:
                    mean_min = first_derived[min_idx]
                else:
                    mean_min = np.average(
                        [first_derived[min_idx - 1], first_derived[min_idx], first_derived[min_idx + 1]])

                # print('min {0}, mean min{2} index {1}'.format(min, min_idx, mean_min))
                # make slopeline for plot
                num = int(len(curve) / 6)
                if num > 5:
                    num = 5
                if num < 3:
                    num = 3
                Y_on_curve = curve_orig[min_idx]
                min_X_range_slope = min_idx - num
                max_X_range_slope = min_idx + num
                range_slope = np.arange(min_X_range_slope, max_X_range_slope, 0.4)
                Ys_slope = ((range_slope - min_idx) * (mean_min * -1)) + Y_on_curve
                fig = plt.figure()
                plt.plot(curve_orig)
                plt.plot(range_slope, Ys_slope, 'r')
                plt.plot(peak_idx, (max * -1), 'rs', label='peak strain')
                plt.plot(min_idx, Y_on_curve, 'rx', label='diastolic SR')
                plt.legend(loc='lower right')
                fig.savefig('{0}{1}.png'.format(results_dir, Name_curve))
                plt.close(fig)

                # diast_LV_mid_mapse_la_4Ch[d] = mean_min
                diast_strain = (mean_min) / (dt / 1000)
                TPK_diast_strain = (min_idx * (dt / 1000))
            elif min_idx == 0:
                mean_min = -1
                diast_strain = mean_min
                TPK_diast_strain = mean_min
            else:
                mean_min = -1
                diast_strain = mean_min
                TPK_diast_strain = mean_min
        else:
            mean_min = -1
            diast_strain = mean_min
            TPK_diast_strain = mean_min

    except:
        # TODO how to handle exception (similar to other data, but not sure how to do this.
        print('{0}: {1} Case failed')
        mean_min = -1
        diast_strain = mean_min
        TPK_diast_strain = mean_min

    return peak_strain, TPK_strain, diast_strain, TPK_diast_strain

