"""
$ python3 calculate_flow.py  --help
usage: calculate_flow.py [-h] [-i JSON_FILE]

Calculate flow params using segmentations generated using nnunet

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib
import os
import nibabel as nib
import numpy as np

# imports - 3rd party:
from datetime import datetime
from common_utils.load_args import Params
from common_utils.utils import set_logger, save_nifti
from scipy.signal import find_peaks
from sklearn.metrics import auc
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep


# def smooth_curve(curve, num_frames):
#     window_size, poly_order = 13, 3  # For savgol filter
#     x = np.linspace(0, num_frames - 1, num_frames)
#     xx = np.linspace(x.min(), x.max(), num_frames)
#     itp = interp1d(x, curve, kind='linear')
#     try:
#         yy_sg = savgol_filter(itp(xx), window_size, poly_order)
#         curve_smooth = yy_sg  # Area(mm²)
#     except:
#         curve_smooth = curve

#     return curve_smooth


def smooth_curve(curve, num_frames, s, num_points=None, x_values=None):
    """Smooth via spline interpolation, with smoothing condition s
    Use num_points to increase number of points in curve
    If changing number of points, also change x axis values to match"""
    x = np.linspace(0, num_frames - 1, num_frames)

    if num_points is not None:
        xx = np.linspace(x.min(), x.max(), num_points)
        curve_interp = interp1d(x, curve, kind='linear')(xx)
        x_interp = interp1d(x, x_values, kind='linear')(xx)
        spl = splrep(xx, curve_interp, s=s)
        curve_smooth = splev(xx, spl)
        return curve_smooth, x_interp
    else:
        spl = splrep(x, curve, s=s)
        curve_smooth = splev(x, spl)
        return curve_smooth


def calculate_flow(seg_array, flow_array, tt, num_frames, _venc, _area_per_voxel):
    mean_velocity_nnunet = np.zeros(num_frames, dtype=object)
    area_nnunet = np.zeros(num_frames)
    for fr in range(num_frames):
        area_nnunet[fr] = np.sum(seg_array[:, :, fr] == 1) * _area_per_voxel * 100
        aa, bb = np.where(seg_array[:, :, fr] == 1)
        if len(aa) > 1 and len(bb) > 1:
            velocity_fr_nnunet = flow_array[aa, bb, fr]
            mean_velocity_nnunet[fr] = np.mean(velocity_fr_nnunet) / 4096 * _venc
    flow_rate_nnunet = mean_velocity_nnunet * area_nnunet / 100

    # Smooth area curve
    s = 1500
    area_smooth = smooth_curve(area_nnunet, num_frames, s, 100, tt)

    try:
        flow_rate_smooth, tt_smooth = smooth_curve(flow_rate_nnunet, num_frames,
        s, 100, tt)
    except:
        flow_rate_smooth = flow_rate_nnunet
        tt_smooth = tt

    # Find peaks within 10%-65% of cardiac cycle that are above 0.4 * max peak
    # 0.35 * is to deal with noisy signals
    # Our peak of interest should be the first one - if negative, flip curve
    max_ind = np.nanargmax(np.abs(flow_rate_smooth))
    max_frames = len(flow_rate_smooth)
    ind_10pc = int(0.1 * max_frames)
    ind_65pc = int(0.65 * max_frames)
    flow_rate_10_65 = flow_rate_smooth[ind_10pc:ind_65pc]
    peak_threshold = 0.35 * np.abs(flow_rate_10_65).max()
    try:
        _peaks_pos, _ = find_peaks(flow_rate_smooth)
        _peaks_neg, _ = find_peaks(-flow_rate_smooth)
        _peaks = sorted(np.concatenate((_peaks_neg, _peaks_pos)))
        _peaks = [p for p in _peaks if p >= ind_10pc and p <= ind_65pc]
        main_peak = [p for p in _peaks if np.abs(flow_rate_smooth[p]) > peak_threshold][0]
        if flow_rate_smooth[main_peak] < 0:
            flow_rate_smooth = flow_rate_smooth * -1
            flow_rate_nnunet = flow_rate_nnunet * -1
    except:
        if flow_rate_smooth[max_ind] < 0:
            flow_rate_smooth = flow_rate_smooth * -1
            flow_rate_nnunet = flow_rate_nnunet * -1

    return area_nnunet, area_smooth, mean_velocity_nnunet, flow_rate_nnunet,\
        flow_rate_smooth, tt_smooth


def do_process(_local_dir, nifti_dir, target_dir, task_name, _logger):
    target_base = os.path.join(target_dir, task_name, 'Results')
    target_imagesTs = os.path.join(target_dir, task_name, "imagesTs")
    Results_test = os.path.join(target_base, 'ensemble')
    curve_save_dir = os.path.join(target_base, 'curves')
    tt_save_dir = os.path.join(target_base, 'tt_flow')
    gif_dir = os.path.join(_local_dir, 'gifs_flow')

    if not os.path.exists(curve_save_dir):
        os.mkdir(curve_save_dir)
    if not os.path.exists(tt_save_dir):
        os.mkdir(tt_save_dir)
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    Xte = sorted(os.listdir(Results_test))
    volumes_test_flow = np.zeros((len(Xte), 5), dtype=object)
    peak_calc_failed = []

    for d, file_path in enumerate(Xte):
        _logger.info(f'[{d+1}/{len(Xte)}]: {file_path}')

        img_name = file_path[10:]  # e.g. flow_v300in_0.nii.gz
        study_ID = file_path[:9]  # e.g. A_S000001
        eid = file_path.replace('.nii.gz', '')  # e.g. A_S000001_flow_v300in_0

        seg_path = os.path.join(Results_test, file_path)
        subject_dir = os.path.join(nifti_dir, study_ID)
        tt_flow = np.loadtxt(os.path.join(subject_dir, f'{img_name.replace(".nii.gz","")}_tt.txt'))

        # Load images
        img_mag = nib.load(os.path.join(target_imagesTs, file_path.replace('.nii.gz', '_0000.nii.gz')))
        img_flow = nib.load(os.path.join(target_imagesTs, file_path.replace('.nii.gz', '_0001.nii.gz')))
        img_flow_array = img_flow.get_fdata()
        dx, dy, _, _ = img_mag.header['pixdim'][1:5]
        area_per_voxel = dx * dy * 1e-2  # calculate the area per pixel of the image
        seg_nim = nib.load(seg_path)
        seg = seg_nim.get_fdata()
        _, _, T = seg.shape
        venc = int(seg_path.split('_v')[1].split('in_')[0])

        # ---------------------------------------------------
        # Flow calculation
        # ---------------------------------------------------
        ao_area_nnunet, ao_area_smooth, ao_mean_velocity_nnunet, \
            ao_flow_rate_nnunet, ao_flow_rate_smooth, tt_smooth = \
            calculate_flow(seg, img_flow_array, tt_flow, T, venc, area_per_voxel)

        volumes_test_flow[d, 0] = eid
        try:
            # ---------------------------------------------------
            # Find peak aortic flow
            # ---------------------------------------------------
            peaks, _ = find_peaks(ao_flow_rate_smooth)
            peaks = peaks[ao_flow_rate_smooth[peaks].argmax()]

            tt_flow_max_nnunet = tt_smooth[peaks]
            ao_flow_rate_nnunet_max = ao_flow_rate_smooth[peaks]
            auc_nnunet = auc(tt_smooth, ao_flow_rate_smooth) / 1000

            volumes_test_flow[d, 1] = peaks
            volumes_test_flow[d, 2] = ao_flow_rate_nnunet_max
            volumes_test_flow[d, 3] = auc_nnunet
            volumes_test_flow[d, 4] = tt_flow_max_nnunet
        except:
            peak_calc_failed.append(eid)
            _logger.info(f'Peak calculation failed. Skipping {eid}')

        # # Save curves
        np.savetxt(os.path.join(curve_save_dir, f'{eid}_ao_area.txt'), ao_area_nnunet)
        np.savetxt(os.path.join(curve_save_dir, f'{eid}_ao_area_smooth.txt'), ao_area_smooth)
        np.savetxt(os.path.join(curve_save_dir, f'{eid}_ao_flow_rate.txt'), ao_flow_rate_nnunet)
        np.savetxt(os.path.join(curve_save_dir, f'{eid}_ao_flow_rate_smooth.txt'), ao_flow_rate_smooth)
        np.savetxt(os.path.join(curve_save_dir, f'{eid}_ao_mean_velocity.txt'),
                   ao_mean_velocity_nnunet)

        # Save tt_flow
        np.savetxt(os.path.join(tt_save_dir, f'{eid}_tt_flow.txt'), tt_flow)
        np.savetxt(os.path.join(tt_save_dir, f'{eid}_tt_flow_smooth.txt'), 
            tt_smooth)

        # ---------------------------------------------------
        # Also save seg and curves in original nifti folders
        # ---------------------------------------------------
        flow_label_path = os.path.join(subject_dir, img_name).replace(".nii.gz", "_seg_nnUnet.nii.gz")
        if not os.path.exists(flow_label_path):
            seg = np.expand_dims(seg, 2)
            save_nifti(seg_nim.affine, seg, seg_nim.header, flow_label_path)

        results_flow_subdir = os.path.join(subject_dir, "results_flow")
        if not os.path.exists(results_flow_subdir):
            os.mkdir(results_flow_subdir)

        # Copy all curve txt files
        txt_files = [f for f in os.listdir(curve_save_dir) if study_ID in f and f.endswith("txt")]
        for tt in txt_files:
            os.system(f"cp {os.path.join(curve_save_dir, tt)} {results_flow_subdir}/")

        # Copy all tt txt files
        txt_files = [f for f in os.listdir(tt_save_dir) if study_ID in f and f.endswith("txt")]
        for tt in txt_files:
            os.system(f"cp {os.path.join(tt_save_dir, tt)} {results_flow_subdir}/")

    _logger.info(f'Eids failed - peak: {peak_calc_failed}')
    _logger.info(f'Total failed: {len(peak_calc_failed)}')

    df = pd.DataFrame(volumes_test_flow)
    df.to_csv(f'{target_base}/volumes_test_stack.csv',
              header=['eid', 'max_frame_nnunet', 'ao_flow_rate_nnunet_max', 'auc_nnunet', 'tt_flow_max_nnunet'],
              index=False)
    os.system(f"cp {target_base}/volumes_test_stack.csv {results_flow_subdir}/report_aortic_flow.csv")


def main(json_config_path):
    if os.path.exists(json_config_path):
        cfg = Params(json_config_path).dict
    else:
        raise FileNotFoundError
    local_dir = cfg['DEFAULT_LOCAL_DIR']
    nifti_dir = os.path.join(local_dir, cfg['DEFAULT_SUBDIR_NIFTI'])
    target_dir = os.path.join(local_dir, cfg['DEFAULT_NNUNET_NIFTI'])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    log_dir = os.path.join(local_dir, cfg['DEFAULT_LOG_DIR'])
    # Start logging console prints
    time_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_txt_file = os.path.join(log_dir, 'calculate_flow_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting flow calculation\n')
    do_process(local_dir, nifti_dir, target_dir, "Task118_AscAoFlow", logger)
    logger.info('Closing calculate_flow_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
