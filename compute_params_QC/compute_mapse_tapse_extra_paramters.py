# imports - stdlib:
import os
import numpy as np

# imports 3rd party
from datetime import datetime
import nibabel as nib
import pylab as plt
import pandas as pd

# imports - local
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, set_logger


def calc_valve_velo_diast(study_ID, curve, dt, valve_curve_name, save_dir):
    # TODO:how to deal with error/exception, not '-1' but maybe like in the other def using []?
    if curve[0] > -2:
        # get first derivatives and a second more smooth line to check peak point
        first_derived = np.gradient(curve)

        ## FIND Peak descend using different smoothened curves.
        # Find mapse to check only minima after, but not atrial kick, so taking out last 25% of data (idx_at)
        idx_at = int(len(curve) / 4)
        vapse_idx = np.argmax(curve)
        min_idx = np.argmin(first_derived[:-idx_at])
        min = np.min(first_derived[:-idx_at])
        max_idx = np.argmax(first_derived[:-idx_at])
        max = np.max(curve)

        # get e'
        if min_idx > vapse_idx:
            if len(curve)<45:
                mean_min = first_derived[min_idx]
            else:
                mean_min = np.average([first_derived[min_idx - 1], first_derived[min_idx], first_derived[min_idx + 2]])
            num = int(len(curve) / 6)
            if num > 5:
                num = 5
            if num < 3:
                num = 3

            # make slopeline for plot
            Y_on_MAPSE = curve[min_idx]
            min_X_range_slope = min_idx - num
            max_X_range_slope = min_idx + num
            range_slope = np.arange(min_X_range_slope, max_X_range_slope, 0.4)
            Ys_slope = ((range_slope - min_idx) * mean_min) + Y_on_MAPSE

            fig = plt.figure()
            plt.plot(curve)
            plt.plot(range_slope, Ys_slope, 'r')
            plt.plot(vapse_idx, max, 'rs', label= 'max')
            plt.plot(min_idx, Y_on_MAPSE, 'rx', label='diast velocity')
            plt.annotate(valve_curve_name.split('_')[0], (vapse_idx, max))
            plt.title('{}: {}'.format(study_ID, valve_curve_name))
            plt.legend(loc='upper right')
            fig.savefig('{0}/{1}.png'.format(save_dir,valve_curve_name))
            plt.close(fig)

            diast_vapse = mean_min / (dt / 1000)
            vapse = max
        elif min_idx == 0:
            mean_min = -1
            diast_vapse = mean_min
            vapse = max
        else:
            mean_min = -1
            diast_vapse = mean_min
            vapse = max
    else:
        mean_min = -1
        diast_vapse = mean_min
        vapse = -1

    return vapse, diast_vapse


def do_studies(study_IDs, nifti_dir, logger):
    for study_ID_counter, study_ID in enumerate(study_IDs):
        logger.info(f"do_studies() [{study_ID_counter}]: {study_ID}")
        subject_dir = os.path.join(nifti_dir, study_ID)
        results_dir = os.path.join(subject_dir, 'results')
        if not os.path.exists(results_dir):
            os.system('mkdir -p {0}'.format(results_dir))

        if os.path.exists(os.path.join(results_dir, 'LV_mid_mapse_smooth_2Ch.txt')) and os.path.exists(os.path.join(subject_dir, 'la_2Ch.nii.gz')):
            mapse_2Ch = np.loadtxt(os.path.join(results_dir, 'LV_mid_mapse_smooth_2Ch.txt'))
            nim = nib.load(os.path.join(subject_dir, 'la_2Ch.nii.gz'))
            dt = nim.header['pixdim'][5]
            mapse_2Ch, diast_mapse_2Ch = calc_valve_velo_diast(study_ID, mapse_2Ch, dt, 'MAPSE_2Ch', results_dir)
        else:
            mapse_2Ch = -1
            diast_mapse_2Ch = -1
        
        # 4Ch MAPSE
        if os.path.exists(os.path.join(results_dir, 'LV_mid_mapse_smooth_4Ch.txt')) and os.path.exists(os.path.join(subject_dir, 'la_4Ch.nii.gz')):
            mapse_4Ch = np.loadtxt(os.path.join(results_dir, 'LV_mid_mapse_smooth_4Ch.txt'))
            nim = nib.load(os.path.join(subject_dir, 'la_4Ch.nii.gz'))
            dt = nim.header['pixdim'][5]
            mapse_4Ch, diast_mapse_4Ch = calc_valve_velo_diast(study_ID, mapse_4Ch, dt, 'MAPSE_4Ch', results_dir)
        else:
            mapse_4Ch = -1
            diast_mapse_4Ch = -1
        
        # 4Ch TAPSE
        if os.path.exists(os.path.join(results_dir, 'RA_tapse_smooth_la4Ch.txt')) and os.path.exists(os.path.join(subject_dir, 'la_4Ch.nii.gz')):
            tapse = np.loadtxt(os.path.join(results_dir, 'RA_tapse_smooth_la4Ch.txt'))
            nim = nib.load('{0}/la_4Ch.nii.gz'.format(subject_dir))
            dt = nim.header['pixdim'][5]
            tapse_4Ch, diast_tapse_4Ch = calc_valve_velo_diast(study_ID, tapse, dt, 'TAPSE_4Ch', results_dir)
        else:
            tapse_4Ch = -1
            diast_tapse_4Ch = -1

        # ####### Combine in a csv ###########
        out_csv = -1 * np.ones((1, 6))
        out_csv[0, 0] = mapse_2Ch
        out_csv[0, 1] = diast_mapse_2Ch
        out_csv[0, 2] = mapse_4Ch
        out_csv[0, 3] = diast_mapse_4Ch
        out_csv[0, 4] = tapse_4Ch
        out_csv[0, 5] = diast_tapse_4Ch

        header= ['mapse_2Ch','diast_mapse_2Ch','mapse_4Ch','diast_mapse_4Ch',
                    'tapse_4Ch','diast_tapse_4Ch']
        df=pd.DataFrame(out_csv)
        df.to_csv('{0}/report_mapse_tapse.csv'.format(results_dir),
                header=header, index=False)

# =============================================================================
# MAIN
# =============================================================================
def main(json_config_path):
    if os.path.exists(json_config_path):
        cfg = Params(json_config_path).dict
    else:
        raise FileNotFoundError
    local_dir = cfg['DEFAULT_LOCAL_DIR']
    nifti_dir = os.path.join(local_dir, cfg['DEFAULT_SUBDIR_NIFTI'])
    log_dir = os.path.join(local_dir, cfg['DEFAULT_LOG_DIR'])
    # Start logging console prints
    time_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_txt_file = os.path.join(log_dir, 'compute_mapse_tapse_extra_paramters_log_'+time_file+'.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting computing parameters and volumes for lax\n')
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    do_studies(study_IDs, nifti_dir, logger)
    logger.info('Closing compute_mapse_tapse_extra_paramters_log_{}.txt'.format(time_file))


if __name__ == '__main__':
    import sys
    sys.path.append('/home/bram/Scripts/AI_CMR_QC')
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)

