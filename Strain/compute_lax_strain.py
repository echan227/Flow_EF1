"""
$ python3 compute_lax_strain.py  --help
usage: compute_lax_strain.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib
import argparse
import os

# imports 3rd party
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# imports - 3rd party:
from datetime import datetime
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, set_logger
from Strain.strain_functions import get_points_2Ch_longit_strain, get_points_4Ch_longit_strain, calc_peak_diast_strain, compute_line_strain, compute_MAPSE, strain_Tshape


def do_studies(study_IDs, nifti_dir, logger):
    for study_ID_counter, study_ID in enumerate(study_IDs):
        logger.info(f"do_studies() [{study_ID_counter}]: {study_ID}")
        subject_dir = os.path.join(nifti_dir, study_ID)
        results_dir = os.path.join(subject_dir, 'results_strain')
        if not os.path.exists(results_dir):
            os.system('mkdir -p {0}'.format(results_dir))

        # =============================================================================
        # Longit strain la_2Ch
        # =============================================================================
        # Load data
        la_2Ch_img_dir = os.path.join(subject_dir, 'la_2Ch.nii.gz')
        la_2Ch_seg_dir = os.path.join(subject_dir, 'la_2Ch_seg_nnUnet.nii.gz')
        if not (os.path.exists(la_2Ch_img_dir) or os.path.exists(la_2Ch_seg_dir)):
            logger.info('No la_2Ch data for subject {}'.format(study_ID))
            continue
        try:
            N_frames_2Ch, points_myo_la_2Ch, valve_points_2Ch, dx, dt_2Ch, distance_apex_mid_valve_2Ch, tt_2Ch = \
                get_points_2Ch_longit_strain(la_2Ch_img_dir, la_2Ch_seg_dir, results_dir)
        except:
            logger.info('Failed to compute 2Ch strain points for subject {}'.format(study_ID))
            continue
        
        # =============================================================================
        # 1. Longit line strain
        # =============================================================================
        try:
            longit_strain_la_2Ch_smooth = compute_line_strain(N_frames_2Ch, points_myo_la_2Ch, results_dir, chamber = '2Ch')           
        except:
            logger.info('Failed to compute 2Ch line strain for subject {}'.format(study_ID))
            continue

        # =============================================================================
        # 2. Longit strain  - MAPSE
        # =============================================================================
        try:
             mapse_la_2Ch_smooth = compute_MAPSE(N_frames_2Ch, valve_points_2Ch, dx, results_dir, chamber = '2Ch')
        except:
            logger.info('Failed to compute 2Ch MAPSE for subject {}'.format(study_ID))
            continue
        
        # =============================================================================
        # 3. Longit strain  - T shape
        # =============================================================================
        try:
            longit_strain_T_la_2Ch_smooth = strain_Tshape(distance_apex_mid_valve_2Ch, results_dir, N_frames_2Ch, chamber = '4Ch')
        except:
            logger.info('Failed to compute 2Ch T shape for subject {}'.format(study_ID))
            continue    
    
        ### Calculate peak strain (here using long strain, not MAPSE/T (can be changed))##
        Name_curve = '2Ch_long_strain'
        Long_peak_strain_2Ch, TPK_long_strain_2Ch, diast_long_strain_2Ch, TPK_diast_long_strain_2Ch  = calc_peak_diast_strain(longit_strain_la_2Ch_smooth, dt_2Ch, results_dir, Name_curve)


        # =============================================================================
        # Longit strain la_4Ch
        # =============================================================================
        # Load data
        la_4Ch_img_dir = os.path.join(subject_dir, 'la_4Ch.nii.gz')
        la_4Ch_seg_dir = os.path.join(subject_dir, 'la_4Ch_seg_nnUnet.nii.gz')
        if not (os.path.exists(la_4Ch_img_dir) or os.path.exists(la_4Ch_seg_dir)):
            logger.info('No la_4Ch data for subject {}'.format(study_ID))
            continue
        try:
            N_frames_4Ch, points_myo_la_4Ch, valve_points_4Ch, dx, dt_4Ch, distance_apex_mid_valve_4Ch, tt_4Ch = \
                get_points_4Ch_longit_strain(la_4Ch_img_dir, la_4Ch_seg_dir, results_dir)
        except:
            logger.info('Failed to compute 4Ch strain points for subject {}'.format(study_ID))


        # =============================================================================
        # 1. Longit line strain
        # =============================================================================
        try:
           longit_strain_la_4Ch_smooth = compute_line_strain(N_frames_4Ch, points_myo_la_4Ch, results_dir, chamber = '4Ch')      
        except:
            logger.info('Failed to compute 4Ch line strain for subject {}'.format(study_ID))
            continue

        # =============================================================================
        # 2. Longit strain  - MAPSE
        # =============================================================================
        try:
             mapse_la_4Ch_smooth = compute_MAPSE(N_frames_4Ch, valve_points_4Ch, dx, results_dir, chamber = '4Ch')
        except:
            logger.info('Failed to compute 4Ch MAPSE for subject {}'.format(study_ID))
            continue
        
        # =============================================================================
        # 3. Longit strain  - T shape
        # =============================================================================
        try:
            longit_strain_T_la_4Ch_smooth = strain_Tshape(distance_apex_mid_valve_4Ch, results_dir, N_frames_4Ch, chamber = '4Ch')
        except:
            logger.info('Failed to compute 4Ch T shape for subject {}'.format(study_ID))
            continue    
    
        ### Calculate peak strain (here using long strain, not MAPSE/T (can be changed))##
        Name_curve = '4Ch_long_strain'
        Long_peak_strain_4Ch, TPK_long_strain_4Ch, diast_long_strain_4Ch, TPK_diast_long_strain_4Ch  = calc_peak_diast_strain(longit_strain_la_4Ch_smooth, dt_4Ch, results_dir, Name_curve)

        # Plots and store results
        plt.figure()
        if os.path.exists(la_2Ch_img_dir):
            plt.plot(longit_strain_la_2Ch_smooth, label='Ell 2Ch')
            plt.plot(longit_strain_T_la_2Ch_smooth, label='Ell 2Ch T shape')
            plt.plot(mapse_la_2Ch_smooth, label='MAPSE 2Ch')
        if os.path.exists(la_4Ch_img_dir):
            plt.plot(longit_strain_la_4Ch_smooth, label='Ell 4Ch')
            plt.plot(longit_strain_T_la_4Ch_smooth, label='Ell 4Ch T shape')
            plt.plot(mapse_la_4Ch_smooth, label='MAPSE 4Ch')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'Ell.png'))
        plt.close('all')

        if os.path.exists(la_2Ch_img_dir) and os.path.exists(la_4Ch_img_dir):
            vec = np.zeros((max(N_frames_4Ch, N_frames_2Ch) + 1, 11), dtype=object)
            vec[0:len(tt_2Ch) + 1, 0] = np.hstack(['trigger time 2Ch', tt_2Ch])
            vec[0:len(tt_4Ch) + 1, 1] = np.hstack(['trigger time 4Ch', tt_4Ch])
            #  Strain
            if len(longit_strain_la_2Ch_smooth) == len(longit_strain_la_4Ch_smooth):
                vec[0:len(tt_2Ch) + 1, 2] = np.hstack(
                    ['Average line strain', np.mean([longit_strain_la_2Ch_smooth, longit_strain_la_4Ch_smooth], axis=0)])
            else:
                vec[0, 1] = 'Average line strain'
            vec[0:len(longit_strain_la_2Ch_smooth) + 1, 3] = np.hstack(['Ell 2Ch line strain', longit_strain_la_2Ch_smooth])
            vec[0:len(longit_strain_la_4Ch_smooth) + 1, 4] = np.hstack(['Ell 4Ch line strain', longit_strain_la_4Ch_smooth])
            # T strain
            if len(longit_strain_T_la_2Ch_smooth) == len(longit_strain_T_la_4Ch_smooth):
                vec[0:len(tt_2Ch) + 1, 5] = np.hstack(
                    ['Average T shape', np.mean([longit_strain_T_la_2Ch_smooth, longit_strain_T_la_4Ch_smooth], axis=0)])
            else:
                vec[0, 4] = 'Average T shape'
            vec[0:len(longit_strain_T_la_2Ch_smooth) + 1, 6] = np.hstack(['Ell 2Ch T shape', longit_strain_T_la_2Ch_smooth])
            vec[0:len(longit_strain_T_la_4Ch_smooth) + 1, 7] = np.hstack(['Ell 4Ch T shape', longit_strain_T_la_4Ch_smooth])
            # MAPSE
            if len(mapse_la_2Ch_smooth) == len(mapse_la_4Ch_smooth):
                vec[0:len(tt_2Ch) + 1, 8] = np.hstack(
                    ['Average MAPSE', np.mean([mapse_la_2Ch_smooth, mapse_la_4Ch_smooth], axis=0)])
            else:
                vec[0, 7] = 'Average MAPSE'
            vec[0:len(mapse_la_2Ch_smooth) + 1, 9] = np.hstack(['Ell 2Ch MAPSE', mapse_la_2Ch_smooth])
            vec[0:len(mapse_la_4Ch_smooth) + 1, 10] = np.hstack(['Ell 4Ch MAPSE', mapse_la_4Ch_smooth])

            df = pd.DataFrame(vec.T)
            new_name = '{0}/Ell.csv'.format(results_dir)
            df.to_csv(new_name, header=None, index=False, sep=',', encoding='utf-8')

        elif os.path.exists(la_2Ch_img_dir):
            vec = np.zeros((max(N_frames_4Ch, N_frames_2Ch) + 1, 11), dtype=object)
            vec[0:len(tt_2Ch) + 1, 0] = np.hstack(['trigger time 2Ch', tt_2Ch])
            vec[0:len(tt_4Ch) + 1, 1] = 'trigger time 4Ch'
            #  Strain
            vec[0:len(tt_2Ch) + 1, 2] = np.hstack(
                ['Average line strain', longit_strain_la_2Ch_smooth])
            vec[0:len(longit_strain_la_2Ch_smooth) + 1, 3] = np.hstack(['Ell 2Ch line strain', longit_strain_la_2Ch_smooth])
            vec[0:len(longit_strain_la_4Ch_smooth) + 1, 4] = 'Ell 4Ch line strain'
            # T strain
            vec[0:len(tt_2Ch) + 1, 5] = np.hstack(
                ['Average T shape', longit_strain_T_la_2Ch_smooth])
            vec[0:len(longit_strain_T_la_2Ch_smooth) + 1, 6] = np.hstack(['Ell 2Ch T shape', longit_strain_T_la_2Ch_smooth])
            vec[0:len(longit_strain_T_la_4Ch_smooth) + 1, 7] = 'Ell 4Ch T shape'
            # MAPSE
            vec[0:len(tt_2Ch) + 1, 8] = np.hstack(['Average MAPSE', mapse_la_2Ch_smooth])
            vec[0:len(mapse_la_2Ch_smooth) + 1, 9] = np.hstack(['Ell 2Ch MAPSE', mapse_la_2Ch_smooth])
            vec[0:len(mapse_la_4Ch_smooth) + 1, 10] = 'Ell 4Ch MAPSE'

            df = pd.DataFrame(vec.T)
            new_name = '{0}/Ell.csv'.format(results_dir)
            df.to_csv(new_name, header=None, index=False, sep=',', encoding='utf-8')

        elif os.path.exists(la_4Ch_img_dir):
            vec = np.zeros((max(N_frames_4Ch, N_frames_2Ch) + 1, 11), dtype=object)
            vec[0:len(tt_2Ch) + 1, 0] = 'trigger time 2Ch' 
            vec[0:len(tt_4Ch) + 1, 1] = np.hstack(['trigger time 4Ch', tt_2Ch])
            #  Strain
            vec[0:len(tt_2Ch) + 1, 2] = np.hstack(['Average line strain', longit_strain_la_4Ch_smooth])
            vec[0:len(longit_strain_la_2Ch_smooth) + 1, 3] = 'Ell 2Ch line strain'
            vec[0:len(longit_strain_la_4Ch_smooth) + 1, 4] = np.hstack(['Ell 4Ch line strain', longit_strain_la_4Ch_smooth])
            # T strain
            vec[0:len(tt_2Ch) + 1, 5] = np.hstack(['Average T shape', longit_strain_T_la_4Ch_smooth])
            vec[0:len(longit_strain_T_la_2Ch_smooth) + 1, 6] =  'Ell 2Ch T shape'
            vec[0:len(longit_strain_T_la_4Ch_smooth) + 1, 7] = np.hstack(['Ell 4Ch T shape', longit_strain_T_la_4Ch_smooth]) 
            # MAPSE
            vec[0:len(tt_2Ch) + 1, 8] = np.hstack(['Average MAPSE', mapse_la_4Ch_smooth])
            vec[0:len(mapse_la_2Ch_smooth) + 1, 9] = 'Ell 2Ch MAPSE' 
            vec[0:len(mapse_la_4Ch_smooth) + 1, 10] = np.hstack(['Ell 4Ch MAPSE', mapse_la_2Ch_smooth])

            df = pd.DataFrame(vec.T)
            new_name = '{0}/Ell.csv'.format(results_dir)
            df.to_csv(new_name, header=None, index=False, sep=',', encoding='utf-8')

    # Save CSV file
    out_csv = -1 * np.ones((1, 8))
    if os.path.exists(la_4Ch_img_dir) and os.path.exists(la_4Ch_seg_dir):
        out_csv[0, 0] = Long_peak_strain_4Ch
        out_csv[0, 1] = TPK_long_strain_4Ch
        out_csv[0, 4] = diast_long_strain_4Ch
        out_csv[0, 6] = TPK_diast_long_strain_4Ch
    if os.path.exists(la_2Ch_img_dir) and os.path.exists(la_2Ch_seg_dir):
        out_csv[0, 2] = Long_peak_strain_2Ch
        out_csv[0, 3] = TPK_long_strain_2Ch
        out_csv[0, 5] = diast_long_strain_2Ch
        out_csv[0, 7] = TPK_diast_long_strain_2Ch

    header= ['long_peak_strain_4Ch','TPK_long_strain_4Ch','long_peak_strain_2Ch','TPK_long_strain_2Ch',
                'diast_long_strain_4Ch','diast_long_strain_2Ch','TPK_long_diast_strain_4Ch','TPK_long_diast_strain_2Ch',]
    df=pd.DataFrame(out_csv)
    df.to_csv('{0}/report_long_strain.csv'.format(results_dir),
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
    target_dir = os.path.join(local_dir, cfg['DEFAULT_NNUNET_NIFTI'])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    log_dir = os.path.join(local_dir, cfg['DEFAULT_LOG_DIR'])
    # Start logging console prints
    time_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_txt_file = os.path.join(log_dir, 'compute_lax_strain_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting computing LAX strain\n')
    # Find studies to analyse
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    do_studies(study_IDs, nifti_dir, logger)
    logger.info('Closing compute_lax_strain_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
