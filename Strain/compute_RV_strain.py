"""
$ python3 compute_RV_strain.py  --help
usage: compute_RV_strain.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib
import argparse
import os
from unittest import result

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
from Strain.strain_functions import get_points_4Ch_RV_longit_strain, compute_RV_longit_strain, get_points_sax_RV_circ_strain, compute_RV_cric_strain




def do_studies(study_IDs, nifti_dir, logger):
    for study_ID_counter, study_ID in enumerate(study_IDs):
        logger.info(f"do_studies() [{study_ID_counter}]: {study_ID}")
        subject_dir = os.path.join(nifti_dir, study_ID)
        results_dir = os.path.join(subject_dir, 'results_strain')
        if not os.path.exists(results_dir):
            os.system('mkdir -p {0}'.format(results_dir))
        
        # =============================================================================
        # Longit RV strain la_4Ch
        # =============================================================================
        # Load data
        la_4Ch_img_dir = os.path.join(subject_dir, 'la_4Ch.nii.gz')
        la_4Ch_seg_dir = os.path.join(subject_dir, 'la_4Ch_seg_nnUnet.nii.gz')
        if not (os.path.exists(la_4Ch_img_dir) or os.path.exists(la_4Ch_seg_dir)):
            logger.info('No la_4Ch data for subject {}'.format(study_ID))
            continue
        
        try:
            total_length_4Ch, total_apex_free, total_apex_rvlv, N_frames_4Ch, dt = \
                get_points_4Ch_RV_longit_strain(la_4Ch_img_dir, la_4Ch_seg_dir, results_dir)
        except:
            logger.info('Failed to compute 4Ch strain points for subject {}'.format(study_ID))

        # =============================================================================
        # Calculate RV longit strain
        # =============================================================================
        try:
            strain_longit = compute_RV_longit_strain(total_length_4Ch,  total_apex_free, total_apex_rvlv, N_frames_4Ch, results_dir, dt)
        except:
            strain_longit = -np.ones(11)
        
        # =============================================================================
        # Circ RV strain sax
        # =============================================================================
        # Load data
        sa_img_dir = os.path.join(subject_dir, 'sa.nii.gz')
        sa_seg_dir = os.path.join(subject_dir, 'sa_seg_nnUnet.nii.gz')
        if not (os.path.exists(sa_img_dir) or os.path.exists(sa_seg_dir)):
            logger.info('No SAX data for subject {}'.format(study_ID))
            continue

        try:
            strain_circ_RV, strain_circ_RV_raw, dt = \
                get_points_sax_RV_circ_strain(sa_img_dir, sa_seg_dir, results_dir)
        except:
            logger.info('Failed to compute 4Ch strain points for subject {}'.format(study_ID))

        # =============================================================================
        # Calculate RV circ strain
        # =============================================================================
        try:
            strain_circ = compute_RV_cric_strain(strain_circ_RV,strain_circ_RV_raw, results_dir, dt)
        except:
            strain_circ = -np.ones(5)

        RV_strain = np.concatenate([strain_longit, strain_circ])

        # Save strain csv
        header= header=['peak_long_RV_strain', 'TPK_long_RV_strain','peak_long_RV_strain_freewall', 'TPK_long_RV_strain_freewall', 'diast_SR_long','TPK_diast_SR_long','diast_SR_long_freewall','TPK_diast_SR_long_freewall', 'peak_circ_strain', 'TPK_circ_strain','diast_SR_circ','TPK_diast_SR_circ','flag_long_peak','error_long','flag_circ_peak','error_circ']
        df=pd.DataFrame(RV_strain.reshape(1,-1))
        df.to_csv('{0}/report_RV_strain.csv'.format(results_dir),
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
    log_txt_file = os.path.join(log_dir, 'compute_rv_strain_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting computing RV strain\n')
    # Find studies to analyse
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    do_studies(study_IDs, nifti_dir, logger)
    logger.info('Closing compute_rv_strain_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
