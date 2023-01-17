"""
$ python3 compute_volumes_paramters_SAX.py  --help
usage: compute_volumes_paramters_SAX.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib:
import argparse
import os

# imports 3rd party
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model
from datetime import datetime

# imports - local
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, set_logger
from compute_params_QC.functions_SAX_QC import sa_pass_quality_volumes, \
    sa_pass_quality_control_images, sa_pass_quality_LSTM, compute_volumes


# --------- --------- --------- ---------
# *PyTorch*
# PyTorch models:
def tf_load_models(model_dir, name):
    model_name = os.path.join(model_dir, name)
    model = load_model(model_name)
    return model


def do_studies(study_IDs, nifti_dir, model, logger):
    for study_ID_counter, study_ID in enumerate(study_IDs):
        logger.info(f"do_studies() [{study_ID_counter}]: {study_ID}")
        subject_dir = os.path.join(nifti_dir, study_ID)
        filename_seg = os.path.join(subject_dir, 'sa_seg_nnUnet.nii.gz')
        results_dir = os.path.join(subject_dir, 'results_SAX')
        if not os.path.exists(results_dir):
            os.system('mkdir -p {0}'.format(results_dir))
        try:
            volume_LV, volume_RV = compute_volumes(logger, subject_dir, filename_seg, results_dir, study_ID)
        except:
            logger.info('Error computing the LV or RV volume')

        # =============================================================================
        # QC
        # =============================================================================
        if len(volume_LV) > 1:
            QC = np.zeros((4, 1))
            QC[0] = sa_pass_quality_volumes(volume_LV, volume_RV)
            QC[1] = sa_pass_quality_control_images(logger, filename_seg)
            QC[2] = sa_pass_quality_LSTM(logger, model, volume_LV, 'LV')
            QC[3] = sa_pass_quality_LSTM(logger, model, volume_RV, 'RV')

            df = pd.DataFrame(QC.T)
            df.to_csv('{0}/QC.csv'.format(results_dir), header=['QC1', 'QC2', 'QC3', 'QC4'], index=False)


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
    # Start logger console prints
    time_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_txt_file = os.path.join(log_dir, 'compute_volume_parameters_sax_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting computing parameters and volumes for sax\n')
    model_dir = os.path.join(os.getcwd(), cfg['DEFAULT_MODELS_DIR'])
    model = tf_load_models(model_dir, cfg['DEFAULT_MODEL_QC'])
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    do_studies(study_IDs, nifti_dir, model, logger)
    logger.info('Closing compute_volume_parameters_sax_log_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/basic_opt.json'
    main(DEFAULT_JSON_FILE)
