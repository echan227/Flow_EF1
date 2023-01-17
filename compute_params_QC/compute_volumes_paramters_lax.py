# !/usr/bin/env python3 pylint: disable=invalid-name,unused-import,missing-module-docstring,missing-docstring,
# line-too-long,too-many-arguments,too-many-locals,too-many-branches,too-many-statements,bad-whitespace,
# pointless-string-statement,trailing-newlines

"""
$ python3 v.py  --help
usage: generate_SAX_panel3.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib:
import os

# imports 3rd party
from datetime import datetime

# imports - local
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, set_logger
from compute_params_QC.functions_lax import compute_atria_MAPSE_TAPSE_params


# =============================================================================
# FUNCTIONS
# =============================================================================
def do_studies(study_IDs, nifti_dir, logger):
    for study_ID_counter, study_ID in enumerate(study_IDs):
        logger.info(f"do_studies() [{study_ID_counter}]: {study_ID}")
        subject_dir = os.path.join(nifti_dir, study_ID)
        results_dir = os.path.join(subject_dir, 'results')
        if not os.path.exists(results_dir):
            os.system('mkdir -p {0}'.format(results_dir))
        compute_atria_MAPSE_TAPSE_params(study_ID, subject_dir, results_dir, logger)


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
    log_txt_file = os.path.join(log_dir, 'compute_volume_parameters_lax_'+time_file+'.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting computing parameters and volumes for lax\n')
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    do_studies(study_IDs, nifti_dir, logger)
    logger.info('Closing compute_volume_parameters_lax_log_{}.txt'.format(time_file))


if __name__ == '__main__':
    import sys
    sys.path.append('/home/bram/Scripts/AI_CMR_QC')
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
