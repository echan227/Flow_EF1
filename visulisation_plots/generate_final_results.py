# !/usr/bin/env python3 pylint: disable=invalid-name,unused-import,missing-module-docstring,missing-docstring,
# line-too-long,too-many-arguments,too-many-locals,too-many-branches,too-many-statements,bad-whitespace,
# pointless-string-statement,trailing-newlines

"""
$ python3 generate_final_results.py  --help
usage: generate_final_results.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""
# imports 3rd party
import pandas as pd
import numpy as np
from datetime import datetime

# imports - local
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, set_logger

import os


# =============================================================================
# FUNCTIONS
# =============================================================================
def do_studies(study_IDs, nifti_dir, server_dir, logger):
    out_csv = -1 * np.ones((len(study_IDs), 17), dtype=object)
    gifs_dir = os.path.join(server_dir, 'gifs_SAX')
    if not os.path.exists(gifs_dir):
        os.system('mkdir -p {0}'.format(gifs_dir))

    for study_ID_counter, study_ID in enumerate(study_IDs):
        logger.info(f"do_studies() [{study_ID_counter+1}]: {study_ID}")
        subject_dir = os.path.join(nifti_dir, study_ID)
        results_dir = os.path.join(subject_dir, 'results_SAX')
        out_csv[study_ID_counter, 0] = study_ID

        if os.path.exists(os.path.join(results_dir, 'report_volumes.csv')):
            report_volumes = np.squeeze(pd.read_csv(os.path.
                                                    join(results_dir, 'report_volumes.csv')).values)
            out_csv[study_ID_counter, 1] = report_volumes[0]  # LVEDV
            out_csv[study_ID_counter, 2] = report_volumes[1]  # LVESV
            out_csv[study_ID_counter, 3] = report_volumes[2]  # LVSV
            out_csv[study_ID_counter, 4] = report_volumes[3]  # LVEF
            out_csv[study_ID_counter, 5] = report_volumes[4]  # LVM
            out_csv[study_ID_counter, 6] = report_volumes[5]  # RVEDV
            out_csv[study_ID_counter, 7] = report_volumes[6]  # RVESV
            out_csv[study_ID_counter, 8] = report_volumes[7]  # RVSV
            out_csv[study_ID_counter, 9] = report_volumes[8]  # RVEF
            out_csv[study_ID_counter, 10] = report_volumes[9]  # LV ED frame
            out_csv[study_ID_counter, 11] = report_volumes[10]  # LV ES frame
            out_csv[study_ID_counter, 12] = report_volumes[11]  # RV ED frame
            out_csv[study_ID_counter, 13] = report_volumes[12]  # RV ES frame
            out_csv[study_ID_counter, 14] = report_volumes[13]  # HR
            out_csv[study_ID_counter, 15] = report_volumes[14]  # LVPER
            out_csv[study_ID_counter, 16] = report_volumes[15]  # point LVPER

        df = pd.DataFrame(out_csv)
        df.to_csv('{0}/report_volumes.csv'.format(server_dir),
                  header=['eid', 'LVEDV', 'LVESV', 'LVSV', 'LVEF', 'LVM', 'RVEDV',
                          'RVESV', 'RVSV', 'RVEF', 'LV ED frame',
                          'LV ES frame', 'RV ED frame', 'RV ES frame', 'HR',
                          'PER', 'point_PER'],
                  index=False)

        # Plots
        if os.path.exists(os.path.join(results_dir, '{}_SAX.gif'.
                                       format(study_ID))):
            os.system('cp {} {}'.format(os.path.join(
                results_dir, '{}_SAX.gif'.format(study_ID)), gifs_dir))


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
    log_txt_file = os.path.join(
        log_dir, 'generate_final_results_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting Generate final results\n')
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    do_studies(study_IDs, nifti_dir, local_dir, logger)
    logger.info('Closing generate_final_results__log_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
