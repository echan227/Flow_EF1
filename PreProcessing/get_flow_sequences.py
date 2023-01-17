# !/usr/bin/env python3 pylint: disable=line-too-long,invalid-name,missing-module-docstring,
# missing-function-docstring,too-many-locals,invalid-name,trailing-newlines,bad-whitespace,too-many-arguments

"""
$ python3 generate_numpy_files.py  --help
usage: generate_numpy_files_classification_pipeline.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib
import os
import shutil

# imports - 3rd party:
import numpy as np
from pydicom.dicomio import read_file
from datetime import datetime
import pandas as pd

from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, set_logger, convert_dicom_to_nifti, \
    get_temporal_sequences

# =============================================================================
# DEFAULT PARAMS
# =============================================================================
analysis_in_bulk = True
CLI_DRYRUN = True

# --------- --------- --------- --------- --------- --------- --------- ---------
def ranges(nums, series_gap):
    nums = sorted(set(nums))
    rr = []
    rej = np.empty(0)
    for si, s in enumerate(nums):
        initial = s
        if s not in rej:
            final = s
            for e, ei in enumerate(nums[si + 1:]):
                if s + series_gap * (e + 1) == ei:
                    rej = np.hstack([rej, ei])
                    final = ei
                else:
                    break
            rr.append([initial, final])
    return rr


def do_studies(study_IDs, dicom_dir, nifti_dir, log_dir, logger):
    flow_sequences = np.zeros((len(study_IDs), 31), dtype= object)
    header_metadata = np.hstack([['study_ID'],  np.tile(['sequence', 'sequences_number'], 15)])
    flow_csv_file = os.path.join(log_dir, 'flow_sequences_v1.csv')
    for idx, study_ID in enumerate(study_IDs):
        try:
            logger.info(f"{idx}: {study_ID}")
            source_ID_dir = os.path.join(dicom_dir, study_ID)
            flow_sequences[idx, 0] = study_ID
            sequences = np.array(get_list_of_dirs(source_ID_dir, full_path=False))
            sequences_numbers = np.array([int(f.split('_')[-1]) for f in sequences])
            index = np.where(sequences_numbers>len(sequences_numbers))
            if len(index) > 0:
                sequences_numbers = np.delete(sequences_numbers, index[0])
                sequences = np.delete(sequences, index[0])
            if len(sequences_numbers) > 1:
                series_gap = np.min(np.diff(np.sort(np.unique(sequences_numbers))))
            else:
                series_gap = 1
            subject_dir = os.path.join(nifti_dir, study_ID)
            if not os.path.exists(subject_dir):
                os.makedirs(subject_dir, exist_ok=True)

            # Discard (non-temporal) sequences with trigger times <= 10
            tt_per_seq, temporal_seq, sequences_non_temp, dcm_params, axis_dcm, dcm_files_seq, dcm_files_seq_all = \
                get_temporal_sequences(source_ID_dir, sequences, 10, logger)

            seriesNumber = np.array([fname.split('_')[-1] for fname in temporal_seq]).astype(int)
            temporal_seq = temporal_seq[seriesNumber.argsort()]
            k = 1
            for seq in temporal_seq:
                if 'ThroughPlane_Flow_Breath_Hold' in seq:
                    flow_sequences[idx, k] = seq
                    flow_sequences[idx, k+1] = seq.split('_')[-1]
                    k += 2

            df_temp = pd.DataFrame(flow_sequences, columns=header_metadata)
            df_temp.to_csv(flow_csv_file, index=False)
        except:
            logger.error('Error')

def main(json_config_path):
    if os.path.exists(json_config_path):
        cfg = Params(json_config_path).dict
    else:
        raise FileNotFoundError
    local_dir = cfg['DEFAULT_LOCAL_DIR']
    dicom_dir = os.path.join(local_dir, cfg['DEFAULT_DICOM_SUBDIR'])
    nifti_dir = os.path.join(local_dir, cfg['DEFAULT_SUBDIR_NIFTI'])
    log_dir = os.path.join(local_dir, cfg['DEFAULT_LOG_DIR'])

    # Start logging console prints
    time_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_txt_file = os.path.join(log_dir, 'get_flow_sequences_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting get flow sequences\n')

    study_IDs = get_list_of_dirs(dicom_dir, full_path=False)
    do_studies(study_IDs,dicom_dir,nifti_dir,log_dir,logger)

    logger.info('Closing generate_numpy_files_classification_log_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
