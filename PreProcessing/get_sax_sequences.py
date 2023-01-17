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
from common_utils.utils import get_list_of_dirs, set_logger, get_temporal_sequences

# =============================================================================
# DEFAULT PARAMS
# =============================================================================
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


analysis_in_bulk = True
CLI_DRYRUN = True

def do_studies(study_IDs, dicom_dir, nifti_dir, log_dir, logger):
    sax_sequences = np.zeros((len(study_IDs), 51), dtype= object)
    header_metadata = np.hstack([['study_ID'],  np.tile(['sequence'],50)])
    sax_csv_file = os.path.join(log_dir, 'sax_sequences_v2_aux.csv')
    for idx, study_ID in enumerate(study_IDs):
        try:
            logger.info(f"{idx}: {study_ID}")
            source_ID_dir = os.path.join(dicom_dir, study_ID)
            sax_sequences[idx, 0] = study_ID
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
            dcm_params =  dcm_params[seriesNumber.argsort(),:]
            axis_dcm =  axis_dcm[seriesNumber.argsort(),:]

            index = []
            for s in range(len(temporal_seq)):
                seq = temporal_seq[s]
                if 'Wall_motion_stack_3_slices_per_breath_hold' in seq or 'tf2d14_retro_iPAT_ES32_SA_Multi_Slice' in seq :
                    index.append(s)
            index = np.array(index)

            
            if len(index) > 0:
                temporal_seq = temporal_seq[index]    
                dcm_params = dcm_params[index, :]
                axis_dcm = axis_dcm[index, :]
                aux_sax_sequences = []
                for axs in np.unique(axis_dcm, axis=0):
                    inds = np.where((axs == axis_dcm).all(axis=1))[0]
                    inds = inds[dcm_params[inds, -1].argsort()]
                    aux_dcm = dcm_params[inds, :-2]
                    aux_temporal_seq = temporal_seq[inds]
                    aux_seriesNumber =  dcm_params[inds, -1]
                    rr = ranges(np.sort(aux_seriesNumber), series_gap)
                    if len(inds) > 1 and (aux_dcm == aux_dcm[0]).all() and len(rr) == 1:
                        aux = []
                        for seq in aux_temporal_seq:
                            aux.append(seq)
                        aux_sax_sequences.append(aux)
                    elif len(rr) > 1:
                        for ri in rr:
                            ind3 = []
                            for rj in range(int(ri[0]), int(ri[1]+1)):
                                ind3.append(np.where(aux_seriesNumber == rj)[0][0])
                            aux_dcm_2 = aux_dcm[np.array(ind3), :]
                            aux_temporal_seq_2 = aux_temporal_seq[np.array(ind3)]
                            if (aux_dcm_2 == aux_dcm_2[0]).all():
                                aux = []
                                for seq in aux_temporal_seq_2:
                                    aux.append(seq)
                                aux_sax_sequences.append(aux)
                            else:
                                print('A')
                    else:
                        print('B')
                
                if len(aux_sax_sequences) > 1:
                    # seq_check.append(study_ID)
                    if study_ID == 'A_S000021':
                        aux_sax_sequences = aux_sax_sequences[0]
                    elif study_ID == 'A_S000022':
                        aux_sax_sequences = np.concatenate([aux_sax_sequences[1], aux_sax_sequences[2]])                    
                    elif study_ID == 'A_S000031':
                        aux_sax_sequences = np.concatenate([aux_sax_sequences[0], aux_sax_sequences[1]])          
                    elif study_ID == 'A_S000034':
                            aux_sax_sequences = np.concatenate([aux_sax_sequences[0], aux_sax_sequences[1]])   
                    elif study_ID == 'A_S000035':
                            aux_sax_sequences = np.concatenate([aux_sax_sequences[0], aux_sax_sequences[1]])  
                    elif study_ID == 'A_S000041':
                        aux_sax_sequences = np.concatenate([aux_sax_sequences[0], aux_sax_sequences[1]])  
                    elif study_ID == 'A_S000058':
                        aux_sax_sequences = np.concatenate([aux_sax_sequences[1], aux_sax_sequences[2]])  
                    elif study_ID == 'A_S000198':
                        aux_sax_sequences = np.concatenate([aux_sax_sequences[0], aux_sax_sequences[1]])  
                    elif study_ID == 'A_S000241':
                        aux_sax_sequences = np.concatenate([aux_sax_sequences[0], aux_sax_sequences[1]]) 
                    else:
                        aux_sax_sequences = aux_sax_sequences[np.array([len(fname) for fname in aux_sax_sequences]).argmax()]
                else:
                    aux_sax_sequences = np.squeeze(np.array(aux_sax_sequences))

                k = 1
                for seq in aux_sax_sequences:
                    sax_sequences[idx, k] = seq
                    k +=1

            df_temp = pd.DataFrame(sax_sequences, columns=header_metadata)
            df_temp.to_csv(sax_csv_file, index=False)
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
    log_txt_file = os.path.join(log_dir, 'get_sax_sequences' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting ge saxsequences numpy\n')

    study_IDs = get_list_of_dirs(dicom_dir, full_path=False)
    do_studies(study_IDs,dicom_dir,nifti_dir,log_dir,logger)

    logger.info('Closing get_flow_sequences_log_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
