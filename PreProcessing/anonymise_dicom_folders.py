# !/usr/bin/env python3 pylint: disable=line-too-long,invalid-name,missing-module-docstring,
# missing-function-docstring,invalid-name,too-many-arguments,too-many-locals,too-many-branches,too-many-statements,
# bad-whitespace,trailing-newlines

"""
$ python3 anonymise_dicom_folders.py  --help
usage: anonymise_dicom_folders.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib:
import os
from time import time
import shutil
from fnmatch import fnmatch

# imports 3rd party
import numpy as np
import pandas as pd
from pydicom.dicomio import read_file
from datetime import datetime

# imports - local
from common_utils.load_args import Params
from common_utils.utils import set_logger, get_list_of_dirs, store_anonymised_metadata, perform_anonymisation, \
    DirNotEmpty, CsvNotUpdated


# =============================================================================
# FUNCTIONS
# =============================================================================
def anonymise_dicom_folders(local_dir, DICOM_logs_dir, DICOM_formatted_dir, DICOM_anon_dir, DICOM_anon_temp_dir,
                            study_anonymised_csv_file, logger):
    # =============================================================================
    # CHECK DIRS
    # =============================================================================
    assert os.path.exists(DICOM_logs_dir), "Path does not exist: %r" % DICOM_logs_dir
    assert os.path.exists(DICOM_formatted_dir), "Path does not exist: %r" % DICOM_formatted_dir
    assert os.path.exists(DICOM_anon_dir), "Path does not exist: %r" % DICOM_anon_dir

    # =============================================================================
    # SANITY CHECKS:
    #   Is DICOM_anon_temp_dir emtpy?
    #   Are folder names in DICOM_anon_dir correct?
    #   Does study_anonymised.csv contain updated information?
    # =============================================================================
    # Check that DICOM_anon_temp_dir exists and/or is empty
    if os.path.isdir(DICOM_anon_temp_dir):
        anon_temp_folders = get_list_of_dirs(DICOM_anon_temp_dir, full_path=False)
        if anon_temp_folders:
            os.rmdir(anon_temp_folders)
            logger.error(DirNotEmpty(DICOM_anon_temp_dir))
    else:
        os.mkdir(DICOM_anon_temp_dir)

    # Look into anon_folder (source and target), get the highest SXXXXXX number, and start with the next one
    anon_folders = get_list_of_dirs(DICOM_anon_dir, full_path=False)
    anon_folders = [anon_folder for anon_folder in anon_folders if
                    not anon_folder.startswith('WS_A_S')]  # ignore cases whose workspaces were manually extracted
    if len(anon_folders) > 0:
        assert all([anon_folder.startswith('A_S') for anon_folder in
                    anon_folders]), "At least one folder does not start with 'A_S'. Please revise."
        try:
            # extract numbers from folders, calculate max and add 1
            anon_index_folder = np.max(sorted([int(anon_folder[-6:]) for anon_folder in
                                               anon_folders])) + 1
        except ValueError:
            logger.error(
                "At least one folder does not follow the current naming convention (i.e. the last 6 characters must "
                "be an int). Please revise.")
            raise
    else:
        anon_index_folder = 1  # no folders? start at 000001

    if os.path.exists(study_anonymised_csv_file):
        # Load df_anonymised with anon_index, patientID, date
        df_anonymised = pd.read_csv(study_anonymised_csv_file)
        anon_index_csv = np.max(df_anonymised["Anonymised ID"].replace({'A_S': ''}, regex=True).astype(int)) + 1
    else:
        anon_index_csv = 1
        df_anonymised = pd.DataFrame()

    # Sanity check: anon_index and anon_index_csv must coincide
    if anon_index_folder != anon_index_csv:
        logger.error('The information on the .csv file is not updated. Please check anon/ folder.')
        raise CsvNotUpdated

    anon_index = anon_index_folder

    # =============================================================================
    # ANONYMISE STUDY FOLDERS AND MOVE TO DICOM_anon_dir
    # =============================================================================
    # Get keys for anonymised dicom tags
    tags = pd.read_excel(os.path.join(local_dir, 'dicom_tags_to_anonymise_PHILIPS.xlsx'), sheet_name='pseudo')
    tags = tags[tags['anonymise'] != False]  # discard tags with anonymise=FALSE
    tags_to_anonymise = {}
    for key in tags['key']:
        key_left, key_right = key[1:-1].split(',')
        hex_key = (int(key_left, 16), int(key_right, 16))
        tags_to_anonymise[hex_key] = None

    # Get keys and keywords for stored dicom tags
    tags = pd.read_excel(os.path.join(local_dir, 'dicom_tags_to_store.xlsx'), sheet_name='v1')
    tags_to_store = {}
    for key, value in zip(tags['key'], tags['keyword']):
        key_left, key_right = key[1:-1].split(',')
        hex_key = (int(key_left, 16), int(key_right, 16))
        tags_to_store[hex_key] = value
    header_metadata = ["Anonymised ID"]
    header_metadata.extend(tags_to_store.values())

    # Get all PatientIDs/Dates/ which have already been anonymised
    if df_anonymised.empty:
        anonymised_folders = []
    else:
        anonymised_folders = df_anonymised['Patient ID'].astype(str) + os.path.sep + df_anonymised['Study Date'].astype(str)
        anonymised_folders = anonymised_folders.to_list()

    # Get all patient_IDs within Workspace/ or No_Workspace
    patient_IDs = get_list_of_dirs(DICOM_formatted_dir, full_path=False)

    # for patient_ID in tqdm(patient_IDs):
    for patient_ID in patient_IDs:
        try:
            # Get all study_dates for current Workspace/Patient_ID/
            patient_ID_folder = os.path.join(DICOM_formatted_dir, patient_ID)
            dates = get_list_of_dirs(patient_ID_folder, full_path=False)
            for date in dates:
                # Check if current Patient_ID/date/ has already been anonymised
                current_study = patient_ID + os.path.sep + date
                if current_study in anonymised_folders:
                    logger.info("Study '{}' has already been anonymised. Continue.".format(current_study))
                    continue

                logger.info("Anonymising study '{}'".format(current_study))

                # 1. Copy study to a temporary location
                anon_index_str = str(anon_index).zfill(6)
                source_dir = os.path.join(patient_ID_folder, date)
                tmp_anon_dir = os.path.join(DICOM_anon_temp_dir, 'S' + anon_index_str)
                # Make tmp_anon_dir and copy files from DICOM_formatted_dir/Study_ID/Dates/ to DICOM_anon_temp_dir/
                _ = shutil.copytree(source_dir, tmp_anon_dir)

                # 2. Read DICOM metadata before it is anonymised
                metadata = store_anonymised_metadata(tags_to_store, anon_index_str,
                                                     source_dir)  # get values for each dicom file
                metadata = np.unique(metadata, axis=0)  # get unique combinations of values
                metadata = [np.unique(metadata[:, i]) for i in
                            range(len(metadata[0]))]  # get unique values for each column
                metadata = [i[i != '-1'] if ('-1' in i and len(i) > 1) else i for i in
                            metadata]  # remove '-1' if there is at least another value
                metadata = [i[0] if len(i) == 1 else i for i in
                            metadata]  # keep as array only if there is more than one value

                # 3. Anonymise folder and copy it to the final location
                target_anon_dir = os.path.join(DICOM_anon_dir, 'A_S' + anon_index_str)
                shutil.move(tmp_anon_dir, target_anon_dir)

                # 4. Add the anonymised study to study_anonymised.csv
                if not df_anonymised.empty and (
                        set(header_metadata) - set(df_anonymised.columns) or set(df_anonymised.columns) - set(
                        header_metadata)):
                    logger.error(
                        'The .csv headers do not coincide with the current dicom tags. Please check .csv and tags.')
                    raise CsvNotUpdated
                df_temp = pd.DataFrame([metadata], columns=header_metadata)
                df_anonymised = df_anonymised.append(df_temp, ignore_index=True)
                df_anonymised.to_csv(study_anonymised_csv_file, index=False)

                anon_index += 1

        except Exception as error:
            shutil.rmtree(tmp_anon_dir)
            error_string = repr(error)
            os.system('echo "ERROR for {}: {}" >> {}/anonymisation_errors.txt'.format(patient_ID,
                                                                                      error_string.replace('"', ''),
                                                                                      DICOM_logs_dir))


# =============================================================================
# MAIN
# =============================================================================
def main(json_config_path):
    if os.path.exists(json_config_path):
        cfg = Params(json_config_path).dict
    else:
        raise FileNotFoundError
    local_dir = cfg['DEFAULT_LOCAL_DIR']
    log_dir = os.path.join(local_dir, cfg['DEFAULT_LOG_DIR'])
    DICOM_formatted_dir = os.path.join(local_dir, cfg['DEFAULT_DICOM_formatted_SUBDIR'])
    DICOM_anon_dir = os.path.join(local_dir, cfg['DEFAULT_DICOM_SUBDIR'])
    DICOM_anon_temp_dir = os.path.join(local_dir, 'DICOM_anon_temp')
    for path in [DICOM_anon_temp_dir, DICOM_anon_dir]:
        os.makedirs(path, exist_ok=True)
    study_anonymised_csv_file = os.path.join(log_dir, 'study_anonymised_v1.csv')
    Anonymisation_dir = os.path.join(cfg['DEFAULT_CODE_DIR'], 'Anonymisation')

    # Start logging console prints
    time_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_txt_file = os.path.join(log_dir, 'anonymisation_dicom_folders_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting anonymisation\n')

    _ = anonymise_dicom_folders(Anonymisation_dir, log_dir, DICOM_formatted_dir, DICOM_anon_dir, DICOM_anon_temp_dir,
                                study_anonymised_csv_file, logger)

    logger.info('Closing anonymisation_dicom_folders_log_{}.txt'.format(time_file))


if __name__ == "__main__":
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
