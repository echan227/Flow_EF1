# !/usr/bin/env python3 pylint: disable=line-too-long,invalid-name,missing-module-docstring,
# missing-function-docstring,invalid-name,too-many-arguments,too-many-locals,too-many-branches,too-many-statements,
# bad-whitespace,trailing-newlines

"""
$ python3 format_DICOM_data.py  --help
usage: format_DICOM_data.py [-h] [-i JSON_FILE]

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
from turtle import st
from datetime import datetime

# imports 3rd party
import numpy as np
import pandas as pd
from pydicom.dicomio import read_file

# imports - local
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, set_logger, remove_empty_folders

# =============================================================================
# DIRS
# =============================================================================
# Special characters to remove from each filenames:
STRINGS_TO_REPLACE = ['-', ',', '(', ')', '>', '<', ' ', '/', '#']


# =============================================================================
# FUNCTIONS
# =============================================================================
def get_files(source_dir, filespec='*.dcm'):
    files = [os.path.join(path, name) for path, subdirs, files in os.walk(source_dir)
             for name in files
             if fnmatch(name, filespec)]
    return files


def copy_files(filenames, path_DICOM, study_ID, logger):
    # List of strings to be replaced from SeriesDescription when creating series folder
    chars_to_replace = ['-', ',', '(', ')', '>', '<', '#', '*', '.']
    series_description = []
    series_number = []
    PatientIDs = []
    study_dates = []
    SOPInstanceUID = []
    exclude_dcm = []
    for d, dicom_file in enumerate(filenames):
        try:
            info = read_file(dicom_file)  # read metadata from DICOM file
            if not hasattr(info, 'ImageType'):
                exclude_dcm.append(d)
            if hasattr(info, 'PatientID'):
                PatientIDs.append(info.PatientID)
            else:
                PatientIDs.append('-1')
            if hasattr(info, 'SeriesDescription'):
                series_description.append(str(info.SeriesDescription))
            else:
                series_description.append('-1')
            if hasattr(info, 'SeriesNumber'):
                series_number.append(str(info.SeriesNumber))
            else:
                series_number.append('-1')
            if hasattr(info, 'StudyDate'):
                study_dates.append(str(info.StudyDate))
            else:
                study_dates.append('-1')
            if hasattr(info, 'SOPInstanceUID'):
                SOPInstanceUID.append(info.SOPInstanceUID)
            else:
                SOPInstanceUID.append('-1')
        except:
            series_description.append('-1')
            series_number.append('-1')
            study_dates.append('11111111')
            PatientIDs.append('-1')
            SOPInstanceUID.append('-1')

    series_description = np.array(series_description)
    series_number = np.array(series_number)
    study_dates = np.array(study_dates)
    PatientIDs = np.array(PatientIDs)
    SOPInstanceUID = np.array(SOPInstanceUID)
    filenames = np.array(filenames)
    exclude_dcm = np.array(exclude_dcm).astype(int)

    if len(exclude_dcm) > 0:
        series_description = np.delete(series_description, exclude_dcm)
        series_number = np.delete(series_number, exclude_dcm)
        study_dates = np.delete(study_dates, exclude_dcm)
        PatientIDs = np.delete(PatientIDs, exclude_dcm)
        SOPInstanceUID = np.delete(SOPInstanceUID, exclude_dcm)
        filenames = np.delete(filenames, exclude_dcm)

    for PatientID in np.unique(PatientIDs):
        ind = np.where(PatientID == PatientIDs)[0]
        study_dates_PatientID = study_dates[ind]
        series_description_PatientID = series_description[ind]
        SOPInstanceUID_PatientID = SOPInstanceUID[ind]
        series_number_PatientID = series_number[ind]
        filenames_PatientID = filenames[ind]
        print(study_ID)
        PatientIDs = study_ID
        study_ID = study_ID.replace(' ', '_').replace('/', '_')
        for char_to_replace in chars_to_replace:
            study_ID = study_ID.replace(char_to_replace, '')  # remove potentially problematic characters
        while '__' in study_ID:
            study_ID = study_ID.replace('__', '_')
        for study_date in np.unique(study_dates_PatientID):
            indx = np.where(study_date == study_dates_PatientID)[0]
            series_description_dates = series_description_PatientID[indx]
            series_number_dates = series_number_PatientID[indx]
            dicom_files_dates = filenames_PatientID[indx]
            SOPInstanceUID_dates = SOPInstanceUID_PatientID[indx]
            target_ID_dir = os.path.join(path_DICOM, study_ID, study_date)
            os.makedirs(target_ID_dir, exist_ok=True)

            for sn in np.unique(series_number_dates):  # for each series number...
                idx = np.where(series_number_dates == sn)[0]  # find all appearances of current series
                dicom_files_sn = dicom_files_dates[idx]  # DICOM file names for current series
                series_description_sn = series_description_dates[idx]  # SeriesDescription for current series
                SOPInstanceUID_dates_sn = SOPInstanceUID_dates[idx]  # SeriesDescription for current series

                # for each dicom file and corresponding series description...
                for sUID, dfile, sd in zip(SOPInstanceUID_dates_sn, dicom_files_sn,
                                           series_description_sn):
                    if sd == '-1':  # Check if current series has a description (this name is up to the clinician)
                        logger.warning(
                            'SeriesDescription field not available. DICOM file copied to NoSeriesDescription_{}/'.
                            format(sn))
                        # folder for current series is PatientID/StudyDate_found/SeriesDescription/SeriesNumber/
                        sn_dir = os.path.join(target_ID_dir, 'NoSeriesDescription_{}'.format(sn))
                        if not os.path.exists(sn_dir):
                            os.mkdir(sn_dir)
                    else:
                        sd = sd.replace(' ', '_').replace('/', '_')
                        for char_to_replace in chars_to_replace:
                            sd = sd.replace(char_to_replace, '')  # remove potentially problematic characters
                        while '__' in sd:
                            sd = sd.replace('__', '_')
                        # Folder for current series is PatientID/StudyDate_found/SeriesDescription/SeriesNumber/
                        sn_dir = os.path.join(target_ID_dir, '{}_{}'.format(sd, sn))
                        if not os.path.exists(sn_dir):
                            os.mkdir(sn_dir)
                    target_file = os.path.join(sn_dir, '{}.dcm'.format(sUID))
                    if not os.path.isfile(target_file):  # skip copy if file already exists
                        try:
                            shutil.copy(dfile, target_file)
                        except IOError:
                            logger.info("Unable to copy file {} to {}".format(dfile, target_file))

        # Remove small (useless) folders in DICOM_formatted_dir
        logger.info('Removing empty folders\n')
        remove_empty_folders(target_ID_dir)


# =============================================================================
# MAIN
# =============================================================================
def main(json_config_path):
    if os.path.exists(json_config_path):
        cfg = Params(json_config_path).dict
    else:
        raise FileNotFoundError
    input_dir = cfg['DEFAULT_INPUT_DIR']
    local_dir = cfg['DEFAULT_LOCAL_DIR']
    log_dir = os.path.join(local_dir, cfg['DEFAULT_LOG_DIR'])
    path_DICOM = os.path.join(local_dir, cfg['DEFAULT_DICOM_formatted_SUBDIR'])
    for path in [local_dir, log_dir, path_DICOM]:
        os.makedirs(path, exist_ok=True)

    # Start logging console prints
    time_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_txt_file = os.path.join(log_dir, 'format_DICOM_data_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting formatting\n')

    # Find studies to analyse
    study_IDs = get_list_of_dirs(input_dir, full_path=False)
    logger.info(f"DEBUG: get_list_of_dirs(): {study_IDs}")
    t1a = time()
    files_processed = 0
    for study_ID_counter, study_ID in enumerate(study_IDs):
        source_ID_dir = os.path.join(input_dir, study_ID)
        filenames = []
        for root, dirs, files in os.walk(source_ID_dir):
            for file in files:
                # if file.endswith('.dcm') and not file.startswith('.'):
                if os.path.isfile(os.path.join(root, file)) and not file.startswith('.'):
                    filenames.append(os.path.join(root, file))

        files_processed += len(filenames)
        copy_files(filenames, path_DICOM, study_ID, logger)

    t1b = time()
    logger.info(f"DEBUG  studies:{len(study_IDs)},  files_processed:{files_processed}")
    logger.info(f"DEBUG Time taken: %.3f sec" % (t1b - t1a))
    logger.info('Closing format_DICOM_data_log_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
