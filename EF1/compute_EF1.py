# !/usr/bin/env python3 pylint: disable=invalid-name,unused-import,
# missing-module-docstring,missing-docstring,line-too-long,too-many-arguments,
# too-many-locals,too-many-branches,too-many-statements,bad-whitespace,
# pointless-string-statement,trailing-newlines

"""
$ python3 v.py  --help
usage: compute_EF1.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib:
import os
from glob import glob

# imports 3rd party
from datetime import datetime
import pandas as pd
import numpy as np

# imports - local
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, set_logger
from EF1.EF1_functions import compute_EF1


# =============================================================================
# FUNCTIONS
# =============================================================================
def do_studies(local_dir, study_IDs, nifti_dir, logger):
    # For linking anon ID to original patient ID
    anon_csv = os.path.join(local_dir, "log", "study_anonymised_v1.csv")
    anon_df = pd.read_csv(anon_csv).values
    anon_IDs = anon_df[:, 0]
    patient_IDs = anon_df[:, 8]

    # Overall EF1 spreadsheet
    EF1_summary_csv = os.path.join(local_dir, "report_EF1.csv")
    if os.path.exists(EF1_summary_csv):
        os.remove(EF1_summary_csv)

    # Folder for all graphs - to look through all cases more easily
    all_plots_dir = os.path.join(local_dir, 'EF1_plots')
    all_curves_dir = os.path.join(all_plots_dir, 'Flow_LVV_curves')
    if not os.path.exists(all_plots_dir):
        os.mkdir(all_plots_dir)
    if not os.path.exists(all_curves_dir):
        os.mkdir(all_curves_dir)

    for study_ID_counter, study_ID in enumerate(study_IDs):
        logger.info(f"do_studies() [{study_ID_counter}]: {study_ID}")
        subject_dir = os.path.join(nifti_dir, study_ID)
        results_dir_flow = os.path.join(subject_dir, "results_flow")
        results_dir_sax = os.path.join(subject_dir, "results_SAX")
        if not os.path.exists(results_dir_flow):
            os.system("mkdir -p {0}".format(results_dir_flow))

        # Get original patient ID
        anon_ind = np.where(study_ID == anon_IDs)[0][0]
        patient_ID = patient_IDs[anon_ind]
        if 'CMR - ' in patient_ID:
            patient_ID = patient_ID[6:]
        elif len(patient_ID) > 9:
            patient_ID = patient_ID[:9]

        compute_EF1(
            study_ID, subject_dir, results_dir_sax, results_dir_flow,
            EF1_summary_csv, logger, patient_ID
        )

        # Copy generated graphs to folder containing all graphs
        all_jpgs = glob(f'{results_dir_flow}/*jpg')
        for jpg in all_jpgs:
            os.system(f'cp {jpg} {all_curves_dir}')

    # Calculate mean and std for each EF1 method
    df_EF1 = pd.read_csv(EF1_summary_csv).values
    LVEF1 = df_EF1[:, 7]
    LVEF1_2 = df_EF1[:, 14]
    LVEF1_Ao = df_EF1[:, 24]

    # Save mean and std in summary spreadsheet
    data_EF1 = np.zeros((1, df_EF1.shape[1]), dtype=object)
    data_EF1[:, 7] = f"{LVEF1.mean():.2f} ({LVEF1.std():.2f})"
    data_EF1[:, 14] = f"{LVEF1_2.mean():.2f} ({LVEF1_2.std():.2f})"
    data_EF1[:, 24] = f"{LVEF1_Ao.mean():.2f} ({LVEF1_Ao.std():.2f})"
    data_EF1[data_EF1 == 0] = np.nan
    df2 = pd.DataFrame(data_EF1)
    df2.to_csv(EF1_summary_csv, mode='a', header=False, index=False)


# =============================================================================
# MAIN
# =============================================================================
def main(json_config_path):
    if os.path.exists(json_config_path):
        cfg = Params(json_config_path).dict
    else:
        raise FileNotFoundError
    local_dir = cfg["DEFAULT_LOCAL_DIR"]
    nifti_dir = os.path.join(local_dir, cfg["DEFAULT_SUBDIR_NIFTI"])
    log_dir = os.path.join(local_dir, cfg["DEFAULT_LOG_DIR"])
    # Start logging console prints
    time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_txt_file = os.path.join(log_dir, f"compute_volume_EF1_{time_file}.txt")
    logger = set_logger(log_txt_file)
    logger.info("Starting computing parameters and volumes for lax\n")
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    do_studies(local_dir, study_IDs, nifti_dir, logger)
    logger.info(f"Closing compute_volume_EF1_log_{time_file}.txt")


if __name__ == "__main__":
    import sys

    sys.path.append("/home/bram/Scripts/AI_CMR_QC")
    DEFAULT_JSON_FILE = "/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json"
    main(DEFAULT_JSON_FILE)
