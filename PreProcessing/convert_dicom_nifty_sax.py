import logging
import os
from symbol import typedargslist
from turtle import st

# import shutil
import numpy as np

# imports - 3rd party:
from nibabel.nifti1 import (
    unit_codes,
    xform_codes,
    data_type_codes,
)  # units, transforms, data types
from pydicom.dicomio import read_file
import pandas as pd
from datetime import datetime

from common_utils.load_args import Params
from common_utils.utils import (
    get_list_of_dirs,
    set_logger,
    convert_dicom_to_nifti,
    get_list_of_files,
)


def do_studies(study_IDs, dicom_dir, nifti_dir, log_dir, logger):
    info = pd.read_csv(os.path.join(log_dir, "EF1_data.csv"))
    df_sax = pd.read_csv(os.path.join(log_dir, "sax_sequences_v1.csv")).values

    for idx, study_ID in enumerate(study_IDs):
        try:
            logger.info(f"{idx}: {study_ID}")
            source_ID_dir = os.path.join(dicom_dir, study_ID)
            sax_seq = df_sax[idx, :]

            if info["Processed"].iloc[idx] == 1:
                taget_ID_dir = os.path.join(nifti_dir, study_ID)
                if not os.path.exists(taget_ID_dir):
                    os.mkdir(taget_ID_dir)

                sequence_dicom_files_dir = []
                for seq in sax_seq[1:]:
                    if str(seq) != "0":
                        sequence_dicom_files_dir.append(
                            get_list_of_files(
                                os.path.join(source_ID_dir, seq),
                                full_path=True,
                                ext_str=".dcm",
                            )
                        )
                dest_nifti_image = os.path.join(taget_ID_dir, "sa.nii.gz")
                convert_dicom_to_nifti(
                    np.squeeze(np.hstack(sequence_dicom_files_dir)),
                    dest_nifti_image,
                    "sa",
                    logger,
                )
        except:
            logger.error("Error")


def main(json_config_path):
    if os.path.exists(json_config_path):
        cfg = Params(json_config_path).dict
    else:
        raise FileNotFoundError
    local_dir = cfg["DEFAULT_LOCAL_DIR"]
    dicom_dir = os.path.join(local_dir, cfg["DEFAULT_DICOM_SUBDIR"])
    nifti_dir = os.path.join(local_dir, cfg["DEFAULT_SUBDIR_NIFTI"])
    log_dir = os.path.join(local_dir, cfg["DEFAULT_LOG_DIR"])

    # Start logging console prints
    time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_txt_file = os.path.join(log_dir, "convert_dicom_nifty_sax" + time_file + ".txt")
    logger = set_logger(log_txt_file)
    logger.info("Starting convert dicom nifty sax\n")

    study_IDs = get_list_of_dirs(dicom_dir, full_path=False)
    do_studies(study_IDs, dicom_dir, nifti_dir, log_dir, logger)

    logger.info("Closing convert_dicom_nifty_sax_log_{}.txt".format(time_file))


if __name__ == "__main__":
    DEFAULT_JSON_FILE = "/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json"
    main(DEFAULT_JSON_FILE)
