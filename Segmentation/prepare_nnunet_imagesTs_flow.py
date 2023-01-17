"""
$ python3 prepare_nnunet_imagesTs_flow.py  --help
usage: prepare_nnunet_imagesTs_flow.py [-h] [-i JSON_FILE]

Change header of images so niis are 3D rather than 2D+t - for segmentation
Save new images in imagesTs directory with nnunnet naming convention (_0000 and _0001 for mag and phase)

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib
import argparse
import os
import nibabel as nib
from glob import glob

# imports - 3rd party:
from datetime import datetime
from common_utils.load_args import Params
from common_utils.utils import save_nifti, get_list_of_dirs, set_logger


def create_imagesTs(study_IDs, nifti_dir, target_dir, _logger):
    target_imagesTs = os.path.join(target_dir, 'Task118_AscAoFlow', "imagesTs")
    if not os.path.exists(target_imagesTs):
        os.mkdir(target_imagesTs)

    for idx, study_ID in enumerate(study_IDs):
        _logger.info(f'[{idx+1}/{len(study_IDs)}]: {study_ID}')
        current_dir = os.path.join(nifti_dir, study_ID)

        # Get list of all flow nii files for the patient (ignore SAX)
        nii_files = glob(f'{current_dir}/*.nii.gz')
        nii_files = [os.path.basename(f) for f in nii_files if 'sa.nii.gz' not in f and 'sa_seg_nnUnet.nii.gz' not in f]

        phase_files = [f for f in nii_files if '_P' in f]

        # Save files with studyID name in output filename
        for current_phase_file in phase_files:
            mag_file = current_phase_file.replace('_P', '')

            # Phase
            input_file = os.path.join(current_dir, current_phase_file)
            output_file = os.path.join(target_imagesTs, f'{study_ID}_{mag_file}'.replace('.nii.gz', '_0001.nii.gz'))

            img = nib.load(input_file)
            img_hdr = img.header
            img_hdr['dim'][4] = 1
            img_hdr['pixdim'][4] = 0
            img_data = img.get_fdata()[:, :, 0, :]
            save_nifti(img.affine, img_data, img_hdr, output_file)

            # Mag
            input_file = os.path.join(current_dir, mag_file)
            output_file = os.path.join(target_imagesTs, f'{study_ID}_{mag_file}'.replace('.nii.gz', '_0000.nii.gz'))
            mag_data = nib.load(input_file).get_fdata()[:, :, 0, :]
            save_nifti(img.affine, mag_data, img_hdr, output_file)


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
    log_txt_file = os.path.join(log_dir, 'prepare_nnunet_imagesTs_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting processing images\n')
    # Find studies to analyse
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    create_imagesTs(study_IDs, nifti_dir, target_dir, logger)
    logger.info('Closing prepare_nnunet_imagesTs_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
