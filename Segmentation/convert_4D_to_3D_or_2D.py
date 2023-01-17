"""
$ python3 4D_to_3D_or_2D.py  --help
usage: 4D_to_3D_or_2D.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib
import argparse
import os
import nibabel

# imports - 3rd party:
from datetime import datetime
from common_utils.load_args import Params
from common_utils.utils import save_nifti, get_list_of_dirs, set_logger


# --------- --------- --------- --------- --------- --------- --------- ---------
def volume4D_to_frames3D(source_image, save_dir, study_name, seq_name, format_str="_0000.nii.gz", vol_type='3D'):
    # Read Nifti
    img = nibabel.load(source_image)
    img_fdata = img.get_fdata()
    _, _, _Z, T = img_fdata.shape
    img_hdr = img.header
    img_hdr['dim'][4] = 1
    img_hdr['pixdim'][4] = 0
    # Export separate nifti image files
    if vol_type == '3D':  # 3D volumes
        for fr in range(T):
            filename = f"{study_name}_{seq_name}_fr_{fr:02d}{format_str}"
            filename_img_fr = os.path.join(save_dir, filename)
            save_nifti(img.affine, img_fdata[:, :, :, fr], img_hdr, filename_img_fr)


def do_studies(study_IDs, nifti_dir, target_dir, _cfg):
    sax_img_name = 'sa.nii.gz'
    v4ds = [
        # img_name,       save_dir,              seq_name
        (sax_img_name, 'Task301_SAX', 'sa'),
    ]

    for idx, study_ID in enumerate(study_IDs):
        source_dir_tmp = os.path.join(nifti_dir, study_ID)
        for img_name, save_dir, seq_name in v4ds:
            source_image = os.path.join(source_dir_tmp, img_name)
            if os.path.exists(source_image):
                # create subdirectories  'imagesTs'
                target_imagesTs = os.path.join(target_dir, save_dir, "imagesTs")
                os.makedirs(target_imagesTs, exist_ok=True)
                # 4D to  save_nifti
                source_image = os.path.join(source_dir_tmp, img_name)
                exist_case = False
                for ss in os.listdir(target_imagesTs):
                    if study_ID in ss:
                        exist_case = True
                        break
                if not exist_case:
                    volume4D_to_frames3D(source_image, target_imagesTs, study_ID, seq_name)


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
    log_txt_file = os.path.join(log_dir, 'convert_4D_to_3D_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting conversion\n')
    # Find studies to analyse
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    do_studies(study_IDs, nifti_dir, target_dir, cfg)
    logger.info('Closing convert_4D_to_3D_log_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
