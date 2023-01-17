"""
$ python3 nnunet_segmentation.py  --help
usage: nnunet_segmentation.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib
import os
import subprocess
import nibabel as nib
import numpy as np
import argparse

# imports - 3rd party:
from datetime import datetime
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, save_nifti, set_logger


# --------- --------- --------- --------- --------- --------- --------- ---------
def combine_LAX_segmentations(_subject_dir, seq):
    lvrv_path = os.path.join(_subject_dir, f"{seq}_seg_LVRV_nnUnet.nii.gz")
    lara_path = os.path.join(_subject_dir, f"{seq}_seg_LARA_nnUnet.nii.gz")
    nim_path = os.path.join(_subject_dir, f"{seq}.nii.gz")
    output_path = os.path.join(_subject_dir, f"{seq}_seg_nnUnet.nii.gz")
    # Load Nifti files:
    if os.path.exists(lvrv_path) and os.path.exists(lara_path):
        la_LV_seg = nib.load(lvrv_path).get_fdata()
        la_atria_seg = nib.load(lara_path).get_fdata()
        nim = nib.load(nim_path)
        # iterate frames:
        fused_seg = np.zeros_like(la_LV_seg)
        N_frames = la_LV_seg.shape[3]
        for fr in range(N_frames):
            if seq == 'la_2Ch':
                aux = la_LV_seg[:, :, 0, fr] + 3 * la_atria_seg[:, :, 0, fr]
                if len(np.unique(aux)) > 4:
                    aux[aux > 3] = 3
                fused_seg[:, :, 0, fr] = aux
            else:
                aa = np.copy(la_LV_seg[:, :, 0, fr])
                aa[aa == 3] = 0

                bb = np.copy(la_LV_seg[:, :, 0, fr])
                bb[bb == 2] = 0
                bb[bb == 1] = 0

                c = np.copy(la_atria_seg[:, :, 0, fr])
                c[c == 2] = 0
                c = c * 4

                d = np.copy(la_atria_seg[:, :, 0, fr])
                d[d == 1] = 0
                d[d == 2] = 5

                fused_seg_1 = aa + c
                if len(np.unique(fused_seg_1)) > 3:
                    fused_seg_1[fused_seg_1 > 4] = 4

                fused_seg_2 = bb + d
                if len(np.unique(fused_seg_2)) > 3:
                    fused_seg_2[fused_seg_2 > 5] = 5

                aux = fused_seg_1 + fused_seg_2

                if len(np.unique(aux)) > 6:
                    aux[aux == 7] = 2
                fused_seg[:, :, 0, fr] = aux

        # Save:
        save_nifti(nim.affine, fused_seg, nim.header, output_path)


def run_nnunet_inference(target_dir, task_folder, script_no, script_dir):
    target_imagesTs = os.path.join(target_dir, task_folder, "imagesTs")
    inference_dir = os.path.join(target_dir, task_folder, "Results")
    subprocess.run(f'bash {script_dir} -a {target_imagesTs} -b {inference_dir} -c {script_no}',
                   shell=True)


def get_nifti_filenames(target_dir, task_folder, sid):
    inference_dir = os.path.join(target_dir, task_folder, "Results")
    files = sorted([os.path.join(inference_dir, f)
                    for f in os.listdir(inference_dir)
                    if (not f.startswith('.') and f.startswith(sid) and f.endswith('.nii.gz'))])
    return files


def _doit(nifti_dir, target_dir, sid, task_folder, images, seg):
    files = get_nifti_filenames(target_dir, task_folder, sid)
    # Load Nifti file:
    img_dir = os.path.join(nifti_dir, sid, images)
    if os.path.exists(img_dir):
        nim = nib.load(img_dir)
        X, Y, Z, T = nim.header['dim'][1:5]
        seg_4D = np.zeros((X, Y, Z, T))
        if len(files) != T:
            raise Exception(f"Patient {sid}: Wrong number of segmentations, only contains {len(files)} frames")
        for f, fi in enumerate(files):
            if int(fi.split('_')[-1].strip('nii.gz')) != f:
                raise Exception(f"Patient {sid}: Frames not correspond number")
            try:
                seg_4D[:, :, :, f] = nib.load(fi).get_fdata()
            except:
                print('Error')

        # Save Nifti file:
        filename_seg = os.path.join(nifti_dir, sid, seg)
        save_nifti(nim.affine, seg_4D, nim.header, filename_seg)


def _doit1(nifti_dir, target_dir, study_IDs, msg, script_no, task_folder, images, seg, script_dir, logger):
    run_nnunet_inference(target_dir, task_folder, script_no, script_dir)
    for idx, sid in enumerate(study_IDs):
        logger.info(f'{msg}: {idx}: {sid}')
        if not os.path.exists(os.path.join(nifti_dir, sid, seg)):
            _doit(nifti_dir, target_dir, sid, task_folder, images, seg)


def main(json_config_path):
    if os.path.exists(json_config_path):
        cfg = Params(json_config_path).dict
    else:
        raise FileNotFoundError
    local_dir = cfg['DEFAULT_LOCAL_DIR']
    nifti_dir = os.path.join(local_dir, cfg['DEFAULT_SUBDIR_NIFTI'])
    target_dir = os.path.join(local_dir, cfg['DEFAULT_NNUNET_NIFTI'])
    log_dir = os.path.join(local_dir, cfg['DEFAULT_LOG_DIR'])
    # Start logging console prints
    time_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_txt_file = os.path.join(log_dir, 'nnunet_segmentation_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting segmentation\n')
    script_dir = '/home/br14/code/Python/AI_CMR_QC/Segmentation/inference_nnunet.sh'

    # Find studies to analyse
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)

    # =============================================================================
    # SAX
    # =============================================================================
    _doit1(nifti_dir, target_dir, study_IDs, 'SAX', 301, 'Task301_SAX', 'sa.nii.gz', 'sa_seg_nnUnet.nii.gz',
           script_dir, logger)

    logger.info('Closing nnunet_segmentation_log_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
