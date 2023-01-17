# !/usr/bin/env python3 pylint: disable=invalid-name,unused-import,missing-module-docstring,missing-docstring,
# line-too-long,too-many-arguments,too-many-locals,too-many-branches,too-many-statements,bad-whitespace,
# pointless-string-statement,trailing-newlines

"""
$ python3 generate_SAX_panel.py  --help
usage: generate_SAX_panel.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib:
import argparse
import os
import logging

# imports 3rd party
import numpy as np
import pylab as plt
from pygifsicle import optimize
from skimage import measure
import nibabel as nib
import imageio
from datetime import datetime
import pandas as pd

# imports - local
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, set_logger

# =============================================================================
# DEFAULT DIR
# =============================================================================
DEFAULT_SEG_FMT_LVBP = 'c'  # 'c' cyan
DEFAULT_SEG_FMT_MYO = 'b'  # 'b' blue
DEFAULT_SEG_FMT_RVBP = 'r'  # 'r' red


# =============================================================================
# SA Segmentation validation over time
# =============================================================================
def segmentation_validation(study_ID, img_data, seg_data, LVV, RVV, tt, results_dir):
    filenames = []
    seg_bool = []
    # <0.5: larger segmentation contour,  0.5: balanced,  >0.5: smaller segmentation contour
    interp_value = 0.5
    nb_slices = seg_data.shape[2] + 2  # get number of z-slices
    nb_frames = seg_data.shape[3]  # get number of t-frames
    ncols = 6
    if 1 < nb_slices <= 12:
        nrows = 2
        no_of_plots = 12
    elif 12 < nb_slices <= 18:
        nrows = 3
        no_of_plots = 18
    elif 18 < nb_slices <= 24:
        nrows = 4
        no_of_plots = 24
    elif 24 < nb_slices <= 30:
        nrows = 5
        no_of_plots = 30
    else:
        print(
            f"More than 30 slices (nb_slices), for {study_ID}! Is this correct? (Skipping)")
        return filenames, seg_bool

    # 1. Plot individual png for each frame
    for fr in range(nb_frames):
        if np.sum(seg_data[..., fr]) == 0:
            seg_bool.append(False)
        else:
            seg_bool.append(True)

        seg_fr_LVBP = (seg_data[:, :, :, fr] == 1).astype(int)
        seg_fr_MYO = (seg_data[:, :, :, fr] == 2).astype(int)
        seg_fr_RVBP = (seg_data[:, :, :, fr] == 3).astype(int)
        img_fr = img_data[:, :, :, fr]

        # --------- --------- --------- ---------
        # subplots:
        # --------- --------- --------- ---------
        fig = plt.figure(figsize=(18, 9))  # Notice the equal aspect ratio
        axs = [fig.add_subplot(nrows, ncols, i + 1)
               for i in range(no_of_plots)]

        for i, a in enumerate(axs):
            if i >= 2:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_aspect('equal')
                a.axis('off')

        # --------- --------- --------- ---------
        # 1st + 2nd graphs common to all frames:
        # --------- --------- --------- ---------
        # 1st graph - LVV:
        if os.path.exists(os.path.join(results_dir, 'report_volumes.csv')):
            _global_volume = np.squeeze(pd.read_csv(
                os.path.join(results_dir, 'report_volumes.csv')).values)
            LV_ED_frame = int(_global_volume[9])
            LV_ES_frame = int(_global_volume[10])
            LV_point_PER = int(_global_volume[15])
        else:
            LV_ED_frame = -1
            LV_ES_frame = -1
            LV_point_PER = -1

        axs[0].plot(tt, LVV)
        axs[0].title.set_text('LVV')
        if LV_ED_frame != -1:
            axs[0].plot(tt[LV_ED_frame], LVV[LV_ED_frame], 'g*')
            axs[0].annotate('ED', (tt[LV_ED_frame], LVV[LV_ED_frame]))
        if LV_ES_frame != -1:
            axs[0].plot(tt[LV_ES_frame], LVV[LV_ES_frame], 'g*')
            axs[0].annotate('ES', (tt[LV_ES_frame], LVV[LV_ES_frame]))
        if LV_point_PER != -1:
            axs[0].plot(tt[LV_point_PER], LVV[LV_point_PER], 'ro')
            axs[0].annotate('PER', (tt[LV_point_PER], LVV[LV_point_PER]))
        # 2nd graph - RVV:
        axs[1].plot(tt, RVV)  # y=RVV
        axs[1].title.set_text('RVV')
        # --------- ---------

        # 3rd + graphs:
        for s in range(len(axs)):
            if s < nb_slices - 2:
                axs[s + 2].imshow(img_fr[:, :, s], cmap=plt.cm.gray)
                axs[s + 2].axis('off')
                for seg, color in [(seg_fr_LVBP, DEFAULT_SEG_FMT_LVBP),  # 'c' cyan
                                   (seg_fr_MYO, DEFAULT_SEG_FMT_MYO),  # 'b' blue
                                   (seg_fr_RVBP, DEFAULT_SEG_FMT_RVBP)]:  # 'r' red
                    contours = measure.find_contours(
                        seg[:, :, s], interp_value)
                    for cc in contours:
                        x = cc[:, 1]
                        y = cc[:, 0]
                        # linestyle='dashed',
                        axs[s + 2].plot(x, y, linewidth=2, color=color)
                axs[s + 2].axis('off')
            else:
                axs[s].axis('off')

        # title for entire sheet:
        fig.suptitle(f"Subject {study_ID} -- frame {fr}", fontsize=24)

        # Save image and contour
        filename = os.path.join(results_dir, f"{study_ID}_panel_fr_{fr}.png")
        filenames.append(filename)
        fig.savefig(filename)
        plt.close(fig)

    return filenames, seg_bool


# =============================================================================
# Generate Images (.png, animates .gif)
# =============================================================================
def generate_images(nifti_dir, study_IDs, logger,
                    img_name='sa.nii.gz',
                    seg_name='sa_seg_nnUnet.nii.gz'):
    for _study_ID_counter, study_ID in enumerate(study_IDs):
        logger.info(f'{_study_ID_counter}/{len(study_IDs)}: {study_ID}')

        # a) load data
        subject_dir = os.path.join(nifti_dir, study_ID)
        results_dir = os.path.join(subject_dir, 'results_SAX')
        # Load image and segmentation
        try:
            img = nib.load(os.path.join(subject_dir, img_name))
            seg = nib.load(os.path.join(subject_dir, seg_name))
            # TBV should these be in the input 'subject_dir'?
            LVV = np.loadtxt(os.path.join(results_dir, 'LVV.txt'))
            RVV = np.loadtxt(os.path.join(results_dir, 'RVV.txt'))
            tt = np.loadtxt(os.path.join(results_dir, 'sa_tt.txt'))
        except FileNotFoundError:
            logger.info(
                'Could not find SAX nifti file or volume for {}'.format(subject_dir))
            continue
        img_data = img.get_fdata()
        seg_data = seg.get_fdata()

        # b) generate image files:
        filenames, seg_bool = segmentation_validation(
            study_ID, img_data, seg_data, LVV, RVV, tt, results_dir)

        generate_animated_image(results_dir, study_ID, filenames)
        delete_temp_images(results_dir, study_ID)


def generate_animated_image(results_dir, study_ID, filenames, ext='gif', seq='SAX'):
    gif_dir = os.path.join(results_dir, f"{study_ID}_{seq}.{ext}")
    with imageio.get_writer(gif_dir, mode='I', fps=4) as writer:
        for fr, filename in enumerate(filenames):
            image = imageio.imread(filename)
            writer.append_data(image)
    optimize(gif_dir)


def delete_temp_images(results_dir, study_ID):
    os.system('rm -rf {0}/{1}_panel*.png'.format(results_dir, study_ID))


# =============================================================================
# MAIN
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
        log_dir, 'generate_SAX_panel_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting Generate SAX panel\n')
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    generate_images(nifti_dir, study_IDs, logger)
    logger.info('Closing generate_SAX_panel_log_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
