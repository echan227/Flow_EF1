# !/usr/bin/env python3 pylint: disable=invalid-name,unused-import,missing-module-docstring,missing-docstring,
# line-too-long,too-many-arguments,too-many-locals,too-many-branches,too-many-statements,bad-whitespace,
# pointless-string-statement,trailing-newlines

"""
$ python3 generate_LAX_panel.py  --help
usage: generate_LAX_panel.py [-h] [-i JSON_FILE]

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
import pandas as pd
import pylab as plt
from pygifsicle import optimize
from skimage import measure
import nibabel as nib
import imageio
import matplotlib.gridspec as gridspec
from datetime import datetime

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
def segmentation_validation(study_ID, img_data, seg_data, img_data2, seg_data2, LAV, RAV, mapse, tapse,
                            points, results_dir):
    filenames = []
    nb_frames = int(np.max([seg_data.shape[3], seg_data2.shape[3]]))
    interp_value = 0.5

    for fr in range(nb_frames):
        fig = plt.figure(figsize=(18, 9))
        gs = gridspec.GridSpec(3, 4, figure=fig)
        ax1 = plt.subplot(gs[0:2, 0:2])
        ax2 = plt.subplot(gs[0:2, 2:])
        ax3 = plt.subplot(gs[-1, 0])
        ax4 = plt.subplot(gs[-1, 1])
        ax5 = plt.subplot(gs[-1, 2])
        ax6 = plt.subplot(gs[-1, 3])

        ax3.plot(LAV)
        ax3.set_title('LA volume')
        if points[0] != -1:
            ax3.plot(points[0], LAV[points[0]], 'ro')
            ax3.annotate('LA max', (points[0], LAV[points[0]]))
        if points[1] != -1:
            ax3.plot(points[1], LAV[points[1]], 'ro')
            ax3.annotate('LA_reservoir', (points[1], LAV[points[1]]))
        if points[2] != -1:    
            ax3.plot(points[2], LAV[points[2]], 'ro')
            ax3.annotate('LA pump', (points[2], LAV[points[2]]))
            y_min, y_max = ax3.get_ylim()
            ax3.set_ylim([y_min, y_max + 10])
            x_min, x_max = ax3.get_xlim()
            ax3.set_xlim([x_min, x_max + 5])

        ax4.plot(RAV)
        ax4.set_title('RA volume')
        if points[3] != -1:
            ax4.plot(points[3], RAV[points[3]], 'ro')
            ax4.annotate('RA max', (points[3], RAV[points[3]]))
        if points[4] != -1:
            ax4.plot(points[4], RAV[points[4]], 'ro')
            ax4.annotate('RA_reservoir', (points[4], RAV[points[4]]))
        if points[5] != -1:
            ax4.plot(points[5], RAV[points[5]], 'ro')
            ax4.annotate('RA pump', (points[5], RAV[points[5]]))
            y_min, y_max = ax4.get_ylim()
            ax4.set_ylim([y_min, y_max + 10])
            x_min, x_max = ax4.get_xlim()
            ax4.set_xlim([x_min, x_max + 5])

        ax5.plot(mapse)
        ax5.plot(mapse.argmax(), mapse[mapse.argmax()], 'ro')
        ax5.set_title('Mapse')

        ax6.plot(tapse)
        ax6.plot(tapse.argmax(), tapse[tapse.argmax()], 'ro')
        ax6.set_title('Tapse')

        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.set_title('la 2Ch')

        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_aspect('equal')
        ax2.set_title('la 4Ch')

        if fr < img_data.shape[-1]:
            ax1.imshow(img_data[:, :, 0, fr], cmap='gray')
            if not np.sum(seg_data[..., fr]) == 0:
                for labels, color in enumerate(['g', 'b', 'y']):
                    seg_aux = (seg_data[:, :, 0, fr] == labels).astype(int)
                    contours = measure.find_contours(seg_aux, interp_value)
                    for cc in contours:
                        ax1.plot(cc[:, 1], cc[:, 0], linewidth=2, color=color)
                        ax1.axis('off')

        if fr < img_data2.shape[-1]:
            ax2.imshow(img_data2[:, :, 0, fr], cmap='gray')
            if not np.sum(seg_data2[..., fr]) == 0:
                for labels, color in enumerate(['r', 'b', 'y', 'c', 'g']):
                    seg_aux = (seg_data2[:, :, 0, fr] == labels).astype(int)
                    contours = measure.find_contours(seg_aux, interp_value)
                    for cc in contours:
                        ax2.plot(cc[:, 1], cc[:, 0], linewidth=2, color=color)
                        ax2.axis('off')

        # title for entire sheet:
        fig.suptitle(f"Subject {study_ID} -- frame {fr}", fontsize=24)

        # Save image and contour
        filename = os.path.join(results_dir, f"{study_ID}_panel_fr_{fr}.png")
        filenames.append(filename)
        fig.savefig(filename)
        plt.close(fig)

    return filenames


# =============================================================================
# Generate Images (.png, animates .gif)
# =============================================================================
def generate_images(nifti_dir, study_IDs,
                    img_name='la_2Ch.nii.gz',
                    seg_name='la_2Ch_seg_nnUnet.nii.gz',
                    img_name2='la_4Ch.nii.gz',
                    seg_name2='la_4Ch_seg_nnUnet.nii.gz'):
    for _study_ID_counter, study_ID in enumerate(study_IDs):
        # a) load data
        subject_dir = os.path.join(nifti_dir, study_ID)
        results_dir = os.path.join(subject_dir, 'results')
        # Load image and segmentation
        try:
            img = nib.load(os.path.join(subject_dir, img_name))
            seg = nib.load(os.path.join(subject_dir, seg_name))
            img_data = img.get_fdata()
            seg_data = seg.get_fdata()
        except FileNotFoundError:
            img_data = np.zeros((200, 200, 1, 20))
            seg_data = np.zeros((200, 200, 1, 20))
        try:
            img2 = nib.load(os.path.join(subject_dir, img_name2))
            seg2 = nib.load(os.path.join(subject_dir, seg_name2))
            img_data2 = img2.get_fdata()
            seg_data2 = seg2.get_fdata()
        except FileNotFoundError:
            img_data2 = np.zeros((200, 200, 1, 20))
            seg_data2 = np.zeros((200, 200, 1, 20))
        if os.path.exists(os.path.join(results_dir, 'LA_volumes_SR_smooth.txt')):
            LAV = np.loadtxt(os.path.join(results_dir, 'LA_volumes_SR_smooth.txt'))
        else:
            LAV = np.zeros(50)
        if os.path.exists(os.path.join(results_dir, 'LA_volumes_SR_smooth.txt')):
            RAV = np.loadtxt(os.path.join(results_dir, 'RA_volumes_SR_smooth.txt'))
        else:
            RAV = np.zeros(50)
        if os.path.exists(os.path.join(results_dir, 'LA_volumes_SR_smooth.txt')):
            mapse = np.loadtxt(os.path.join(results_dir, 'LV_mid_mapse_smooth_2Ch.txt'))
        else:
            mapse = np.zeros(50)
        if os.path.exists(os.path.join(results_dir, 'LA_volumes_SR_smooth.txt')):
            tapse = np.loadtxt(os.path.join(results_dir, 'RA_tapse_smooth_la4Ch.txt'))
        else:
            tapse = np.zeros(50)
        if os.path.exists(os.path.join(results_dir, 'clinical_measure_atria.csv')):
            df = pd.read_csv(os.path.join(results_dir, 'clinical_measure_atria.csv')).values
            points = np.concatenate([df[0, 12:15].astype(int), df[0, 22:25].astype(int)])
        else:
            points = -1*np.ones(6)

        # b) generate image files:
        filenames = segmentation_validation(study_ID, img_data, seg_data, img_data2, seg_data2,
                                            LAV, RAV, mapse, tapse, points, results_dir)

        generate_animated_image(results_dir, study_ID, filenames)
        delete_temp_images(results_dir, study_ID)


def generate_animated_image(results_dir, study_ID, filenames, ext='gif', seq='LAX'):
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
# =============================================================================
def main(json_config_path):
    if os.path.exists(json_config_path):
        cfg = Params(json_config_path).dict
    else:
        raise FileNotFoundError
    local_dir = cfg['DEFAULT_LOCAL_DIR']
    nifti_dir = os.path.join(local_dir, cfg['DEFAULT_SUBDIR_NIFTI'])
    log_dir = os.path.join(local_dir, cfg['DEFAULT_LOG_DIR'])
    time_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_txt_file = os.path.join(log_dir, 'generate_LAX_panel_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting generating LAX panel\n')
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    generate_images(nifti_dir, study_IDs)
    logger.info('Closing generate_LAX_panel_log_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
