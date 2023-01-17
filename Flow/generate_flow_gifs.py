"""
$ python3 generate_flow_gifs.py  --help
usage: generate_flow_gifs.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib
import os
import nibabel as nib
import numpy as np

# imports - 3rd party:
from datetime import datetime
from common_utils.load_args import Params
from common_utils.utils import set_logger
import matplotlib.pyplot as plt
from skimage import measure
from pygifsicle import optimize
import imageio
from scipy.signal import find_peaks


def crop_image(image, size):
    X, Y, z, t = image.shape
    cx, cy = int(X / 2), int(Y / 2)
    r = int(size / 2)
    x1, x2 = cx - r, cx + r
    y1, y2 = cy - r, cy + r
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_, :, :]

    return crop


def do_generate(_local_dir, target_dir, task_name, _logger):
    target_base = os.path.join(target_dir, task_name, 'Results')
    target_imagesTs = os.path.join(target_dir, task_name, "imagesTs")
    Results_test = os.path.join(target_base, 'ensemble')
    curve_save_dir = os.path.join(target_base, 'curves')
    tt_dir = os.path.join(target_base, "tt_flow")

    gif_dir = os.path.join(_local_dir, 'gifs_flow')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    Xte = sorted(os.listdir(Results_test))
    gif_failed = []

    for d, file_path in enumerate(Xte):
        _logger.info(f'[{d+1}/{len(Xte)}]: {file_path}')

        seg_path = os.path.join(Results_test, file_path)
        eid = file_path.replace('.nii.gz', '')  # e.g. A_S000001_flow_v300in_0
        ao_flow_rate_path = os.path.join(curve_save_dir, f'{eid}_ao_flow_rate_smooth.txt')
        ao_flow_rate_smooth = np.loadtxt(ao_flow_rate_path).astype(float)
        tt_flow = np.loadtxt(os.path.join(tt_dir, f'{eid}_tt_flow_smooth.txt')).astype(float)

        # Load images
        img_mag = nib.load(os.path.join(target_imagesTs, file_path.replace('.nii.gz', '_0000.nii.gz')))
        img_mag_array = img_mag.get_fdata()
        img_flow = nib.load(os.path.join(target_imagesTs, file_path.replace('.nii.gz', '_0001.nii.gz')))
        img_flow_array = img_flow.get_fdata()
        seg = nib.load(seg_path).get_fdata()
        _, _, T = seg.shape
        
        # ---------------------------------------------------
        # Create gifs
        # ---------------------------------------------------
        img_mag_crop = crop_image(np.expand_dims(img_mag_array, 2), 100)
        img_flow_crop = crop_image(np.expand_dims(img_flow_array, 2), 100)
        seg_crop = crop_image(np.expand_dims(seg, 2), 100)

        filenames = []
        gif_file = os.path.join(gif_dir, f'{eid}.gif')
        _logger.info(gif_file)
        try:
            for fr in range(T):
                f = plt.figure(figsize=(12, 6))  # Notice the equal aspect ratio
                axs = [f.add_subplot(1, 3, i + 1) for i in range(3)]
                f.subplots_adjust(wspace=0, hspace=0)
                axs[0].plot(tt_flow, ao_flow_rate_smooth, label='Aortic flow rate (ml/s)')
                axs[0].legend(loc="upper center")
                axs[1].title.set_text(f'{eid} - tt {tt_flow[fr]:.2f} - frame {fr}')
                axs[1].set_xticklabels([])
                axs[1].set_yticklabels([])
                axs[1].set_aspect('equal')
                axs[1].axis('off')
                axs[1].imshow(img_mag_crop[:, :, 0, fr], cmap=plt.cm.gray)
                axs[2].set_xticklabels([])
                axs[2].set_yticklabels([])
                axs[2].set_aspect('equal')
                axs[2].axis('off')
                axs[2].imshow(img_flow_crop[:, :, 0, fr], cmap=plt.cm.gray)
                contours = measure.find_contours(seg_crop[:, :, 0, fr], 0.7)
                for cc in contours:
                    axs[1].plot(cc[:, 1], cc[:, 0], linewidth=2, color='r')

                f.tight_layout(rect=[0, 0.01, 1, 0.99])
                plt.subplots_adjust(top=0.99, bottom=0.05, left=0.05, right=0.95, hspace=0.25,
                                    wspace=0.2)
                f.savefig(f'{gif_dir}/{eid}_panel_fr_{fr}.png')
                filenames.append(f'{gif_dir}/{eid}_panel_fr_{fr}.png')
                plt.close('all')

            # Save the image as a gif file
            with imageio.get_writer(gif_file, mode='I') as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            optimize(gif_file)

            os.system('rm -rf {0}/*.png'.format(gif_dir))
        except:
            _logger.info('Failed to create gif')
            gif_failed.append(eid)

    _logger.info(f'Eids failed - gifs: {gif_failed}')
    _logger.info(f'Total failed: {len(gif_failed)}')


def main(json_config_path):
    if os.path.exists(json_config_path):
        cfg = Params(json_config_path).dict
    else:
        raise FileNotFoundError
    local_dir = cfg['DEFAULT_LOCAL_DIR']
    target_dir = os.path.join(local_dir, cfg['DEFAULT_NNUNET_NIFTI'])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    log_dir = os.path.join(local_dir, cfg['DEFAULT_LOG_DIR'])
    # Start logging console prints
    time_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_txt_file = os.path.join(log_dir, 'generate_flow_gifs_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting generating flow gifs\n')
    do_generate(local_dir, target_dir, "Task118_AscAoFlow", logger)
    logger.info('Closing generate_flow_gifs_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
