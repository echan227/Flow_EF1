"""
$ python3 compute_sax_strain.py  --help
usage: compute_sax_strain.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib
import argparse
import os

# imports 3rd party
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# imports - 3rd party:
from datetime import datetime
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, set_logger
from Strain.strain_functions import compute_myo_points, calc_peak_diast_strain


def do_studies(study_IDs, nifti_dir, logger, nb_layers_SA = 3, n_samples = 24, window_size = 7,  poly_order = 3):
    for study_ID_counter, study_ID in enumerate(study_IDs):
        logger.info(f"do_studies() [{study_ID_counter}]: {study_ID}")
        subject_dir = os.path.join(nifti_dir, study_ID)
        results_dir = os.path.join(subject_dir, 'results_strain')
        if not os.path.exists(results_dir):
            os.system('mkdir -p {0}'.format(results_dir))
        # Load data
        img_dir = os.path.join(subject_dir, 'sa.nii.gz')
        seg_dir = os.path.join(subject_dir, 'sa_seg_nnUnet.nii.gz')
        if not (os.path.exists(img_dir) or os.path.exists(seg_dir)):
            logger.info('No SAX data for subject {}'.format(study_ID))
            continue

        try:
            N_frames,slices_selected, points_myo, tt, dt = compute_myo_points(img_dir, seg_dir, results_dir)
        except:
            logger.info('Failed to compute strain points for subject {}'.format(study_ID))
            continue

        # =============================================================================
        # Circ strain
        # =============================================================================
        try:
            dist = np.zeros((N_frames, len(slices_selected), nb_layers_SA))
            for fr in range(N_frames):
                for sl in range(len(slices_selected)):
                    for ly in range(nb_layers_SA):
                        dd = 0
                        for ss in range(n_samples - 1):
                            dd += np.linalg.norm(points_myo[fr, sl, ly, ss, :] - points_myo[fr, sl, ly, ss + 1, :])
                        dist[fr, sl, ly] = dd

            cric_strain_per_slice = np.zeros((len(slices_selected), N_frames))
            slice_to_reject = []
            for sl in range(len(slices_selected)):
                if len(np.unique((np.where(points_myo[:, sl, :, :, :] == 0)[0]))) < 5:
                    cric_strain_per_slice[sl, :] = np.mean((dist[:, sl, :] - dist[0, sl, :]) / dist[0, sl, :], axis=1)
                    cric_strain_per_slice[sl, :] *= 100
                    if np.sum(cric_strain_per_slice[sl, :] < -55) > 0:
                        cric_strain = cric_strain_per_slice[sl, :]
                        indx = np.where(cric_strain < -55)[0]
                        cric_strain[indx] = np.NaN
                        s = pd.Series(cric_strain)
                        cric_strain = s.interpolate(method='polynomial', order=2).to_numpy()
                        cric_strain_per_slice[sl, :] = cric_strain
                else:
                    slice_to_reject.append(sl)

            if len(slice_to_reject) >= 1:
                slice_to_reject = np.unique(np.array(slice_to_reject))
                slices_selected = np.delete(slices_selected, slice_to_reject)
                points_myo = np.delete(points_myo, slice_to_reject, axis=1)
                cric_strain_per_slice = np.delete(cric_strain_per_slice, slice_to_reject, axis=0)
        
            smooth_circ_strain = np.zeros((len(slices_selected) + 1, N_frames + 1), dtype=object)
            plt.figure()
            for s, sl in enumerate(slices_selected):
                x = np.linspace(0, N_frames - 1, N_frames)
                xx = np.linspace(np.min(x), np.max(x), N_frames)
                itp = interp1d(x, cric_strain_per_slice[s, :])
                cric_strain_smooth = savgol_filter(itp(xx), window_size, poly_order)
                plt.plot(cric_strain_smooth, label='slice ' + str(sl))
                np.savetxt(os.path.join(results_dir, 'circ_strain_slice_{}.txt'.format(sl + 1)), cric_strain_per_slice[s, :])
                smooth_circ_strain[s + 1, 0] = sl + 1
                smooth_circ_strain[s + 1, 1:] = cric_strain_smooth
                                
            plt.title('{}: ECC per slice'.format(study_ID))
            plt.legend()
            plt.savefig(os.path.join(results_dir, 'ECC_per_slice.png'))
            plt.close('all')

            smooth_circ_strain[0, 0] = 'all slices'
            smooth_circ_strain[0, 1:] = np.mean(smooth_circ_strain[1:, 1:], axis=0)
            smooth_circ_strain = np.vstack([np.hstack(['trigger time', tt]), smooth_circ_strain])
            df = pd.DataFrame(smooth_circ_strain)
            new_name = '{0}/Ecc.csv'.format(results_dir, study_ID)
            df.to_csv(new_name, header=None, index=False, sep=',', encoding='utf-8')

            plt.figure()
            plt.plot(smooth_circ_strain[1, 1:])
            plt.axis([0, N_frames, -40, 5])
            plt.title('{}: ECC'.format(study_ID))
            plt.savefig(os.path.join(results_dir, 'ECC.png'))
            plt.close('all')

            # Calculate peak strain 
            Name_curve = 'Circ_strain'
            mean_strain_curve = np.mean(smooth_circ_strain[1:, 1:], axis=0)
            circ_peak_strain, TPK_circ_strain, diast_circ_strain, TPK_diast_circ_strain = calc_peak_diast_strain(
            mean_strain_curve, dt, results_dir, Name_curve)

            out_csv = -1 * np.ones((1,4))
            out_csv[0, 0] = circ_peak_strain
            out_csv[0, 1] = TPK_circ_strain
            out_csv[0, 2] = diast_circ_strain
            out_csv[0, 3] = TPK_diast_circ_strain

            header = ['circ_peak_strain', 'TPK_circ_strain', 'diast_circ_strain', 'TPK_diast_circ_strain' ]
            df = pd.DataFrame(out_csv)
            df.to_csv('{0}/report_SAX_circ_strain.csv'.format(results_dir),
                    header=header, index=False)

        except:
            logger.info('Failed to compute circ strain subject {}'.format(study_ID))

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
    target_dir = os.path.join(local_dir, cfg['DEFAULT_NNUNET_NIFTI'])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    log_dir = os.path.join(local_dir, cfg['DEFAULT_LOG_DIR'])
    # Start logging console prints
    time_file = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_txt_file = os.path.join(log_dir, 'compute_sax_strain_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting computing SAX strain\n')
    # Find studies to analyse
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    do_studies(study_IDs, nifti_dir, logger)
    logger.info('Closing compute_sax_strain_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
