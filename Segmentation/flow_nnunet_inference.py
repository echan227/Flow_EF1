"""
$ python3 flow_nnunet_inference.py  --help
usage: flow_nnunet_inference.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib
import os
import subprocess

# imports - 3rd party:
from datetime import datetime
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, save_nifti, set_logger


# --------- --------- --------- --------- --------- --------- --------- ---------
def run_nnunet_inference(target_dir, task_folder, script_no, script_dir, log_file):
    target_imagesTs = os.path.join(target_dir, task_folder, "imagesTs")
    inference_dir = os.path.join(target_dir, task_folder, "Results")
    with open(log_file, "w+") as f:
        subprocess.run(f'bash {script_dir} -a {target_imagesTs} -b {inference_dir} -c {script_no}',
                       shell=True, stdout=f)


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
    log_txt_file = os.path.join(log_dir, 'flow_nnunet_inference_' + time_file + '.txt')
    logger = set_logger(log_txt_file)
    logger.info('Starting segmentation inference\n')
    script_dir = '/home/br14/code/Python/AI_centre/Flow_project_Carlota_Ciaran/AI_CMR_QC/Segmentation/' \
                 'inference_nnunet_ensemble.sh'

    # Run segmentation inference
    run_nnunet_inference(target_dir, 'Task118_AscAoFlow', 118, script_dir, log_txt_file)

    logger.info('Closing nnunet_segmentation_log_{}.txt'.format(time_file))


if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
