# imports - stdlib
import os
import shutil
import subprocess

# imports - 3rd party:
import numpy as np
from os.path import join as join_dirs

# imports - local
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, get_list_of_files

def remove_small_files(study_IDs, nifti_dir):
    for idx, study_ID in enumerate(study_IDs):
        try:
            source_ID_dir = os.path.join(nifti_dir, study_ID)
            nifty_files = np.array(get_list_of_files(source_ID_dir, full_path=False, ext_str='.nii.gz'))
            for nifty_file in nifty_files:
                # Remove study folders smaller than 5MB
                path_nifty_file = join_dirs(source_ID_dir, nifty_file)
                bash_size = 'du -shm ' + path_nifty_file
                file_size = subprocess.check_output(bash_size.split(), universal_newlines=True).split()[0]
                if int(file_size) <= 0.5:
                    print('Removing {} ({}MB)\n'.format(path_nifty_file, file_size))
                    # shutil.rmtree(path_nifty_file)
        except Exception as e:
            print('Caught exception "{}" while revising study {}. Continue.'.format(e, study_ID))

def main(json_config_path):
    if os.path.exists(json_config_path):
        cfg = Params(json_config_path).dict
    else:
        raise FileNotFoundError
    local_dir = cfg['DEFAULT_LOCAL_DIR']
    nifti_dir = os.path.join(local_dir, cfg['DEFAULT_SUBDIR_NIFTI'])
    study_IDs = get_list_of_dirs(nifti_dir, full_path=False)
    remove_small_files(study_IDs, nifti_dir)

if __name__ == '__main__':
    DEFAULT_JSON_FILE = '/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json'
    main(DEFAULT_JSON_FILE)
