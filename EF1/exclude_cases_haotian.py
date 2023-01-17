"""
Move any cases excluded by Haotian from analysis for fair comparison
"""
import os
import shutil
from glob import glob

import pandas as pd

ef1_dir = "/data/Datasets/Flow/data_EF1"
nifti_dir = os.path.join(ef1_dir, "nifti")
sax_dir = os.path.join(ef1_dir, 'gifs_SAX')
excluded_dir = os.path.join(ef1_dir, "nifti_excluded_haotian")
if not os.path.exists(excluded_dir):
    os.mkdir(excluded_dir)

report_automatic = os.path.join(ef1_dir, "report_EF1.csv")  # our report
df_auto = pd.read_csv(report_automatic).values
report_manual = os.path.join(ef1_dir, "EF1 in CMR and Echo.xlsx")  # haotian's report
df_manual = pd.read_excel(report_manual).values

pids_manual = df_manual[:, 0]  # patient ids - manual
pids_auto = df_auto[:-1, 1]  # patient ids - automatic
eids_auto = df_auto[:-1, 0]  # anon ids - automatic

flow_dir = os.path.join(ef1_dir, 'gifs_flow')
all_flow_imgs = glob(f'{flow_dir}/*.gif')

nnunet_dir = os.path.join(ef1_dir, 'nnUNet_raw_data/Task118_AscAoFlow/Results/ensemble')
nnunet_excluded = nnunet_dir.replace('ensemble', 'ensemble_excluded_haotian')
if not os.path.exists(nnunet_excluded):
    os.mkdir(nnunet_excluded)
all_nnunet_imgs = glob(f'{nnunet_dir}/*.nii.gz')

for i, (eid_auto, pid_auto) in enumerate(zip(eids_auto, pids_auto)):
    # Make sure pid_auto is in a consistent format
    if 'Cmras' in pid_auto:
        pid_auto = pid_auto.replace('Cmras', 'CMRAS')
    if 'CMR - ' in pid_auto:
        pid_auto = pid_auto.replace('CMR - ', '')
    if len(pid_auto) > 9:
        pid_auto = pid_auto[:9]

    print(f'{i}/{len(eids_auto)}: {pid_auto}, {eid_auto}')

    if pid_auto not in pids_manual:
        # Nifti image folder
        nifti_folder = os.path.join(nifti_dir, eid_auto)
        if os.path.exists(os.path.join(excluded_dir, eid_auto)):
            continue
        shutil.move(nifti_folder, excluded_dir)

        # SAX images
        sax_img = os.path.join(sax_dir, f'{eid_auto}_SAX.gif')
        shutil.move(sax_img, excluded_dir)

        # Flow images
        to_remove_flow = [f for f in all_flow_imgs if eid_auto in f]
        for f in to_remove_flow:
            shutil.move(f, excluded_dir)

        # nnunet ensemble results
        to_remove_nnunet = [f for f in all_nnunet_imgs if eid_auto in f]
        for f in to_remove_nnunet:
            shutil.move(f, nnunet_excluded)
