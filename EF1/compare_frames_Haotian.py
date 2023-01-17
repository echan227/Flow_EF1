"""
Use frame numbers that Haotian used to calculate tt, ESV and V1
Compare to our own results
"""

import os
from datetime import datetime
import sys

import pandas as pd
import numpy as np

sys.path.append('/home/br14/code/Python/AI_centre/Flow_project_Carlota_Ciaran/AI_CMR_QC')
from EF1.plot_EF1_bland_altman import plot_graphs, calc_pearson_coefficient
from common_utils.utils import set_logger

EF1_folder = "/data/Datasets/Flow/data_EF1"
HG_data_path = os.path.join(EF1_folder, "EF1 in CMR and Echo.xlsx")
our_data_path = os.path.join(EF1_folder, "report_EF1.csv")
data_path = os.path.join(EF1_folder, "nifti")
save_path = os.path.join(EF1_folder, "EF1_plots/BA_HG_frames")
if not os.path.exists(save_path):
    os.mkdir(save_path)

time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(EF1_folder, "log")
log_txt_file = os.path.join(log_dir, f"compare_HG_frames_{time_file}.txt")
logger = set_logger(log_txt_file)

# -----------------------------------------------------------------------------
# Load HG's data and match to our IDs
# -----------------------------------------------------------------------------
# Load Haotian's (HG) data
df = pd.read_excel(HG_data_path)
non_nan_inds = df["MR SAX EF1"].notna()
data_HG_all = df.values
data_HG = data_HG_all[:, [2, 3, 5, 8, 4, 7]]
data_HG = data_HG[non_nan_inds, :]  # Remove rows without EF1
IDs_HG = data_HG_all[non_nan_inds, 0]

# Load our IDs
data_all = pd.read_csv(our_data_path).values
eids = data_all[:-1, [0, 1, 2]]

# Make sure data points are in same order for comparison
inds = [np.where(ID == eids[:, 1])[0][0] for ID in IDs_HG
        if ID in eids[:, 1]]
eids_sorted = eids[inds, :]

# Remove from Haotian's data if we excluded from our analysis
remove_IDs_HG = [ind for ind, ID in enumerate(IDs_HG)
                 if ID not in eids[:, 1]]
data_HG = np.delete(data_HG, remove_IDs_HG, axis=0)
IDs_HG = np.delete(IDs_HG, remove_IDs_HG, axis=0)

# -----------------------------------------------------------------------------
# Use HG's frames to find tt_ES, ESV, tt_V1 and V1 using our generated curves
# -----------------------------------------------------------------------------
data_ours = np.zeros((data_HG.shape[0], data_HG.shape[1]-2))
for i, (studyID, _, _) in enumerate(eids_sorted):
    logger.info(f'[{i}/{len(eids_sorted)}]: {studyID}')
    current_folder = os.path.join(data_path, studyID, "results_SAX")

    LVV = np.loadtxt(os.path.join(current_folder, "LVV.txt"))
    tt = np.loadtxt(os.path.join(current_folder, "sa_tt.txt"))
    tt_orig = np.loadtxt(os.path.join(current_folder, "sa_tt_orig.txt"))

    # Since we interpolate, find nearest frame in our own curve to Haotian's
    frame_ES = int((data_HG[i, 0] / len(tt_orig)) * len(tt))
    frame_V1 = int((data_HG[i, 1] / len(tt_orig)) * len(tt))

    data_ours[i, 0] = tt[frame_ES]  # tt_ES
    data_ours[i, 1] = LVV[frame_ES]  # ESV
    data_ours[i, 2] = tt[frame_V1]  # tt_V1
    data_ours[i, 3] = LVV[frame_V1]  # V1

# -----------------------------------------------------------------------------
# Plot correlation and BA plots
# -----------------------------------------------------------------------------
# Options for BA plots
titles = ['Time to ES', 'ESV', 'Time to Ao peak', 'V1']
save_names2 = ['tt_ES', 'ESV', 'tt_ao_peak', 'V1']
xlims = [[240, 450], [15, 210], [70, 190], [60, 300]]
ylims = [[-200, 60], [-40, 30], [-80, 100], [-30, 65]]

# Options for correlation plot
x_title = "Haotian's"
y_title = 'Ours'

pearson_data = np.zeros((data_ours.shape[1], 2), dtype=object)
for i, (title, save_name, xlim, ylim) in enumerate(zip(titles, save_names2,
                                                       xlims, ylims)):
    data1 = np.asarray(data_HG[:, i+2].astype(float))
    data2 = np.asarray(data_ours[:, i].astype(float))

    save_path_png = os.path.join(save_path, f'{save_name}.png')
    logger.info(f'[{i+1}/{data_ours.shape[1]}]: {save_name}')

    pearson_data[i, 0], pearson_data[i, 1] = \
        calc_pearson_coefficient(data1, data2, logger)
    plot_graphs(save_path_png, data1, data2, eids, title, xlim, ylim,
                x_title, y_title, pearson_data[i, 0], pearson_data[i, 1],
                logger)
