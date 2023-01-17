# !/usr/bin/env python3 pylint: disable=invalid-name,unused-import,missing-module-docstring,missing-docstring,
# line-too-long,too-many-arguments,too-many-locals,too-many-branches,too-many-statements,bad-whitespace,
# pointless-string-statement,trailing-newlines

"""
$ python3 v.py  --help
usage: plot_EF1_bland_altman.py [-h] [-i JSON_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -i JSON_FILE, --json_config_path JSON_FILE
                        Json file with config
"""

# imports - stdlib:
import os
from itertools import combinations

# imports - 3rd party
from datetime import datetime
from re import X
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from pyCompare._calculateConfidenceIntervals import calculateConfidenceIntervals
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr

# imports - local
from common_utils.load_args import Params
from common_utils.utils import set_logger


def calc_pearson_coefficient(data1, data2, logger):
    # Concordance correlation coefficient
    ccc, ppp = pearsonr(data1, data2)
    logger.info('Pearson coefficient: {0:.3f}'.format(ccc))
    logger.info('p-value pearson coefficient: {0}'.format(ppp))

    return ccc, ppp


def plot_graphs(save_path_png, data1, data2, eids, title, xlims, ylims,
                xtitle, ytitle, r, p, logger):
    limitOfAgreement = 1.96
    confidenceInterval = 95
    # 'exact paired' uses the exact paired method described by Carkeet
    confidenceIntervalMethod = 'approximate'
    dpi = 75
    save_path_pdf = save_path_png.replace(".png", ".pdf")

    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)

    print(f'Mean: {mean.min()} - {mean.max()}')
    print(f'Diff: {diff.min()} - {diff.max()}')

    confidenceIntervals = calculateConfidenceIntervals(
        md, sd, len(diff), limitOfAgreement, confidenceInterval,
        confidenceIntervalMethod)

    figureSize = (24, 9.5)
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 24}
    plt.rc('font', **font)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=figureSize, dpi=dpi)

    if 'mean' in confidenceIntervals.keys():
        ax.axhspan(confidenceIntervals['mean'][0],
                   confidenceIntervals['mean'][1],
                   facecolor='silver', alpha=0.2)

    if 'upperLoA' in confidenceIntervals.keys():
        ax.axhspan(confidenceIntervals['upperLoA'][0],
                   confidenceIntervals['upperLoA'][1],
                   facecolor='coral', alpha=0.2)

    if 'lowerLoA' in confidenceIntervals.keys():
        ax.axhspan(confidenceIntervals['lowerLoA'][0],
                   confidenceIntervals['lowerLoA'][1],
                   facecolor='coral', alpha=0.2)

    # Plot the mean diff and LoA
    ax.axhline(0, color='k')
    ax.axhline(md, color='grey', linestyle='--')
    ax.axhline(md + limitOfAgreement * sd, color='red', linestyle='--')
    ax.axhline(md - limitOfAgreement * sd, color='red', linestyle='--')

    ax.scatter(mean, diff, color='k', marker='o', s=60)

    ax.set_xlim((xlims[0], xlims[1]))
    ax.set_ylim((ylims[0], ylims[1]))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    limitOfAgreementRange = (md + (limitOfAgreement * sd)) - \
        (md - limitOfAgreement * sd)
    offset = (limitOfAgreementRange / 100.0) * 1.5

    ax.text(1.01, md + offset, 'Mean', ha="left", va="bottom", transform=trans)
    ax.text(1.01, md - offset, '%.2f' % (md), ha="left", va="top",
            transform=trans)

    ax.text(1.01, md + (limitOfAgreement * sd) + offset, '+%.2f SD'
            % (limitOfAgreement), ha="left", va="bottom",
            transform=trans)
    ax.text(1.01, md + (limitOfAgreement * sd) - offset, '%.2f'
            % (md + limitOfAgreement * sd), ha="left", va="top",
            transform=trans)

    ax.text(1.01, md - (limitOfAgreement * sd) - offset, '-%.2f SD'
            % (limitOfAgreement), ha="left", va="top", transform=trans)
    ax.text(1.01, md - (limitOfAgreement * sd) + offset, '%.2f'
            % (md - limitOfAgreement * sd), ha="left", va="bottom",
            transform=trans)

    ax.set_ylabel('Difference')
    ax.set_xlabel('Mean')

    ax.patch.set_alpha(0)
    ax.set_title(title)

    # Correlation plot
    ax2.scatter(data1, data2, color='k', marker='o', s=60)
    # a, b = np.polyfit(data1, data2, 1)
    # ax2.plot(data1, a*data1+b)
    ax2.set_ylabel(ytitle)
    ax2.set_xlabel(xtitle)
    ax2.set_title(f'r = {r:.4f}, p = {p:.4f}')

    plt.tight_layout()
    fig.savefig(save_path_png, dpi=dpi)
    fig.savefig(save_path_pdf, dpi=dpi)
    plt.close()

    # List outlier IDs
    logger.info('Outliers:')
    for current_eid, current_mean, current_diff in zip(eids, mean, diff):
        if current_diff > md + limitOfAgreement * sd or \
          current_diff < md - limitOfAgreement * sd:
            logger.info(f'{current_eid[0]}_{current_eid[2]}: '
                        f'mean {current_mean:.2f}, diff {current_diff:.2f}')


def do_generate(local_dir, our_data_path, HG_data_path, logger):
    BA_dir = os.path.join(local_dir, 'EF1_plots', 'Bland_altman_plots')
    if not os.path.exists(BA_dir):
        os.mkdir(BA_dir)

    # Load our data (remove last row which contains mean (std))
    data_all = pd.read_csv(our_data_path).values
    data_ours = data_all[:-1, [4, 12, 19, 29, 36, 30, 37, 3, 5, 6, 7, 14, 24, 31, 8, 15,
                               25, 32]]
    EF1_data = data_ours[:, -3:]
    eids = data_all[:-1, [0, 1, 2]]

    # Load Haotian's (HG) data
    df = pd.read_excel(HG_data_path)
    non_nan_inds = df["MR SAX EF1"].notna()
    data_HG_all = df.values
    data_HG = data_HG_all[:, [2, 3, 3, 3, 3, 4, 4, 6, 5, 8, 7, 7, 7, 7, 10, 10, 10, 10]]
    data_HG = data_HG[non_nan_inds, :]  # Remove rows without EF1
    IDs_HG = data_HG_all[non_nan_inds, 0]
    
    # ---------------------------------------------------------------------
    # Compare our methods to one another
    # ---------------------------------------------------------------------
    # Options for BA plots
    titles = ['EF1: dV/dt Smoothing vs dV/dt No Smoothing',
              'EF1: dV/dt Smoothing vs Flow',
              'EF1: dV/dt No Smoothing vs Flow']
    save_names = ['EF1_dVdt_smooth_vs_dvdt_noSmooth',
                  'EF1_dVdt_smooth_vs_Ao',
                  'EF1_dVdt_noSmooth_vs_Ao']
    xlims = [[0, 60], [0, 60], [0, 60]]
    ylims = [[-35, 45], [-35, 45], [-35, 45]]

    # Options for correlation plot
    x_titles = ['dV/dt Smoothing', 'dV/dt Smoothing', 'dV/dt No Smoothing']
    y_titles = ['dV/dt No Smoothing', 'Flow', 'Flow']

    # Compare columns pairwise
    ind_pairs = list(combinations(range(EF1_data.shape[1]), 2))
    total_comparisons = len(ind_pairs) + data_ours.shape[1]
    pearson_data = np.zeros((total_comparisons, 2), dtype=object)
    for i, (data_inds, title, save_name, xlim, ylim, xtitle, ytitle)\
        in enumerate(zip(ind_pairs, titles, save_names, xlims, ylims,
                         x_titles, y_titles)):
        data1 = np.asarray(EF1_data[:, data_inds[0]].astype(float))
        data2 = np.asarray(EF1_data[:, data_inds[1]].astype(float))

        save_path_png = os.path.join(BA_dir, f'{save_name}.png')
        logger.info(f'[{i+1}/{total_comparisons}]: {save_name}')

        pearson_data[i, 0], pearson_data[i, 1] = \
            calc_pearson_coefficient(data1, data2, logger)
        plot_graphs(save_path_png, data1, data2, eids, title, xlim, ylim,
                    xtitle, ytitle, pearson_data[i, 0], pearson_data[i, 1],
                    logger)

    # ---------------------------------------------------------------------
    # Compare our methods to Haotian's
    # ---------------------------------------------------------------------
    # Options for BA plots
    titles = ['Frame ES', 'Frame V1 (1st deriv smooth)',
              'Frame V1 (1st deriv)', 'Frame V1 (Flow)',
              'Frame V1 (Flow % frames)', 'Time to Ao peak',
              'Time to Ao peak (% frames)', 'EDV', 'Time to ES', 'ESV',
              'V1 (1st deriv smooth)', 'V1 (1st deriv)', 'V1 (flow)',
              'V1 (flow % frames)', 'EF1 (1st deriv smooth)',
              'EF1 (1st deriv)', 'EF1 (flow)', 'EF1 (flow % frames)']
    save_names2 = ['frame_ES', 'frame_V1_deriv_smooth',
                   'frame_V1_deriv', 'frame_V1_flow',
                   'frame_V1_flow_pc', 'tt_ao_peak',
                   'tt_ao_pc_peak', 'EDV', 'tt_ES', 'ESV',
                   'V1_deriv_smooth', 'V1_deriv', 'V1_flow',
                   'V1_flow_pc', 'EF1_deriv_smooth',
                   'EF1_deriv', 'EF1_flow', 'EF1_flow_pc']
    xlims = [[0, 25], [0, 10], [0, 10], [0, 10], [0, 10],
             [70, 200], [70, 200], [80, 360], [230, 410], [10, 210], [55, 305],
             [55, 305], [55, 305], [55, 305], [0, 50], [0, 50], [0, 50],
             [0, 50]]
    ylims = [[-15, 10], [-10, 10], [-10, 10], [-10, 10], [-10, 10],
             [-70, 150], [-70, 150], [-80, 35], [-125, 165], [-35, 35], [-100, 45],
             [-100, 45], [-100, 45], [-100, 45], [-35, 45], [-35, 45], [-35, 45],
             [-35, 45]]

    # Options for correlation plot
    x_title = 'Automatic'
    y_title = 'Manual'

    # Make sure data points are in same order for comparison
    inds = [np.where(ID == eids[:, 1])[0][0] for ID in IDs_HG
            if ID in eids[:, 1]]
    data_ours_sorted = data_ours[inds, :]
    eids_sorted = eids[inds, :]

    # Remove from Haotian's data if we excluded from our analysis
    remove_IDs_HG = [ind for ind, ID in enumerate(IDs_HG)
                     if ID not in eids[:, 1]]
    data_HG = np.delete(data_HG, remove_IDs_HG, axis=0)
    IDs_HG = np.delete(IDs_HG, remove_IDs_HG, axis=0)

    for i, (title, save_name, xlim, ylim) in\
        enumerate(zip(titles, save_names2, xlims,
                      ylims)):
        data1 = np.asarray(data_ours_sorted[:, i].astype(float))
        data2 = np.asarray(data_HG[:, i].astype(float))

        save_path_png = os.path.join(BA_dir, f'{save_name}.png')
        logger.info(f'[{i+1+len(ind_pairs)}/{total_comparisons}]: {save_name}')

        pearson_ind = i + len(ind_pairs)
        pearson_data[pearson_ind, 0], pearson_data[pearson_ind, 1] = \
            calc_pearson_coefficient(data1, data2, logger)
        plot_graphs(save_path_png, data1, data2, eids_sorted, title, xlim,
                    ylim, x_title, y_title, pearson_data[pearson_ind, 0],
                    pearson_data[pearson_ind, 1], logger)

    # ---------------------------------------------------------------------
    # Save pearson coefficient results
    # ---------------------------------------------------------------------
    df_pearson = pd.DataFrame(pearson_data, index=save_names + save_names2)
    df_pearson.to_csv(os.path.join(BA_dir, 'EF1_pearson_coefficient.csv'),
                      header=['coef', 'p-value'])


# =============================================================================
# MAIN
# =============================================================================
def main(json_config_path):
    if os.path.exists(json_config_path):
        cfg = Params(json_config_path).dict
    else:
        raise FileNotFoundError
    local_dir = cfg["DEFAULT_LOCAL_DIR"]
    log_dir = os.path.join(local_dir, cfg["DEFAULT_LOG_DIR"])
    # Start logging console prints
    time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_txt_file = os.path.join(log_dir, "plot_BA_" + time_file + ".txt")
    logger = set_logger(log_txt_file)
    logger.info("Starting generating Bland-Altman plots\n")
    data_ours_file = os.path.join(local_dir, 'report_EF1.csv')
    data_HG_file = os.path.join(local_dir, 'EF1 in CMR and Echo.xlsx')
    do_generate(local_dir, data_ours_file, data_HG_file, logger)
    logger.info("Closing plot_BA_{}.txt".format(time_file))


if __name__ == "__main__":
    import sys

    sys.path.append("/home/bram/Scripts/AI_CMR_QC")
    DEFAULT_JSON_FILE = "/home/bram/Scripts/AI_CMR_QC/configs/basic_opt.json"
    main(DEFAULT_JSON_FILE)
