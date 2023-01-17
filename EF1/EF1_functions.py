# imports - stdlib:
import os
import numpy as np

# imports 3rd party
import pylab as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import splev, splrep


def compute_EF1(study_ID, subject_dir, results_dir_SAX, results_dir_flow,
                summary_csv, logger, unanonID=None):
    """
    Optional arg unanonID which is the original patient ID - if need to relate
    anonymisedID back
    """
    flow_sequences = [
        fname[:-7]
        for fname in os.listdir(subject_dir)
        if "flow_v" in fname and fname.endswith("nii.gz") and not
        fname.endswith("_seg_nnUnet.nii.gz")
    ]

    for seq in flow_sequences:
        logger.info(f"{seq}")

        LVV_path = os.path.join(results_dir_SAX, "LVV.txt")
        flow_rate_path = os.path.join(results_dir_flow,
                                      f"{study_ID}_{seq}_ao_flow_rate_smooth.txt")
        tt_sax_path = os.path.join(subject_dir, "results_SAX", "sa_tt.txt")
        tt_sax_orig_path = os.path.join(subject_dir, "results_SAX", "sa_tt_orig.txt")
        tt_flow_path = os.path.join(subject_dir, "results_flow",
                                    f"{study_ID}_{seq}_tt_flow_smooth.txt")

        LVV = np.loadtxt(LVV_path).astype(float)
        ao_flow_rate = np.loadtxt(flow_rate_path).astype(float)
        tt_sax = np.loadtxt(tt_sax_path).astype(float)
        tt_flow = np.loadtxt(tt_flow_path).astype(float)
        peaks, _ = find_peaks(ao_flow_rate)

        # ---------------------------------------------------------------------
        # EF1 based on dV/dt
        # ---------------------------------------------------------------------
        _ES_frame = LVV.argmin()
        volume_LV_ES = LVV[0:_ES_frame]
        LVEDV = LVV[0]

        # Choose max point of 15% of the cycle as EDV
        # LVV_15 = int(len(LVV) * 0.15)
        # LVEDV = LVV[np.argmax(LVV[:LVV_15])]

        # Using smoothing
        xp = np.linspace(0, _ES_frame, _ES_frame)
        spl = splrep(xp, volume_LV_ES, s=2)
        volume_LV_ES_fit = splev(xp, spl)
        first_dervolume_ES_fit = np.gradient(volume_LV_ES_fit)
        _point_PER = first_dervolume_ES_fit.argmin()
        if _point_PER == 0:
            _point_PER = int(_ES_frame / 2)
        LVV1_deriv_smooth = np.round((LVV[_point_PER] + LVV[_point_PER - 1] +
                                      LVV[_point_PER + 1]) / 3)
        LVEF1_avg = (LVEDV - LVV1_deriv_smooth) / LVEDV * 100
        LVEF1_p1 = (LVEDV - LVV[_point_PER - 1]) / LVEDV * 100
        LVEF1_p2 = (LVEDV - LVV[_point_PER]) / LVEDV * 100
        LVEF1_p3 = (LVEDV - LVV[_point_PER + 1]) / LVEDV * 100

        # Without smoothing
        vol_first_deriv_ES = np.gradient(volume_LV_ES)
        _point_PER2 = vol_first_deriv_ES.argmin()
        LVV1_deriv = np.round((LVV[_point_PER2] + LVV[_point_PER2 - 1] +
                               LVV[_point_PER2 + 1]) / 3)
        LVEF1_2_avg = (LVEDV - LVV1_deriv) / LVEDV * 100
        LVEF1_2_p1 = (LVEDV - LVV[_point_PER2 - 1]) / LVEDV * 100
        LVEF1_2_p2 = (LVEDV - LVV[_point_PER2]) / LVEDV * 100
        LVEF1_2_p3 = (LVEDV - LVV[_point_PER2 + 1]) / LVEDV * 100

        # ---------------------------------------------------------------------
        # EF1 based on Ao flow
        # ---------------------------------------------------------------------
        # Find the peak with higher Ao flow
        peaks = peaks[ao_flow_rate[peaks].argmax()]
        tt_flow_max = tt_flow[peaks]
        ao_flow_rate_max = ao_flow_rate[peaks]

        # Skip if ao peak tt very different from existing sax tts
        if np.abs(tt_sax - tt_flow_max).min() > 50:
            logger.info("Skipped - trigger times too different for Ao and SAX")
            continue

        _point_PER_Ao = np.abs(tt_sax - tt_flow_max).argmin()
        LVV1_Ao = np.round((LVV[_point_PER_Ao] + LVV[_point_PER_Ao - 1] +
                            LVV[_point_PER_Ao + 1]) / 3)
        LVEF1_Ao_avg = (LVEDV - LVV1_Ao) / LVEDV * 100
        LVEF1_Ao_p1 = (LVEDV - LVV[_point_PER_Ao - 1]) / LVEDV * 100
        LVEF1_Ao_p2 = (LVEDV - LVV[_point_PER_Ao]) / LVEDV * 100
        LVEF1_Ao_p3 = (LVEDV - LVV[_point_PER_Ao + 1]) / LVEDV * 100

        # ---------------------------------------------------------------------
        # EF1 based on Ao frames - using percentage frames
        # ---------------------------------------------------------------------
        # Detect the first point just before flow curve crosses 0 after peak
        flow_after_peak = ao_flow_rate[peaks:]
        zero_cross = np.where(np.diff(np.signbit(flow_after_peak)))[0][0]
        zero_frame = peaks + zero_cross

        # Above code always finds point BEFORE crossing 0
        # So check if point after is actually closer to 0
        if np.abs(ao_flow_rate[zero_frame+1]) < np.abs(ao_flow_rate[zero_frame]):
            zero_frame = zero_frame + 1

        peak_pc = peaks / zero_frame
        _point_PER_Ao_pc = int(len(volume_LV_ES) * peak_pc)
        LVV1_Ao_pc = np.round((LVV[_point_PER_Ao_pc] +
                               LVV[_point_PER_Ao_pc - 1] +
                               LVV[_point_PER_Ao_pc + 1]) / 3)
        LVEF1_Ao_pc_avg = (LVEDV - LVV1_Ao_pc) / LVEDV * 100
        LVEF1_Ao_pc_p1 = (LVEDV - LVV[_point_PER_Ao_pc - 1]) / LVEDV * 100
        LVEF1_Ao_pc_p2 = (LVEDV - LVV[_point_PER_Ao_pc]) / LVEDV * 100
        LVEF1_Ao_pc_p3 = (LVEDV - LVV[_point_PER_Ao_pc + 1]) / LVEDV * 100


        # ---------------------------------------------------------------------
        # V1 using Haotian's (HG) frame number
        # ---------------------------------------------------------------------
        tt_sax_orig = np.loadtxt(tt_sax_orig_path).astype(float)
        data_HG_file = '/data/Datasets/Flow/data_EF1/EF1 in CMR and Echo.xlsx'
        df = pd.read_excel(data_HG_file)
        non_nan_inds = df["MR SAX EF1"].notna()
        data_HG = df.values
        data_HG = data_HG[non_nan_inds, :]  # Remove rows without EF1
        try:
            subj_ind_HG = np.where(data_HG[:, 0] == unanonID)[0][0]
            V1_frame_HG = data_HG[subj_ind_HG, 3]
            V_frame_HG_interp = int((V1_frame_HG / len(tt_sax_orig)) * len(tt_sax))
            plot_HG_flag = True
        except:
            plot_HG_flag = False

        # ---------------------------------------------------------------------
        # Plot flow/LVV and derivatives
        # ---------------------------------------------------------------------
        # Create figure and axis objects with subplots()
        fig, (ax, ax3) = plt.subplots(1, 2, figsize=(12, 6))
        # make a plot
        ax.plot(tt_flow, ao_flow_rate, "b-")
        ax.plot(tt_flow[peaks], ao_flow_rate[peaks], "r*")
        ax.plot(tt_flow[zero_frame], ao_flow_rate[zero_frame], "k^")
        # set x-axis label
        ax.set_xlabel("Trigger Time", fontsize=14)
        # set y-axis label
        ax.set_ylabel("Ao flow", color="blue", fontsize=14)
        # twin object for two different y-axis 3on the sample plot
        ax2 = ax.twinx()
        # make a plot with different y-axis using second axis object
        ax2.plot(tt_sax, LVV, "k-")
        # plt.plot(tt_sax[_point_PER], LVV[_point_PER], "mo",
        #          label="Smoothed Deriv")
        if plot_HG_flag:
            plt.plot(tt_sax[V_frame_HG_interp], LVV[V_frame_HG_interp], "go",
                     fillstyle="none", markersize=8, label="Haotian")
        plt.plot(tt_sax[_point_PER2], LVV[_point_PER2], "mo",
                 fillstyle="none", markersize=8, label="1st Derivative")
        plt.plot(tt_sax[_point_PER_Ao], LVV[_point_PER_Ao], "r*",
                 label="Flow")
        plt.plot(tt_sax[_point_PER_Ao_pc], LVV[_point_PER_Ao_pc], "k*",
                 label="Flow percentage frames")
        ax2.set_ylabel("LVV", color="black", fontsize=14)
        ax2.legend(loc="upper right")

        # Plot derivatives
        ax3.plot(tt_sax[:_ES_frame], vol_first_deriv_ES, label="No smoothing")
        ax3.scatter(tt_sax[:_ES_frame], vol_first_deriv_ES)
        ax3.plot(tt_sax[:_ES_frame], first_dervolume_ES_fit, label="Smoothing")
        ax3.set_ylabel("LVV 1st derivative", fontsize=14)
        ax3.set_xlabel("Trigger Time", fontsize=14)
        ax3.legend(loc="upper right")

        # Save the plot as a file
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.3)
        fig.savefig(  # Save in individual subject folder
            "{}/{}_{}.jpg".format(results_dir_flow, study_ID, seq),
            format="jpeg", dpi=100, bbox_inches="tight",
        )
        plt.close("all")

        # ---------------------------------------------------------------------
        # Get true frame numbers (wrt the original number of frames)
        # ---------------------------------------------------------------------
        ES_frame_orig = (_ES_frame / len(tt_sax)) * len(tt_sax_orig)
        point_PER_orig = (_point_PER / len(tt_sax)) * len(tt_sax_orig)
        point_PER2_orig = (_point_PER2 / len(tt_sax)) * len(tt_sax_orig)
        point_Ao_orig = (_point_PER_Ao / len(tt_sax)) * len(tt_sax_orig)
        point_Ao_pc_orig = (_point_PER_Ao_pc / len(tt_sax)) * len(tt_sax_orig)

        # ---------------------------------------------------------------------
        # Save results in csv files
        # ---------------------------------------------------------------------
        data_EF1 = np.zeros((1, 37), dtype=object)
        data_EF1[:, 0] = study_ID
        data_EF1[:, 1] = seq
        data_EF1[:, 2] = LVEDV
        data_EF1[:, 3] = ES_frame_orig  # frame num ES
        data_EF1[:, 4] = tt_sax[_ES_frame]  # Time to ES
        data_EF1[:, 5] = LVV[_ES_frame]  # ESV
        data_EF1[:, 6] = LVV1_deriv_smooth  # V1 (smoothing)
        data_EF1[:, 7] = LVEF1_avg
        data_EF1[:, 8] = LVEF1_p1
        data_EF1[:, 9] = LVEF1_p2
        data_EF1[:, 10] = LVEF1_p3
        data_EF1[:, 11] = point_PER_orig
        data_EF1[:, 12] = tt_sax[_point_PER]  # Time to PER (smoothing)
        data_EF1[:, 13] = LVV1_deriv  # V1 (no smoothing)
        data_EF1[:, 14] = LVEF1_2_avg
        data_EF1[:, 15] = LVEF1_2_p1
        data_EF1[:, 16] = LVEF1_2_p2
        data_EF1[:, 17] = LVEF1_2_p3
        data_EF1[:, 18] = point_PER2_orig
        data_EF1[:, 19] = tt_sax[_point_PER2]  # Time to PER (no smoothing)
        data_EF1[:, 20] = ao_flow_rate_max
        data_EF1[:, 21] = tt_flow_max  # Time in terms of flow sequence
        data_EF1[:, 22] = peaks  # Flow peak frame number
        data_EF1[:, 23] = LVV1_Ao  # V1 (flow)
        data_EF1[:, 24] = LVEF1_Ao_avg
        data_EF1[:, 25] = LVEF1_Ao_p1
        data_EF1[:, 26] = LVEF1_Ao_p2
        data_EF1[:, 27] = LVEF1_Ao_p3
        data_EF1[:, 28] = point_Ao_orig
        data_EF1[:, 29] = tt_sax[_point_PER_Ao]  # Time to PER in terms of SAX
        data_EF1[:, 30] = LVV1_Ao_pc  # V1 (flow - percentage frames)
        data_EF1[:, 31] = LVEF1_Ao_pc_avg
        data_EF1[:, 32] = LVEF1_Ao_pc_p1
        data_EF1[:, 33] = LVEF1_Ao_pc_p2
        data_EF1[:, 34] = LVEF1_Ao_pc_p3
        data_EF1[:, 35] = point_Ao_pc_orig
        data_EF1[:, 36] = tt_sax[_point_PER_Ao_pc]  # Time to PER in terms of SAX

        header = [
            "Anonymised ID",
            "Flow Seq",
            "EDV",
            "Point ES",
            "tt ES",
            "ESV",
            "V1",
            "LVEF1 dV/dt",
            "LVEF1 p1 dV/dt",
            "LVEF1 p2 dV/dt",
            "LVEF1 p3 dV/dt",
            "Point EF1 (dv/dt)",
            "tt EF1 (dv/dt)",
            "V1 2",
            "LVEF1 dV/dt 2 ",
            "LVEF1 dV/dt 2 p1",
            "LVEF1 dV/dt 2 p2",
            "LVEF1 dV/dt 2 p3",
            "Point EF1 (dv/dt) 2 ",
            "tt EF1 (dv/dt) 2",
            "Ao flow max",
            "tt Ao flow max",
            "Point Ao flow max",
            "V1 flow",
            "LVEF1 Ao",
            "LVEF1 Ao p1",
            "LVEF1 Ao p2",
            "LVEF1 Ao p3",
            "Point EF1 (Ao)",
            "tt EF1 (Ao)",
            "V1 flow (% frames)",
            "LVEF1 Ao (% frames)",
            "LVEF1 Ao p1 (% frames)",
            "LVEF1 Ao p2 (% frames)",
            "LVEF1 Ao p3 (% frames)",
            "Point EF1 (Ao % frames)",
            "tt EF1 (Ao % frames)",
        ]

        # If an unanoymised patient ID is provided, insert into second column
        if unanonID:
            data_EF1 = np.insert(data_EF1, 1, unanonID)[None, ...]
            header.insert(1, "Patient ID")

        # Save individual results in results_flow dir
        df2 = pd.DataFrame(data_EF1)
        df2.to_csv(
            f"{results_dir_flow}/{study_ID}_{seq}_EF1_params.csv",
            header=header,
            index=False,
            sep=",",
            encoding="utf-8",
        )

        # Save results in summary spreadsheet
        write_header = header if not os.path.exists(summary_csv) else False
        df2.to_csv(summary_csv, mode="a", header=write_header, index=False)
