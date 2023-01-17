import os
import numpy as np

flow_path = "/data/Datasets/Flow/data_EF1/EF1_plots/Flow_LVV_curves"
nifti_path = "/data/Datasets/Flow/data_EF1/nifti"
all_cases = sorted(os.listdir(flow_path))

all_diff = np.zeros(len(all_cases))
for ind, current_case in enumerate(all_cases):
    eid = current_case.split("_flow")[0]
    img_name = current_case.split(f"{eid}_")[1].replace(".jpg", "")

    data_path = os.path.join(nifti_path, eid)
    phase_HR_file = img_name.replace("flow_", "flow_P_") + "_HR.txt"
    phase_HR_path = os.path.join(data_path, phase_HR_file)
    sa_HR_path = os.path.join(data_path, "sa_HR.txt")

    phase_HR = np.loadtxt(phase_HR_path)
    sa_HR = np.loadtxt(sa_HR_path)
    diff = sa_HR - phase_HR
    all_diff[ind] = diff

    print(f"{current_case}: {diff}")

print(f"Difference: Mean {all_diff.mean()}, Std {all_diff.std()}")
print(f"Abs difference: Mean {np.abs(all_diff).mean()}, Std {np.abs(all_diff).std()}")
