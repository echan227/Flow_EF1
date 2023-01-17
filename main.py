from PreProcessing import (
    convert_dicom_nifty_sax,
    convert_dicom_nifty_flow,
    format_DICOM_data,
)
from PreProcessing import get_flow_sequences, get_sax_sequences
from PreProcessing import check_consistency_folders, anonymise_dicom_folders
from Segmentation import (
    nnunet_segmentation,
    convert_4D_to_3D_or_2D,
    prepare_nnunet_imagesTs_flow,
    flow_nnunet_inference,
)
from compute_params_QC import compute_volumes_paramters_SAX
from visulisation_plots import generate_SAX_panel, generate_final_results
from Flow import calculate_flow, generate_flow_gifs
from EF1 import compute_EF1, plot_EF1_bland_altman


DEFAULT_JSON_FILE = ("/home/br14/code/Python/AI_centre/"
                     "Flow_project_Carlota_Ciaran/AI_CMR_QC/"
                     "configs/basic_opt_EF1.json")

# # ######################### Preprocessing ###################################
# format_DICOM_data.main(DEFAULT_JSON_FILE)
# anonymise_dicom_folders.main(DEFAULT_JSON_FILE)
# get_flow_sequences.main(DEFAULT_JSON_FILE
# get_sax_sequences.main(DEFAULT_JSON_FILE)
# convert_dicom_nifty_flow.main(DEFAULT_JSON_FILE)
# convert_dicom_nifty_sax.main(DEFAULT_JSON_FILE)
# check_consistency_folders.main(DEFAULT_JSON_FILE)

# # ############################## SAX ########################################
# convert_4D_to_3D_or_2D.main(DEFAULT_JSON_FILE)
# nnunet_segmentation.main(DEFAULT_JSON_FILE)
# compute_volumes_paramters_SAX.main(DEFAULT_JSON_FILE)
# generate_SAX_panel.main(DEFAULT_JSON_FILE)
# generate_final_results.main(DEFAULT_JSON_FILE)

# # ############################## Flow #######################################
# prepare_nnunet_imagesTs_flow.main(DEFAULT_JSON_FILE)
# flow_nnunet_inference.main(DEFAULT_JSON_FILE)
# calculate_flow.main(DEFAULT_JSON_FILE)
# generate_flow_gifs.main(DEFAULT_JSON_FILE)

# # ######################################### EF1 #############################
compute_EF1.main(DEFAULT_JSON_FILE)
# plot_EF1_bland_altman.main(DEFAULT_JSON_FILE)
