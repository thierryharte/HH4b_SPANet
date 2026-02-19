"""Script with configurations for each of the datasets that are to be tested for efficiency.

There are two dictionaries; one is a dictionary showing the actual datasets and the other is a list of true data, to which compare the predictions.
"""

import logging

# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(funcName)s | %(message)s",
#     datefmt="%d-%b-%y %H-%M-%S",
# )
logger = logging.getLogger(__name__)

spanet_dir = "/eos/user/t/tharte/Analysis_data/predictions/"
spanet_dir_matteo = (
    "/eos/user/m/mmalucch/spanet_infos/spanet_inputs/out_prediction_files/"
)
new_spanet_dir_matteo = (
    "/eos/user/m/mmalucch/spanet_infos/spanet_outputs/out_spanet_outputs/"
)



true_dir_thierry = "/eos/user/t/tharte/Analysis_data/spanet_samples/"
true_dir_matteo = (
    "/eos/user/m/mmalucch/spanet_infos/spanet_inputs/out_prediction_files/true_files/"
)
new_true_dir_matteo = (
    "/eos/user/m/mmalucch/spanet_infos/spanet_inputs/"
)

# uncomment the configurations that you want to use

run2_dataset_MC = "9_jets_vbf_ggf_all_Klambda"
run2_dataset_DATA = ""

spanet_dict = {
    # --- VBF pairing ---
    # 'hh4b_pairing_vbf_ggf_pairing_classification': {
    #     'file': f'{new_spanet_dir_matteo}/out_hh4b_pairing_vbf_ggf_pairing_classification/out_seed_trainings_100/version_2/predicitons.h5',
    #     'true': '9_jets_vbf_ggf_SM',
    #     'label': 'SPANet - VBF/ggF',
    #     'color': 'orange',
    #     'vbf': True,
    # },
    'hh4b_pairing_vbf_ggf_pairing_classification_allKalmbda': {
        'file': f'{new_spanet_dir_matteo}/out_hh4b_pairing_vbf_ggf_pairing_classification_allKlambda/out_seed_trainings_100/version_1/JetTotalSPANetPtFlattenPadded_kl_combined_test_vbf_all_Klambda_predicitons.h5',
        'true': '9_jets_vbf_ggf_all_Klambda',
        'label': 'SPANet - VBF/ggF',
        'color': 'orange',
        'vbf': True,
    },
    # ============================================= DATA ===========================================================
}


# The `klambda` parameter so far only determines, if there is different klambdas or not. The type if not `none` doesn't matter.
true_dict = {
    "9_jets_vbf_ggf_SM": {
        "name": f"{new_true_dir_matteo}/vbf/vbf_SM/JetTotalSPANetPtFlattenPadded_pad9999_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },
    "9_jets_vbf_ggf_all_Klambda": {
        "name": f"{new_true_dir_matteo}/vbf/vbf_all_Klambda/JetTotalSPANetPtFlattenPadded_kl_combined_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },
}
