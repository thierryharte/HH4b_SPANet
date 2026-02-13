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
    "/eos/home-m/mmalucch/spanet_infos/spanet_inputs/out_prediction_files/"
)

true_dir_thierry = "/eos/user/t/tharte/Analysis_data/spanet_samples/"
true_dir_matteo = (
    "/eos/home-m/mmalucch/spanet_infos/spanet_inputs/out_prediction_files/true_files/"
)

# uncomment the configurations that you want to use

run2_dataset_MC = "5_jets_pt_true_5wp_3L1cuts_allklambda_inclusive"
# run2_dataset_MC = '5_jets_pt_true_inclusive_b_region_postEE_allklambda'
run2_dataset_DATA = "5_jets_pt_true_wp_DATA_oldWP"

spanet_dict = {
    # New cuts
    # '5_jets_ptvary_btag_wp_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp.h5',
    #     'true': '5_jets_pt_true_wp_allklambda',
    #     'label': 'SPANet btag 5 WP - Flattened pt [0.3,1.7]',
    #     'color': 'orange'},
    # '5_jets_ptvary_btag_5wp_300e_3L1cuts_allklambda': {
    #     'file': f'{spanet_dir}1_13_2_spanet_loose_MC_postEE_pt_vary_btagWP_s100_newLeptonVeto_3L1Cut_UpdateJetVetoMap_MC.h5',
    #     'true': '5_jets_pt_true_5wp_3L1cuts_allklambda',
    #     'label': 'SPANet btag 5 WP - Flattened pt [0.3,1.7] - 3L1 triggers',
    #     'color': 'deepskyblue'},
    # '5_jets_ptvary_btag_5wp_300e_NoL1cuts_allklambda': {
    #     'file': f'{spanet_dir}1_13_2_spanet_loose_MC_postEE_pt_vary_btagWP_s100_newLeptonVeto_NoL1Cut_UpdateJetVetoMap_MC.h5',
    #     'true': '5_jets_pt_true_5wp_NoL1cuts_allklambda',
    #     'label': 'SPANet btag 5 WP - Flattened pt [0.3,1.7] - no L1 triggers',
    #     'color': 'forestgreen'},
    # Inclusive region allklambda
    "5_jets_ptvary_btag_wp_300e_allklambda": {
        "file": f"{spanet_dir}1_14_7_spanet_hh4b_5jets_ptvary_loose_300_btag_wp_eval_inclusive_allklambda.h5",
        "true": "5_jets_pt_true_5wp_3L1cuts_allklambda_inclusive",
        "label": "SPANet btag 5 WP - Flattened pt [0.3,1.7]",
        "color": "orange",
    },
    "5_jets_ptvary_btag_5wp_300e_3L1cuts_allklambda": {
        "file": f"{spanet_dir}1_14_7_spanet_loose_MC_postEE_pt_vary_btagWP_s100_newLeptonVeto_L3Cut_UpdateJetVetoMap_eval_inclusive_allklambda.h5",
        "true": "5_jets_pt_true_5wp_3L1cuts_allklambda_inclusive",
        "label": "SPANet btag 5 WP - Flattened pt [0.3,1.7] - 3L1 triggers",
        "color": "deepskyblue",
    },
    "5_jets_ptvary_btag_5wp_300e_NoL1cuts_allklambda": {
        "file": f"{spanet_dir}1_14_7_spanet_loose_MC_postEE_pt_vary_btagWP_s100_newLeptonVeto_NoCut_UpdateJetVetoMap_eval_inclusive_allklambda.h5",
        "true": "5_jets_pt_true_5wp_3L1cuts_allklambda_inclusive",
        "label": "SPANet btag 5 WP - Flattened pt [0.3,1.7] - no L1 triggers",
        "color": "forestgreen",
    },
    # # DEBUG TEST
    # '5_jets_ptvary_loose_inclusive_b_region_btag_300e_03_17_allklambda': {
    #     'file': f'{spanet_dir}1_14_7_hh4b_5jets_ptvary_loose_300_btag_wide_inclusive_eval_inclusive_allklambda_debug.h5',
    #     'true': '5_jets_pt_true_inclusive_b_region_postEE_allklambda',
    #     'label': 'SPANet - Inclusive b region, flattened pt btag',
    #     'color': 'red'},
    # ============================================= DATA ===========================================================
}


# true_dir = '/eos/home-r/ramellar/out_prediction_files/true_files/'
# true_dir = '/afs/cern.ch/user/m/mmalucch/public/out_prediction_files/true_files/'
# The `klambda` parameter so far only determines, if there is different klambdas or not. The type if not `none` doesn't matter.
true_dict = {
    "5_jets_pt_true_5wp_3L1cuts_allklambda": {
        "name": f"{true_dir_thierry}../spanet_samples/1_13_2_loose_MC_postEE_pt_nominal_btagWP_newLeptonVeto_3L1Cut_UpdateJetVetoMap/output_JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_5wp_3L1cuts_allklambda_inclusive": {
        "name": f"{true_dir_thierry}../spanet_samples/1_14_7_cutcomparison_3L1/h5_allklambda/JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_5wp_3L1cuts_SM_inclusive": {
        "name": f"{true_dir_thierry}../spanet_samples/1_14_7_cutcomparison_3L1/h5_files_SM/JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_5wp_3L1cuts_kl245_inclusive": {
        "name": f"{true_dir_thierry}../spanet_samples/1_14_7_cutcomparison_3L1/h5_files_kl245/JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_5wp_3L1cuts_kl5_inclusive": {
        "name": f"{true_dir_thierry}../spanet_samples/1_14_7_cutcomparison_3L1/h5_files_kl5/JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_inclusive_b_region_postEE_allklambda": {
        "name": "/eos/user/t/tharte/Analysis_data/spanet_samples/loose_MC_all2022_inclusive_region/postEE/output_JetGood_test.h5",
        "klambda": "postEE",
    },
}
