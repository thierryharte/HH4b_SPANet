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

true_dir_thierry = "/eos/user/t/tharte/Analysis_data/spanet_samples/"
true_dir_matteo = (
    "/eos/user/m/mmalucch/spanet_infos/spanet_inputs/out_prediction_files/true_files/"
)

# uncomment the configurations that you want to use

run2_dataset_MC = "5_jets_pt_true_5wp_3L1cuts_allklambda"
run2_dataset_DATA = "5_jets_pt_true_wp_DATA_oldWP"

spanet_dict = {
    #     '4_jets_allklambda_newkl_newCuts': {
    #         'file': f'{true_dir_matteo}output_JetGoodHiggs_test_allkl_new_kl_newcuts.h5',
    #         'true': '4_jets_allklambda_newkl_newCuts',
    #         'label': 'DHH true file',
    #         'color': 'pink',
    #         },
    # --- For baseline with btag ---
    # '5_jets_pt_btag_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_predict_s160_btag.h5',
    #     'true': '5_jets_pt_true_btag_allklambda',
    #     'label': 'SPANet baseline postEE train and eval',
    #     'color': 'darkblue'},
    # --- trained on inclusive
    # '5_jets_pt_btag_300e_allklambda_inclusive_train_eval': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_predict_s100_btag_inclusive_train_eval.h5',
    #     'true': '5_jets_pt_true_inclusive_allklambda',
    #     'label': 'SPANet baseline preEE+postEE',
    #     'color': 'royalblue'},
    # '5_jets_pt_btag_300e_allklambda_inclusive_train_eval_alt': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_predict_s100_btag_inclusive_train_eval_alt.h5',
    #     'true': '5_jets_pt_true_new_btag_bratio_inclusive_allklambda',
    #     'label': 'SPANet baseline preEE+postEE alt',
    #     'color': 'skyblue'},
    # --- [0.1, 10] pt vary ---
    # '5_jets_ptvary_loose_btag_300e_01_10_allklambda': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_01_10_loose_s100_btag.h5',  # THIS
    # '5_jets_ptvary_loose_btag_300e_01_10_allklambda_rerun': {
    #     'file': f'{spanet_dir}rerun/spanet_rerun_hh4b_5jets_300_ptvary_loose_s100_btag_01_10.h5',
    #     'true': '5_jets_pt_true_btag_allklambda',
    #     'label': 'SPANet - Flattened pt [0.1,10]',
    #     'color': 'tan'},
    # --- btag WP ---
    # '5_jets_pt_btag_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptreg_loose_s100_btag.h5',
    #     'true': '5_jets_pt_true_wp_allklambda_oldWP',
    #     'label': 'SPANet baseline',
    #     'color': 'darkblue'},
    # '5_jets_pt_btag_wp_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptreg_loose_s100_btag_wp.h5',
    #     'true': '5_jets_pt_true_wp_allklambda_oldWP',
    #     'label': 'SPANet btag 5 WP',
    #     'color': 'royalblue'},
    # '5_jets_pt_btag_3wp_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptreg_loose_s100_btag_3wp.h5',
    #     'true': '5_jets_pt_true_wp_allklambda_oldWP',
    #     'label': 'SPANet btag 3 WP',
    #     'color': 'skyblue'},
    # '5_jets_ptvary_loose_btag_300e_03_17_allklambda_rerun': {
    #     'file': f'{spanet_dir}rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17_eval_on_btagWP.h5',
    #     'true': '5_jets_pt_true_wp_allklambda_oldWP',
    #     'label': 'SPANet - Flattened pt [0.3,1.7]',
    #     'color': 'firebrick'},
    # '5_jets_ptvary_btag_wp_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp.h5',
    #     'true': '5_jets_pt_true_wp_allklambda',
    #     'label': 'SPANet btag 5 WP - Flattened pt [0.3,1.7]',
    #     'color': 'orange'},
    # '5_jets_ptvary_btag_3wp_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_3wp.h5',
    #     'true': '5_jets_pt_true_wp_allklambda',
    #     'label': 'SPANet btag 3 WP - Flattened pt [0.3,1.7]',
    #     'color': 'forestgreen'},
    # '5_jets_ptvary_btag_wp_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp_oldWP.h5',
    #     'true': '5_jets_pt_true_wp_allklambda_oldWP',
    #     'label': 'SPANet btag 5 WP - Flattened pt [0.3,1.7]',
    #     'color': 'orangered'},
    # '5_jets_ptvary_btag_3wp_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_3wp_oldWP.h5',
    #     'true': '5_jets_pt_true_wp_allklambda_oldWP',
    #     'label': 'SPANet btag 3 WP - Flattened pt [0.3,1.7]',
    #     'color': 'coral'},
    "5_jets_ptvary_btag_1wp_300e_allklambda": {
        "file": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_1wp.h5",
        "true": "5_jets_pt_true_wp_allklambda_oldWP",
        "label": "SPANet btag dummy WP - Flattened pt [0.3,1.7]",
        "color": "orange",
    },
    # '5_jets_pt_btag_300e_allklambda_eval_on_WPfile': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_predict_s160_btag_eval_on_btagWP.h5',
    #     'true': '5_jets_pt_true_wp_allklambda_oldWP',
    #     'label': 'SPANet baseline',
    #     'color': 'skyblue'},
    # '5_jets_ptvary_btag_wp_diff_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp_diff.h5',
    #     'true': '5_jets_pt_true_wp_allklambda',
    #     'label': 'SPANet btag 5 DeltaWP - Flattened pt [0.3,1.7]',
    #     'color': 'purple'},
    # '5_jets_ptvary_btag_wp_diff_inclusive_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp_diff_inclusive.h5',
    #     'true': '5_jets_pt_true_wp_allklambda',
    #     'label': 'SPANet btag 5 DeltaWP - Flattened pt [0.3,1.7] - inclusive',
    #     'color': 'mediumseagreen'},
    "5_jets_ptvary_btag_wp_inclusive_300e_allklambda": {
        "file": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp_inclusive.h5",
        "true": "5_jets_pt_true_wp_allklambda",
        "label": "SPANet btag 5 WP - Flattened pt [0.3,1.7] - inclusive",
        "color": "deepskyblue",
    },
    # New cuts
    # '5_jets_ptvary_btag_wp_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp.h5',
    #     'true': '5_jets_pt_true_wp_allklambda',
    #     'label': 'SPANet btag 5 WP - Flattened pt [0.3,1.7]',
    #     'color': 'orange'},
    "5_jets_ptvary_btag_5wp_300e_3L1cuts_allklambda": {
        "file": f"{spanet_dir}1_13_2_spanet_loose_MC_postEE_pt_vary_btagWP_s100_newLeptonVeto_3L1Cut_UpdateJetVetoMap_MC.h5",
        "true": "5_jets_pt_true_5wp_3L1cuts_allklambda",
        "label": "SPANet btag 5 WP - Flattened pt [0.3,1.7] - 3L1 triggers",
        "color": "firebrick",
    },
    # '5_jets_ptvary_btag_5wp_300e_NoL1cuts_allklambda': {
    #     'file': f'{spanet_dir}1_13_2_spanet_loose_MC_postEE_pt_vary_btagWP_s100_newLeptonVeto_NoL1Cut_UpdateJetVetoMap_MC.h5',
    #     'true': '5_jets_pt_true_5wp_NoL1cuts_allklambda',
    #     'label': 'SPANet btag 5 WP - Flattened pt [0.3,1.7] - no L1 triggers',
    #     'color': 'forestgreen'},
    # --- [0.3,1.7] ---
    # '5_jets_ptvary_loose_btag_300e_wide_allklambda': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_wide_loose_s100_btag.h5',  # THIS ## Chosen model
    # '5_jets_ptvary_loose_btag_300e_wide_allklambda_SM': f'{spanet_dir}SM_train/spanet_hh4b_5jets_300_ptvary_wide_loose_s100_btag.h5',  # THIS
    # '5_jets_ptvary_loose_btag_300e_wide_allklambda_onlylog': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_wide_onlylog_loose_s100_btag.h5',  # THIS
    # '5_jets_ptvary_loose_btag_300e_03_17_allklambda_rerun': {
    #     'file': f'{spanet_dir}rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17.h5',
    #     'true': '5_jets_pt_true_btag_allklambda',
    #     'label': 'SPANet - Flattened pt [0.3,1.7]',
    #     'color': 'darkgoldenrod'},
    # --- [0.5,1.5] ---
    # '5_jets_ptvary_loose_btag_300e_allklambda': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s50_btag.h5',  # THIS
    #     'true': '5_jets_pt_true_btag_allklambda',
    #     'label': 'SPANet - Flattened pt [0.5,1.5]',
    #     'color': 'gold'},
    #  no pT/phi/btag
    # '5_jets_ptnone_btag_300e_allklambda': f'{spanet_dir}spanet_hh4b_5jets_300_ptnone_loose_s100_btag.h5',
    # '5_jets_pt_300e_allklambda_no_btag': f'{spanet_dir}spanet_hh4b_5jets_50_ptreg_loose_s100_no_btag.h5',  # THIS
    # '5_jets_ptreg_loose_btag_300e_allklambda_nophi': f'{spanet_dir}spanet_hh4b_5jets_300_ptreg_loose_s100_btag_nophi.h5',
    # '5_jets_ptvary_loose_btag_300e_01_10_allklambda_rerun_nophi': f'{spanet_dir}rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_01_10_nophi.h5',
    # --- Tight selection ---
    #  '5_jets_ptvary_tight_btag_300e_allklambda': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_tight_s100_btag.h5',  # THIS
    # --- Matteo comparison (b-tag ordering) ---
    #  originating from: '/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5',  # HERE
    #  '5_jets_ATLAS_ptreg_allklambda_train': f'{spanet_dir_matteo}out_spanet_prediction_5jets_lr1e4_noevkl_300e.h5',
    #  '5_jets_ATLAS_ptreg_allklambda_eval': f'{spanet_dir_matteo}out_spanet_prediction_SMtraining_lr1e4_evkl.h5',
    # ' 5_jets_data_ATLAS_ptreg_5train_allklambda_noklinput_oldCuts_newCutsEval': f'{spanet_dir_matteo}spanet_prediction_oc_kl3p5_noklinp_data_nc.h5',
    # '5_jets_ATLAS_ptreg_allklambda_train_noklinput_newkl_oldCuts_newCutsEval': f'{spanet_dir_matteo}spanet_prediction_oc_kl3p5_noklinp_nc.h5',
    # --- ERA preEE ---
    # '5_jets_preEE_300e_allklambda_preEE_eval': {
    #     'file': f'{spanet_dir}/spanet_hh4b_preEE_5jets_100_pvary_loose_s300_eval_on_preEE.h5',
    #     'true': '5_jets_pt_allklambda_preEE_eval',
    #     'label': 'pt Flattened preEE trained preEE eval',
    #     'color': 'skyblue'},
    # '5_jets_postEE_300e_allklambda_preEE_eval': {
    #     'file': f'{spanet_dir}/spanet_hh4b_postEE_5jets_100_pvary_loose_s300_eval_on_preEE.h5',
    #     'true': '5_jets_pt_allklambda_preEE_eval',
    #     'label': 'pT Flatened postEE trained preEE eval',
    #     'color': 'royalblue'},
    # '5_jets_inclusive_300e_allklambda_preEE_eval': {
    #     'file': f'{spanet_dir}/spanet_hh4b_inclusive_5jets_100_pvary_loose_s300_eval_on_preEE.h5',
    #     'true': '5_jets_pt_allklambda_preEE_eval',
    #     'label': 'pT Flatened inclusive trained preEE eval',
    #     'color': 'darkblue'},
    # '5_jets_inclusive_rebalance_300e_allklambda_preEE_eval': {
    #     'file': f'{spanet_dir}/spanet_hh4b_inclusive_rebalance_5jets_100_pvary_loose_s300_eval_on_preEE.h5',
    #     'true': '5_jets_pt_allklambda_preEE_eval_weighted',
    #     'label': 'pT Flatened inclusive year balanced trained preEE eval',
    #     'color': 'purple'},
    # --- ERA postEE ---
    # '5_jets_preEE_300e_allklambda_postEE_eval': {
    #     'file': f'{spanet_dir}/spanet_hh4b_preEE_5jets_100_pvary_loose_s300_eval_on_postEE.h5',
    #     'true': '5_jets_pt_allklambda_postEE_eval',
    #     'label': 'pt Flattened preEE trained postEE eval',
    #     'color': 'gold'},
    # '5_jets_ptvary_loose_btag_300e_03_17_allklambda_rerun': {
    #     'file': f'{spanet_dir}rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17.h5',
    #     'true': '5_jets_pt_true_btag_allklambda',
    #     'label': 'pT Flattened postEE trained postEE eval',
    #     'color': 'orange'},
    # '5_jets_inclusive_300e_allklambda_postEE_eval': {
    #     'file': f'{spanet_dir}/spanet_hh4b_inclusive_5jets_100_pvary_loose_s300_eval_on_postEE.h5',
    #     'true': '5_jets_pt_allklambda_postEE_eval',
    #     'label': 'pT Flatened inclusive trained postEE eval',
    #     'color': 'darkorange'},
    # '5_jets_inclusive_rebalance_300e_allklambda_postEE_eval': {
    #     'file': f'{spanet_dir}/spanet_hh4b_inclusive_rebalance_5jets_100_pvary_loose_s300_eval_on_postEE.h5',
    #     'true': '5_jets_pt_allklambda_postEE_eval_weighted',
    #     'label': 'pT Flatened inclusive year balanced trained postEE eval',
    #     'color': 'deeppink'},
    # --- ERA pre+postEE ---
    # '5_jets_inclusive_300e_allklambda_inclusive_eval': {
    #     'file': f'{spanet_dir}/spanet_hh4b_inclusive_5jets_100_pvary_loose_s300_eval_on_inclusive.h5',
    #     'true': '5_jets_pt_true_inclusive_allklambda',
    #     'label': 'pt Flattened preEE+postEE',
    #     'color': 'goldenrod'},
    # '5_jets_inclusive_300e_allklambda_inclusive_eval_alt': {
    #     'file': f'{spanet_dir}/spanet_hh4b_inclusive_5jets_100_pvary_loose_s300_eval_on_inclusive_alt.h5',
    #     'true': '5_jets_pt_true_new_btag_bratio_inclusive_allklambda',
    #     'label': 'pt Flattened preEE+postEE alt',
    #     'color': 'gold'},
    # --- Different BTag ---
    # '5_jets_ptvary_loose_btag_300e_03_17_allklambda_btag12': {
    #     'file': f'{spanet_dir}spanet_hh4b_inclusive_5jets_100_pvary_loose_s300_btag12.h5',
    #     'true': '5_jets_pt_true_new_btag_bratio_inclusive_allklambda',
    #     'label': 'SPANet - Flattened pt btag 1,2',
    #     'color': 'cyan'},
    # '5_jets_ptvary_loose_btag_300e_03_17_allklambda_btag12_bratio': {
    #     'file': f'{spanet_dir}spanet_hh4b_inclusive_5jets_100_pvary_loose_s300_btag12_bratio.h5',
    #     'true': '5_jets_pt_true_new_btag_bratio_inclusive_allklambda',
    #     'label': 'SPANet - Flattened pt btag 1,2 bratio 3,4',
    #     'color': 'teal'},
    # '5_jets_ptvary_loose_btag_300e_03_17_allklambda_bratio_all': {
    #     'file': f'{spanet_dir}spanet_hh4b_inclusive_5jets_100_pvary_loose_s300_bratio_all.h5',
    #     'true': '5_jets_pt_true_new_btag_bratio_inclusive_allklambda',
    #     'label': 'SPANet - Flattened pt bratio 1,2,3,4,5',
    #     'color': 'mediumaquamarine'},
    # '5_jets_ptvary_loose_btag_300e_03_17_allklambda_bratio_all_seed101': {
    #     'file': f'{spanet_dir}spanet_hh4b_inclusive_5jets_100_pvary_loose_s300_bratio_all_seed101.h5',
    #     'true': '5_jets_pt_true_new_btag_bratio_inclusive_allklambda',
    #     'label': 'SPANet - Flattened pt bratio 1,2,3,4,5 seed 101',
    #     'color': 'aquamarine'},
    # --- Inclusive b-tag region (Only cut on btag1 and 2): ---
    # '5_jets_ptvary_loose_inclusive_b_region_btag_300e_03_17_allklambda_bratio_all': {
    #     'file': f'{spanet_dir}spanet_hh4b_inclusive_b_region_5jets_loose_ptvary_wide_bratio_300_inclusive_years_s100.h5',
    #     'true': '5_jets_pt_true_inclusive_b_region_bratio_inclusive_allklambda',
    #     'label': 'SPANet - Inclusive b region, flattened pt bratio 1,2,3,4,5',
    #     'color': 'red'},
    # ============================================= DATA ===========================================================
    #  baseline with b-tag
    # '5_jets_pt_data_btag_300e': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5', # ---DATA
    #  For baseline no btag:
    # '5_jets_pt_data_300e_allklambda_no_btag': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_50_ptreg_loose_s100_no_btag.h5',  # THIS
    # --- For baseline with btag ---
    # '5_jets_pt_data_btag_300e': {
    #     'file': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5',
    #     'true': '5_jets_pt_data',
    #     'label': 'SPANet baseline EraE',
    #     'color': 'darkblue'},
    # --- btag WP ---
    #  '5_jets_pt_data_btag_300e_allklambda_eval_on_WPfile': {
    #      'file': f'{spanet_dir}spanet_hh4b_5jets_300_predict_s160_btag_eval_on_btagWP_DATA_postEE.h5',
    #     'true': '5_jets_pt_true_wp_DATA_oldWP',
    #      'label': 'SPANet baseline',
    #      'color': 'darkblue'},
    # '5_jets_pt_data_btag_wp_300e': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptreg_loose_s100_btag_wp_DATA_postEE.h5',
    #     'true': '5_jets_pt_true_wp_DATA_oldWP',
    #     'label': 'SPANet btag 5 WP full postEE',
    #     'color': 'royalblue'},
    # '5_jets_pt_data_btag_3wp_300e': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptreg_loose_s100_btag_3wp_DATA_postEE.h5',
    #     'true': '5_jets_pt_true_wp_DATA_oldWP',
    #     'label': 'SPANet btag 3 WP full postEE',
    #     'color': 'skyblue'},
    # '5_jets_pt_data_vary_loose_btag_300e_03_17_allklambda_rerun': {
    #      'file': f'{spanet_dir}rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17_eval_on_btagWP_DATA_postEE.h5',
    #     'true': '5_jets_pt_true_wp_DATA_oldWP',
    #      'label': 'SPANet - Flattened pt [0.3,1.7]',
    #      'color': 'firebrick'},
    # '5_jets_pt_data_vary_btag_wp_300e': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp_DATA_postEE.h5',
    #     'true': '5_jets_pt_true_wp_DATA',
    #     'label': 'SPANet btag 5 WP - Flattened pt [0.3,1.7]',
    #     'color': 'orange'},
    # '5_jets_pt_data_vary_btag_3wp_300e': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_3wp_DATA_postEE.h5',
    #     'true': '5_jets_pt_true_wp_DATA',
    #     'label': 'SPANet btag 3 WP - Flattened pt [0.3,1.7]',
    #     'color': 'forestgreen'},
    # '5_jets_pt_data_vary_btag_wp_300e': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp_DATA_postEE_oldWP.h5',
    #     'true': '5_jets_pt_true_wp_DATA_oldWP',
    #     'label': 'SPANet btag 5 WP - Flattened pt [0.3,1.7]',
    #     'color': 'orangered'},
    # '5_jets_pt_data_vary_btag_3wp_300e': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_3wp_DATA_postEE_oldWP.h5',
    #     'true': '5_jets_pt_true_wp_DATA_oldWP',
    #     'label': 'SPANet btag 3 WP - Flattened pt [0.3,1.7]',
    #     'color': 'coral'},
    # '5_jets_pt_data_vary_btag_1wp_300e': {
    #     'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_1wp_DATA_postEE.h5',
    #     'true': '5_jets_pt_true_wp_DATA_oldWP',
    #     'label': 'SPANet btag 1 WP - Flattened pt [0.3,1.7]',
    #     'color': 'skyblue'},
    #  '5_jets_pt_data_vary_btag_wp_diff_300e_allklambda': {
    #      'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp_diff_DATA_postEE.h5',
    #      'true': '5_jets_pt_true_wp_DATA',
    #      'label': 'SPANet btag 5 DeltaWP - Flattened pt [0.3,1.7]',
    #      'color': 'purple'},
    #  '5_jets_pt_data_vary_btag_wp_diff_inclusive_300e_allklambda': {
    #      'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp_diff_inclusive_DATA_postEE.h5',
    #      'true': '5_jets_pt_true_wp_DATA',
    #      'label': 'SPANet btag 5 DeltaWP - Flattened pt [0.3,1.7] - inclusive',
    #      'color': 'mediumseagreen'},
    #  '5_jets_pt_data_vary_btag_wp_inclusive_300e_allklambda': {
    #      'file': f'{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s100_btag_wp_inclusive_DATA_postEE.h5',
    #      'true': '5_jets_pt_true_wp_DATA',
    #      'label': 'SPANet btag 5 WP - Flattened pt [0.3,1.7] - inclusive',
    #      'color': 'deepskyblue'},
    # --- [0.1,10] ---
    # '5_jets_pt_data_vary_loose_btag_300e_01_10': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptvary_01_10_loose_s100_btag.h5',
    # '5_jets_pt_data_vary_loose_btag_300e_01_10_rerun': f'{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_01_10.h5',
    # --- [0.3,1.7] ---
    # '5_jets_pt_data_vary_loose_btag_300e_wide': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptvary_wide_loose_s100_btag.h5',
    # # normalizations:
    # '5_jets_pt_data_vary_loose_btag_300e_wide_onlylog': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptvary_wide_onlylog_loose_s100_btag.h5',
    # '5_jets_pt_data_vary_loose_btag_300e_03_17_rerun': f'{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17.h5',
    # '5_jets_pt_data_vary_loose_btag_300e_03_17_rerun': {
    #     'file': f'{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17.h5',
    #     'true': '5_jets_pt_data',
    #     'label': 'SPANet EraE - Flattened pt [0.3,1.7]',
    #     'color': 'darkgoldenrod'},
    # --- [0.5,1.5] ---
    # '5_jets_pt_data_vary_loose_btag_300e': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptvary_loose_s50_btag.h5',
    # --- no pT ---
    # '5_jets_pt_data_none_btag_300e': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptnone_loose_s100_btag.h5', # ---DATA
    # --- no phi ---
    # '5_jets_pt_data_vary_loose_btag_300e_01_10_rerun_nophi': f'{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_01_10_nophi.h5',
    # '5_jets_pt_data_loose_btag_300e_nophi': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptreg_loose_s100_btag_nophi.h5',
    # --- baseline ---
    # '5_jets_pt_data_btag_300e': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5', # ---DATA
    # --- pt_vary ---
    # '5_jets_pt_data_btag_300e': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5', # ---DATA
    # '5_jets_pt_data_vary_loose_btag_300e_01_10_rerun': f'{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_01_10.h5',
    # '5_jets_pt_data_vary_loose_btag_300e_03_17_rerun': f'{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17.h5',
    # '5_jets_pt_data_vary_loose_btag_300e': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptvary_loose_s50_btag.h5',
    # --- No pt/phi/btag ---
    # '5_jets_pt_data_btag_300e': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5', # ---DATA
    # '5_jets_pt_data_300e_allklambda_no_btag': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_50_ptreg_loose_s100_no_btag.h5',  # THIS
    # '5_jets_pt_data_none_btag_300e': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptnone_loose_s100_btag.h5', # ---DATA
    # #'5_jets_pt_data_vary_loose_btag_300e_01_10_rerun_nophi': f'{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_01_10_nophi.h5',
    # '5_jets_pt_data_loose_btag_300e_nophi': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptreg_loose_s100_btag_nophi.h5',
    # --- onlylog ---
    # '5_jets_pt_data_btag_300e': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5', # ---DATA
    # '5_jets_pt_data_vary_loose_btag_300e_03_17_rerun': f'{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17.h5',
    # '5_jets_pt_data_vary_loose_btag_300e_wide_onlylog': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptvary_wide_onlylog_loose_s100_btag.h5',
    # --- final_comp ---#
    # '5_jets_pt_data_btag_300e': f'{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5', # ---DATA
    # '5_jets_pt_data_vary_loose_btag_300e_03_17_rerun': f'{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17.h5',
    # '5_jets_pt_true_vary_loose_btag_allklambda': f'{spanet_dir}../samplesets/loose_selection_random_pt_mass/output_JetGood_train.h5',  # THIS
    # '5_jets_pt_true_vary_loose_btag_wide_allklambda': f'{spanet_dir}../samplesets/loose_selection_random_pt_mass_wide/output_JetGood_train.h5',  # THIS
    # '5_jets_pt_true_vary_loose_btag_01_10_allklambda': f'{spanet_dir}../samplesets/loose_selection_random_pt_mass_01_10/output_JetGood_train.h5',  # THIS
    # '5_jets_pt_true_btag_allklambda': f'{spanet_dir}../samplesets/jet5global_pt/output_JetGood_train.h5',  # THIS
}


# true_dir = '/eos/home-r/ramellar/out_prediction_files/true_files/'
# true_dir = '/afs/cern.ch/user/m/mmalucch/public/out_prediction_files/true_files/'
# The `klambda` parameter so far only determines, if there is different klambdas or not. The type if not `none` doesn't matter.
true_dict = {
    "4 jets": {
        "name": f"{true_dir_matteo}output_JetGoodHiggs_test.h5",
        "klambda": "none",
    },
    "5 jets": {"name": f"{true_dir_matteo}output_JetGood_test.h5", "klambda": "none"},
    "5_jets_btag_presel": {
        "name": f"{true_dir_matteo}output_JetGood_btag_presel_test.h5",
        "klambda": "none",
    },
    "4_jets_klambda0": {
        "name": f"{true_dir_matteo}kl0_output_JetGoodHiggs_test.h5",
        "klambda": "none",
    },
    "4_jets_klambda2p45": {
        "name": f"{true_dir_matteo}kl2p45_output_JetGoodHiggs_test.h5",
        "klambda": "none",
    },
    "4_jets_klambda5": {
        "name": f"{true_dir_matteo}kl5_output_JetGoodHiggs_test.h5",
        "klambda": "none",
    },
    "5_jets_klambda0": {
        "name": f"{true_dir_matteo}kl0_output_JetGood_test.h5",
        "klambda": "none",
    },
    "5_jets_klambda2p45": {
        "name": f"{true_dir_matteo}kl2p45_output_JetGood_test.h5",
        "klambda": "none",
    },
    "5_jets_klambda5": {
        "name": f"{true_dir_matteo}kl5_output_JetGood_test.h5",
        "klambda": "none",
    },
    "4_jets_data": {
        "name": f"{spanet_dir_matteo}out_spanet_prediction_data_ev4jets_training5jet_ptreg_ATLAS.h5",
        "klambda": "none",
    },
    "5_jets_data": {
        "name": f"{spanet_dir_matteo}out_spanet_prediction_data_ev5jets_training5jet_ptreg_ATLAS.h5",
        "klambda": "none",
    },
    "5_jets_data_oldCuts": {
        "name": f"{spanet_dir_matteo}spanet_prediction_sm_on_data_oc.h5",
        "klambda": "none",
    },
    "5_jets_data_newCuts": {
        "name": f"{spanet_dir_matteo}spanet_prediction_nc_noklinp_on_data.h5",
        "klambda": "none",
    },
    "4_jets_allklambda": {
        "name": f"{true_dir_matteo}output_JetGood_test_allkl_new_kl_newcuts.h5",
        "klambda": "postEE",
    },  # output_JetGoodHiggs_allkl_test
    "5_jets_allklambda": {
        "name": f"{true_dir_matteo}output_JetGood_test_allkl_new_kl_newcuts.h5",
        "klambda": "postEE",
    },  # output_JetGood_allkl_test
    "5_jets_allklambda_newkl_oldCuts": {
        "name": f"{true_dir_matteo}output_JetGood_test_allkl_new_kl_oldcuts.h5",
        "klambda": "postEE",
    },  # '/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5'},
    "5_jets_allklambda_newkl_newCuts": {
        "name": f"{true_dir_matteo}output_JetGood_test_allkl_new_kl_newcuts.h5",
        "klambda": "postEE",
    },  # '/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5'},
    "4_jets_allklambda_newkl_newCuts": {
        "name": f"{true_dir_matteo}output_JetGoodHiggs_test_allkl_new_kl_newcuts.h5",
        "klambda": "postEE",
    },  # '/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5'},
    "5_jets_pt_allklambda": {
        "name": f"{true_dir_thierry}/loose_selection/output_JetGood_test.h5",
        "klambda": "postEE",
    },
    "4_jets_pt_allklambda": {
        "name": f"{true_dir_thierry}/loose_selection/output_JetGoodHiggs_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_data": {
        "name": f"{true_dir_thierry}/DATA_loose_cut/output_JetGood_test.h5",
        "klambda": "none",
    },
    "5_jets_pt_true_btag_allklambda": {
        "name": f"{spanet_dir}../spanet_samples/loose_selection/output_JetGood_test.h5",
        "klambda": "postEE",
    },  # THIS
    "5_jets_pt_true_vary_loose_btag_allklambda": {
        "name": f"{spanet_dir}../spanet_samples/loose_selection_random_pt_mass/output_JetGood_train.h5",
        "klambda": "postEE",
    },  # THIS
    "5_jets_pt_true_vary_loose_btag_wide_allklambda": {
        "name": f"{spanet_dir}../spanet_samples/loose_selection_random_pt_mass_wide/output_JetGood_train.h5",
        "klambda": "postEE",
    },  # THIS
    "5_jets_pt_true_vary_loose_btag_01_10_allklambda": {
        "name": f"{spanet_dir}../spanet_samples/loose_selection_random_pt_mass_01_10/output_JetGood_train.h5",
        "klambda": "postEE",
    },  # THIS
    "5_jets_pt_allklambda_preEE_eval": {
        "name": f"{true_dir_thierry}../spanet_samples/loose_all2022/preEE/output_JetGood_test.h5",
        "klambda": "preEE",
    },
    "5_jets_pt_allklambda_postEE_eval": {
        "name": f"{true_dir_thierry}../spanet_samples/loose_all2022/postEE/output_JetGood_test.h5",
        "klambda": "preEE",
    },
    "5_jets_pt_allklambda_preEE_eval_weighted": {
        "name": f"{true_dir_thierry}../spanet_samples/loose_all2022/preEE/with_weights/output_JetGood_test.h5",
        "klambda": "preEE",
    },
    "5_jets_pt_allklambda_postEE_eval_weighted": {
        "name": f"{true_dir_thierry}../spanet_samples/loose_all2022/postEE/with_weights/output_JetGood_test.h5",
        "klambda": "preEE",
    },
    "5_jets_pt_true_inclusive_allklambda": {
        "name": f"{true_dir_thierry}../spanet_samples/loose_all2022/inclusive/output_JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_new_btag_bratio_inclusive_allklambda": {
        "name": f"{true_dir_thierry}../spanet_samples/loose_all2022/inclusive/new_btag_values/output_JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_inclusive_b_region_bratio_inclusive_allklambda": {
        "name": f"{true_dir_thierry}../spanet_samples/loose_MC_all2022_inclusive_region/inclusive/output_JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_wp_allklambda_oldWP": {
        "name": f"{true_dir_thierry}../spanet_samples/loose_MC_postEE_btagWP/old_WP/output_JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_wp_allklambda": {
        "name": f"{true_dir_thierry}../spanet_samples/loose_MC_postEE_btagWP/output_JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_wp_DATA_oldWP": {
        "name": f"{true_dir_thierry}../spanet_samples/loose_DATA_postEE_btagWP/old_WP/output_JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_wp_DATA": {
        "name": f"{true_dir_thierry}../spanet_samples/loose_DATA_postEE_btagWP/output_JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_5wp_3L1cuts_allklambda": {
        "name": f"{true_dir_thierry}../spanet_samples/1_13_2_loose_MC_postEE_pt_nominal_btagWP_newLeptonVeto_3L1Cut_UpdateJetVetoMap/output_JetGood_test.h5",
        "klambda": "postEE",
    },
    "5_jets_pt_true_5wp_NoL1cuts_allklambda": {
        "name": f"{true_dir_thierry}../spanet_samples/1_13_2_loose_MC_postEE_pt_nominal_btagWP_newLeptonVeto_NoL1Cut_UpdateJetVetoMap/output_JetGood_test.h5",
        "klambda": "postEE",
    },
}
