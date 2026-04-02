"""Script with configurations for each of the datasets that are to be tested for efficiency.

There are two dictionaries; one is a dictionary showing the actual datasets and the other is a list of true data, to which compare the predictions.
"""

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
    # --- ggF pairing ---
    # "5_jets_ptvary_btag_5wp_300e_3L1cuts_allklambda": {
    #     "file": f"/eos/user/t/tharte/Analysis_data/predictions/1_13_2_spanet_loose_MC_postEE_pt_vary_btagWP_s100_newLeptonVeto_3L1Cut_UpdateJetVetoMap_MC.h5",
    #     "true": "5_jets_pt_true_5wp_3L1cuts_allklambda",
    #     "label": "SPANet btag 5 WP - Flattened pt [0.3,1.7] - 3L1 triggers",
    #     "color": "firebrick",
    # },
    # --- VBF/ggF pairing ---
    # 'hh4b_pairing_vbf_ggf_pairing_classification': {
    #     'file': f'{new_spanet_dir_matteo}/out_hh4b_pairing_vbf_ggf_pairing_classification/out_seed_trainings_100/version_2/predicitons.h5',
    #     'true': '9_jets_vbf_ggf_SM',
    #     'label': 'SPANet - VBF/ggF SM - pairing+classification',
    #     'color': 'orange',
    #     'vbf': True,
    # },

    ##'hh4b_pairing_vbf_ggf_pairing_allKalmbda': {
    ##    'file': f'{new_spanet_dir_matteo}/out_hh4b_pairing_vbf_ggf_pairing_classification_allKlambda/out_seed_trainings_100/version_1/JetTotalSPANetPadded_kl_combined_test_vbf_all_Klambda_predicitons.h5',
    ##    'true': '9_jets_vbf_ggf_all_Klambda',
    ##    'label': 'SPANet - VBF/ggF - pairing',
    ##    'color': 'orange',
    ##    'vbf': True,
    ##},
    ##'hh4b_pairing_vbf_ggf_pairing_classification_allKalmbda': {
    ##    'file': f'{new_spanet_dir_matteo}/out_hh4b_pairing_vbf_ggf_pairing_classification_allKlambda/out_seed_trainings_100/version_2/JetTotalSPANetPadded_kl_combined_EVENT_AllKlambda_classification_ptvarytraining_reverse_test.h5',
    ##    'true': '9_jets_vbf_ggf_all_Klambda',
    ##    'label': 'SPANet - VBF/ggF - pairing+classification',
    ##    'color': 'blue',
    ##    'vbf': True,
    ##},

    # 'hh4b_pairing_vbf_ggf_pairing_classification_allKalmbda_SeparateHiggsVBF': {
    #     'file': f'{new_spanet_dir_matteo}/out_hh4b_pairing_vbf_ggf_pairing_classification_allKlambda_SeparateHiggsVBF/out_seed_trainings_100/version_0/hh4b_pairing_vbf_ggf_pairing_classification_AllKlambda_SeparateHiggsVBF_JetGoodProvHiggsPadded_JetGoodVBFMergedProvVBFPadded_test_eval.h5',
    #     'true': '9_jets_vbf_ggf_all_Klambda_SeparateHiggsVBF',
    #     'label': 'SPANet - VBF/ggF - pairing+classification SeparateHiggsVBF',
    #     'color': 'green',
    #     'vbf': True,
    # },
}


# The `klambda` parameter so far only determines, if there is different klambdas or not. The type if not `none` doesn't matter.
true_dict = {
    # --- ggF pairing ---
    "5_jets_pt_true_5wp_3L1cuts_allklambda": {
        "name": f"{true_dir_thierry}/1_13_2_loose_MC_postEE_pt_nominal_btagWP_newLeptonVeto_3L1Cut_UpdateJetVetoMap/output_JetGood_test.h5",
        "klambda": "postEE",
    },
    # --- VBF/ggF pairing ---
    "9_jets_vbf_ggf_SM": {
        "name": f"{new_true_dir_matteo}/vbf/vbf_SM/JetTotalSPANetPtFlattenPadded_pad9999_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },
    "9_jets_vbf_ggf_all_Klambda": {
        "name": f"{new_true_dir_matteo}/vbf/vbf_all_Klambda/JetTotalSPANetPadded_kl_combined_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },
    "7_jets_vbf_ggf_all_Klambda": {
        "name": f"{new_true_dir_matteo}/vbf/vbf_all_Klambda/7jets_JetTotalSPANetPadded_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },
    "9_jets_vbf_ggf_all_Klambda_SeparateHiggsVBF": {
        "name": f"{new_true_dir_matteo}/vbf/vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj_SeparateHiggsVBF/AllKlambda_DetaMjj_SeparateHiggsVBF_JetGoodProvHiggsPadded_JetGoodVBFMergedProvVBFPadded_test.h5",
        "klambda": "postEE",
    },
}


# =============================================================================
# (nkontaxa) MODIFICATIONS below
# =============================================================================
spanet_dir_nestor = "/eos/user/n/nkontaxa/semester_project/spanet_outputs/"
true_dir_nestor = "/eos/user/m/mmalucch/spanet_infos/spanet_inputs/"

spanet_dict.update({
    #AllKlambda 7 jets
    f"{spanet_dir_nestor}vbf/predictions_allKlambda_7jets_JetTotalSPANETPadded.h5": {
        "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_7jets_JetTotalSPANETPadded.h5",
        "true": "true_7jets_allklambda",
        "label": "SPANet 7 jets all_Kl_PtFlattened_training (nkontaxa)",
        "color": "royalblue",
        "vbf": True,
    },

    #500 epochs all_Klambda 9 jets
    f"{spanet_dir_nestor}vbf/predictions_allKlambda_500epochs.h5": {
        "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_500epochs.h5",
        "true": "true_9jets_allklambda",
        "label": "SPANet 9 jets all_Klambda 500 epochs (nkontaxa)",
        "color": "blue",
        "vbf": True,
    },
    
    #allKlambda_DetaMjj_NGW_noextravars
    f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_novars_NormGenWeights.h5": {
        "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_novars_NormGenWeights.h5",
        "true": "true_detamjj_ngw",
        "label": "SPANet DetaMjj NGW noextravars (nkontaxa)",
        "color": "black",
        "vbf": True,
    },

    #allKlambda_DetaMjj_NNGW_noextravars
        f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_novars_Non-NormGenWeights.h5": {
            "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_novars_Non-NormGenWeights.h5",
            "true": "true_detamjj_nngw",
            "label": "SPANet DetaMjj NNGW noextravars (nkontaxa)",
            "color": "green",
            "vbf": True,
        },

    #allKlambda_DetaMjj_NNGW_extravars
        f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_extravars_Non-NormGenWeights.h5": {
            "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_extravars_Non-NormGenWeights.h5",
            "true": "true_detamjj_nngw",
            "label": "SPANet DetaMjj NNGW extravars (nkontaxa)",
            "color": "yellow",
            "vbf": True,
        },

    #ptFlattenedMatchedHiggs_allKlambda_DetaMjj
        f"{spanet_dir_nestor}vbf/predictions_vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj.h5": {
            "file": f"{spanet_dir_nestor}vbf/predictions_vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj.h5",
            "true": "true_ptFlattenMatchedHiggs_detamjj",
            "label": "SPANet ptFlattenedMatchedHiggs DetaMjj (nkontaxa)",
            "color": "lightblue",
            "vbf": True,
        },

    #ptFlattenedMatchedHiggs_allKlambda_DetaMjj_AddVBFJetOrder
        f"{spanet_dir_nestor}vbf/predictions_vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj_AddVBFJetOrder.h5": {
            "file": f"{spanet_dir_nestor}vbf/predictions_vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj_AddVBFJetOrder.h5",
            "true": "true_ptFlattenMatchedHiggs_addVBFJetOrder",
            "label": "SPANet ptFlattenedMatchedHiggs DetaMjj AddVBFJetOrder (nkontaxa)",
            "color": "pink",
            "vbf": True,
        },

    #ptFlattenedMatchedHiggs_allKlambda_DetaMjj_SeparateHiggsVBF_MergedCollections_1
        f"{spanet_dir_nestor}vbf/predictions_vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj_SeparateHiggsVBF_MergedCollections_1.h5": {
            "file": f"{spanet_dir_nestor}vbf/predictions_vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj_SeparateHiggsVBF_MergedCollections_1.h5",
            "true": "true_ptFlattenMatchedHiggs_separateHiggsVBF",
            "label": "SPANet ptFlattenedMatchedHiggs DetaMjj SeparateHiggsVBF (nkontaxa)",
            "color": "brown",
            "vbf": True,
        },

    #ptFlattenedMatchedHiggs_allKlambda_DetaMjj_SeparateHiggsVBF_MergedCollections_2
        f"{spanet_dir_nestor}vbf/predictions_vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj_SeparateHiggsVBF_MergedCollections_2.h5": {
            "file": f"{spanet_dir_nestor}vbf/predictions_vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj_SeparateHiggsVBF_MergedCollections_2.h5",
            "true": "true_ptFlattenMatchedHiggs_separateHiggsVBF",
            "label": "SPANet ptFlattenedMatchedHiggs DetaMjj SeparateHiggsVBF OnlyHiggsPadding (nkontaxa)",
            "color": "lightgrey",
            "vbf": True,
        },

    #AllKlambda 9 jets
    f"{spanet_dir_nestor}vbf/predictions_allKlambda.h5": {
        "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda.h5",
        "true": "true_9jets_allklambda",
        "label": "SPANet 9 jets all_Klambda 100e (nkontaxa)",
        "color": "red",
        "vbf": True,
    },

})

true_dict.update({
    "true_7jets_allklambda": {
        "name": f"{true_dir_nestor}vbf/vbf_all_Klambda/7jets_JetTotalSPANetPadded_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },

    "true_9jets_allklambda": {
        "name": f"{true_dir_nestor}vbf/vbf_all_Klambda/JetTotalSPANetPadded_kl_combined_EVENT_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },

    "true_detamjj_ngw": {
        "name": f"{true_dir_nestor}vbf/vbf_all_Klambda_DetaMjj/SaveMjjDeta_NormGenWeights_JetTotalSPANetPadded_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },

    "true_detamjj_nngw": {
        "name": f"{true_dir_nestor}vbf/vbf_all_Klambda_DetaMjj/SaveMjjDeta_JetTotalSPANetPadded_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },

    "true_ptFlattenMatchedHiggs_detamjj": {
        "name": f"{true_dir_nestor}vbf/vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj/AllKlambda_DetaMjjJetTotalSPANetPtFlattenHiggsMatchedPadded_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },

    "true_ptFlattenMatchedHiggs_addVBFJetOrder": {
        "name": f"{true_dir_nestor}vbf/vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj_AddVBFJetPtOrder/AllKlambda_DetaMjj_AddVBFJetPtOrder_JetTotalSPANetPtFlattenPadded_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },

    "true_ptFlattenMatchedHiggs_separateHiggsVBF": {
        "name": f"{true_dir_nestor}vbf/vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj_SeparateHiggsVBF_MergedCollections/AllKlambda_DetaMjj_SeparateHiggsVBFMergeColl_JetTotalSPANetSeparateProvHiggsVBFPadded_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },
})
# =============================================================================
# END MODIFICATIONS SECTION
# =============================================================================