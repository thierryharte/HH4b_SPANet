"""Script with configurations for each of the datasets that are to be tested for efficiency.

There are two dictionaries; one is a dictionary showing the actual datasets and the other is a list of true data, to which compare the predictions.
"""

# uncomment the configurations that you want to use
new_spanet_dir_matteo = (
    "/eos/user/m/mmalucch/spanet_infos/spanet_outputs/out_spanet_outputs/"
)
new_true_dir_matteo = (
    "/eos/user/m/mmalucch/spanet_infos/spanet_inputs/"
)

spanet_dict = {
    # --- VBF/ggF pairing ---
    'hh4b_pairing_vbf_ggf_pairing_classification': {
        'file': f'{new_spanet_dir_matteo}/out_hh4b_pairing_vbf_ggf_pairing_classification/out_seed_trainings_100/version_2/predicitons.h5',
        'true': '9_jets_vbf_ggf_SM',
        'label': 'SPANet - VBF/ggF SM - pairing+classification',
        'color': 'orange',
        'vbf': True,
    },
    'hh4b_pairing_vbf_ggf_pairing_classification_allKalmbda': {
        'file': f'{new_spanet_dir_matteo}/out_hh4b_pairing_vbf_ggf_pairing_classification_allKlambda/out_seed_trainings_100/version_2/JetTotalSPANetPadded_kl_combined_EVENT_AllKlambda_classification_ptvarytraining_reverse_test.h5',
        'true': '9_jets_vbf_ggf_all_Klambda',
        'label': 'SPANet - VBF/ggF - pairing+classification',
        'color': 'blue',
        'vbf': True,
    },
}

true_dict = {
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
}


# =============================================================================
# (nkontaxa) MODIFICATIONS TO spanet_dict below
# =============================================================================
spanet_dir_nestor = "/eos/user/n/nkontaxa/semester_project/spanet_outputs/"
true_dir_nestor = "/eos/user/m/mmalucch/spanet_infos/spanet_inputs/"

spanet_dict.update({
    ###AllKlambda 7 jets
    ##f"{spanet_dir_nestor}vbf/predictions_allKlambda_7jets_JetTotalSPANETPadded.h5": {
    ##    "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_7jets_JetTotalSPANETPadded.h5",
    ##    "true": "vbf_jet_total_padded",
    ##    "label": "SPANet 7 jets all_Kl_PtFlattened_training (nkontaxa)",
    ##    "color": "royalblue",
    ##    "vbf": True,
    ##},
    
    #AllKlambda 9 jets
    f"{spanet_dir_nestor}vbf/predictions_allKlambda.h5": {
        "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda.h5",
        "true": "vbf_jet_total_padded",
        "label": "SPANet 9 jets all_Klambda (nkontaxa)",
        "color": "red",
        "vbf": True,
    },

    #500 epochs all_Klambda 9 jets
    f"{spanet_dir_nestor}vbf/predictions_allKlambda_500epochs.h5": {
        "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_500epochs.h5",
        "true": "vbf_jet_total_padded",
        "label": "SPANet 9 jets all_Klambda 500 epochs (nkontaxa)",
        "color": "black",
        "vbf": True,
    },
    
    ###allKlambda_DetaMjj_NGW_noextravars
    ##f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_novars_NormGenWeights.h5": {
    ##    "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_novars_NormGenWeights.h5",
    ##    "true": "vbf_jet_total_padded",
    ##    "label": "SPANet DetaMjj NGW noextravars (nkontaxa)",
    ##    "color": "black",
    ##    "vbf": True,
    ##},

    ###allKlambda_DetaMjj_NNGW_noextravars
    ##    f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_novars_Non-NormGenWeights.h5": {
    ##        "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_novars_Non-NormGenWeights.h5",
    ##        "true": "vbf_jet_total_padded",
    ##        "label": "SPANet DetaMjj NNGW noextravars (nkontaxa)",
    ##        "color": "green",
    ##        "vbf": True,
    ##    },

})

true_dict.update({
    #AllKlambda 7 jets
    "vbf_jet_total_padded": {
        "name": f"{true_dir_nestor}vbf/vbf_all_Klambda/7jets_JetTotalSPANetPadded_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },
    
    #AllKlambda 9 jets
    "vbf_jet_total_padded": {
        "name": f"{true_dir_nestor}vbf/vbf_all_Klambda/JetTotalSPANetPadded_kl_combined_EVENT_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },

    #AllKlambda DetaMjj NNGW noextravars jets
    "vbf_jet_total_padded": {
        "name": f"{true_dir_nestor}vbf/vbf_all_Klambda_DetaMjj/SaveMjjDeta_JetTotalSPANetPadded_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },

    #AllKlambda DetaMjj NGW noextravars jets
    "vbf_jet_total_padded": {
        "name": f"{true_dir_nestor}vbf/vbf_all_Klambda_DetaMjj/SaveMjjDeta_NormGenWeights_JetTotalSPANetPadded_test.h5",
        "klambda": "postEE",
        "vbf": True,
    },
    # Add more entries here...
})

# =============================================================================
# END MODIFICATIONS SECTION
# =============================================================================

