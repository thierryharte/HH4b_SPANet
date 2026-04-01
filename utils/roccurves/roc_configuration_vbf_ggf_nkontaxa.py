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

spanet_dict = {}

true_dict = {}


# =============================================================================
# (nkontaxa) MODIFICATIONS below
# =============================================================================
spanet_dir_nestor = "/eos/user/n/nkontaxa/semester_project/spanet_outputs/"
true_dir_nestor = "/eos/user/m/mmalucch/spanet_infos/spanet_inputs/"

spanet_dict.update({
    #AllKlambda 7 jets
    ##f"{spanet_dir_nestor}vbf/predictions_allKlambda_7jets_JetTotalSPANETPadded.h5": {
    ##    "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_7jets_JetTotalSPANETPadded.h5",
    ##    "true": "true_7jets_allklambda",
    ##    "label": "SPANet 7 jets all_Kl_PtFlattened_training (nkontaxa)",
    ##    "color": "royalblue",
    ##    "vbf": True,
    ##},

    #500 epochs all_Klambda 9 jets
    ##f"{spanet_dir_nestor}vbf/predictions_allKlambda_500epochs.h5": {
    ##    "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_500epochs.h5",
    ##    "true": "true_9jets_allklambda",
    ##   "label": "SPANet 9 jets all_Klambda 500 epochs (nkontaxa)",
    ##    "color": "blue",
    ##    "vbf": True,
    ##},
    
    #allKlambda_DetaMjj_NGW_noextravars
    ##f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_novars_NormGenWeights.h5": {
    ##    "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_novars_NormGenWeights.h5",
    ##    "true": "true_detamjj_ngw",
    ##    "label": "SPANet DetaMjj NGW noextravars (nkontaxa)",
    ##    "color": "black",
    ##    "vbf": True,
    ##},

    #allKlambda_DetaMjj_NNGW_noextravars
    ##    f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_novars_Non-NormGenWeights.h5": {
    ##        "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_novars_Non-NormGenWeights.h5",
    ##        "true": "true_detamjj_nngw",
    ##        "label": "SPANet DetaMjj NNGW noextravars (nkontaxa)",
    ##        "color": "green",
    ##        "vbf": True,
    ##    },

    #allKlambda_DetaMjj_NNGW_extravars
    ##    f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_extravars_Non-NormGenWeights.h5": {
    ##        "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda_DetaMjj_extravars_Non-NormGenWeights.h5",
    ##        "true": "true_detamjj_nngw",
    ##        "label": "SPANet DetaMjj NNGW extravars (nkontaxa)",
    ##        "color": "yellow",
    ##        "vbf": True,
    ##    },

    #ptFlattenedMatchedHiggs_allKlambda_DetaMjj
    ##    f"{spanet_dir_nestor}vbf/predictions_vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj.h5": {
    ##        "file": f"{spanet_dir_nestor}vbf/predictions_vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj.h5",
    ##        "true": "true_ptFlattenMatchedHiggs_detamjj",
    ##        "label": "SPANet ptFlattenedMatchedHiggs DetaMjj (nkontaxa)",
    ##        "color": "lightblue",
    ##        "vbf": True,
    ##    },

    #ptFlattenedMatchedHiggs_allKlambda_DetaMjj_AddVBFJetOrder
    ##    f"{spanet_dir_nestor}vbf/predictions_vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj_AddVBFJetOrder.h5": {
    ##        "file": f"{spanet_dir_nestor}vbf/predictions_vbf_ptFlattenMatchedHiggs_all_Klambda_DetaMjj_AddVBFJetOrder.h5",
    ##        "true": "true_ptFlattenMatchedHiggs_addVBFJetOrder",
    ##        "label": "SPANet ptFlattenedMatchedHiggs DetaMjj AddVBFJetOrder (nkontaxa)",
    ##        "color": "pink",
    ##        "vbf": True,
    ##    },

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
            "label": "SPANet ptFlattenedMatchedHiggs DetaMjj SeparateHiggsVBF OnlyHiggsFlattening (nkontaxa)",
            "color": "lightgrey",
            "vbf": True,
        },

    #AllKlambda 9 jets
    ##f"{spanet_dir_nestor}vbf/predictions_allKlambda.h5": {
    ##    "file": f"{spanet_dir_nestor}vbf/predictions_allKlambda.h5",
    ##    "true": "true_9jets_allklambda",
    ##    "label": "SPANet 9 jets all_Klambda 100e (nkontaxa)",
    ##    "color": "red",
    ##    "vbf": True,
    ##},

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
