

spanet_dir = "/eos/user/t/tharte/Analysis_Project/predictions/"
spanet_dir_matteo = "/eos/home-m/mmalucch/spanet_inputs/out_prediction_files/"

# uncomment the configurations that you want to use
print(
    "WARNING: the naming has to follow the convetions in",
    " efficiency_functions.check_names function in order to work properly",
    " associating the spanet predicted files with the true files",
)
# uncomment the configurations that you want to use
spanet_dict = {
    # "5_jets_pt_300e_allklambda": f"{spanet_dir}spanet_hh4b_5jets_300_predict_s101.h5",  # THIS
    # "5_jets_pt_SM_50e_allklambda": f"{spanet_dir}spanet_hh4b_5jets_50_predict_s100_SM.h5",  # THIS
    # "5_jets_pt_btag_50e_allklambda": f"{spanet_dir}spanet_hh4b_5jets_50_predict_s155_btag.h5",  # THIS
    "5_jets_ptnone_btag_300e_allklambda": f"{spanet_dir}spanet_hh4b_5jets_300_ptnone_loose_s100_btag.h5",  # THIS
    # "5_jets_pt_btag_300e_allklambda": f"{spanet_dir}spanet_hh4b_5jets_300_predict_s160_btag.h5",  # THIS
    "5_jets_ptvary_loose_btag_300e_01_10_allklambda": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_01_10_loose_s100_btag.h5",  # THIS
    # "5_jets_ptvary_loose_btag_300e_wide_allklambda": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_wide_loose_s100_btag.h5",  # THIS
    "5_jets_ptvary_loose_btag_300e_allklambda": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s50_btag.h5",  # THIS
    "5_jets_ptvary_loose_btag_300e_wide_allklambda_SM": f"{spanet_dir}SM_train/spanet_hh4b_5jets_300_ptvary_wide_loose_s100_btag.h5",  # THIS
    "5_jets_ptvary_loose_btag_300e_wide_allklambda_onlylog": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_wide_onlylog_loose_s100_btag.h5",  # THIS
    # "5_jets_ptvary_tight_btag_300e_wide_allklambda": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_wide_tight_s100_btag.h5",  # THIS
    # "5_jets_ptvary_tight_btag_300e_allklambda": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_tight_s100_btag.h5",  # THIS
    # "5_jets_ATLAS_ptreg_5train_klambda0": f"{spanet_dir}out_spanet_prediction_5jets_klambda0.h5",
    # "5_jets_ATLAS_ptreg_5train_klambda2p45": f"{spanet_dir}out_spanet_prediction_5jets_klambda2p45.h5",
    # "5_jets_ATLAS_ptreg_5train_klambda5": f"{spanet_dir}out_spanet_prediction_5jets_klambda5.h5",
    #
    # "4_jets_ATLAS_ptreg_5train": f"{spanet_dir}out_spanet_prediction_5jets_ptreg_ATLAS.h5",  # THIS
    # "4_jets_ATLAS_ptreg_5train_klambda0": f"{spanet_dir}out_spanet_prediction_5jets_klambda0.h5",
    # "4_jets_ATLAS_ptreg_5train_klambda2p45": f"{spanet_dir}out_spanet_prediction_5jets_klambda2p45.h5",
    # "4_jets_ATLAS_ptreg_5train_klambda5": f"{spanet_dir}out_spanet_prediction_5jets_klambda5.h5",
    #
    # "4_jets_ATLAS_ptreg_5train": f"{spanet_dir}out_spanet_prediction_4jets_5training.h5", # THIS
    # "4_jets_ATLAS_ptreg_5train_klambda0": f"{spanet_dir}out_spanet_prediction_4jets_klambda0_5jetstrainig.h5",
    # "4_jets_ATLAS_ptreg_5train_klambda2p45": f"{spanet_dir}out_spanet_prediction_4jets_klambda2p45_5jetstrainig.h5",
    # "4_jets_ATLAS_ptreg_5train_klambda5": f"{spanet_dir}out_spanet_prediction_4jets_klambda5_5jetstrainig.h5",
    #
    #"4_jets_5global_ATLAS_ptreg": f"{spanet_dir}out_9_spanet_prediction_4jets_5global_ATLAS_ptreg_klambda1.h5",  # THIS
    # "4_jets_5global_ATLAS_ptreg_klambda0": f"{spanet_dir}out_9_spanet_prediction_4jets_5global_ATLAS_ptreg_klambda0.h5",
    # "4_jets_5global_ATLAS_ptreg_klambda2p45": f"{spanet_dir}out_9_spanet_prediction_4jets_5global_ATLAS_ptreg_klambda2p45.h5",
    # "4_jets_5global_ATLAS_ptreg_klambda5": f"{spanet_dir}out_9_spanet_prediction_4jets_5global_ATLAS_ptreg_klambda5.h5",
    #
    # "5_jets_ATLAS_ptreg_allklambda_train_klinput": f"{spanet_dir}out_spanet_prediction_5jets_lr1e4_kl_300e.h5",
    # # "/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5",  # HERE
    # "5_jets_ATLAS_ptreg_allklambda_train": f"{spanet_dir_matteo}out_spanet_prediction_5jets_lr1e4_noevkl_300e.h5",
    # # "/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5",  # HERE
    # "5_jets_ATLAS_ptreg_allklambda_eval": f"{spanet_dir_matteo}out_spanet_prediction_SMtraining_lr1e4_evkl.h5",
    # "/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5",  # HERE
    #
    # "5_jets_ATLAS_ptreg_allklambda_train_klinput_newkl_newCuts_newCutsEval": f"{spanet_dir}spanet_prediction_nc_on_nc_300e.h5",
    # "5_jets_ATLAS_ptreg_allklambda_train_klinput_newkl_newCuts_oldCutsEval": f"{spanet_dir}spanet_prediction_nc_on_oc_kl3p5.h5",
    # "5_jets_ATLAS_ptreg_allklambda_train_klinput_newkl_oldCuts_oldCutsEval": f"{spanet_dir}spanet_prediction_oc_on_oc_kl3p5.h5",
    # "5_jets_ATLAS_ptreg_5train_allklambda_noklinput_oldCuts_newCutsEval": f"{spanet_dir}spanet_prediction_oc_kl3p5_noklinp_data_nc.h5",
    # "5_jets_ATLAS_ptreg_5train_allklambda_noklinput_newCuts_newCutsEval": f"{spanet_dir}spanet_prediction_nc_noklinp_on_data.h5",
    # "5_jets_ATLAS_ptreg_allklambda_train_klinput_newkl_oldCuts_newCutsEval": f"{spanet_dir_matteo}spanet_prediction_oc_kl3p5_on_nc.h5",
    "5_jets_ATLAS_ptreg_allklambda_train_noklinput_newkl_oldCuts_newCutsEval": f"{spanet_dir_matteo}spanet_prediction_oc_kl3p5_noklinp_nc.h5",
}


# true_dir = "/eos/home-r/ramellar/out_prediction_files/true_files/"
# true_dir = "/afs/cern.ch/user/m/mmalucch/public/out_prediction_files/true_files/"
true_dir_thierry = "/eos/user/t/tharte/Analysis_Project/samplesets"
true_dir_matteo = "/eos/home-m/mmalucch/spanet_inputs/out_prediction_files/true_files/"

print(
    "WARNING: do not comment the items of this dictionary",
    " if you add a new true file you have to update the efficiency_functions.check_names",
    " and add a new if statement in the function",
)
true_dict = {
    "4 jets": f"{true_dir_matteo}output_JetGoodHiggs_test.h5",
    "5 jets": f"{true_dir_matteo}output_JetGood_test.h5",
    "5_jets_btag_presel": f"{true_dir_matteo}output_JetGood_btag_presel_test.h5",
    "4_jets_klambda0": f"{true_dir_matteo}kl0_output_JetGoodHiggs_test.h5",
    "4_jets_klambda2p45": f"{true_dir_matteo}kl2p45_output_JetGoodHiggs_test.h5",
    "4_jets_klambda5": f"{true_dir_matteo}kl5_output_JetGoodHiggs_test.h5",
    "5_jets_klambda0": f"{true_dir_matteo}kl0_output_JetGood_test.h5",
    "5_jets_klambda2p45": f"{true_dir_matteo}kl2p45_output_JetGood_test.h5",
    "5_jets_klambda5": f"{true_dir_matteo}kl5_output_JetGood_test.h5",
    "4_jets_data": f"{spanet_dir_matteo}out_spanet_prediction_data_ev4jets_training5jet_ptreg_ATLAS.h5",
    "5_jets_data": f"{spanet_dir_matteo}out_spanet_prediction_data_ev5jets_training5jet_ptreg_ATLAS.h5",
    "5_jets_data_oldCuts": f"{spanet_dir_matteo}spanet_prediction_sm_on_data_oc.h5",
    "5_jets_data_newCuts": f"{spanet_dir_matteo}spanet_prediction_nc_noklinp_on_data.h5",
    "4_jets_allklambda": f"{true_dir_matteo}output_JetGood_test_allkl_new_kl_newcuts.h5",  # output_JetGoodHiggs_allkl_test
    "5_jets_allklambda": f"{true_dir_matteo}output_JetGood_test_allkl_new_kl_newcuts.h5",  # output_JetGood_allkl_test
    "5_jets_allklambda_newkl_oldCuts": f"{true_dir_matteo}output_JetGood_test_allkl_new_kl_oldcuts.h5",  # "/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5",
    "5_jets_allklambda_newkl_newCuts": f"{true_dir_matteo}output_JetGood_test_allkl_new_kl_newcuts.h5",  # "/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5",
    "4_jets_allklambda_newkl_newCuts": f"{true_dir_matteo}output_JetGoodHiggs_test_allkl_new_kl_newcuts.h5",  # "/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5",
    "5_jets_pt_allklambda": f"{true_dir_thierry}/jet5global_pt/output_JetGood_test.h5",
    "4_jets_pt_allklambda": f"{true_dir_thierry}/jet5global_pt/output_JetGoodHiggs_test.h5",
}


names_dict = {
    "total_diff_eff_spanet": "Total Pairing Efficiency",
    "diff_eff_spanet": "Pairing Efficiency",
    "total_diff_eff_mask30": r"Total Efficiency ($\Delta D_{HH} > 30$ GeV)",
    "diff_eff_mask30": r"Efficiency ($\Delta D_{HH} > 30$ GeV)",
    "5_jets_pt_300e_allklambda": "SPANet 5 jets, 5th jet pT ordered",
    "5_jets_pt_SM_50e_allklambda": "SPANet 5 jets, 5th jet pT ordered SM train",
    "5_jets_pt_btag_50e_allklambda": "SPANet 5 jets pt order with btag 50e",
    "5_jets_pt_btag_300e_allklambda": "SPANet 5 jets pt order with btag",
    "5_jets_ptnone_btag_300e_allklambda": "SPANet 5 jets no pt with btag",
    "5_jets_ptvary_loose_btag_300e_wide_allklambda": "SPANet 5 jets pt order, ptvary [0.3,1.7], loose",
    "5_jets_ptvary_loose_btag_300e_wide_allklambda_SM": "SPANet 5 jets pt order, ptvary [0.3,1.7], loose, SM",
    "5_jets_ptvary_loose_btag_300e_wide_allklambda_onlylog": "SPANet 5 jets pt order, ptvary [0.3,1.7], loose, no normalization",
    "5_jets_ptvary_loose_btag_300e_01_10_allklambda": "SPANet 5 jets pt order, ptvary [0.1,10], loose",
    "5_jets_ptvary_loose_btag_300e_allklambda": "SPANet 5 jets pt order, ptvary [0.5,1.5], loose",
    "5_jets_ptvary_tight_btag_300e_wide_allklambda": "SPANet 5 jets pt order, ptvary [0.3,1.7], tignt",
    "5_jets_ptvary_tight_btag_300e_allklambda": "SPANet 5 jets pt order, ptvary [0.5,1.5], tight",
    "4_jets_ATLAS_ptreg_5train": "SPANet Lite 5 jets (4 jets eval)",
    "4_jets_5global_ATLAS_ptreg": "SPANet Lite 4 jets",
    "5_jets_data_ATLAS_ptreg_5train": "SPANet Lite 5 jets",
    "4_jets_data_ATLAS_ptreg_5train": "SPANet Lite 5 jets (4 jets eval)",
    "4_jets_data_ATLAS_5global_ptreg": "SPANet Lite 4 jets",
    "5_jets_ATLAS_ptreg_allklambda_train_klinput": r"SPANet Lite 5 jets all $\kappa_{\lambda}$ ($\kappa_{\lambda}$ inputs)",
    "5_jets_ATLAS_ptreg_allklambda_train_klinput_newkl_newCuts": r"SPANet Lite 5 jets new $\kappa_{\lambda}$ ($\kappa_{\lambda}$ inputs)",
    "5_jets_ATLAS_ptreg_allklambda_train": r"SPANet Lite 5 jets all $\kappa_{\lambda}$",
    "5_jets_ATLAS_ptreg_allklambda_eval": "SPANet Lite 5 jets SM",
    "4_jets_allklambda": "Run 2",
    "eff_fully_matched_allklambda": "Pairing Efficiency",
    "tot_eff_fully_matched_allklambda": "Total Pairing Efficiency",
    "eff_fully_matched_mask30_allklambda": r"Pairing Efficiency ($\Delta D_{HH} > 30$ GeV)",
    "tot_eff_fully_matched_mask30_allklambda": r"Total Pairing Efficiency ($\Delta D_{HH} > 30$ GeV)",
    "5_jets_ATLAS_ptreg_allklambda_train_klinput_newkl_oldCuts_newCutsEval": r"SPANet - $\kappa_{\lambda}$ ($\kappa_{\lambda}$ inputs) - Tight Selection",
    "5_jets_ATLAS_ptreg_allklambda_train_noklinput_newkl_oldCuts_newCutsEval": r"SPANet - $\kappa_{\lambda}$ - Tight Selection",
    "5_jets_ATLAS_ptreg_sm_train_allklambda_eval_noklinput_newkl_oldCuts_newCutsEval": "SPANet - SM - Tight Selection",
    "5_jets_ATLAS_ptreg_allklambda_train_noklinput_newkl_newCuts_newCutsEval": r"SPANet - $\kappa_{\lambda}$ - Loose Selection",
    "5_jets_data_ATLAS_ptreg_5train_allklambda_noklinput_oldCuts_newCutsEval": r"SPANet - $\kappa_{\lambda}$ - Tight Selection",
    "5_jets_data_ATLAS_ptreg_5train_allklambda_noklinput_newCuts_newCutsEval": r"SPANet - $\kappa_{\lambda}$ - Loose Selection",
    "5_jets_allklambda_newkl_newCuts": "$D_{HH}$-method",
    "4_jets_allklambda_newkl_newCuts": "$D_{HH}$-method",
}


color_dict = {
    "5_jets_ATLAS_ptreg_allklambda_train_klinput_newkl_oldCuts_newCutsEval": "tab:blue",
    "5_jets_ATLAS_ptreg_allklambda_train_noklinput_newkl_oldCuts_newCutsEval": "tab:orange",
    "5_jets_ATLAS_ptreg_sm_train_allklambda_eval_noklinput_newkl_oldCuts_newCutsEval": "tab:green",
    "5_jets_ATLAS_ptreg_allklambda_train_noklinput_newkl_newCuts_newCutsEval": "purple",
    "5_jets_data_ATLAS_ptreg_5train_allklambda_noklinput_oldCuts_newCutsEval": "deepskyblue",
    "5_jets_data_ATLAS_ptreg_5train_allklambda_noklinput_newCuts_newCutsEval": "coral",
    "5_jets_allklambda_newkl_newCuts": "yellowgreen",
    "4_jets_allklambda_newkl_newCuts": "yellowgreen",
    "5_jets_ATLAS_ptreg_allklambda_train_noklinput_newkl_newCuts_newCutsEval": "coral",
    "5_jets_ATLAS_ptreg_allklambda_train_noklinput_newkl_oldCuts_newCutsEval": "orangered",
    "5_jets_ATLAS_ptreg_allklambda_train_klinput_newkl_oldCuts_newCutsEval": "darkorange",
    "5_jets_ATLAS_ptreg_sm_train_allklambda_eval_noklinput_newkl_oldCuts_newCutsEval": "deeppink",
    "5_jets_pt_300e_allklambda": "goldenrod",
    "5_jets_pt_SM_50e_allklambda": "gold",
    "5_jets_pt_btag_50e_allklambda": "royalblue",
    "5_jets_pt_btag_300e_allklambda": "navy",
    "5_jets_ptvary_loose_btag_300e_wide_allklambda": "goldenrod",
    "5_jets_ptvary_loose_btag_300e_wide_allklambda_SM": "lime",
    "5_jets_ptvary_loose_btag_300e_wide_allklambda_onlylog": "deeppink",
    "5_jets_ptvary_loose_btag_300e_01_10_allklambda": "orange",
    "5_jets_ptvary_loose_btag_300e_allklambda": "gold",
    "5_jets_pt_SM_50e_allklambda": "gold",
    "5_jets_ptnone_300e_allklambda": "cyan",
    "5_jets_ptvary_tight_btag_300e_wide_allklambda": "cyan",
    "5_jets_ptvary_tight_btag_300e_allklambda": "darkcyan",
}
