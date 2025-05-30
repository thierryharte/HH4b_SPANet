

spanet_dir = "/eos/user/t/tharte/Analysis_data/predictions/"
spanet_dir_matteo = "/eos/user/m/mmalucch/spanet_inputs/out_prediction_files/"

true_dir_thierry = "/eos/user/t/tharte/Analysis_data/spanet_samples/"
true_dir_matteo = "/eos/user/m/mmalucch/spanet_inputs/out_prediction_files/true_files/"

# uncomment the configurations that you want to use
print(
    "WARNING: the naming has to follow the convetions in",
    " efficiency_functions.check_names function in order to work properly",
    " associating the spanet predicted files with the true files",
)
# This is rather special
# We need a run2 dataset. This is here defined over the spanet model. However, it only depends on the true file defined in the spanet dictionary. MIGHT HAVE TO BE IMPROVED.
# The reason not to go directly with the true file is, that we are not reading out all the true files anymore...
run2_dataset = "5_jets_postEE_300e_allklambda_preEE_eval"
#run2_dataset = "5_jets_pt_btag_300e_allklambda"

spanet_dict = {
    #### For baseline with btag ####
    #"5_jets_pt_btag_300e_allklambda": {
    #    "file": f"{spanet_dir}spanet_hh4b_5jets_300_predict_s160_btag.h5",
    #    "true": "5_jets_pt_true_btag_allklambda",
    #    "label": "SPANet baseline",
    #    "color": "darkblue"},

    #### [0.1, 10] pt vary ####
    #"5_jets_ptvary_loose_btag_300e_01_10_allklambda": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_01_10_loose_s100_btag.h5",  # THIS
    #"5_jets_ptvary_loose_btag_300e_01_10_allklambda_rerun": {
    #    "file": f"{spanet_dir}rerun/spanet_rerun_hh4b_5jets_300_ptvary_loose_s100_btag_01_10.h5",
    #    "true": "5_jets_pt_true_btag_allklambda",
    #    "label": "SPANet - Flattened pt [0.1,10]",
    #    "color": "tan"},

    #### [0.3,1.7] ####
    #"5_jets_ptvary_loose_btag_300e_wide_allklambda": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_wide_loose_s100_btag.h5",  # THIS ## Chosen model
    #"5_jets_ptvary_loose_btag_300e_wide_allklambda_SM": f"{spanet_dir}SM_train/spanet_hh4b_5jets_300_ptvary_wide_loose_s100_btag.h5",  # THIS
    #"5_jets_ptvary_loose_btag_300e_wide_allklambda_onlylog": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_wide_onlylog_loose_s100_btag.h5",  # THIS
    #"5_jets_ptvary_loose_btag_300e_03_17_allklambda_rerun": {
    #    "file": f"{spanet_dir}rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17.h5",
    #    "true": "5_jets_pt_true_btag_allklambda",
    #    "label": "SPANet - Flattened pt [0.3,1.7]",
    #    "color": "darkgoldenrod"},

    #### [0.5,1.5] ####
    #"5_jets_ptvary_loose_btag_300e_allklambda": {
    #    "file": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_loose_s50_btag.h5",  # THIS
    #    "true": "5_jets_pt_true_btag_allklambda",
    #    "label": "SPANet - Flattened pt [0.5,1.5]",
    #    "color": "gold"},

    # no pT/phi/btag
    #"5_jets_ptnone_btag_300e_allklambda": f"{spanet_dir}spanet_hh4b_5jets_300_ptnone_loose_s100_btag.h5",
    #"5_jets_pt_300e_allklambda_no_btag": f"{spanet_dir}spanet_hh4b_5jets_50_ptreg_loose_s100_no_btag.h5",  # THIS
    #"5_jets_ptreg_loose_btag_300e_allklambda_nophi": f"{spanet_dir}spanet_hh4b_5jets_300_ptreg_loose_s100_btag_nophi.h5",
    #"5_jets_ptvary_loose_btag_300e_01_10_allklambda_rerun_nophi": f"{spanet_dir}rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_01_10_nophi.h5",

    #### Tight selection ####
    # "5_jets_ptvary_tight_btag_300e_allklambda": f"{spanet_dir}spanet_hh4b_5jets_300_ptvary_tight_s100_btag.h5",  # THIS
    
    #### Matteo comparison (b-tag ordering) ####
    # originating from: "/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5",  # HERE
    # "5_jets_ATLAS_ptreg_allklambda_train": f"{spanet_dir_matteo}out_spanet_prediction_5jets_lr1e4_noevkl_300e.h5",
    # "5_jets_ATLAS_ptreg_allklambda_eval": f"{spanet_dir_matteo}out_spanet_prediction_SMtraining_lr1e4_evkl.h5",
  # "5_jets_data_ATLAS_ptreg_5train_allklambda_noklinput_oldCuts_newCutsEval": f"{spanet_dir_matteo}spanet_prediction_oc_kl3p5_noklinp_data_nc.h5",
    #"5_jets_ATLAS_ptreg_allklambda_train_noklinput_newkl_oldCuts_newCutsEval": f"{spanet_dir_matteo}spanet_prediction_oc_kl3p5_noklinp_nc.h5",
    #

    ##### ERA preEE #####
    #"5_jets_preEE_300e_allklambda_preEE_eval": {
    #    "file": f"{spanet_dir}/spanet_hh4b_preEE_5jets_100_pvary_loose_s300_eval_on_preEE.h5",
    #    "true": "5_jets_pt_allklambda_preEE_eval",
    #    "label": "pt Flattened preEE trained",
    #    "color": "darkblue"},
    "5_jets_postEE_300e_allklambda_preEE_eval": {
        "file": f"{spanet_dir}/spanet_hh4b_postEE_5jets_100_pvary_loose_s300_eval_on_preEE.h5",
        "true": "5_jets_pt_allklambda_preEE_eval",
        "label": "pT Flatened postEE trained",
        "color": "royalblue"},


    ############################################## DATA ###########################################################
    # baseline with b-tag
    #"5_jets_pt_data_btag_300e": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5", ### DATA
    # For baseline no btag:
    #"5_jets_pt_data_300e_allklambda_no_btag": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_50_ptreg_loose_s100_no_btag.h5",  # THIS
    
    #### [0.1,10] ####
    #"5_jets_pt_data_vary_loose_btag_300e_01_10": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptvary_01_10_loose_s100_btag.h5",
    #"5_jets_pt_data_vary_loose_btag_300e_01_10_rerun": f"{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_01_10.h5",

    #### [0.3,1.7] ####
    #"5_jets_pt_data_vary_loose_btag_300e_wide": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptvary_wide_loose_s100_btag.h5",
    ## normalizations:
    #"5_jets_pt_data_vary_loose_btag_300e_wide_onlylog": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptvary_wide_onlylog_loose_s100_btag.h5",
    #"5_jets_pt_data_vary_loose_btag_300e_03_17_rerun": f"{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17.h5",
    
    #### [0.5,1.5] ####
    #"5_jets_pt_data_vary_loose_btag_300e": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptvary_loose_s50_btag.h5",

    #### no pT ####
    #"5_jets_pt_data_none_btag_300e": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptnone_loose_s100_btag.h5", ### DATA
    
    #### no phi ####
    #"5_jets_pt_data_vary_loose_btag_300e_01_10_rerun_nophi": f"{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_01_10_nophi.h5",
    #"5_jets_pt_data_loose_btag_300e_nophi": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptreg_loose_s100_btag_nophi.h5",

    ##### baseline ####
    #"5_jets_pt_data_btag_300e": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5", ### DATA

    #### pt_vary ####
    #"5_jets_pt_data_btag_300e": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5", ### DATA
    #"5_jets_pt_data_vary_loose_btag_300e_01_10_rerun": f"{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_01_10.h5",
    #"5_jets_pt_data_vary_loose_btag_300e_03_17_rerun": f"{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17.h5",
    #"5_jets_pt_data_vary_loose_btag_300e": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptvary_loose_s50_btag.h5",

    #### No pt/phi/btag ####
    #"5_jets_pt_data_btag_300e": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5", ### DATA
    #"5_jets_pt_data_300e_allklambda_no_btag": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_50_ptreg_loose_s100_no_btag.h5",  # THIS
    #"5_jets_pt_data_none_btag_300e": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptnone_loose_s100_btag.h5", ### DATA
    ##"5_jets_pt_data_vary_loose_btag_300e_01_10_rerun_nophi": f"{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_01_10_nophi.h5",
    #"5_jets_pt_data_loose_btag_300e_nophi": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptreg_loose_s100_btag_nophi.h5",

    #### onlylog ####
    #"5_jets_pt_data_btag_300e": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5", ### DATA
    #"5_jets_pt_data_vary_loose_btag_300e_03_17_rerun": f"{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17.h5",
    #"5_jets_pt_data_vary_loose_btag_300e_wide_onlylog": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_ptvary_wide_onlylog_loose_s100_btag.h5",
    
    ##### final_comp #####
    #"5_jets_pt_data_btag_300e": f"{spanet_dir}DATA/spanet_hh4b_data_5jets_300_predict_s160_btag.h5", ### DATA
    #"5_jets_pt_data_vary_loose_btag_300e_03_17_rerun": f"{spanet_dir}DATA/rerun/spanet_rerun_hh4b_data_5jets_300_ptvary_loose_s100_btag_03_17.h5",

  # "5_jets_pt_true_vary_loose_btag_allklambda": f"{spanet_dir}../samplesets/loose_selection_random_pt_mass/output_JetGood_train.h5",  # THIS
  # "5_jets_pt_true_vary_loose_btag_wide_allklambda": f"{spanet_dir}../samplesets/loose_selection_random_pt_mass_wide/output_JetGood_train.h5",  # THIS
  # "5_jets_pt_true_vary_loose_btag_01_10_allklambda": f"{spanet_dir}../samplesets/loose_selection_random_pt_mass_01_10/output_JetGood_train.h5",  # THIS
  # "5_jets_pt_true_btag_allklambda": f"{spanet_dir}../samplesets/jet5global_pt/output_JetGood_train.h5",  # THIS
  }


# true_dir = "/eos/home-r/ramellar/out_prediction_files/true_files/"
# true_dir = "/afs/cern.ch/user/m/mmalucch/public/out_prediction_files/true_files/"

print(
    "WARNING: do not comment the items of this dictionary",
    " if you add a new true file you have to update the efficiency_functions.check_names",
    " and add a new if statement in the function",
)
true_dict = {
    "4 jets" : {"name": f"{true_dir_matteo}output_JetGoodHiggs_test.h5" , "klambda": "none"},
    "5 jets" : {"name": f"{true_dir_matteo}output_JetGood_test.h5" , "klambda": "none"},
    "5_jets_btag_presel" : {"name": f"{true_dir_matteo}output_JetGood_btag_presel_test.h5" , "klambda": "none"},
    "4_jets_klambda0" : {"name": f"{true_dir_matteo}kl0_output_JetGoodHiggs_test.h5" , "klambda": "none"},
    "4_jets_klambda2p45" : {"name": f"{true_dir_matteo}kl2p45_output_JetGoodHiggs_test.h5" , "klambda": "none"},
    "4_jets_klambda5" : {"name": f"{true_dir_matteo}kl5_output_JetGoodHiggs_test.h5" , "klambda": "none"},
    "5_jets_klambda0" : {"name": f"{true_dir_matteo}kl0_output_JetGood_test.h5" , "klambda": "none"},
    "5_jets_klambda2p45" : {"name": f"{true_dir_matteo}kl2p45_output_JetGood_test.h5" , "klambda": "none"},
    "5_jets_klambda5" : {"name": f"{true_dir_matteo}kl5_output_JetGood_test.h5" , "klambda": "none"},
    "4_jets_data" : {"name": f"{spanet_dir_matteo}out_spanet_prediction_data_ev4jets_training5jet_ptreg_ATLAS.h5" , "klambda": "none"},
    "5_jets_data" : {"name": f"{spanet_dir_matteo}out_spanet_prediction_data_ev5jets_training5jet_ptreg_ATLAS.h5" , "klambda": "none"},
    "5_jets_data_oldCuts" : {"name": f"{spanet_dir_matteo}spanet_prediction_sm_on_data_oc.h5" , "klambda": "none"},
    "5_jets_data_newCuts" : {"name": f"{spanet_dir_matteo}spanet_prediction_nc_noklinp_on_data.h5" , "klambda": "none"},
    "4_jets_allklambda" : {"name": f"{true_dir_matteo}output_JetGood_test_allkl_new_kl_newcuts.h5" , "klambda": "postEE"},  # output_JetGoodHiggs_allkl_test
    "5_jets_allklambda" : {"name": f"{true_dir_matteo}output_JetGood_test_allkl_new_kl_newcuts.h5" , "klambda": "postEE"},  # output_JetGood_allkl_test
    "5_jets_allklambda_newkl_oldCuts" : {"name": f"{true_dir_matteo}output_JetGood_test_allkl_new_kl_oldcuts.h5" , "klambda": "postEE"},  # "/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5"},
    "5_jets_allklambda_newkl_newCuts" : {"name": f"{true_dir_matteo}output_JetGood_test_allkl_new_kl_newcuts.h5" , "klambda": "postEE"},  # "/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5"},
    "4_jets_allklambda_newkl_newCuts" : {"name": f"{true_dir_matteo}output_JetGoodHiggs_test_allkl_new_kl_newcuts.h5" , "klambda": "postEE"},  # "/work/mmalucch/out_hh4b/out_spanet/output_JetGood_test.h5"},
    "5_jets_pt_allklambda" : {"name": f"{true_dir_thierry}/loose_selection/output_JetGood_test.h5" , "klambda": "postEE"},
    "4_jets_pt_allklambda" : {"name": f"{true_dir_thierry}/loose_selection/output_JetGoodHiggs_test.h5" , "klambda": "postEE"},
    "5_jets_pt_data" : {"name": f"{true_dir_thierry}/DATA_loose_cut/output_JetGood_test.h5" , "klambda": "none"},
    "5_jets_pt_true_btag_allklambda" : {"name": f"{spanet_dir}../spanet_samples/loose_selection/output_JetGood_test.h5" , "klambda": "postEE"},  # THIS
    "5_jets_pt_true_vary_loose_btag_allklambda" : {"name": f"{spanet_dir}../spanet_samples/loose_selection_random_pt_mass/output_JetGood_train.h5" , "klambda": "postEE"},  # THIS
    "5_jets_pt_true_vary_loose_btag_wide_allklambda" : {"name": f"{spanet_dir}../spanet_samples/loose_selection_random_pt_mass_wide/output_JetGood_train.h5" , "klambda": "postEE"},  # THIS
    "5_jets_pt_true_vary_loose_btag_01_10_allklambda" : {"name": f"{spanet_dir}../spanet_samples/loose_selection_random_pt_mass_01_10/output_JetGood_train.h5" , "klambda": "postEE"},  # THIS
    "5_jets_pt_allklambda_preEE_eval" : {"name": f"{true_dir_thierry}../spanet_samples/loose_all2022/preEE/output_JetGood_test.h5" , "klambda": "preEE"},
}
