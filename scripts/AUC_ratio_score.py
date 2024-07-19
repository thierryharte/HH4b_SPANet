from collections import defaultdict
import awkward as ak
import numba
import numpy as np
import pandas as pd
import h5py
import vector
vector.register_numba()
vector.register_awkward()

import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
#import mplhep as hep
#hep.style.use(hep.style.ROOT)

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib
matplotlib.rcParams["figure.dpi"] = 300
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import argparse

from AUC_functions import *

path_p= "/eos/home-r/ramellar/4_classification/classification_predictions/"
path_v= "/eos/home-r/ramellar/4_classification/variability_study"
path_i= "/eos/home-r/ramellar/5_inputs/"
desired_lenght=100
x2=np.linspace(0,0.4,10)
x1=np.linspace(0.4,1,100)

print("\n")
print("For DNN vars and probability difference: \n")
score_2b_data_r_dnn_proba=AUC_scores("spanet_c_4v_dnn_proba_2b_data_out.h5", "spanet_classifier_2b_data/output_JetGood_test.h5")
print(f"AUC_2b_data_r_dnn_proba {score_2b_data_r_dnn_proba}")
score_2b_QCD_r_dnn_proba= AUC_scores("spanet_c_4v_dnn_proba_2b_QCD_new.h5", "spanet_classifier_2b_QCD/output_JetGood_test.h5")
print(f"AUC_2b_QCD_r_dnn_proba {score_2b_QCD_r_dnn_proba}")
score_2b_data_f_dnn_proba= AUC_scores("spanet_prediction_2b_data_f_dnn_proba.h5", "2b_data_full_statistics_c/output_JetGood_test.h5")
print(f"AUC_2b_data_f_dnn_proba {score_2b_data_f_dnn_proba}")
score_4b_QCD_dnn_proba= AUC_scores("prediction__4b_QCD_dnn_vars_proba_seeds/", "spanet_classifier_4b_QCD_working/output_JetGood_test.h5", True)
print(f"AUC_4b_QCD_dnn_proba {score_4b_QCD_dnn_proba}")

model_ratio_dnn_proba= score_2b_QCD_r_dnn_proba/score_2b_data_r_dnn_proba
btag_ratio_dnn_proba= [score_4b_QCD_dnn_proba_i/score_2b_QCD_r_dnn_proba for score_4b_QCD_dnn_proba_i in score_4b_QCD_dnn_proba]

AUC_4b_data_dnn_proba= [btag_ratio_dnn_proba_i * score_2b_data_f_dnn_proba for btag_ratio_dnn_proba_i in btag_ratio_dnn_proba]

print( "AUC 4b data", AUC_4b_data_dnn_proba)
print(f"max {max(AUC_4b_data_dnn_proba):.3f}")
print(f"min {min(AUC_4b_data_dnn_proba):.3f}")
print(f"mean {(sum(AUC_4b_data_dnn_proba)/len(AUC_4b_data_dnn_proba)):.3f}")

print("\n")
print("For DNN vars: \n")
score_2b_data_r_dnn=AUC_scores("spanet_c_4v_dnn_2b_data_out.h5", "spanet_classifier_2b_data/output_JetGood_test.h5")
print(f"AUC_2b_data_r_dnn {score_2b_data_r_dnn}")
score_2b_QCD_r_dnn= AUC_scores("spanet_c_4v_dnn_2b_QCD_new.h5", "spanet_classifier_2b_QCD/output_JetGood_test.h5")
print(f"AUC_2b_QCD_r_dnn {score_2b_QCD_r_dnn}")
score_2b_data_f_dnn= AUC_scores("spanet_prediction_2b_data_f_dnn.h5", "2b_data_full_statistics_c/output_JetGood_test.h5")
print(f"AUC_2b_data_f_dnn {score_2b_data_f_dnn}")
score_4b_QCD_dnn= AUC_scores("prediction__4b_QCD_dnn_vars_seeds/", "spanet_classifier_4b_QCD_working/output_JetGood_test.h5", True)
print(f"AUC_4b_QCD_dnn {score_4b_QCD_dnn}")

model_ratio_dnn= score_2b_QCD_r_dnn/score_2b_data_r_dnn
btag_ratio_dnn= [score_4b_QCD_dnn_i/score_2b_QCD_r_dnn for score_4b_QCD_dnn_i in score_4b_QCD_dnn]

AUC_4b_data_dnn= [btag_ratio_dnn_i * score_2b_data_f_dnn for btag_ratio_dnn_i in btag_ratio_dnn]

print( "AUC 4b data", AUC_4b_data_dnn)
print(f"max{max(AUC_4b_data_dnn):.3f}")
print(f"min{min(AUC_4b_data_dnn):.3f}")
print(f"mean{(sum(AUC_4b_data_dnn)/len(AUC_4b_data_dnn)):.3f}")

#For dnn and dnn + prob but ev on sr
score_2b_data_r_dnn_sr=AUC_scores("spanet_c_4v_dnn_2b_data_out.h5", "spanet_classifier_2b_data/output_JetGood_test.h5")
# print(f"AUC_2b_data_r_dnn {score_2b_data_r_dnn}")
score_2b_QCD_r_dnn_sr= AUC_scores("spanet_c_4v_dnn_2b_QCD_new.h5", "spanet_classifier_2b_QCD/output_JetGood_test.h5")
# print(f"AUC_2b_QCD_r_dnn {score_2b_QCD_r_dnn}")
score_2b_data_f_dnn_sr= AUC_scores("spanet_c_pred_training_inclusive_2b_data_f_eval_sr_2b_data_full_stats_dnn.h5","/signal_region/2b_data_full_stats/output_JetGood_test.h5")
# print(f"AUC_2b_data_f_dnn {score_2b_data_f_dnn}")
score_4b_QCD_dnn_sr= AUC_scores("prediction__4b_QCD_dnn_vars_seeds/", "spanet_classifier_4b_QCD_working/output_JetGood_test.h5", True)
# print(f"AUC_4b_QCD_dnn {score_4b_QCD_dnn}")

model_ratio_dnn_sr= score_2b_QCD_r_dnn_sr/score_2b_data_r_dnn_sr
btag_ratio_dnn_sr= [score_4b_QCD_dnn_i/score_2b_QCD_r_dnn_sr for score_4b_QCD_dnn_i in score_4b_QCD_dnn_sr]

AUC_4b_data_dnn_sr= [btag_ratio_dnn_i * score_2b_data_f_dnn_sr for btag_ratio_dnn_i in btag_ratio_dnn_sr]


score_2b_data_r_dnn_proba_sr=AUC_scores("spanet_c_4v_dnn_proba_2b_data_out.h5", "spanet_classifier_2b_data/output_JetGood_test.h5")
# print(f"AUC_2b_data_r_dnn_proba {score_2b_data_r_dnn_proba}")
score_2b_QCD_r_dnn_proba_sr= AUC_scores("spanet_c_4v_dnn_proba_2b_QCD_new.h5", "spanet_classifier_2b_QCD/output_JetGood_test.h5")
# print(f"AUC_2b_QCD_r_dnn_proba {score_2b_QCD_r_dnn_proba}")
score_2b_data_f_dnn_proba_sr= AUC_scores("spanet_c_pred_training_inclusive_2b_data_f_eval_sr_2b_data_full_stats_dnn_proba.h5","/signal_region/2b_data_full_stats/output_JetGood_test.h5")
# print(f"AUC_2b_data_f_dnn_proba {score_2b_data_f_dnn_proba}")
score_4b_QCD_dnn_proba_sr= AUC_scores("prediction__4b_QCD_dnn_vars_proba_seeds/", "spanet_classifier_4b_QCD_working/output_JetGood_test.h5", True)
# print(f"AUC_4b_QCD_dnn_proba {score_4b_QCD_dnn_proba}")

model_ratio_dnn_proba_sr= score_2b_QCD_r_dnn_proba_sr/score_2b_data_r_dnn_proba_sr
btag_ratio_dnn_proba_sr= [score_4b_QCD_dnn_proba_i/score_2b_QCD_r_dnn_proba_sr for score_4b_QCD_dnn_proba_i in score_4b_QCD_dnn_proba_sr]

AUC_4b_data_dnn_proba_sr= [btag_ratio_dnn_proba_i * score_2b_data_f_dnn_proba_sr for btag_ratio_dnn_proba_i in btag_ratio_dnn_proba_sr]


#For the morphing of the ROC curve

print("\n")
print("For the FPR: \n")

model_ratio_dnn, fpr_2b_QCD_dnn, fpr_2b_data_dnn, tpr_2b_QCD_dnn, tpr_2b_data_dnn = ratio_fpr("spanet_c_4v_dnn_2b_QCD_new.h5", "spanet_classifier_2b_QCD/output_JetGood_test.h5", "spanet_c_4v_dnn_2b_data_out.h5", "spanet_classifier_2b_data/output_JetGood_test.h5", x1, x2)
btag_ratio_dnn, fpr_4b_QCD_dnn, fpr_2b_QCD, tpr_4b_QCD_dnn, tpr_2b_QCD_dnn = ratio_fpr("prediction__4b_QCD_dnn_vars_seeds/", "spanet_classifier_4b_QCD_working/output_JetGood_test.h5", "spanet_c_4v_dnn_2b_QCD_new.h5", "spanet_classifier_2b_QCD/output_JetGood_test.h5", x1,x2, True)


model_ratio_dnn_proba, fpr_2b_QCD_dnn_proba, fpr_2b_data_dnn_proba, tpr_2b_QCD_dnn_proba, tpr_2b_data_dnn_proba = ratio_fpr("spanet_c_4v_dnn_proba_2b_QCD_new.h5", "spanet_classifier_2b_QCD/output_JetGood_test.h5", "spanet_c_4v_dnn_proba_2b_data_out.h5", "spanet_classifier_2b_data/output_JetGood_test.h5", x1,x2)
btag_ratio_dnn_proba, fpr_4b_QCD_dnn_proba, fpr_2b_QCD_proba, tpr_4b_QCD_dnn_proba, tpr_2b_QCD_dnn_proba = ratio_fpr("prediction__4b_QCD_dnn_vars_proba_seeds/", "spanet_classifier_4b_QCD_working/output_JetGood_test.h5", "spanet_c_4v_dnn_proba_2b_QCD_new.h5", "spanet_classifier_2b_QCD/output_JetGood_test.h5", x1,x2, True)
# ratio_fpr("prediction__4b_QCD_dnn_vars_seeds/", "spanet_classifier_4b_QCD_working/output_JetGood_test.h5", "spanet_c_4v_dnn_2b_QCD_new.h5", "spanet_classifier_2b_QCD/output_JetGood_test.h5", True)


fpr_2b_data_full_stats_dnn, tpr_2b_data_full_stats_dnn= fpr_tpr("spanet_prediction_2b_data_f_dnn.h5", "2b_data_full_statistics_c/output_JetGood_test.h5")
fpr_2b_data_full_stats_dnn_proba, tpr_2b_data_full_stats_dnn_proba= fpr_tpr("spanet_prediction_2b_data_f_dnn_proba.h5","2b_data_full_statistics_c/output_JetGood_test.h5" )

fpr_ratio(fpr_2b_data_full_stats_dnn, tpr_2b_data_full_stats_dnn, AUC_4b_data_dnn , btag_ratio_dnn, model_ratio_dnn, "4b_data_dnn", x1, x2, True)
fpr_ratio(fpr_2b_data_full_stats_dnn_proba, tpr_2b_data_full_stats_dnn_proba, AUC_4b_data_dnn_proba , btag_ratio_dnn_proba, model_ratio_dnn_proba, "4b_data_dnn_proba", x1, x2, True)

#Evaluation on the sr but inclusive training
fpr_2b_data_full_stats_dnn_proba_ev_on_sr, tpr_2b_data_full_stats_dnn_proba_ev_on_sr= fpr_tpr("spanet_c_pred_training_inclusive_2b_data_f_eval_sr_2b_data_full_stats_dnn_proba.h5","/signal_region/2b_data_full_stats/output_JetGood_test.h5" )
fpr_2b_data_full_stats_dnn_ev_on_sr, tpr_2b_data_full_stats_dnn_ev_on_sr= fpr_tpr("spanet_c_pred_training_inclusive_2b_data_f_eval_sr_2b_data_full_stats_dnn.h5", "signal_region/2b_data_full_stats/output_JetGood_test.h5")

fpr_ratio(fpr_2b_data_full_stats_dnn_proba_ev_on_sr, tpr_2b_data_full_stats_dnn_proba_ev_on_sr, AUC_4b_data_dnn_proba_sr , btag_ratio_dnn_proba, model_ratio_dnn_proba, "4b_data_dnn_proba_eval_on_sr", x1, x2)
fpr_ratio(fpr_2b_data_full_stats_dnn_ev_on_sr, tpr_2b_data_full_stats_dnn_ev_on_sr, AUC_4b_data_dnn_sr , btag_ratio_dnn, model_ratio_dnn, "4b_data_dnn_eval_on_sr", x1, x2)

#Training on sr
fpr_4b_QCD_sr_dnn, tpr_4b_QCD_sr_dnn = fpr_tpr("spanet_pred_c_4b_QCD_sr_dnn.h5", "signal_region/signal_region_4b_QCD/output_JetGood_test.h5")
fpr_4b_QCD_sr_dnn_proba, tpr_4b_QCD_sr_dnn_proba = fpr_tpr("spanet_pred_c_4b_QCD_sr_dnn_proba.h5", "signal_region/signal_region_4b_QCD/output_JetGood_test.h5")

fpr_4b_QCD_dnn, tpr_4b_QCD_dnn = fpr_tpr("spanet_c_pred_4b_QCD_dnn.h5", "spanet_classifier_4b_QCD_working/output_JetGood_test.h5")
fpr_4b_QCD_dnn_proba, tpr_4b_QCD_dnn_proba = fpr_tpr("spanet_c_pred_4b_QCD_dnn_proba.h5", "spanet_classifier_4b_QCD_working/output_JetGood_test.h5")

roc_curve_compare_weights(tpr_4b_QCD_sr_dnn, fpr_4b_QCD_sr_dnn, "4b QCD sr with DNN as inputs", "4b_QCD_sr_dnn", "4b QCD sr with DNN as inputs" )
roc_curve_compare_weights(tpr_4b_QCD_sr_dnn_proba, fpr_4b_QCD_sr_dnn_proba, "4b QCD sr with DNN and prob diff as inputs", "4b_QCD_sr_dnn_proba", "4b QCD sr with DNN and probability difference as inputs" )

tpr_list=[tpr_4b_QCD_sr_dnn, tpr_4b_QCD_sr_dnn_proba]
fpr_list=[fpr_4b_QCD_sr_dnn, fpr_4b_QCD_sr_dnn_proba]
labels=["4b QCD sr with DNN as inputs", "4b QCD sr with DNN and probability difference as inputs"]

roc_curve_compare_weights(tpr_list, fpr_list, labels, "4b_QCD_sr","Comparison for trainings in the 4b QCD signal region", True)

tpr_list=[tpr_4b_QCD_sr_dnn, tpr_4b_QCD_sr_dnn_proba, tpr_4b_QCD_dnn, tpr_4b_QCD_dnn_proba]
fpr_list=[fpr_4b_QCD_sr_dnn, fpr_4b_QCD_sr_dnn_proba, fpr_4b_QCD_dnn, fpr_4b_QCD_dnn_proba]
labels=["4b QCD sr with DNN as inputs", "4b QCD sr with DNN and probability difference as inputs", "4b QCD with DNN as inputs", "4b QCD with DNN and prob diff as inputs"]

roc_curve_compare_weights(tpr_list, fpr_list, labels, "4b_QCD","Comparison for trainings with the 4b QCD signal region vs 4b inclusive", True)

roc_curve_compare_weights(tpr_list, fpr_list, labels, "4b_QCD_AN","Comparison for trainings with the 4b QCD signal region vs 4b inclusive", True, True)



# print("tpr_2b_data_full_stats_dnn_proba", tpr_2b_data_full_stats_dnn_proba)