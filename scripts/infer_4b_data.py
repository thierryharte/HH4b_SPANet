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
x2= np.linspace(0,0.4,1000)
x1= np.linspace(0.4,1,1000)

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

AUC_4b_data_dnn_proba= [btag_ratio_dnn_proba_i * score_2b_data_f_dnn_proba * model_ratio_dnn_proba for btag_ratio_dnn_proba_i in btag_ratio_dnn_proba]

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




#For dnn and dnn + prob but evaluated on sr
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
AUC_4b_data_dnn_proba_sr_m= [AUC_4b_data_dnn_proba_sr_i * model_ratio_dnn_proba_sr for AUC_4b_data_dnn_proba_sr_i in AUC_4b_data_dnn_proba_sr]


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

fpr_ratio(fpr_2b_data_full_stats_dnn_proba_ev_on_sr, tpr_2b_data_full_stats_dnn_proba_ev_on_sr, AUC_4b_data_dnn_proba_sr_m , btag_ratio_dnn_proba, model_ratio_dnn_proba, "4b_data_dnn_proba_eval_on_sr", x1, x2, True)
fpr_ratio(fpr_2b_data_full_stats_dnn_ev_on_sr, tpr_2b_data_full_stats_dnn_ev_on_sr, AUC_4b_data_dnn_sr , btag_ratio_dnn, model_ratio_dnn, "4b_data_dnn_eval_on_sr", x1, x2, True)

#Training on signal region
fpr_4b_QCD_sr_dnn, tpr_4b_QCD_sr_dnn = fpr_tpr("spanet_pred_c_4b_QCD_sr_dnn.h5", "signal_region/signal_region_4b_QCD/output_JetGood_test.h5")
fpr_4b_QCD_sr_dnn_proba, tpr_4b_QCD_sr_dnn_proba = fpr_tpr("spanet_pred_c_4b_QCD_sr_dnn_proba.h5", "signal_region/signal_region_4b_QCD/output_JetGood_test.h5")

fpr_4b_QCD_dnn_2, tpr_4b_QCD_dnn_2 = fpr_tpr("spanet_c_pred_4b_QCD_dnn.h5", "spanet_classifier_4b_QCD_working/output_JetGood_test.h5")
fpr_4b_QCD_dnn_proba_2, tpr_4b_QCD_dnn_proba_2 = fpr_tpr("spanet_c_pred_4b_QCD_dnn_proba.h5", "spanet_classifier_4b_QCD_working/output_JetGood_test.h5")

roc_curve_compare_weights(tpr_4b_QCD_sr_dnn, fpr_4b_QCD_sr_dnn, "4b QCD sr with DNN as inputs", "4b_QCD_sr_dnn", "4b QCD sr with DNN as inputs" )
roc_curve_compare_weights(tpr_4b_QCD_sr_dnn_proba, fpr_4b_QCD_sr_dnn_proba, "4b QCD sr with DNN and prob diff as inputs", "4b_QCD_sr_dnn_proba", "4b QCD sr with DNN and probability difference as inputs" )

tpr_list=[tpr_4b_QCD_sr_dnn, tpr_4b_QCD_sr_dnn_proba]
fpr_list=[fpr_4b_QCD_sr_dnn, fpr_4b_QCD_sr_dnn_proba]
labels=["4b QCD sr with DNN as inputs", "4b QCD sr with DNN and probability difference as inputs"]

roc_curve_compare_weights(tpr_list, fpr_list, labels, "4b_QCD_sr","Comparison for trainings in the 4b QCD signal region", True)
roc_curve_compare_weights(tpr_list, fpr_list, labels, "4b_QCD_sr_AN","Comparison for trainings in the 4b QCD signal region", True, True)


tpr_list=[tpr_4b_QCD_sr_dnn, tpr_4b_QCD_sr_dnn_proba, tpr_4b_QCD_dnn_2, tpr_4b_QCD_dnn_proba_2]
fpr_list=[fpr_4b_QCD_sr_dnn, fpr_4b_QCD_sr_dnn_proba, fpr_4b_QCD_dnn_2, fpr_4b_QCD_dnn_proba_2]
labels=["4b QCD sr with DNN as inputs", "4b QCD sr with DNN and probability difference as inputs", "4b QCD with DNN as inputs", "4b QCD with DNN and prob diff as inputs"]

roc_curve_compare_weights(tpr_list, fpr_list, labels, "4b_QCD","Comparison for trainings with the 4b QCD signal region vs 4b inclusive", True)

roc_curve_compare_weights(tpr_list, fpr_list, labels, "4b_QCD_AN","Comparison for trainings with the 4b QCD signal region vs 4b inclusive", True, True)



# print("tpr_2b_data_full_stats_dnn_proba", tpr_2b_data_full_stats_dnn_proba)


#2b data training and evaluation on signal region

print("\n")
print("For DNN vars in signal region: \n")

score_2b_data_r_dnn_sr=AUC_scores(
    "spanet_prediction_c_dnn_2b_data_r_sr.h5", 
    "signal_region/signal_region_2b_data_reduced/output_JetGood_test.h5"
)
print(f"AUC_2b_data_r_dnn {score_2b_data_r_dnn_sr}")


score_2b_QCD_r_dnn_sr= AUC_scores(
    "spanet_prediction_c_dnn_2b_QCD_sr.h5", 
    "spanet_classifier_2b_QCD_sr/output_JetGood_test.h5"
)
print(f"AUC_2b_QCD_r_dnn {score_2b_QCD_r_dnn_sr}")


score_2b_data_f_dnn_sr= AUC_scores(
    "spanet_prediction_c_dnn_2b_data_full_sr.h5", 
    "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5"
)
print(f"AUC_2b_data_f_dnn {score_2b_data_f_dnn_sr}")


score_4b_QCD_dnn_sr= AUC_scores(
    "spanet_pred_c_4b_QCD_sr_dnn.h5", 
    "signal_region/signal_region_4b_QCD/output_JetGood_test.h5"
)
print(f"AUC_4b_QCD_dnn {score_4b_QCD_dnn_sr}")

model_ratio_dnn_sr= score_2b_QCD_r_dnn_sr/score_2b_data_r_dnn_sr
print("model_ratio_dnn_sr", model_ratio_dnn_sr)
btag_ratio_dnn_sr= score_4b_QCD_dnn_sr/score_2b_QCD_r_dnn_sr
print("btag_ratio_dnn_sr", btag_ratio_dnn_sr)

AUC_4b_data_dnn_sr= score_2b_data_f_dnn_sr * model_ratio_dnn_sr * btag_ratio_dnn_sr

print("In signal region AUC_4b_data_dnn_sr", AUC_4b_data_dnn_sr)


print("\n")
print("For DNN vars and prob diff in signal region: \n")

score_2b_data_r_dnn_proba_sr=AUC_scores("spanet_prediction_c_dnn_proba_2b_data_r_sr.h5", "signal_region/signal_region_2b_data_reduced/output_JetGood_test.h5")
print(f"AUC_2b_data_r_dnn_proba {score_2b_data_r_dnn_proba_sr}")


score_2b_QCD_r_dnn_proba_sr= AUC_scores("spanet_prediction_c_dnn_proba_2b_QCD_sr.h5", "spanet_classifier_2b_QCD_sr/output_JetGood_test.h5")
print(f"AUC_2b_QCD_r_dnn_proba {score_2b_QCD_r_dnn_proba_sr}")


score_2b_data_f_dnn_proba_sr= AUC_scores("spanet_prediction_c_dnn_proba_2b_data_full_sr.h5", "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5")
print(f"AUC_2b_data_f_dnn_proba {score_2b_data_f_dnn_proba_sr}")


score_4b_QCD_dnn_proba_sr= AUC_scores("spanet_pred_c_4b_QCD_sr_dnn_proba.h5", "signal_region/signal_region_4b_QCD/output_JetGood_test.h5")
print(f"AUC_4b_QCD_dnn_proba {score_4b_QCD_dnn_proba_sr}")

model_ratio_dnn_proba_sr= score_2b_QCD_r_dnn_proba_sr/score_2b_data_r_dnn_proba_sr
btag_ratio_dnn_proba_sr= score_4b_QCD_dnn_proba_sr/score_2b_QCD_r_dnn_proba_sr

AUC_4b_data_dnn_proba_sr= score_2b_data_f_dnn_proba_sr * model_ratio_dnn_proba_sr * btag_ratio_dnn_proba_sr

print("In signal region AUC_4b_data_dnn_proba_sr", AUC_4b_data_dnn_proba_sr)

#ROC curves comparison
# For DNN vars

fpr_2b_data_sr_dnn, tpr_2b_data_sr_dnn = fpr_tpr("spanet_prediction_c_dnn_2b_data_full_sr.h5", "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5")

roc_curve_compare_weights(tpr_2b_data_sr_dnn, fpr_2b_data_sr_dnn, "2b data sr with DNN as inputs", "2b_data_sr_dnn", "2b data sr with DNN as inputs", None, True)

#For DNN and prob diff

fpr_2b_data_sr_dnn_proba, tpr_2b_data_sr_dnn_proba = fpr_tpr(
    "spanet_prediction_c_dnn_proba_2b_data_full_sr.h5",
    "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5")

roc_curve_compare_weights(
    tpr_2b_data_sr_dnn_proba, 
    fpr_2b_data_sr_dnn_proba, 
    "2b data sr with DNN as inputs", 
    "2b_data_sr_dnn_proba", 
    "2b data sr with DNN as inputs", 
    None, 
    True
)

# DNN vs DNN+pd 
tpr_list=[tpr_2b_data_sr_dnn, tpr_2b_data_sr_dnn_proba, tpr_2b_data_dnn, tpr_2b_data_dnn_proba]
fpr_list=[fpr_2b_data_sr_dnn, fpr_2b_data_sr_dnn_proba, fpr_2b_data_dnn, fpr_2b_data_dnn_proba]
labels=["2b data SR with DNN as inputs", "2b data SR with DNN and PD as inputs", "2b data with DNN as inputs", "2b data with DNN and PD as inputs"]
roc_curve_compare_weights(tpr_list, fpr_list, labels, "2b_data_full","Comparison for trainings with the 2b data signal region vs 2b data inclusive", True)

roc_curve_compare_weights(tpr_list, fpr_list, labels, "2b_data_full_AN","Comparison for trainings with the 2b data signal region vs 2b inclusive", True, True)

#DNN sr vs DNN

tpr_list=[tpr_2b_data_sr_dnn, tpr_2b_data_dnn]
fpr_list=[fpr_2b_data_sr_dnn, fpr_2b_data_dnn]
labels=["2b data sr with DNN as inputs", "2b data with DNN as inputs"]

roc_curve_compare_weights(tpr_list, fpr_list, labels, "2b_data_dnn","Comparison for trainings with the 2b data signal region vs 2b inclusive", True)

roc_curve_compare_weights(tpr_list, fpr_list, labels, "2b_dnn_dnn_AN","Comparison for trainings with the 2b data signal region vs 2b inclusive", True, True)

tpr_list=[tpr_2b_data_dnn, tpr_2b_data_dnn_proba]
fpr_list=[fpr_2b_data_dnn, fpr_2b_data_dnn_proba]
labels=["2b data with DNN as inputs", "2b data with DNN and PD as inputs"]

roc_curve_compare_weights(tpr_list, fpr_list, labels, "2b_data_dnn_vs_pd"," ", True)

tpr_list=[tpr_2b_QCD_dnn, tpr_2b_QCD_dnn_proba]
fpr_list=[fpr_2b_QCD_dnn, fpr_2b_QCD_dnn_proba]
labels=["2b QCD with DNN as inputs", "2b QCD with DNN and PD as inputs"]

roc_curve_compare_weights(tpr_list, fpr_list, labels, "2b_QCD_dnn_vs_pd"," ", True)

tpr_list=[tpr_4b_QCD_dnn_2, tpr_4b_QCD_dnn_proba_2]
fpr_list=[fpr_4b_QCD_dnn_2, fpr_4b_QCD_dnn_proba_2]
labels=["4b QCD with DNN as inputs", "4b QCD with DNN and PD as inputs"]

roc_curve_compare_weights(tpr_list, fpr_list, labels, "4b_QCD_dnn_vs_pd"," ", True)



#Probability difference
diff_proba_weights_comparison(
    "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5",
    "spanet_prediction_c_dnn_proba_2b_data_full_sr.h5",
    "2b_data_sr",
    " "
)

diff_proba_weights_comparison( 
    "spanet_classifier_4b_QCD_working/output_JetGood_test.h5",
    "spanet_c_pred_4b_QCD_dnn.h5" ,
    "4b_QCD",
    " "
)

diff_proba_weights_comparison_arc_tan(
    "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5",
    "spanet_prediction_c_dnn_proba_2b_data_full_sr.h5",
    "2b_data_sr",
    " "
)

diff_proba_weights_comparison_arc_tan( 
    "spanet_classifier_4b_QCD_working/output_JetGood_test.h5",
    "spanet_c_pred_4b_QCD_dnn.h5" ,
    "4b_QCD",
    " "
)

diff_proba_weights_comparison_arc_tan( 
    "spanet_classifier_2b_QCD/output_JetGood_test.h5",
    "spanet_c_4v_dnn_proba_2b_QCD_new.h5" ,
    "2b_QCD",
    " "
)

diff_proba_weights_comparison_arc_tan( 
    "2b_data_full_statistics_c/output_JetGood_test.h5",
    "spanet_prediction_2b_data_f_dnn_proba.h5" ,
    "2b_data_f",
    " "
)

diff_proba_weights_comparison_arc_tan( 
    "spanet_classifier_2b_data/output_JetGood_test.h5",
    "spanet_c_4v_dnn_2b_data_out.h5" ,
    "2b_data_r",
    " "
)

#TODO:finish the plots

#Probability of classification
probabilities(
    "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5",
    "spanet_prediction_c_dnn_proba_2b_data_full_sr.h5",
    "2b_data_dnn_proba_prediction",
    "SPANet predictions with 2b data using DNN and proba diff as inputs"
)
probabilities(
    "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5",
    "spanet_prediction_c_dnn_2b_data_full_sr.h5",
    "2b_data_dnn_prediction",
    "SPANet predictions with 2b data using DNN as inputs"
)




model_ratio_dnn_sr, fpr_2b_QCD_dnn_sr, fpr_2b_data_dnn_sr, tpr_2b_QCD_dnn_sr, tpr_2b_data_dnn_sr = ratio_fpr(
    "spanet_prediction_c_dnn_2b_QCD_sr.h5", 
    "spanet_classifier_2b_QCD_sr/output_JetGood_test.h5", 
    "spanet_prediction_c_dnn_2b_data_full_sr.h5", 
    "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5", 
    x1, 
    x2
)


btag_ratio_dnn_sr, fpr_4b_QCD_dnn_sr, fpr_2b_dnn_QCD_sr, tpr_4b_QCD_dnn_sr, tpr_2b_QCD_dnn_sr = ratio_fpr(
    "spanet_pred_c_4b_QCD_sr_dnn.h5", 
    "signal_region/signal_region_4b_QCD/output_JetGood_test.h5", 
    "spanet_prediction_c_dnn_2b_QCD_sr.h5", 
    "spanet_classifier_2b_QCD_sr/output_JetGood_test.h5", 
    x1,
    x2
)

print("AUC_4b_data_dnn_sr", AUC_4b_data_dnn_sr)
print("btag_ratio_dnn_sr", btag_ratio_dnn_sr)
print("model_ratio_dnn_sr", model_ratio_dnn_sr)

fpr_ratio(fpr_2b_data_dnn_sr, 
          tpr_2b_data_dnn_sr, 
          [AUC_4b_data_dnn_sr] , 
          btag_ratio_dnn_sr, 
          model_ratio_dnn_sr, 
          "4b_data_dnn_sr", 
          x1, 
          x2, 
          True)


model_ratio_dnn_proba_sr, fpr_2b_QCD_dnn_proba_sr, fpr_2b_data_dnn_proba_sr, tpr_2b_QCD_dnn_proba_sr, tpr_2b_data_dnn_proba_sr = ratio_fpr(
    "spanet_prediction_c_dnn_proba_2b_QCD_sr.h5", 
    "spanet_classifier_2b_QCD_sr/output_JetGood_test.h5", 
    "spanet_prediction_c_dnn_proba_2b_data_full_sr.h5", 
    "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5", 
    x1, 
    x2
)


btag_ratio_dnn_proba_sr, fpr_4b_QCD_dnn_proba_sr, fpr_2b_dnn_proba_QCD_sr, tpr_4b_QCD_dnn_proba_sr, tpr_2b_QCD_dnn_proba_sr = ratio_fpr(
    "spanet_pred_c_4b_QCD_sr_dnn_proba.h5", 
    "signal_region/signal_region_4b_QCD/output_JetGood_test.h5", 
    "spanet_prediction_c_dnn_proba_2b_QCD_sr.h5", 
    "spanet_classifier_2b_QCD_sr/output_JetGood_test.h5", 
    x1,
    x2
)

print("AUC_4b_data_dnn_proba_sr", AUC_4b_data_dnn_proba_sr)


fpr_ratio(fpr_2b_data_dnn_proba_sr, 
          tpr_2b_data_dnn_proba_sr, 
          [AUC_4b_data_dnn_proba_sr] , 
          btag_ratio_dnn_proba_sr, 
          model_ratio_dnn_proba_sr, 
          "4b_data_dnn_proba_sr", 
          x1, 
          x2, 
          True)


print("\n")
print("For DNN vars in signal region with variability: \n")

score_2b_data_r_dnn_sr=AUC_scores(
    "spanet_prediction_c_dnn_2b_data_r_sr.h5", 
    "signal_region/signal_region_2b_data_reduced/output_JetGood_test.h5"
)
print(f"AUC_2b_data_r_dnn {score_2b_data_r_dnn_sr}")


score_2b_QCD_r_dnn_sr= AUC_scores(
    "spanet_prediction_c_dnn_2b_QCD_sr.h5", 
    "spanet_classifier_2b_QCD_sr/output_JetGood_test.h5"
)
print(f"AUC_2b_QCD_r_dnn {score_2b_QCD_r_dnn_sr}")


score_2b_data_f_dnn_sr= AUC_scores(
    "spanet_prediction_c_dnn_2b_data_full_sr.h5", 
    "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5"
)
print(f"AUC_2b_data_f_dnn {score_2b_data_f_dnn_sr}")


score_4b_QCD_dnn_sr_var= AUC_scores(
    "prediction_dnn_4b_QCD_sr_seeds/", 
    "signal_region/signal_region_4b_QCD/output_JetGood_test.h5",
    True)
print(f"AUC_4b_QCD_dnn {score_4b_QCD_dnn_sr_var}")

model_ratio_dnn_sr_var= score_2b_QCD_r_dnn_sr/score_2b_data_r_dnn_sr
btag_ratio_dnn_sr_var= [score_4b_QCD_dnn_proba_i/score_2b_QCD_r_dnn_sr for score_4b_QCD_dnn_proba_i in score_4b_QCD_dnn_sr_var]

AUC_4b_data_dnn_sr_var= [btag_ratio_dnn_proba_i * score_2b_data_f_dnn_sr * model_ratio_dnn_sr_var for btag_ratio_dnn_proba_i in btag_ratio_dnn_sr_var]

print( "AUC 4b data", AUC_4b_data_dnn_sr_var)


model_ratio_dnn_sr, fpr_2b_QCD_dnn_sr, fpr_2b_data_dnn_sr, tpr_2b_QCD_dnn_sr, tpr_2b_data_dnn_sr = ratio_fpr(
    "spanet_prediction_c_dnn_2b_QCD_sr.h5", 
    "spanet_classifier_2b_QCD_sr/output_JetGood_test.h5", 
    "spanet_prediction_c_dnn_2b_data_full_sr.h5", 
    "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5", 
    x1, 
    x2
)


btag_ratio_dnn_sr_var_fpr, fpr_4b_QCD_dnn_sr, fpr_2b_dnn_QCD_sr, tpr_4b_QCD_dnn_sr, tpr_2b_QCD_dnn_sr = ratio_fpr(
    "prediction_dnn_4b_QCD_sr_seeds/", 
    "signal_region/signal_region_4b_QCD/output_JetGood_test.h5", 
    "spanet_prediction_c_dnn_2b_QCD_sr.h5", 
    "spanet_classifier_2b_QCD_sr/output_JetGood_test.h5", 
    x1,
    x2,
    True
)

print("len last var",len(btag_ratio_dnn_sr_var_fpr))
print(len(AUC_4b_data_dnn_sr_var))

fpr_ratio(fpr_2b_data_dnn_sr, 
          tpr_2b_data_dnn_sr, 
          AUC_4b_data_dnn_sr_var, 
          btag_ratio_dnn_sr_var_fpr, 
          model_ratio_dnn_sr, 
          "4b_data_dnn_sr_variability", 
          x1, 
          x2,
          True
)

labels=list(range(len(tpr_4b_QCD_dnn_sr)))


roc_curve_compare_weights(
    tpr_4b_QCD_dnn_sr, 
    fpr_4b_QCD_dnn_sr, 
    labels, 
    "4b_QCD_sr_dnn_variability",
    " ", 
    True)

print("\n")
print("For DNN vars and PD in signal region with variability: \n")

score_2b_data_r_dnn_proba_sr=AUC_scores(
    "spanet_prediction_c_dnn_proba_2b_data_r_sr.h5", 
    "signal_region/signal_region_2b_data_reduced/output_JetGood_test.h5"
)
print(f"AUC_2b_data_r_dnn_proba {score_2b_data_r_dnn_proba_sr}")


score_2b_QCD_r_dnn_proba_sr= AUC_scores(
    "spanet_prediction_c_dnn_proba_2b_QCD_sr.h5", 
    "spanet_classifier_2b_QCD_sr/output_JetGood_test.h5"
)
print(f"AUC_2b_QCD_r_dnn_proba {score_2b_QCD_r_dnn_proba_sr}")


score_2b_data_f_dnn_proba_sr= AUC_scores(
    "spanet_prediction_c_dnn_proba_2b_data_full_sr.h5", 
    "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5"
)
print(f"AUC_2b_data_f_dnn_proba {score_2b_data_f_dnn_proba_sr}")


score_4b_QCD_dnn_proba_sr_var= AUC_scores(
    "prediction_dnn_proba_4b_QCD_sr_seeds/", 
    "signal_region/signal_region_4b_QCD/output_JetGood_test.h5",
    True)
print(f"AUC_4b_QCD_dnn_proba {score_4b_QCD_dnn_proba_sr_var}")

model_ratio_dnn_proba_sr_var= score_2b_QCD_r_dnn_proba_sr/score_2b_data_r_dnn_proba_sr
btag_ratio_dnn_proba_sr_var= [score_4b_QCD_dnn_proba_i/score_2b_QCD_r_dnn_proba_sr for score_4b_QCD_dnn_proba_i in score_4b_QCD_dnn_proba_sr_var]

AUC_4b_data_dnn_proba_sr_var= [btag_ratio_dnn_proba_i * score_2b_data_f_dnn_proba_sr * model_ratio_dnn_proba_sr_var for btag_ratio_dnn_proba_i in btag_ratio_dnn_proba_sr_var]

print( "AUC 4b data", AUC_4b_data_dnn_proba_sr_var)


model_ratio_dnn_proba_sr, fpr_2b_QCD_dnn_proba_sr, fpr_2b_data_dnn_proba_sr, tpr_2b_QCD_dnn_proba_sr, tpr_2b_data_dnn_proba_sr = ratio_fpr(
    "spanet_prediction_c_dnn_proba_2b_QCD_sr.h5", 
    "spanet_classifier_2b_QCD_sr/output_JetGood_test.h5", 
    "spanet_prediction_c_dnn_proba_2b_data_full_sr.h5", 
    "signal_region/spanet_full_dataset_2b_data_sr/output_JetGood_test.h5", 
    x1, 
    x2
)


btag_ratio_dnn_proba_sr_var_fpr, fpr_4b_QCD_dnn_proba_sr, fpr_2b_dnn_proba_QCD_sr, tpr_4b_QCD_dnn_proba_sr, tpr_2b_QCD_dnn_proba_sr = ratio_fpr(
    "prediction_dnn_proba_4b_QCD_sr_seeds/", 
    "signal_region/signal_region_4b_QCD/output_JetGood_test.h5", 
    "spanet_prediction_c_dnn_proba_2b_QCD_sr.h5", 
    "spanet_classifier_2b_QCD_sr/output_JetGood_test.h5", 
    x1,
    x2,
    True
)

print("len last var",len(btag_ratio_dnn_proba_sr_var_fpr))
print(len(AUC_4b_data_dnn_proba_sr_var))

fpr_ratio(fpr_2b_data_dnn_proba_sr, 
          tpr_2b_data_dnn_proba_sr, 
          AUC_4b_data_dnn_proba_sr_var, 
          btag_ratio_dnn_proba_sr_var_fpr, 
          model_ratio_dnn_proba_sr, 
          "4b_data_dnn_proba_sr_variability", 
          x1, 
          x2,
          True
)

roc_curve_compare_weights(
    tpr_4b_QCD_dnn_proba_sr, 
    fpr_4b_QCD_dnn_proba_sr, 
    labels, 
    "4b_QCD_sr_dnn_proba_variability",
    " ", 
    True
)

# fpr_2b_data_dnn, tpr_2b_data_dnn= fpr_tpr_variability(
#     "prediction__2b_data_dnn_vars_seeds", 
#     "spanet_classifier_2b_data/output_JetGood_test.h5")
# fpr_2b_data_dnn_proba, tpr_2b_data_dnn_proba= fpr_tpr_variability(
#     "prediction__2b_data_dnn_proba_vars_seeds", 
#     "spanet_classifier_2b_data/output_JetGood_test.h5")

# fpr_2b_QCD_dnn, tpr_2b_QCD_dnn= fpr_tpr_variability(
#     "prediction__2b_QCD_dnn_vars_seeds", 
#     "spanet_classifier_2b_data/output_JetGood_test.h5")
# fpr_2b_QCD_dnn_proba, tpr_2b_QCD_dnn_proba= fpr_tpr_variability(
#     "prediction__2b_QCD_dnn_proba_vars_seeds", 
#     "spanet_classifier_2b_data/output_JetGood_test.h5")



# Redifne the AUC so that they are with the seeds  

# roc_curve_compare_weights(
#     tpr_2b_QCD_dnn_proba, 
#     fpr_2b_QCD_dnn_proba, 
#     labels, 
#     "2b_QCD_dnn_proba_variability",
#     " ", 
#     True
# )

# roc_curve_compare_weights(
#     tpr_2b_QCD_dnn, 
#     fpr_2b_QCD_dnn, 
#     labels, 
#     "2b_QCD_dnn_variability",
#     " ", 
#     True
# )

# roc_curve_compare_weights(
#     tpr_2b_data_dnn_proba, 
#     fpr_2b_data_dnn_proba, 
#     labels, 
#     "2b_data_dnn_proba_variability",
#     " ", 
#     True
# )

# roc_curve_compare_weights(
#     tpr_2b_data_dnn, 
#     fpr_2b_data_dnn, 
#     labels, 
#     "2b_data_dnn_variability",
#     " ", 
#     True
# )

# roc_curve_compare_weights(
#     tpr_4b_QCD_dnn, 
#     fpr_4b_QCD_dnn, 
#     labels, 
#     "4b_QCD_dnn_variability",
#     " ", 
#     True
# )
# roc_curve_compare_weights(
#     tpr_4b_QCD_dnn_proba, 
#     fpr_4b_QCD_dnn_proba, 
#     labels, 
#     "4b_QCD_dnn_proba_variability",
#     " ", 
#     True
# )



#TODO: oversampling but just need the FPR and TPR and plot the curve

probabilities(
    "signal_region/signal_region_4b_QCD/output_JetGood_test.h5",
    "spanet_pred_c_4b_QCD_sr_dnn_proba.h5", 
    "4b_QCD_sr_dnn_proba",
    ""
)

probabilities(
    "spanet_classifier_4b_QCD_working/output_JetGood_test.h5",
    "spanet_c_pred_4b_QCD_dnn.h5", 
    "4b_QCD_dnn",
    ""
)

probabilities(
    "spanet_classifier_4b_QCD_working/output_JetGood_test.h5",
    "spanet_c_pred_4b_QCD_dnn_proba.h5", 
    "4b_QCD_dnn_proba",
    ""
)


pred_file= "/eos/home-r/ramellar/4_classification/classification_predictions/" + "spanet_c_pred_4b_QCD_dnn.h5"
pred_file_2= "/eos/home-r/ramellar/4_classification/classification_predictions/" + "spanet_c_pred_4b_QCD_dnn_proba.h5"
test_file="/eos/home-r/ramellar/5_inputs/spanet_classifier_4b_QCD_working/output_JetGood_test.h5"

df_pred= h5py.File(pred_file, "r")
df_pred_2= h5py.File(pred_file_2, "r")
df_test= h5py.File(test_file, "r")

list_pred=[df_pred,df_pred_2]
list_test=[df_test,df_test]
labels_list=["4b QCD with DNN as inputs", "4b QCD with DNN and probability difference as inputs"]

roc_curve_compare_no_weights(df_test, list_pred, labels_list, "4b_QCD","")

fpr_ratio_2(fpr_2b_data_dnn_proba_sr, 
          tpr_2b_data_dnn_proba_sr, 
          fpr_2b_data_dnn_sr, 
          tpr_2b_data_dnn_sr,
          AUC_4b_data_dnn_proba_sr_var, 
          AUC_4b_data_dnn_sr_var,
          btag_ratio_dnn_proba_sr_var_fpr,
          btag_ratio_dnn_sr_var_fpr, 
          model_ratio_dnn_proba_sr, 
          model_ratio_dnn_sr,
          "4b_data_max_comp", 
          x1, 
          x2,
          True
)