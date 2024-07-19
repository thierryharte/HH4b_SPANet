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

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-t", "--true", type=str, required=True, help="True file"
)
arg_parser.add_argument(
    "-p", "--prediction", type=str, required=True, help="Folder with prediction files"
)
arg_parser.add_argument(
    "-n", "--name", type=str, required=True, help="Plot name"
)
arg_parser.add_argument(
    "-l", "--title", type=str, required=True, help="Title of the plot"
)

arg_parser.add_argument(
    "-w", "--weights", action="store_true", default=False, help="Compute the ROC with weights"
)
args = arg_parser.parse_args()

#Load datatasets

true_file= args.true

# true_file= "/afs/cern.ch/user/r/ramellar/public/inputs_files/spanet_classifier_4b_QCD/output_JetGood_test.h5"
df_true= h5py.File(true_file, "r")



def load_h5_files_from_folder(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    # Filter out the h5 files
    h5_files = [f for f in files if f.endswith('.h5')]

    # Initialize an empty list to store dataframes
    dataframes = []

    # Loop through each h5 file and read it into a dataframe
    for h5_file in h5_files:
        file_path = os.path.join(folder_path, h5_file)
        df =  h5py.File(file_path, "r")
        dataframes.append(df)
        print(f"Loaded {h5_file}")

    return dataframes

# folder_path = '/eos/home-r/ramellar/4_classification/variability_study/predictions/prediction__4b_QCD_dnn_vars_seeds/'
folder_path = args.prediction
dataframes = load_h5_files_from_folder(folder_path)
numbers=list(range(len(dataframes)))



def roc_curve_compare_no_weights(list_test,list_pred,list_labels,name, title):
    true= list_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]
    proba_signal= [pred["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()] for pred in list_pred]
    print(len(proba_signal))
    score= [roc_auc_score(true, proba_signal[i]) for i in range(len(proba_signal))]
    print(score)
    print("max", max(score))
    print("min", min(score))
    fpr, tpr, thresholds = zip(*[roc_curve(true, proba_signal[i]) for i in range(len(proba_signal))])
    fig=plt.figure()
    for i in range(len(fpr)):
        plt.plot(tpr[i],fpr[i],label=f"{list_labels[i]} with AUC={score[i]:.3f}")
        plt.legend(fontsize= 'small', loc="center left")
        plt.xlabel("tpr")
        plt.ylabel("fpr")
        plt.grid(linestyle=":")
        plt.title(f"{title}", fontsize= 'small')
    plt.savefig(f"/afs/cern.ch/user/r/ramellar/HH4b_SPANet/ROC_curves/{name}_no_w")
    plt.close(fig)
    
def roc_curve_compare_weights(list_test,list_pred,list_labels,name, title):
    true= list_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]
    proba_signal= [pred["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()] for pred in list_pred]
    print(len(proba_signal))
    weights= list_test["WEIGHTS"]['weight'][()]
    score= [roc_auc_score(true, proba_signal[i], sample_weight= weights) for i in range(len(proba_signal))]
    print("max", max(score))
    print("min", min(score))
    print(score)
    fpr, tpr, thresholds = zip(*[roc_curve(true, proba_signal[i], sample_weight=weights) for i in range(len(proba_signal))])
    fig=plt.figure()
    for i in range(len(fpr)):
        plt.plot(tpr[i],fpr[i],label=f"{list_labels[i]} with AUC={score[i]:.3f}")
        plt.legend(fontsize= 'small')
        plt.xlabel("tpr")
        plt.ylabel("fpr")
        plt.grid(linestyle=":")
        plt.title(f"{title}", fontsize= 'small')
    plt.savefig(f"/afs/cern.ch/user/r/ramellar/HH4b_SPANet/ROC_curves/{name}_w")
    plt.close(fig)
    

# roc_curve_compare_no_weights(df_true,dataframes ,numbers, "seed_comparison_4b_QCD_dnn", "Comparision of 4b-QCD trainings with dnn vars as inputs using different seeds")
# roc_curve_compare_weights(df_true,dataframes ,numbers, "seed_comparison_4b_QCD_dnn", "Comparision of 4b-QCD trainings with dnn vars as inputs using different seeds")

if args.weights:
   roc_curve_compare_weights(df_true,dataframes ,numbers, args.name, args.title)
else:
    roc_curve_compare_no_weights(df_true,dataframes ,numbers, args.name, args.title)