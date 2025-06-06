from collections import defaultdict
import awkward as ak
import numba
import numpy as np
import pandas as pd
import re
import h5py
import vector

vector.register_numba()
vector.register_awkward()

import matplotlib.pyplot as plt

# from matplotlib.colors import LogNorm
# import mplhep as hep
# hep.style.use(hep.style.ROOT)

from infer_4b_data_functions import roc_curve_compare_weights

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib

matplotlib.rcParams["figure.dpi"] = 300
import os
from scipy.integrate import trapz

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-t", "--true", type=str, required=True, help="True file")
arg_parser.add_argument(
    "-tp",
    "--true_pd",
    type=str,
    required=True,
    help="True file for with probability_difference",
)
arg_parser.add_argument(
    "-p", "--prediction", type=str, required=True, help="Folder with prediction files"
)
arg_parser.add_argument("-n", "--name", type=str, required=True, help="Plot name")
arg_parser.add_argument(
    "-l", "--title", type=str, required=True, help="Title of the plot"
)

arg_parser.add_argument(
    "-w",
    "--weights",
    type=bool,
    required=True,
    default=False,
    help="Compute the ROC with weights",
)
arg_parser.add_argument(
    "-c",
    "--compare",
    type=bool,
    required=True,
    default=True,
    help="Compare the ROC with other method",
)
args = arg_parser.parse_args()

# Load datatasets

true_file = args.true

# true_file= "/afs/cern.ch/user/r/ramellar/public/inputs_files/spanet_classifier_4b_QCD/output_JetGood_test.h5"
df_true = h5py.File(true_file, "r")
df_true_pd = h5py.File(args.true_pd, "r")


def load_h5_files_from_folder(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    # Filter out the h5 files
    h5_files = [f for f in files if f.endswith(".h5")]
    numbers = [get_fileseed(x) for x in h5_files]

    # Initialize an empty list to store dataframes
    dataframes = []

    # Loop through each h5 file and read it into a dataframe
    for idx, h5_file in enumerate(h5_files):
        file_path = os.path.join(folder_path, h5_file)
        df = h5py.File(file_path, "r")
        dataframes.append(df)
        print(f"Loaded {h5_file}")
        # Checking, if probability_diference file, then change naming
        if "pd" in h5_file:
            numbers[idx] = f"{numbers[idx]}_pd"
            if "arctan" in h5_file:
                numbers[idx] = f"{numbers[idx]}_arctan"

    return numbers, dataframes


def get_fileseed(filename):
    match = re.search(r"seed(\d+)\.h5$", filename)
    if match:
        return int(match.group(1))
    else:
        return None


# folder_path = '/eos/home-r/ramellar/4_classification/variability_study/predictions/prediction__4b_QCD_dnn_vars_seeds/'
folder_path = args.prediction
numbers, dataframes = load_h5_files_from_folder(folder_path)

def roc_curve_compare_no_weights(list_test, list_pred, list_labels, name, title):
    true = list_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]
    proba_signal = [
        pred["CLASSIFICATIONS"]["EVENT"]["signal"][:, 1][()] for pred in list_pred
    ]
    print(len(proba_signal))
    score = [roc_auc_score(true, proba_signal[i]) for i in range(len(proba_signal))]
    print(score)
    print("max", max(score))
    print("min", min(score))
    fpr, tpr, thresholds = zip(
        *[roc_curve(true, proba_signal[i]) for i in range(len(proba_signal))]
    )
    fig = plt.figure()
    for i in range(len(fpr)):
        plt.plot(tpr[i], fpr[i], label=f"{list_labels[i]} with AUC={score[i]:.3f}")
        plt.legend(fontsize="small", loc="center left")
        plt.xlabel("tpr")
        plt.ylabel("fpr")
        plt.grid(linestyle=":")
        plt.title(f"{title}", fontsize="small")
    plt.savefig(f"{name}_no_w")
    plt.close(fig)


def roc_curve_compare_weights(
    list_test, list_test_pd, list_pred, list_labels, name, title, compare=True
):
    print(list_labels[1])
    true = list_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]
    true_pd = list_test_pd["CLASSIFICATIONS"]["EVENT"]["signal"][()]
    proba_signal = [
        pred["CLASSIFICATIONS"]["EVENT"]["signal"][:, 1][()] for pred in list_pred
    ]
    print(len(proba_signal))
    weights = list_test["WEIGHTS"]["weight"][()]
    print(len(true))
    print([len(x) for x in proba_signal])
    print(len(weights))

    scores = {}
    cutoff_fpr = 1e-2
    

    roc_curve_list = []
    for i in range(len(proba_signal)):
        if 'pd' in str(list_labels[i]):
            roc_curve_list.append(roc_curve(true_pd, proba_signal[i], sample_weight=weights))
        else:
            roc_curve_list.append(roc_curve(true, proba_signal[i], sample_weight=weights))
    print(roc_curve_list)
    fpr, tpr, thresholds = zip(*roc_curve_list)
    
    for i in range(len(proba_signal)):
        fpr_part = [t for t in fpr[i] if t <= cutoff_fpr]
        tpr_part = tpr[i][: len(fpr_part)]
        scores[i] = auc(fpr_part, tpr_part)
    print("max", max(scores.values()))
    print("min", min(scores.values()))
    if compare:
        roc_points_0 = "/eos/user/t/tharte/Analysis_Project/samplesets/raffaele/roc_points_era_1_fold_0.npy"
        df_roc_0 = np.load(roc_points_0)
        fpr_r = df_roc_0[:, 0]
        tpr_r = df_roc_0[:, 1]
        fpr_part = [t for t in fpr_r if t <= cutoff_fpr]
        tpr_part = tpr_r[: len(fpr_part)]
        score_2_int = trapz(tpr_part, fpr_part)
        score_2 = auc(fpr_part, tpr_part)
        print(score_2_int)
        print(score_2)
    fig = plt.figure()
    idx_max = max(scores, key=scores.get)
    idx_min = min(scores, key=scores.get)
    idx_pd = [i for i, name in enumerate(list_labels) if 'pd' in str(name)]
    for i in [idx_max, *idx_pd, idx_min]:
        if "pd" in str(list_labels[i]):
            if "arctan" in str(list_labels[i]):
                plt.plot(
                    tpr[i],
                    fpr[i],
                    label=f"with pd_arctan, seed: {list_labels[i].split('_')[0]} with AUC(fpr[0,1e-2])={scores[i]:.2e}",
                )
            else:
                plt.plot(
                    tpr[i],
                    fpr[i],
                    label=f"with pd, seed: {list_labels[i].split('_')[0]} with AUC(fpr[0,1e-2])={scores[i]:.2e}",
                )
        else:
            plt.plot(
                tpr[i],
                fpr[i],
                label=f"normal inputs, seed: {list_labels[i]} with AUC(fpr[0,1e-2])={scores[i]:.2e}",
            )
        # plt.plot(tpr[i],fpr[i],label=f"Training Seed {list_labels[i]} with AUC(fpr[0,1e-2])={scores[i]:.2e}")
    plt.plot(
        tpr_r,
        fpr_r,
        label=f"AUC(fpr[0,1e-2])={score_2:.2e} from the AN",
        linewidth=1,
        color="red",
    )
    plt.legend(loc="lower right", fontsize="small")
    plt.xlabel("tpr")
    plt.ylabel("fpr")
    plt.yscale("log")
    plt.xlim(0, 0.4)
    plt.ylim(1e-6, 1e-1)
    plt.grid(linestyle=":")
    plt.title(f"{title}", fontsize="small")
    plt.savefig(f"{name}_w")
    plt.close(fig)


# roc_curve_compare_no_weights(df_true,dataframes ,numbers, "seed_comparison_4b_QCD_dnn", "Comparision of 4b-QCD trainings with dnn vars as inputs using different seeds")
# roc_curve_compare_weights(df_true,dataframes ,numbers, "seed_comparison_4b_QCD_dnn", "Comparision of 4b-QCD trainings with dnn vars as inputs using different seeds")

if args.weights:
    roc_curve_compare_weights(df_true, df_true_pd, dataframes, numbers, args.name, args.title)
else:
    roc_curve_compare_no_weights(df_true, df_true_pd, dataframes, numbers, args.name, args.title)
