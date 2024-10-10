import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mplhep as hep
hep.style.use(hep.style.ROOT)

import matplotlib
matplotlib.rcParams["figure.dpi"] = 300


#Loading files
#no specific label, means with "normal" pt, btag, eta, phi
path="/eos/home-r/ramellar/tensorboard_training_csv/validation_accuracy/"
out_4j_1= path + "run-version_4_event_files_hh4b_4jets-tag-validation_accuracy.csv"
out_4j_2= path + "run-version_0_4jets-tag-validation_accuracy.csv"
out_5j= path+ "run-version_1_5jets-tag-validation_accuracy.csv"
out_4j_5g= path+ "run-version_3_4jets5global_9999pad-tag-validation_accuracy.csv"


data_4j_1= pd.read_csv(out_4j_1)
data_4j_2= pd.read_csv(out_4j_2)
data_5j= pd.read_csv(out_5j)
data_4j_5g= pd.read_csv(out_4j_5g)



import os


def load_csv_files_from_folder(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    # Filter out the CSV files
    csv_files = [f for f in files if f.endswith('.csv')]

    # Initialize an empty list to store dataframes
    dataframes = []

    # Loop through each CSV file and read it into a dataframe
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
        print(f"Loaded {csv_file}")

    return dataframes


# folder_path = '/eos/home-r/ramellar/4_classification/Seeds_dnn'
# dataframes = load_csv_files_from_folder(folder_path)

folder_path = '/eos/home-r/ramellar/Variability_tests'
dataframes = load_csv_files_from_folder(folder_path)
numbers=list(range(len(dataframes)))

#Plotting

def validation_accuracy(data, labels, name, colors):
    column1 = [d['Step'].values for d in data]
    column2 = [d['Value'].values for d in data]
    fig=plt.figure()
    plt.figure(figsize=(10, 7))
    for i in range(len(data)):
        plt.plot(column1[i], column2[i], label=f"{labels[i]}", linewidth=2, color=colors[i])
        plt.legend(loc="lower right", frameon=True,  bbox_to_anchor=(1, 0), fontsize='xx-small')
        plt.grid(linestyle=":")
        plt.ylabel("Validation accuracy", loc="center")
        plt.xlabel("Batch", loc="center")
        # plt.ylim(top=0.98)
        plt.ylim(bottom=0.89)
        # plt.xlim(15000)
    hep.cms.label(
            year="2022",
            com="13.6",
            label=f"Private Work",
        )
    plt.savefig(f"/afs/cern.ch/user/r/ramellar/HH4b_SPANet/tensorboard_info/tensorboard_comparisons_plots/{name}")
    return ""

data=[data_4j_2, data_4j_5g, data_5j]
labels=["4 jets", "4 jets 5 global", "5 jets"]
color=["orange", "c","hotpink"]


validation_accuracy(data,labels, "comp_4_4j5_5_zoom", color)
# validation_accuracy(dataframes, numbers, "seeds_5j_variability")