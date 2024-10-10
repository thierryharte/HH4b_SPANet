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
path="/afs/cern.ch/user/r/ramellar/HH4b_SPANet/tensorboard_info/tensorboard_training_csv/loss_info/"
out_4j_1= path + "version_4_event_files_hh4b_4jets.csv"
out_4j_2= path + "version_0_4jets.csv"
out_5j= path+ "version_1_5jets.csv"
out_4j_5g= path+ "version_3_4jets5global_9999pad.csv"


data_4j_1= pd.read_csv(out_4j_1)
data_4j_2= pd.read_csv(out_4j_2)
data_5j= pd.read_csv(out_5j)
data_4j_5g= pd.read_csv(out_4j_5g)

#Plotting

def loss_f(data, labels, name, color):
    column1 = [d['Step'].values for d in data]
    column2 = [d['Value'].values for d in data]
    fig=plt.figure()
    for i in range(len(data)):
        plt.plot(column1[i], column2[i], label=f"{labels[i]}", linewidth=2, color= color[i])
        plt.legend(loc="upper right", frameon=True)
        plt.grid(linestyle=":")
        plt.ylabel("Validation accuracy", loc="center")
        plt.xlabel("Batch", loc="center")
        plt.ylim(top=0.015)
    hep.cms.label(
            year="2022",
            com="13.6",
            label=f"Private Work",
        )
    plt.savefig(f"/afs/cern.ch/user/r/ramellar/HH4b_SPANet/tensorboard_info/loss_plots/{name}")
    return ""

data=[data_4j_2, data_4j_5g, data_5j]
labels=["4 jets", "4 jets 5 global", "5 jets"]
color=["orange", "blue","crimson"]


loss_f(data,labels, "comp_4_4j5_5_loss_zoom", color)