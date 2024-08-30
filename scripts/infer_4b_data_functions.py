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
import mplhep as hep
#hep.style.use(hep.style.ROOT)

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib
matplotlib.rcParams["figure.dpi"] = 300
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


from matplotlib import style

custom_style = {
    'axes.prop_cycle': plt.cycler(color=['lightseagreen', 'darkorange', "hotpink", 'yellowgreen' , "forestgreen"  ]),
    'lines.linewidth': 1,
}

# Apply the custom style
plt.style.use(custom_style)

# import mplhep as hep
# hep.style.use(hep.style.ROOT)


import argparse


path_p= "/eos/home-r/ramellar/4_classification/classification_predictions/"
path_v= "/eos/home-r/ramellar/4_classification/variability_study"
path_i= "/eos/home-r/ramellar/5_inputs/"


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
        # print(f"Loaded {h5_file}")

    return dataframes

#Function to compute the AUC scores
def AUC_scores(prediction_path, input_path, variablility_path=None):
    
    filename_test= path_i + input_path
    df_test= h5py.File(filename_test, "r")

    
    if variablility_path is not None:
        folder_path = f"/eos/home-r/ramellar/4_classification/variability_study/predictions/{prediction_path}"
        dataframes= load_h5_files_from_folder(folder_path)
        true= df_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]
        proba_signal= [pred["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()] for pred in dataframes]
        weights= df_test["WEIGHTS"]['weight'][()]
        score= [roc_auc_score(true, proba_signal[i], sample_weight=weights) for i in range(len(proba_signal))]
    else:
        filename_pred= path_p + prediction_path
        df_pred= h5py.File(filename_pred, "r")
        true= df_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]
        proba_signal=df_pred["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()] 
        weights= df_test["WEIGHTS"]['weight'][()]
        score= roc_auc_score(true,proba_signal, sample_weight= weights)
        
    return score


def mean_element(array, desired_events):
    array = ak.Array(array)
    # print("len",len(array))
    count=int(len(array)//desired_events)
    # print("count",count)
    # print(len(array)%desired_events)
    
    length= np.ones(desired_events, dtype= int)* (count)
    stop=(len(array)-desired_events*count)
    index=np.arange(0,stop, 1)
    length[index]=count+1
    
    array_unflatten=ak.unflatten(array, length)
    
    
    avg=ak.mean(array_unflatten, axis=1)
    avg=ak.to_numpy(avg)
    avg= np.array(avg, dtype= "d")
    return avg

def fpr_tpr(prediction, test):
    filename_test= path_i + test
    df_test= h5py.File(filename_test, "r")
    filename_pred= path_p + prediction
    df_pred= h5py.File(filename_pred, "r")
    
    true= df_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]
    proba_signal= df_pred["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()]
    weights= df_test["WEIGHTS"]['weight'][()]

    fpr, tpr, thresholds_n = roc_curve(true, proba_signal, sample_weight=weights)
    return fpr, tpr
    
def fpr_tpr_variability(prediction, test):
    filename_test_num= path_i + test
    df_test_num= h5py.File(filename_test_num, "r")
    folder_path = f"/eos/home-r/ramellar/4_classification/variability_study/predictions/{prediction}"
    dataframes= load_h5_files_from_folder(folder_path)
    true_n= df_test_num["CLASSIFICATIONS"]["EVENT"]["signal"][()]
    proba_signal_n= [pred["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()] for pred in dataframes]
    weights_n= df_test_num["WEIGHTS"]['weight'][()]

    fpr_n, tpr_n, thresholds_n = zip(*[roc_curve(true_n, proba_signal_n[i], sample_weight=weights_n) for i in range(len(proba_signal_n))])

    return fpr_n, tpr_n
    

path_p= "/eos/home-r/ramellar/4_classification/classification_predictions/"
path_v= "/eos/home-r/ramellar/4_classification/variability_study"
path_i= "/eos/home-r/ramellar/5_inputs/"

def ratio_fpr(num, i_num, den, i_den, x1, x2 ,variability=None):
    
    if variability is not None:
        filename_test_num= path_i + i_num
        df_test_num= h5py.File(filename_test_num, "r")
        folder_path = f"/eos/home-r/ramellar/4_classification/variability_study/predictions/{num}"
        dataframes= load_h5_files_from_folder(folder_path)
        true_n= df_test_num["CLASSIFICATIONS"]["EVENT"]["signal"][()]
        proba_signal_n= [pred["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()] for pred in dataframes]
        weights_n= df_test_num["WEIGHTS"]['weight'][()]
        
        filename_test_den= path_i + i_den
        df_test_den= h5py.File(filename_test_den, "r")
        filename_pred_den= path_p + den
        df_pred_den= h5py.File(filename_pred_den, "r")
        
        true_d= df_test_den["CLASSIFICATIONS"]["EVENT"]["signal"][()]
        proba_signal_d= df_pred_den["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()]
        weights_d= df_test_den["WEIGHTS"]['weight'][()]
        fp_array=np.array([])
        
        fpr_n, tpr_n, thresholds_n = zip(*[roc_curve(true_n, proba_signal_n[i], sample_weight=weights_n) for i in range(len(proba_signal_n))])
        fpr_d, tpr_d, thresholds_d = roc_curve(true_d, proba_signal_d, sample_weight=weights_d)
        
        # print("len",len(fpr_n))
        fpr_d_int=interp1d(tpr_d,fpr_d)
        
        x=np.concatenate((x2,x1))
       
        spline_fpr_d= np.array(fpr_d_int(x))
            
        spline_variability=[]
        for i in range(len(fpr_n)):
            fpr_n_int=interp1d(tpr_n[i],fpr_n[i])
            # print("fpr", fpr_n_int)
            spline_fpr_n= np.array(fpr_n_int(x))
            # print("spline", len(spline_fpr_n))
            spline_variability.append(spline_fpr_n)
            # print("spline_variability", len(spline_variability))
    
        # fpr_n= spline_variability
        
        fpr_ratio= []
        
        for i in range(len(spline_variability)):
            btag_ratio_v= np.nan_to_num(spline_variability[i]/spline_fpr_d,0)
            # print("btag_ratio_v", btag_ratio_v)
            fpr_ratio.append(btag_ratio_v)
            
        # fpr_n=np.array(fpr_n)
        # fpr_d= np.array(fpr_d)    
    
            
    else:
        filename_test_num= path_i + i_num
        df_test_num= h5py.File(filename_test_num, "r")
        filename_pred_num= path_p + num
        df_pred_num= h5py.File(filename_pred_num, "r")
        
        filename_test_den= path_i + i_den
        df_test_den= h5py.File(filename_test_den, "r")
        filename_pred_den= path_p + den
        df_pred_den= h5py.File(filename_pred_den, "r")
        
        true_n= df_test_num["CLASSIFICATIONS"]["EVENT"]["signal"][()]
        proba_signal_n= df_pred_num["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()]
        weights_n= df_test_num["WEIGHTS"]['weight'][()]
           
        true_d= df_test_den["CLASSIFICATIONS"]["EVENT"]["signal"][()]
        proba_signal_d= df_pred_den["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()]
        weights_d= df_test_den["WEIGHTS"]['weight'][()]
        
        fpr_n, tpr_n, thresholds_n = roc_curve(true_n, proba_signal_n, sample_weight=weights_n)
        fpr_d, tpr_d, thresholds_d = roc_curve(true_d, proba_signal_d, sample_weight=weights_d)
        
        fpr_n_int=interp1d(tpr_n,fpr_n)
        fpr_d_int=interp1d(tpr_d,fpr_d)
        
        
        x=np.concatenate((x2,x1))
        spline_fpr_n= np.array(fpr_n_int(x))
        spline_fpr_d= np.array(fpr_d_int(x))
        
        # print(spline_fpr_n)
        
        fpr_ratio= np.nan_to_num(spline_fpr_n/spline_fpr_d,0)
        
    
    
    return fpr_ratio, fpr_n, fpr_d, tpr_n, tpr_d


# def ratio_fpr(num, i_num, den, i_den, desired_lenght,variability=None):
    
#     if variability is not None:
#         filename_test_num= path_i + i_num
#         df_test_num= h5py.File(filename_test_num, "r")
#         folder_path = f"/eos/home-r/ramellar/4_classification/variability_study/predictions/{num}"
#         dataframes= load_h5_files_from_folder(folder_path)
#         true_n= df_test_num["CLASSIFICATIONS"]["EVENT"]["signal"][()]
#         proba_signal_n= [pred["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()] for pred in dataframes]
#         weights_n= df_test_num["WEIGHTS"]['weight'][()]
        
#         filename_test_den= path_i + i_den
#         df_test_den= h5py.File(filename_test_den, "r")
#         filename_pred_den= path_p + den
#         df_pred_den= h5py.File(filename_pred_den, "r")
        
#         true_d= df_test_den["CLASSIFICATIONS"]["EVENT"]["signal"][()]
#         proba_signal_d= df_pred_den["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()]
#         weights_d= df_test_den["WEIGHTS"]['weight'][()]
#         fp_array=np.array([])
        
#         fpr_n, tpr_n, thresholds_n = zip(*[roc_curve(true_n, proba_signal_n[i], sample_weight=weights_n) for i in range(len(proba_signal_n))])
#         fpr_d, tpr_d, thresholds_d = roc_curve(true_d, proba_signal_d, sample_weight=weights_d)
        
#         fpr_n_av=[]
#         tpr_n_av=[]
#         for i in range(len(fpr_n)):
#             fpr_n_e= mean_element(fpr_n[i], desired_lenght)
#             fpr_n_av.append(fpr_n_e)
#             tpr_n_e= mean_element(tpr_n[i], desired_lenght)
#             tpr_n_av.append(tpr_n_e)
        
#         fpr_d= mean_element(fpr_d, desired_lenght)
#         tpr_d= mean_element(tpr_d, desired_lenght)
#         fpr_n= fpr_n_av
#         tpr_n= tpr_n_av
        
#         fpr_ratio= []
        
#         for i in range(len(fpr_n_av)):
#             btag_ratio_v= np.nan_to_num(fpr_n_av[i]/fpr_d)
#             fpr_ratio.append(btag_ratio_v)
            
#         # fpr_n=np.array(fpr_n)
#         # fpr_d= np.array(fpr_d)    
    
            
#     else:
#         filename_test_num= path_i + i_num
#         df_test_num= h5py.File(filename_test_num, "r")
#         filename_pred_num= path_p + num
#         df_pred_num= h5py.File(filename_pred_num, "r")
        
#         filename_test_den= path_i + i_den
#         df_test_den= h5py.File(filename_test_den, "r")
#         filename_pred_den= path_p + den
#         df_pred_den= h5py.File(filename_pred_den, "r")
        
#         true_n= df_test_num["CLASSIFICATIONS"]["EVENT"]["signal"][()]
#         proba_signal_n= df_pred_num["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()]
#         weights_n= df_test_num["WEIGHTS"]['weight'][()]
           
#         true_d= df_test_den["CLASSIFICATIONS"]["EVENT"]["signal"][()]
#         proba_signal_d= df_pred_den["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()]
#         weights_d= df_test_den["WEIGHTS"]['weight'][()]
        
#         fpr_n, tpr_n, thresholds_n = roc_curve(true_n, proba_signal_n, sample_weight=weights_n)
#         fpr_d, tpr_d, thresholds_d = roc_curve(true_d, proba_signal_d, sample_weight=weights_d)
        

        
#         fpr_n= mean_element(fpr_n, desired_lenght)
#         fpr_d= mean_element(fpr_d, desired_lenght)
#         tpr_n= mean_element(tpr_n, desired_lenght)
#         tpr_d= mean_element(tpr_d, desired_lenght)
#         fpr_ratio= np.nan_to_num(fpr_n/fpr_d,0)
        
    
    
#     return fpr_ratio, fpr_n, fpr_d, tpr_n, tpr_d

def fpr_ratio(fpr, tpr, AUC_score, ratio_btag_fpr, model_ratio_fpr,  name, x1, x2, comparison=False):
    #Comparison with the AN
    if comparison is not False:
        roc_points_0= path_i + "roc_raffaele/roc_points_era_1_fold_0.npy"
        roc_points_1= path_i + "roc_raffaele/roc_points_era_1_fold_1.npy"
        df_roc_0 = np.load(roc_points_0)
        df_roc_1 = np.load(roc_points_1)
        fpr_r=df_roc_0[:,0]
        tpr_r=df_roc_0[:,1]
        fpr_r_1=df_roc_1[:,0]
        tpr_r_1=df_roc_1[:,1]
    
        score_2=auc(fpr_r, tpr_r)
        score_3=auc(fpr_r_1, tpr_r_1)

        fig=plt.figure()
        # tpr=ak.to_numpy(tpr)
        # fpr=ak.to_numpy(fpr)
        
        fpr_int=interp1d(tpr,fpr)
        x=np.concatenate((x2,x1))
        
        # print("fpr_array", fpr_array)
        
        fpr_2b_data_full_stats_dnn, tpr_2b_data_full_stats_dnn= fpr_tpr("spanet_prediction_2b_data_f_dnn.h5", "2b_data_full_statistics_c/output_JetGood_test.h5")
        # print(x)
        # print("AUC test", auc(fpr_array[0], x ))
        
        # sorted_indices = np.argsort(x)
        # fpr_sorted = fpr_array[0][sorted_indices]
        # x_sorted = x[sorted_indices]
        
        # print("auc test", auc(fpr_sorted, x_sorted ))
        
        # new_auc_scores=[auc(fpr_array[i], x ) for i in range(len(fpr_array))]
        # # print(fpr_array[0])
        # print("fpr array",fpr_array )
        
        max_auc_index= np.argmax(AUC_score)
        min_auc_index= np.argmin(AUC_score)
        
        # print(len(AUC_score))
        
        if len(AUC_score)>1:
            spline_fpr=np.array(fpr_int(x))
            fpr_array= [spline_fpr * (model_ratio_fpr) * (ratio_btag_fpr_v) for ratio_btag_fpr_v in ratio_btag_fpr]
            fpr_array= [np.where(fpr_v>1, 1, fpr_v) for fpr_v in fpr_array]
            fig=plt.figure()
            for i in range(len(fpr_array)):
                if i==max_auc_index or i==min_auc_index:
                    plt.plot(x,fpr_array[i], label= f"AUC = {AUC_score[i]:.3f}", linewidth=1)
                # plt.plot(tpr,fpr_array[i], linewidth=1)
            # plt.plot(tpr_2b_data_full_stats_dnn,fpr_2b_data_full_stats_dnn, label= "2b")
            plt.plot(tpr_r_1,fpr_r_1,label=f"AUC={score_3:.3f} from the AN", linewidth=1, color="yellowgreen")
            # plt.plot(tpr_r_1,fpr_r_1,label=f"AUC={score_2:.3f} from the AN", linewidth=1)
            plt.legend(fontsize= 'small')
            plt.xlabel("tpr")
            plt.ylabel("fpr")
    
            # plt.yscale("log")
            # plt.ylim(bottom=7.2*10**(-6), top= 7.3*10**(-6))
            plt.grid(linestyle=":")
            
            # if "data" in fpr:
            #     hep.cms.label(
            #                 year="2022",
            #                 com="13.6",
            #                 label=f"Private Work",
            #                 data=True
            #         )
            
            # else:
            #     hep.cms.label(
            #                 year="2022",
            #                 com="13.6",
            #                 label=f"Private Work",
            #             )
                
            hep.cms.label(
                year="2022",
                com="13.6",
                label=f"Private Work",
            )
            plt.savefig(f"{name}_w_comparison")
    
            plt.close(fig)
        else:
            spline_fpr=np.array(fpr_int(x))
            fpr_array= spline_fpr * (model_ratio_fpr) * (ratio_btag_fpr)
            fpr_array= [np.where(fpr_v>1, 1, fpr_v) for fpr_v in fpr_array]
            fig=plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(x,fpr_array, label= f"AUC = {AUC_score[0]:.3f}", linewidth=1)
            # plt.plot(tpr,fpr_array[i], linewidth=1)
            # plt.plot(tpr_2b_data_full_stats_dnn,fpr_2b_data_full_stats_dnn, label= "2b")
            ax.plot(tpr_r_1,fpr_r_1,label=f"AUC={score_3:.3f} from the AN", linewidth=1, color="yellowgreen")
            # plt.plot(tpr_r_1,fpr_r_1,label=f"AUC={score_2:.3f} from the AN", linewidth=1)
            plt.legend(fontsize= 'small')
            plt.xlabel("tpr")
            plt.ylabel("fpr")
            # plt.yscale("log")
            # plt.ylim(bottom=7.2*10**(-6), top= 7.3*10**(-6))
            plt.grid(linestyle=":")
            if "data" in fpr:
                hep.cms.label(
                            year="2022",
                            com="13.6",
                            label=f"Private Work",
                            data=True
                        )
        
            else:
                hep.cms.label(
                            year="2022",
                            com="13.6",
                            label=f"Private Work",
                        )
            plt.savefig(f"{name}_w_comparison")
            plt.close(fig)
        
    else:
        fig=plt.figure()
        # tpr=ak.to_numpy(tpr)
        # fpr=ak.to_numpy(fpr)
        
        fpr_int=interp1d(tpr,fpr)
        x=np.concatenate((x2,x1))
        
        
        spline_fpr=np.array(fpr_int(x))
        fpr_array= [spline_fpr * (model_ratio_fpr) * (ratio_btag_fpr_v) for ratio_btag_fpr_v in ratio_btag_fpr]
        fpr_array= [np.where(fpr_v>1, 1, fpr_v) for fpr_v in fpr_array]
        
        # print("fpr_array", fpr_array)
        
        fpr_2b_data_full_stats_dnn, tpr_2b_data_full_stats_dnn= fpr_tpr("spanet_prediction_2b_data_f_dnn.h5", "2b_data_full_statistics_c/output_JetGood_test.h5")
        fpr_2b_data_full_stats_dnn_proba, tpr_2b_data_full_stats_dnn_proba= fpr_tpr("spanet_prediction_2b_data_f_dnn_proba.h5", "2b_data_full_statistics_c/output_JetGood_test.h5")
        # print(x)
        # print("AUC test", auc(fpr_array[0], x ))
        
        
        fpr_2b_data_full_stats_dnn_ev_on_sr, tpr_2b_data_full_stats_dnn_ev_on_sr= fpr_tpr("spanet_c_pred_training_inclusive_2b_data_f_eval_sr_2b_data_full_stats_dnn.h5","/signal_region/2b_data_full_stats/output_JetGood_test.h5" )
        fpr_2b_data_full_stats_dnn_proba_ev_on_sr, tpr_2b_data_full_stats_dnn_proba_ev_on_sr= fpr_tpr("spanet_c_pred_training_inclusive_2b_data_f_eval_sr_2b_data_full_stats_dnn_proba.h5","/signal_region/2b_data_full_stats/output_JetGood_test.h5" )

        # sorted_indices = np.argsort(x)
        # fpr_sorted = fpr_array[0][sorted_indices]
        # x_sorted = x[sorted_indices]
        
        # print("auc test", auc(fpr_sorted, x_sorted ))
        
        # new_auc_scores=[auc(fpr_array[i], x ) for i in range(len(fpr_array))]
        # # print(fpr_array[0])
        # print("fpr array",fpr_array )
        # for i in range(len(fpr_array)):
        #     plt.plot(x,fpr_array[i], label= f"AUC = {AUC_score[i]:.3f}", linewidth=1)
        #     # plt.plot(tpr,fpr_array[i], linewidth=1)
        plt.plot(tpr_2b_data_full_stats_dnn_ev_on_sr,fpr_2b_data_full_stats_dnn_ev_on_sr, label= f"2b full stats sr DNN AUC {auc(fpr_2b_data_full_stats_dnn_ev_on_sr, tpr_2b_data_full_stats_dnn_ev_on_sr):.3f}")
        plt.plot(tpr_2b_data_full_stats_dnn_proba_ev_on_sr,fpr_2b_data_full_stats_dnn_proba_ev_on_sr, label= f"2b full stats sr DNN + prob diff AUC {auc(fpr_2b_data_full_stats_dnn_proba_ev_on_sr, tpr_2b_data_full_stats_dnn_proba_ev_on_sr):.3f}")
        plt.plot(tpr_2b_data_full_stats_dnn,fpr_2b_data_full_stats_dnn, label= f"2b full stats DNN AUC {auc(fpr_2b_data_full_stats_dnn, tpr_2b_data_full_stats_dnn):.3f}")
        plt.plot(tpr_2b_data_full_stats_dnn_proba,fpr_2b_data_full_stats_dnn_proba, label= f" 2b full stats DNN + prob diff AUC {auc(fpr_2b_data_full_stats_dnn_proba, tpr_2b_data_full_stats_dnn_proba):.3f}")
        plt.legend(fontsize= 'small')
        plt.xlabel("tpr")
        plt.ylabel("fpr")
        # plt.yscale("log")
        # plt.ylim(bottom=7.2*10**(-6), top= 7.3*10**(-6))
        plt.grid(linestyle=":")
        if "data" in fpr:
            hep.cms.label(
                        year="2022",
                        com="13.6",
                        label=f"Private Work",
                        data=True
                    )
        
        else:
            hep.cms.label(
                        year="2022",
                        com="13.6",
                        label=f"Private Work",
                    )
        plt.savefig(f"{name}_w")
        plt.close(fig)
        
def roc_curve_compare_weights(tpr,fpr,label ,name, title, comparison= None, AN=None):
    
    if comparison is not None and AN is None:
        fig=plt.figure()
        print("len fpr", len(fpr))
        for i in range(len(fpr)):
            plt.plot(tpr[i],fpr[i],label=f"{label[i]} with AUC={auc(fpr[i],tpr[i]):.3f}")
        plt.legend(fontsize= 'small')
        plt.xlabel("tpr")
        plt.ylabel("fpr")
        plt.grid(linestyle=":")
        # plt.title(f"{title}", fontsize= 'small')
        if "data" in fpr:
            hep.cms.label(
                        year="2022",
                        com="13.6",
                        label=f"Private Work",
                        data=True
                    )
        
        else:
            hep.cms.label(
                        year="2022",
                        com="13.6",
                        label=f"Private Work",
                    )
        plt.savefig(f"{name}_w_comparison")
        plt.close(fig)
    elif comparison is not None and AN is not None:
        roc_points_0= path_i + "roc_raffaele/roc_points_era_1_fold_0.npy"
        df_roc_0 = np.load(roc_points_0)
        fpr_r=df_roc_0[:,0]
        tpr_r=df_roc_0[:,1]
        score_2=auc(fpr_r, tpr_r)
                
        fig=plt.figure()
        for i in range(len(fpr)):
            plt.plot(tpr[i],fpr[i],label=f"{label[i]} with AUC={auc(fpr[i],tpr[i]):.3f}")
        plt.plot(tpr_r,fpr_r,label=f"AUC={score_2:.3f} from the AN", linewidth=1, color="yellowgreen")
        plt.legend(fontsize= 'x-small')
        plt.xlabel("tpr")
        plt.ylabel("fpr")
        plt.grid(linestyle=":")
        # plt.title(f"{title}", fontsize= 'small')
        if "data" in fpr:
            hep.cms.label(
                        year="2022",
                        com="13.6",
                        label=f"Private Work",
                        data=True
                    )
        
        else:
            hep.cms.label(
                        year="2022",
                        com="13.6",
                        label=f"Private Work",
                    )
        plt.savefig(f"{name}_w_comparison")
        plt.close(fig)
    else:
        fig=plt.figure()
        plt.plot(tpr,fpr,label=f"{label} with AUC={auc(fpr,tpr):.3f}", linewidth= 1)
        plt.legend(fontsize= 'small')
        plt.xlabel("tpr")
        plt.ylabel("fpr")
        plt.grid(linestyle=":")
        # plt.title(f"{title}", fontsize= 'small')
        if "data" in fpr:
            hep.cms.label(
                        year="2022",
                        com="13.6",
                        label=f"Private Work",
                        data=True
                    )
        
        else:
            hep.cms.label(
                        year="2022",
                        com="13.6",
                        label=f"Private Work",
                    )
        plt.savefig(f"{name}_w")
        plt.close(fig)
        
        
def diff_proba_weights_comparison(test ,prediction , name, title):
    
    filename_test= path_i + test
    dataset_test= h5py.File(filename_test, "r")
    filename_pred= path_p + prediction
    dataset_pred= h5py.File(filename_pred, "r")
    
    mask_signal=dataset_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]==1
    mask_background=dataset_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]==0
    diff_proba_signal=(dataset_pred["INPUTS"]["Event"]["Probability_difference"][()][mask_signal])
    diff_proba_bckg=(dataset_pred["INPUTS"]["Event"]["Probability_difference"][()][mask_background])
    
    # for the weights we can define the signal and bckg weights that we ten apply to the histogram, respectively for the signal and the background
    weights_signal= dataset_test["WEIGHTS"]['weight'][()][mask_signal]
    weights_bckg= dataset_test["WEIGHTS"]['weight'][()][mask_background]
    # print("weights_bckg", weights_bckg)
    
    fig = plt.figure()
    plt.hist(diff_proba_signal,bins=40,range=(-0.25,1.1),histtype='step',label="Signal", density=True)
    plt.hist(diff_proba_bckg,bins=40,range=(-0.25,1.1),histtype='step',label="Background", density=True)
    plt.hist(diff_proba_signal,bins=40,range=(-0.25,1.1),histtype='step',label="Signal weighted", density=True, weights=weights_signal)
    plt.hist(diff_proba_bckg,bins=40,range=(-0.25,1.1),histtype='step',label="Background weighted", density=True, weights=weights_bckg)
    plt.legend(loc='upper left', fontsize="x-small")
    plt.xlabel(r"$\Delta$ Probability")
    # plt.yscale("log")
    plt.ylabel("Normalized counts")
    # plt.title(f"Probability difference {title}")
    plt.grid(linestyle=":")
    if "data" in test:
        hep.cms.label(
                    year="2022",
                    com="13.6",
                    label=f"Private Work",
                    data=True
                )
        
    else:
        hep.cms.label(
                    year="2022",
                    com="13.6",
                    label=f"Private Work",
                )
    plt.savefig(f"probability_diff_{name}")
    plt.close(fig)
    return diff_proba_signal, diff_proba_bckg
        
def diff_proba_weights_comparison_arc_tan(test ,prediction , name, title):
    
    filename_test= path_i + test
    dataset_test= h5py.File(filename_test, "r")
    filename_pred= path_p + prediction
    dataset_pred= h5py.File(filename_pred, "r")
    
    mask_signal=dataset_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]==1
    mask_background=dataset_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]==0
    diff_proba_signal=np.arctanh(dataset_pred["INPUTS"]["Event"]["Probability_difference"][()][mask_signal])
    diff_proba_bckg=np.arctanh(dataset_pred["INPUTS"]["Event"]["Probability_difference"][()][mask_background])
    
    # for the weights we can define the signal and bckg weights that we ten apply to the histogram, respectively for the signal and the background
    weights_signal= dataset_test["WEIGHTS"]['weight'][()][mask_signal]
    weights_bckg= dataset_test["WEIGHTS"]['weight'][()][mask_background]
    # print("weights_bckg", weights_bckg)
    
    fig = plt.figure()
    plt.hist(diff_proba_signal,bins=40,range=(0,8),histtype='step',label="Signal", density=True)
    plt.hist(diff_proba_bckg,bins=40,range=(0,8),histtype='step',label="Background", density=True)
    plt.hist(diff_proba_signal,bins=40,range=(0,8),histtype='step',label="Signal weighted", density=True, weights=weights_signal)
    plt.hist(diff_proba_bckg,bins=40,range=(0,8),histtype='step',label="Background weighted", density=True, weights=weights_bckg)
    plt.legend(loc='upper right', fontsize="x-small")
    plt.xlabel(r"arctan($\Delta$ Probability)")
    plt.yscale("log")
    plt.ylabel("Normalized counts")
    # plt.title(f"Probability difference {title}")
    plt.ylim(bottom= 10**(-4))
    plt.grid(linestyle=":")
    
    if "data" in test:
        hep.cms.label(
                    year="2022",
                    com="13.6",
                    label=f"Private Work",
                    data=True
                )
        
    else:
        hep.cms.label(
                    year="2022",
                    com="13.6",
                    label=f"Private Work",
                )
    plt.savefig(f"probability_diff_{name}_arc_tan")
    plt.close(fig)
    return diff_proba_signal, diff_proba_bckg


def probabilities(test, prediction, name, title):
    filename_test= path_i + test
    df_test= h5py.File(filename_test, "r")
    filename_pred= path_p + prediction
    df_pred= h5py.File(filename_pred, "r")
    
    mask_signal=df_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]==1
    mask_background=df_test["CLASSIFICATIONS"]["EVENT"]["signal"][()]==0
    prob_signal=df_pred["CLASSIFICATIONS"]["EVENT"]["signal"][:,1][()]
    proba_signal_signal=prob_signal[mask_signal]
    proba_signal_background=prob_signal[mask_background]
    weights_signal= df_test["WEIGHTS"]['weight'][()][mask_signal]
    print("dataset", df_test)
    print("bckg", len(proba_signal_background))
    print("signal", len(proba_signal_signal))
    weights_bckg= df_test["WEIGHTS"]['weight'][()][mask_background]
    print(weights_bckg)
    fig= plt.figure()
    plt.hist(proba_signal_signal,bins=100,range=(0,1),histtype='step',label="Signal", density=True)
    plt.hist(proba_signal_background,bins=100,range=(0,1),histtype='step',label="Background", density=True)
    plt.hist(proba_signal_signal,bins=100,range=(0,1),histtype='step',label="Signal weighted", density=True, weights=weights_signal)
    plt.hist(proba_signal_background,bins=100,range=(0,1),histtype='step',label="Background weighted", density=True, weights=weights_bckg)
    # plt.title(f"{title}")
    plt.xlabel("Assigned probability")
    plt.ylabel("Normalized counts")
    plt.legend(loc="upper right")
    # plt.yscale("log")
    # plt.show()
    if "data" in test:
        hep.cms.label(
                    year="2022",
                    com="13.6",
                    label=f"Private Work",
                    data=True
                )
        
    else:
        hep.cms.label(
                    year="2022",
                    com="13.6",
                    label=f"Private Work",
                )
    plt.grid(linestyle=":")
    plt.savefig(f"signal_proba_{name}")
    plt.close(fig)
    return proba_signal_signal, proba_signal_background

def roc_curve_compare_no_weights(test_file,list_pred,list_labels,name, title):
    true= test_file["CLASSIFICATIONS"]["EVENT"]["signal"][()]
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
        plt.legend(fontsize= 'small', loc="upper left")
        plt.xlabel("tpr")
        plt.ylabel("fpr")
        plt.grid(linestyle=":")
        # plt.title(f"{title}", fontsize= 'small')
        hep.cms.label(
                year="2022",
                com="13.6",
                label=f"Private Work",
            )
    plt.savefig(f"{name}_no_w")
    plt.close(fig)
    
    
def fpr_ratio_2(fpr, tpr, fpr_2, tpr_2, AUC_score, AUC_score_2, ratio_btag_fpr, ratio_btag_fpr_2, model_ratio_fpr, model_ratio_fpr_2,  name, x1, x2, comparison=False):
    #Comparison with the AN
    if comparison is not False:
        roc_points_0= path_i + "roc_raffaele/roc_points_era_1_fold_0.npy"
        roc_points_1= path_i + "roc_raffaele/roc_points_era_1_fold_1.npy"
        df_roc_0 = np.load(roc_points_0)
        df_roc_1 = np.load(roc_points_1)
        fpr_r=df_roc_0[:,0]
        tpr_r=df_roc_0[:,1]
        fpr_r_1=df_roc_1[:,0]
        tpr_r_1=df_roc_1[:,1]
    
        score_2=auc(fpr_r, tpr_r)
        score_3=auc(fpr_r_1, tpr_r_1)

        fig=plt.figure()
        # tpr=ak.to_numpy(tpr)
        # fpr=ak.to_numpy(fpr)
        
        fpr_int=interp1d(tpr,fpr)
        fpr_int_2=interp1d(tpr_2,fpr_2)
        x=np.concatenate((x2,x1))
        
        # print("fpr_array", fpr_array)
        
        fpr_2b_data_full_stats_dnn, tpr_2b_data_full_stats_dnn= fpr_tpr("spanet_prediction_2b_data_f_dnn.h5", "2b_data_full_statistics_c/output_JetGood_test.h5")
        # print(x)
        # print("AUC test", auc(fpr_array[0], x ))
        
        # sorted_indices = np.argsort(x)
        # fpr_sorted = fpr_array[0][sorted_indices]
        # x_sorted = x[sorted_indices]
        
        # print("auc test", auc(fpr_sorted, x_sorted ))
        
        # new_auc_scores=[auc(fpr_array[i], x ) for i in range(len(fpr_array))]
        # # print(fpr_array[0])
        # print("fpr array",fpr_array )
        
        max_auc_index= np.argmax(AUC_score)
        max_auc_index_2= np.argmax(AUC_score_2)
        min_auc_index= np.argmin(AUC_score)
        min_auc_index_2= np.argmin(AUC_score_2)
        
        # print(len(AUC_score))
        
        if len(AUC_score)>1:
            spline_fpr=np.array(fpr_int(x))
            fpr_array= [spline_fpr * (model_ratio_fpr) * (ratio_btag_fpr_v) for ratio_btag_fpr_v in ratio_btag_fpr]
            fpr_array= [np.where(fpr_v>1, 1, fpr_v) for fpr_v in fpr_array]
            spline_fpr_2=np.array(fpr_int_2(x))
            fpr_array_2= [spline_fpr_2 * (model_ratio_fpr_2) * (ratio_btag_fpr_v) for ratio_btag_fpr_v in ratio_btag_fpr_2]
            fpr_array_2= [np.where(fpr_v>1, 1, fpr_v) for fpr_v in fpr_array_2]
            fig=plt.figure()
            for i in range(len(fpr_array)):
                if i==min_auc_index_2:
                    plt.plot(x,fpr_array_2[i], label= f"4b-data with DNN with AUC = {AUC_score_2[i]:.3f}", linewidth=1, color='lightseagreen')
                if i==min_auc_index :
                    plt.plot(x,fpr_array[i], label= f"4b-data with DNN and PD with AUC = {AUC_score[i]:.3f}", linewidth=1, color= 'darkorange')
                # plt.plot(tpr,fpr_array[i], linewidth=1)
            # plt.plot(tpr_2b_data_full_stats_dnn,fpr_2b_data_full_stats_dnn, label= "2b")
            plt.plot(tpr_r_1,fpr_r_1,label=f"AUC={score_3:.3f} from the AN", linewidth=1, color="yellowgreen")
            # plt.plot(tpr_r_1,fpr_r_1,label=f"AUC={score_2:.3f} from the AN", linewidth=1)
            plt.legend(fontsize= 'small')
            plt.xlabel("tpr")
            plt.ylabel("fpr")
    
            # plt.yscale("log")
            # plt.ylim(bottom=7.2*10**(-6), top= 7.3*10**(-6))
            plt.grid(linestyle=":")
            
            # if "data" in fpr:
            #     hep.cms.label(
            #                 year="2022",
            #                 com="13.6",
            #                 label=f"Private Work",
            #                 data=True
            #         )
            
            # else:
            #     hep.cms.label(
            #                 year="2022",
            #                 com="13.6",
            #                 label=f"Private Work",
            #             )
                
            hep.cms.label(
                year="2022",
                com="13.6",
                label=f"Private Work",
            )
            plt.savefig(f"{name}_w_comparison")
    
            plt.close(fig)
        else:
            spline_fpr=np.array(fpr_int(x))
            fpr_array= spline_fpr * (model_ratio_fpr) * (ratio_btag_fpr)
            fpr_array= [np.where(fpr_v>1, 1, fpr_v) for fpr_v in fpr_array]
            fig=plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(x,fpr_array, label= f"AUC = {AUC_score[0]:.3f}", linewidth=1)
            # plt.plot(tpr,fpr_array[i], linewidth=1)
            # plt.plot(tpr_2b_data_full_stats_dnn,fpr_2b_data_full_stats_dnn, label= "2b")
            ax.plot(tpr_r_1,fpr_r_1,label=f"AUC={score_3:.3f} from the AN", linewidth=1, color="yellowgreen")
            # plt.plot(tpr_r_1,fpr_r_1,label=f"AUC={score_2:.3f} from the AN", linewidth=1)
            plt.legend(fontsize= 'small')
            plt.xlabel("tpr")
            plt.ylabel("fpr")
            # plt.yscale("log")
            # plt.ylim(bottom=7.2*10**(-6), top= 7.3*10**(-6))
            plt.grid(linestyle=":")
            if "data" in fpr:
                hep.cms.label(
                            year="2022",
                            com="13.6",
                            label=f"Private Work",
                            data=True
                        )
        
            else:
                hep.cms.label(
                            year="2022",
                            com="13.6",
                            label=f"Private Work",
                        )
            plt.savefig(f"{name}_w_comparison")
            plt.close(fig)
        
    else:
        fig=plt.figure()
        # tpr=ak.to_numpy(tpr)
        # fpr=ak.to_numpy(fpr)
        
        fpr_int=interp1d(tpr,fpr)
        x=np.concatenate((x2,x1))
        
        
        spline_fpr=np.array(fpr_int(x))
        fpr_array= [spline_fpr * (model_ratio_fpr) * (ratio_btag_fpr_v) for ratio_btag_fpr_v in ratio_btag_fpr]
        fpr_array= [np.where(fpr_v>1, 1, fpr_v) for fpr_v in fpr_array]
        
        # print("fpr_array", fpr_array)
        
        fpr_2b_data_full_stats_dnn, tpr_2b_data_full_stats_dnn= fpr_tpr("spanet_prediction_2b_data_f_dnn.h5", "2b_data_full_statistics_c/output_JetGood_test.h5")
        fpr_2b_data_full_stats_dnn_proba, tpr_2b_data_full_stats_dnn_proba= fpr_tpr("spanet_prediction_2b_data_f_dnn_proba.h5", "2b_data_full_statistics_c/output_JetGood_test.h5")
        # print(x)
        # print("AUC test", auc(fpr_array[0], x ))
        
        
        fpr_2b_data_full_stats_dnn_ev_on_sr, tpr_2b_data_full_stats_dnn_ev_on_sr= fpr_tpr("spanet_c_pred_training_inclusive_2b_data_f_eval_sr_2b_data_full_stats_dnn.h5","/signal_region/2b_data_full_stats/output_JetGood_test.h5" )
        fpr_2b_data_full_stats_dnn_proba_ev_on_sr, tpr_2b_data_full_stats_dnn_proba_ev_on_sr= fpr_tpr("spanet_c_pred_training_inclusive_2b_data_f_eval_sr_2b_data_full_stats_dnn_proba.h5","/signal_region/2b_data_full_stats/output_JetGood_test.h5" )

        # sorted_indices = np.argsort(x)
        # fpr_sorted = fpr_array[0][sorted_indices]
        # x_sorted = x[sorted_indices]
        
        # print("auc test", auc(fpr_sorted, x_sorted ))
        
        # new_auc_scores=[auc(fpr_array[i], x ) for i in range(len(fpr_array))]
        # # print(fpr_array[0])
        # print("fpr array",fpr_array )
        # for i in range(len(fpr_array)):
        #     plt.plot(x,fpr_array[i], label= f"AUC = {AUC_score[i]:.3f}", linewidth=1)
        #     # plt.plot(tpr,fpr_array[i], linewidth=1)
        plt.plot(tpr_2b_data_full_stats_dnn_ev_on_sr,fpr_2b_data_full_stats_dnn_ev_on_sr, label= f"2b full stats sr DNN AUC {auc(fpr_2b_data_full_stats_dnn_ev_on_sr, tpr_2b_data_full_stats_dnn_ev_on_sr):.3f}")
        plt.plot(tpr_2b_data_full_stats_dnn_proba_ev_on_sr,fpr_2b_data_full_stats_dnn_proba_ev_on_sr, label= f"2b full stats sr DNN + prob diff AUC {auc(fpr_2b_data_full_stats_dnn_proba_ev_on_sr, tpr_2b_data_full_stats_dnn_proba_ev_on_sr):.3f}")
        plt.plot(tpr_2b_data_full_stats_dnn,fpr_2b_data_full_stats_dnn, label= f"2b full stats DNN AUC {auc(fpr_2b_data_full_stats_dnn, tpr_2b_data_full_stats_dnn):.3f}")
        plt.plot(tpr_2b_data_full_stats_dnn_proba,fpr_2b_data_full_stats_dnn_proba, label= f" 2b full stats DNN + prob diff AUC {auc(fpr_2b_data_full_stats_dnn_proba, tpr_2b_data_full_stats_dnn_proba):.3f}")
        plt.legend(fontsize= 'small')
        plt.xlabel("tpr")
        plt.ylabel("fpr")
        # plt.yscale("log")
        # plt.ylim(bottom=7.2*10**(-6), top= 7.3*10**(-6))
        plt.grid(linestyle=":")
        if "data" in fpr:
            hep.cms.label(
                        year="2022",
                        com="13.6",
                        label=f"Private Work",
                        data=True
                    )
        
        else:
            hep.cms.label(
                        year="2022",
                        com="13.6",
                        label=f"Private Work",
                    )
        plt.savefig(f"{name}_w")
        plt.close(fig)