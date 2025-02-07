import awkward as ak
import numpy as np
import h5py
import vector
from math import sqrt
import matplotlib.pyplot as plt
import os
import sys
import mplhep as hep

vector.register_numba()
vector.register_awkward()

import argparse

from efficiency_functions import *
from efficiency_configuration import *

parser = argparse.ArgumentParser(
    description="Convert awkward ntuples in parquet files to h5 files."
)
parser.add_argument(
    "-pd",
    "--plot-dir",
    default="plots",
    help="Directory to save the plots",
)
parser.add_argument(
    "-k",
    "--klambda",
    default=False,
    action="store_true",
    help="evaluate on different klambda values",
)
parser.add_argument(
    "-d",
    "--data",
    default=False,
    action="store_true",
    help="evaluate on data",
)

args = parser.parse_args()


# redirect stout
# sys.stdout = open(f"{args.plot_dir}/efficiency.txt", "w")

if args.data:
    # remove non data samples
    spanet_dict = {k: v for k, v in spanet_dict.items() if "data" in k}
else:
    spanet_dict = {k: v for k, v in spanet_dict.items() if "data" not in k}

print(spanet_dict)
# bin definitions
mh_bins = [
    np.linspace(0, 300, n)
    for n in [80, 80, 80, 40, 40, 40, 40, 40, 40, 80, 80, 80, 80, 80, 80, 80]
]
mh_bins_peak = [
    np.linspace(100, 140, n)
    for n in [20, 20, 20, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20]
]
mh_bins_2d = (
    [np.linspace(50, 200, 80) for _ in range(3)]
    + [np.linspace(50, 200, 40) for _ in range(6)]
    + [np.linspace(0, 500, 50) for _ in range(2)]
    + [np.linspace(0, 500, 100) for _ in range(5)]
)

for bins in [mh_bins, mh_bins_peak, mh_bins_2d]:
    if len(bins) != len(list(true_dict.keys())):
        bins += [bins[-1] for _ in range(len(list(true_dict.keys())) - len(bins))]


mhh_bins = np.linspace(250, 700, 10)


plot_dir = args.plot_dir
os.makedirs(plot_dir, exist_ok=True)

# open files
df_true = [h5py.File(f, "r") for f in true_dict.values()]
df_spanet_pred = [h5py.File(f, "r") for f in spanet_dict.values()]


print("df_true", true_dict.keys(), flush=True)
print("df_spanet_pred", spanet_dict.keys(), flush=True)

idx_b1_true = [df["TARGETS"]["h1"]["b1"][()] for df in df_true]
idx_b2_true = [df["TARGETS"]["h1"]["b2"][()] for df in df_true]
idx_b3_true = [df["TARGETS"]["h2"]["b3"][()] for df in df_true]
idx_b4_true = [df["TARGETS"]["h2"]["b4"][()] for df in df_true]

idx_b1_spanet_pred = [df["TARGETS"]["h1"]["b1"][()] for df in df_spanet_pred]
idx_b2_spanet_pred = [df["TARGETS"]["h1"]["b2"][()] for df in df_spanet_pred]
idx_b3_spanet_pred = [df["TARGETS"]["h2"]["b3"][()] for df in df_spanet_pred]
idx_b4_spanet_pred = [df["TARGETS"]["h2"]["b4"][()] for df in df_spanet_pred]

idx_h1_true = [
    ak.concatenate(
        (
            ak.unflatten(idx_b1, ak.ones_like(idx_b1)),
            ak.unflatten(idx_b2, ak.ones_like(idx_b2)),
        ),
        axis=1,
    )
    for idx_b1, idx_b2 in zip(idx_b1_true, idx_b2_true)
]
idx_h2_true = [
    ak.concatenate(
        (
            ak.unflatten(idx_b3, ak.ones_like(idx_b3)),
            ak.unflatten(idx_b4, ak.ones_like(idx_b4)),
        ),
        axis=1,
    )
    for idx_b3, idx_b4 in zip(idx_b3_true, idx_b4_true)
]

idx_h1_spanet_pred = [
    ak.concatenate(
        (
            ak.unflatten(idx_b1, ak.ones_like(idx_b1)),
            ak.unflatten(idx_b2, ak.ones_like(idx_b2)),
        ),
        axis=1,
    )
    for idx_b1, idx_b2 in zip(idx_b1_spanet_pred, idx_b2_spanet_pred)
]
idx_h2_spanet_pred = [
    ak.concatenate(
        (
            ak.unflatten(idx_b3, ak.ones_like(idx_b3)),
            ak.unflatten(idx_b4, ak.ones_like(idx_b4)),
        ),
        axis=1,
    )
    for idx_b3, idx_b4 in zip(idx_b3_spanet_pred, idx_b4_spanet_pred)
]

idx_true = [
    ak.concatenate(
        (
            ak.unflatten(idx_h1, ak.ones_like(idx_h1[:, 0])),
            ak.unflatten(idx_h2, ak.ones_like(idx_h2[:, 0])),
        ),
        axis=1,
    )
    for idx_h1, idx_h2 in zip(idx_h1_true, idx_h2_true)
]
idx_spanet_pred = [
    ak.concatenate(
        (
            ak.unflatten(idx_h1, ak.ones_like(idx_h1[:, 0])),
            ak.unflatten(idx_h2, ak.ones_like(idx_h2[:, 0])),
        ),
        axis=1,
    )
    for idx_h1, idx_h2 in zip(idx_h1_spanet_pred, idx_h2_spanet_pred)
]


# load jet information
jet_ptPNetRegNeutrino = [df["INPUTS"]["Jet"]["ptPnetRegNeutrino"][()] for df in df_true]
jet_eta = [df["INPUTS"]["Jet"]["eta"][()] for df in df_true]
jet_phi = [df["INPUTS"]["Jet"]["phi"][()] for df in df_true]
jet_mass = [df["INPUTS"]["Jet"]["mass"][()] for df in df_true]

jet_infos = [jet_ptPNetRegNeutrino, jet_eta, jet_phi, jet_mass]

if args.klambda:
    # separate the different klambdas
    (
        idx_true,
        idx_spanet_pred,
        true_dict,
        spanet_dict,
        jet_infos_separate_klambda,
        kl_values_true,
        kl_values_spanet,
        allkl_names_true,
        allkl_names_spanet,
    ) = separate_klambda(
        df_true, df_spanet_pred, idx_true, idx_spanet_pred, true_dict, spanet_dict
    )
    mh_bins += [np.linspace(60, 190, 80) for _ in range(len(kl_values_true))]
    mh_bins_peak += [np.linspace(100, 140, 20) for _ in range(len(kl_values_true))]
    mh_bins_2d += [np.linspace(50, 200, 80) for _ in range(len(kl_values_true))]

    print("mh_bins", len(mh_bins), len(mh_bins[0]))
    print("mh_bins_peak", len(mh_bins_peak), len(mh_bins_peak[0]))
    print("mh_bins_2d", len(mh_bins_2d), len(mh_bins_2d[0]))

    for i in range(len(jet_infos_separate_klambda)):
        jet_infos[i].extend(jet_infos_separate_klambda[i])
    print("jet_infos", len(jet_infos), len(jet_infos[0]))

# Fully matched events
mask_fully_matched = [ak.all(ak.all(idx >= 0, axis=-1), axis=-1) for idx in idx_true]

idx_true_fully_matched = [idx[mask] for idx, mask in zip(idx_true, mask_fully_matched)]
print("idx_true_fully_matched", [len(idx) for idx in idx_true_fully_matched])
print("idx_spanet_pred", [len(pred) for pred in idx_spanet_pred])
print("mask_fully_matched", [len(mask) for mask in mask_fully_matched])
print([check_names(list(spanet_dict.keys())[i]) for i in range(len(idx_spanet_pred))])

idx_spanet_pred_fully_matched = [
    idx_spanet_pred[i][mask_fully_matched[check_names(list(spanet_dict.keys())[i])]]
    for i in range(len(idx_spanet_pred))
]
if not args.data:
    correctly_fully_matched_spanet = [
        (
            ak.all(
                idx_true_fully_matched[check_names(list(spanet_dict.keys())[i])][:, 0]
                == idx_spanet_pred_fully_matched[i][:, 0],
                axis=1,
            )
            | ak.all(
                idx_true_fully_matched[check_names(list(spanet_dict.keys())[i])][:, 0]
                == idx_spanet_pred_fully_matched[i][:, 1],
                axis=1,
            )
        )
        & (
            ak.all(
                idx_true_fully_matched[check_names(list(spanet_dict.keys())[i])][:, 1]
                == idx_spanet_pred_fully_matched[i][:, 0],
                axis=1,
            )
            | ak.all(
                idx_true_fully_matched[check_names(list(spanet_dict.keys())[i])][:, 1]
                == idx_spanet_pred_fully_matched[i][:, 1],
                axis=1,
            )
        )
        for i in range(len(idx_spanet_pred_fully_matched))
    ]

    # compute efficiencies for fully matched events
    efficiencies_fully_matched_spanet = [
        ak.sum(correctly_fully_matched) / len(correctly_fully_matched)
        for correctly_fully_matched in correctly_fully_matched_spanet
    ]
    print(
        "correctly_fully_matched_spanet",
        [len(c) for c in correctly_fully_matched_spanet],
    )
    frac_fully_matched = [ak.sum(mask) / len(mask) for mask in mask_fully_matched]
    print("\n")
    for label, frac in zip(list(true_dict.keys()), frac_fully_matched):
        print(f"Fraction of fully matched events for {label}: {frac:.3f}")

    print("\n")
    for label, eff in zip(list(spanet_dict.keys()), efficiencies_fully_matched_spanet):
        print("Efficiency fully matched for {}: {:.3f}".format(label, eff))
    print("\n")

    total_efficiencies_fully_matched_spanet = [
        efficiencies_fully_matched_spanet[i]
        * frac_fully_matched[check_names(list(spanet_dict.keys())[i])]
        for i in range(len(efficiencies_fully_matched_spanet))
    ]
    for i in range(len(total_efficiencies_fully_matched_spanet)):
        print(
            "Total efficiency fully matched for {}: {:.3f}".format(
                list(spanet_dict.keys())[i],
                total_efficiencies_fully_matched_spanet[i],
            )
        )

    if args.klambda:
        efficiencies_fully_matched_spanet_allklambda = (
            efficiencies_fully_matched_spanet[-len(kl_values_spanet) :]
        )
        print(efficiencies_fully_matched_spanet_allklambda)
        total_efficiencies_fully_matched_spanet_allklambda = (
            total_efficiencies_fully_matched_spanet[-len(kl_values_spanet) :]
        )
        print("\n")
        print("Plotting efficiencies fully matched for all klambda values")
        plot_diff_eff_klambda(
            efficiencies_fully_matched_spanet_allklambda,
            kl_values_spanet,
            allkl_names_spanet,
            "eff_fully_matched_allklambda",
            plot_dir,
        )
        plot_diff_eff_klambda(
            total_efficiencies_fully_matched_spanet_allklambda,
            kl_values_spanet,
            allkl_names_spanet,
            "tot_eff_fully_matched_allklambda",
            plot_dir,
        )

    # do the same for partially matched events (only one higgs is matched)
    mask_1h = [
        ak.sum(ak.any(idx == -1, axis=-1) == 1, axis=-1) == 1 for idx in idx_true
    ]
    idx_true_partially_matched_1h = [idx[mask] for idx, mask in zip(idx_true, mask_1h)]
    idx_spanet_pred_partially_matched_1h = [
        idx_spanet_pred[i][mask_1h[check_names(list(spanet_dict.keys())[i])]]
        for i in range(len(idx_spanet_pred))
    ]

    correctly_partially_matched_spanet = [
        (
            ak.all(
                idx_true_partially_matched_1h[check_names(list(spanet_dict.keys())[i])][
                    :, 0
                ]
                == idx_spanet_pred_partially_matched_1h[i][:, 0],
                axis=1,
            )
            | ak.all(
                idx_true_partially_matched_1h[check_names(list(spanet_dict.keys())[i])][
                    :, 0
                ]
                == idx_spanet_pred_partially_matched_1h[i][:, 1],
                axis=1,
            )
            | ak.all(
                idx_true_partially_matched_1h[check_names(list(spanet_dict.keys())[i])][
                    :, 1
                ]
                == idx_spanet_pred_partially_matched_1h[i][:, 0],
                axis=1,
            )
            | ak.all(
                idx_true_partially_matched_1h[check_names(list(spanet_dict.keys())[i])][
                    :, 1
                ]
                == idx_spanet_pred_partially_matched_1h[i][:, 1],
                axis=1,
            )
        )
        for i in range(len(idx_spanet_pred_partially_matched_1h))
    ]
    efficiencies_partially_matched_spanet = [
        ak.sum(correctly_partially_matched_1h) / len(correctly_partially_matched_1h)
        for correctly_partially_matched_1h in correctly_partially_matched_spanet
    ]
    frac_partially_matched_1h = [ak.sum(mask) / len(mask) for mask in mask_1h]
    print("\n")
    for label, frac in zip(list(true_dict.keys()), frac_partially_matched_1h):
        print(f"Fraction of partially matched events for {label}: {frac:.3f}")
    print("\n")
    for label, eff in zip(
        list(spanet_dict.keys()), efficiencies_partially_matched_spanet
    ):
        print("Efficiency partially matched for {}: {:.3f}".format(label, eff))
    print("\n")
    total_efficiencies_partially_matched_spanet = [
        efficiencies_partially_matched_spanet[i]
        * frac_partially_matched_1h[check_names(list(spanet_dict.keys())[i])]
        for i in range(len(efficiencies_partially_matched_spanet))
    ]

    for i in range(len(total_efficiencies_partially_matched_spanet)):
        print(
            "Total efficiency partially matched for {}: {:.3f}".format(
                list(spanet_dict.keys())[i],
                total_efficiencies_partially_matched_spanet[i],
            )
        )

    # compute number of events with 0 higgs matched
    mask_0h = [ak.sum(ak.any(idx == -1, axis=-1), axis=-1) == 2 for idx in idx_true]

    idx_true_unmatched = [idx[mask] for idx, mask in zip(idx_true, mask_0h)]
    frac_unmatched = [ak.sum(mask) / len(mask) for mask in mask_0h]
    print("\n")
    for label, frac in zip(list(true_dict.keys()), frac_unmatched):
        print(f"Fraction of unmatched events for {label}: {frac:.3f}")


# create a LorentzVector for the jets
jet = [
    ak.zip(
        {
            "pt": jet_infos[0][i],
            "eta": jet_infos[1][i],
            "phi": jet_infos[2][i],
            "mass": jet_infos[3][i],
        },
        with_name="Momentum4D",
    )
    for i in range(len(jet_infos[0]))
]

print("len(jet)", len(jet), len(jet[0]))

# HERE
# the jets for the file with 4jets are the same as the 5jets
# jet[0] = jet[1]


# implement the Run 2 pairing algorithm
# TODO: extend to 5 jets cases (more comb idx)
comb_idx = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]

higgs_candidates_unflatten_order = [reco_higgs(j, comb_idx) for j in jet]
distance = [
    distance_func(
        higgs,
        1.04,
    )
    for higgs in higgs_candidates_unflatten_order
]
dist_order_idx = [ak.argsort(d, axis=1, ascending=True) for d in distance]
dist_order = [ak.sort(d, axis=1, ascending=True) for d in distance]

# if the distance between the two best candidates is less than 30, we do not consider the event
min_idx = [
    ak.where(d[:, 1] - d[:, 0] > 30, d_idx[:, 0], -1)
    for d, d_idx in zip(dist_order, dist_order_idx)
]
mask_30 = [m != -1 for m in min_idx]


comb_idx_mask30 = [
    np.tile(comb_idx, (len(m), 1, 1, 1))[mask] for m, mask in zip(min_idx, mask_30)
]
min_idx_mask30 = [m[mask] for m, mask in zip(min_idx, mask_30)]
# given the min_idx, select the correct combination corresponding to the index
comb_idx_min_mask30 = [
    comb[np.arange(len(m)), m] for comb, m in zip(comb_idx_mask30, min_idx_mask30)
]
idx_run2_pred_fully_matched_mask30 = [
    ak.Array(comb)[m[m_30]]
    for comb, m, m_30 in zip(comb_idx_min_mask30, mask_fully_matched, mask_30)
]

idx_true_fully_matched_mask30 = [
    idx[m_30][m[m_30]] for idx, m, m_30 in zip(idx_true, mask_fully_matched, mask_30)
]
idx_spanet_pred_fully_matched_mask30 = [
    idx_spanet_pred[i][mask_30[check_names(list(spanet_dict.keys())[i])]][
        mask_fully_matched[check_names(list(spanet_dict.keys())[i])][
            mask_30[check_names(list(spanet_dict.keys())[i])]
        ]
    ]
    for i in range(len(idx_spanet_pred))
]
if not args.data:
    print("\n")
    for label, m in zip(list(true_dict.keys()), mask_30):
        print(
            f"Fraction of events with DeltaR>30 for {label}: {ak.sum(m) / len(m):.3f}"
        )
    # compute efficiencies for fully matched events for Run 2 pairing
    correctly_fully_matched_run2_mask30 = [
        (
            ak.all(
                i[:, 0] == i2[:, 0],
                axis=1,
            )
            | ak.all(
                i[:, 0] == i2[:, 1],
                axis=1,
            )
        )
        & (
            ak.all(
                i[:, 1] == i2[:, 0],
                axis=1,
            )
            | ak.all(
                i[:, 1] == i2[:, 1],
                axis=1,
            )
        )
        for i, i2 in zip(
            idx_true_fully_matched_mask30, idx_run2_pred_fully_matched_mask30
        )
    ]

    frac_fully_matched_mask30 = [
        ak.sum(m[m_30]) / len(m[m_30]) for m, m_30 in zip(mask_fully_matched, mask_30)
    ]
    print("\n")
    for label, frac in zip(list(true_dict.keys()), frac_fully_matched_mask30):
        print(f"Fraction of fully matched events {label} (DeltaR>30): {frac:.3f}")
    efficiency_fully_matched_run2_mask30 = [
        ak.sum(corr) / len(corr) for corr in correctly_fully_matched_run2_mask30
    ]

    print("\n")
    for label, eff in zip(list(true_dict.keys()), efficiency_fully_matched_run2_mask30):
        print(f"Efficiency fully matched  for Run 2 {label} (DeltaR>30): {eff:.3f}")

    total_efficiency_fully_matched_run2_mask30 = [
        efficiency_fully_matched_run2_mask30[i] * frac_fully_matched_mask30[i]
        for i in range(len(efficiency_fully_matched_run2_mask30))
    ]
    print("\n")
    for i in range(len(total_efficiency_fully_matched_run2_mask30)):
        print(
            "Total efficiency fully matched for Run 2 {} (DeltaR>30): {:.3f}".format(
                list(true_dict.keys())[i],
                total_efficiency_fully_matched_run2_mask30[i],
            )
        )

    # compute efficiencies for fully matched events for spanet
    correctly_fully_matched_spanet_mask30 = [
        (
            ak.all(
                idx_true_fully_matched_mask30[check_names(list(spanet_dict.keys())[i])][
                    :, 0
                ]
                == idx_spanet_pred_fully_matched_mask30[i][:, 0],
                axis=1,
            )
            | ak.all(
                idx_true_fully_matched_mask30[check_names(list(spanet_dict.keys())[i])][
                    :, 0
                ]
                == idx_spanet_pred_fully_matched_mask30[i][:, 1],
                axis=1,
            )
        )
        & (
            ak.all(
                idx_true_fully_matched_mask30[check_names(list(spanet_dict.keys())[i])][
                    :, 1
                ]
                == idx_spanet_pred_fully_matched_mask30[i][:, 0],
                axis=1,
            )
            | ak.all(
                idx_true_fully_matched_mask30[check_names(list(spanet_dict.keys())[i])][
                    :, 1
                ]
                == idx_spanet_pred_fully_matched_mask30[i][:, 1],
                axis=1,
            )
        )
        for i in range(len(idx_spanet_pred_fully_matched_mask30))
    ]
    efficiencies_fully_matched_spanet_mask30 = [
        ak.sum(correctly_fully_matched) / len(correctly_fully_matched)
        for correctly_fully_matched in correctly_fully_matched_spanet_mask30
    ]
    print("\n")
    for label, eff in zip(
        list(spanet_dict.keys()), efficiencies_fully_matched_spanet_mask30
    ):
        print(f"Efficiency fully matched for {label} (DeltaR>30): {eff:.3f}")

    total_efficiencies_fully_matched_spanet_mask30 = [
        efficiencies_fully_matched_spanet_mask30[i]
        * frac_fully_matched_mask30[check_names(list(spanet_dict.keys())[i])]
        for i in range(len(efficiencies_fully_matched_spanet_mask30))
    ]
    for i in range(len(total_efficiencies_fully_matched_spanet_mask30)):
        print(
            "Total efficiency fully matched for {} (DeltaR>30): {:.3f}".format(
                list(spanet_dict.keys())[i],
                total_efficiencies_fully_matched_spanet_mask30[i],
            )
        )

    if args.klambda:
        efficiencies_fully_matched_spanet_mask30_allklambda = (
            efficiencies_fully_matched_spanet_mask30[-len(kl_values_spanet) :]
        )
        total_efficiencies_fully_matched_spanet_mask30_allklambda = (
            total_efficiencies_fully_matched_spanet_mask30[-len(kl_values_spanet) :]
        )
        efficiency_fully_matched_run2_mask30_allklambda = (
            efficiency_fully_matched_run2_mask30[-len(kl_values_true) :]
        )
        total_efficiency_fully_matched_run2_mask30_allklambda = (
            total_efficiency_fully_matched_run2_mask30[-len(kl_values_true) :]
        )
        print(run2_dataset)
        run2_idxs = [check_names(key) for key in true_dict.keys() if "f{run2_dataset}_" in key]
        first_run2_idx = (
            len(kl_values_true) // kl_values_true.count(kl_values_true[0]) * (-1) * 7
        )  # HERE
        last_run2_idx = (
            len(kl_values_true) // kl_values_true.count(kl_values_true[0]) * (-1) * 6
        )  # HERE
        print("first_run2_idx", first_run2_idx)
        print("kl_values_true", [kl_values_true[x] for x in run2_idxs])
        print("allkl_names_true", allkl_names_true)
        #efficiencies_fully_matched_mask30_allklambda = (
        #    efficiencies_fully_matched_spanet_mask30_allklambda
        #    + [efficiency_fully_matched_run2_mask30_allklambda[x] for x in run2_idxs]
        #)
        #total_efficiencies_fully_matched_mask30_allklambda = (
        #    total_efficiencies_fully_matched_spanet_mask30_allklambda
        #    + [total_efficiency_fully_matched_run2_mask30_allklambda[x] for x in run2_idxs]
        #)
        #print(kl_values_true)
        #print("\n")
        #print("Plotting efficiencies fully matched for all klambda values")
        #plot_diff_eff_klambda(
        #    efficiencies_fully_matched_mask30_allklambda,
        #    kl_values_spanet + [kl_values_true[x] for x in run2_idxs],
        #    allkl_names_spanet + allkl_names_true[-4:-3],
        #    "eff_fully_matched_mask30_allklambda",
        #    plot_dir,
        #)
        #plot_diff_eff_klambda(
        #    total_efficiencies_fully_matched_mask30_allklambda,
        #    kl_values_spanet + [kl_values_true[x] for x in run2_idxs],
        #    allkl_names_spanet + allkl_names_true[-4:-3],
        #    "tot_eff_fully_matched_mask30_allklambda",
        #    plot_dir,
        #)
        efficiencies_fully_matched_mask30_allklambda = (
            efficiencies_fully_matched_spanet_mask30_allklambda
            + efficiency_fully_matched_run2_mask30_allklambda[first_run2_idx:last_run2_idx]
        )
        total_efficiencies_fully_matched_mask30_allklambda = (
            total_efficiencies_fully_matched_spanet_mask30_allklambda
            + total_efficiency_fully_matched_run2_mask30_allklambda[first_run2_idx:last_run2_idx]
        )
        print(kl_values_true)
        print("\n")
        print("Plotting efficiencies fully matched for all klambda values")
        plot_diff_eff_klambda(
            efficiencies_fully_matched_mask30_allklambda,
            kl_values_spanet + kl_values_true[first_run2_idx:last_run2_idx],
            allkl_names_spanet + allkl_names_true[-4:-3],
            "eff_fully_matched_mask30_allklambda",
            plot_dir,
        )
        plot_diff_eff_klambda(
            total_efficiencies_fully_matched_mask30_allklambda,
            kl_values_spanet + kl_values_true[first_run2_idx:last_run2_idx],
            allkl_names_spanet + allkl_names_true[-4:-3],
            "tot_eff_fully_matched_mask30_allklambda",
            plot_dir,
        )



# Reconstruct the Higgs boson candidates with the ciency_fully_matched_run2_mask30_allklambda = (
# of the jets considering the true pairings, the spanet pairings
# and the run2 pairings
jet_fully_matched_mask30 = [
    j[m_30][m[m_30]] for j, m_30, m in zip(jet, mask_30, mask_fully_matched)
]
for index, (j, idx) in enumerate(zip(jet_fully_matched_mask30, idx_true_fully_matched_mask30)):
    best_reco_higgs(j,idx)
true_higgs_fully_matched_mask30 = [
    best_reco_higgs(j, idx)
    for j, idx in zip(jet_fully_matched_mask30, idx_true_fully_matched_mask30)
]
true_hh_fully_matched_mask30 = [
    true_higgs_fully_matched_mask30[i][:, 0] + true_higgs_fully_matched_mask30[i][:, 1]
    for i in range(len(true_higgs_fully_matched_mask30))
]

if not args.data:
    # Differential efficiency
    # Mask 30
    diff_eff_run2_mask30 = []
    unc_diff_eff_run2_mask30 = []
    total_diff_eff_run2_mask30 = []
    unc_total_diff_eff_run2_mask30 = []
    diff_eff_spanet_mask30 = []
    unc_diff_eff_spanet_mask30 = []
    total_diff_eff_spanet_mask30 = []
    unc_total_diff_eff_spanet_mask30 = []

    for j in range(len(list(true_dict.keys()))):
        diff_eff_run2_mask30.append([])
        unc_diff_eff_run2_mask30.append([])
        total_diff_eff_run2_mask30.append([])
        unc_total_diff_eff_run2_mask30.append([])
        for i in range(1, len(mhh_bins)):
            mask = (true_hh_fully_matched_mask30[j].mass > mhh_bins[i - 1]) & (
                true_hh_fully_matched_mask30[j].mass < mhh_bins[i]
            )
            eff_run2 = ak.sum(correctly_fully_matched_run2_mask30[j][mask]) / ak.count(
                correctly_fully_matched_run2_mask30[j][mask]
            )
            unc_eff_run2 = sqrt(
                eff_run2
                * (1 - eff_run2)
                / ak.count(correctly_fully_matched_run2_mask30[j][mask])
            )
            frac_fully_matched = ak.sum(mask_fully_matched[j][mask_30[j]][mask]) / len(
                mask_fully_matched[j][mask_30[j]][mask]
            )
            total_eff_run2 = eff_run2 * frac_fully_matched
            unc_total_eff_run2 = sqrt(
                (total_eff_run2 * (1 - total_eff_run2))
                / len(mask_fully_matched[j][mask_30[j]][mask])
            )
            diff_eff_run2_mask30[j].append(eff_run2)
            unc_diff_eff_run2_mask30[j].append(unc_eff_run2)
            total_diff_eff_run2_mask30[j].append(total_eff_run2)
            unc_total_diff_eff_run2_mask30[j].append(unc_total_eff_run2)

    for j in range(len(list(spanet_dict.keys()))):
        diff_eff_spanet_mask30.append([])
        unc_diff_eff_spanet_mask30.append([])
        total_diff_eff_spanet_mask30.append([])
        unc_total_diff_eff_spanet_mask30.append([])
        for i in range(1, len(mhh_bins)):
            mask = (
                true_hh_fully_matched_mask30[
                    check_names(list(spanet_dict.keys())[j])
                ].mass
                > mhh_bins[i - 1]
            ) & (
                true_hh_fully_matched_mask30[
                    check_names(list(spanet_dict.keys())[j])
                ].mass
                < mhh_bins[i]
            )
            eff_spanet = ak.sum(
                correctly_fully_matched_spanet_mask30[j][mask]
            ) / ak.count(correctly_fully_matched_spanet_mask30[j][mask])
            unc_eff_spanet = sqrt(
                eff_spanet
                * (1 - eff_spanet)
                / ak.count(correctly_fully_matched_spanet_mask30[j][mask])
            )
            frac_fully_matched = ak.sum(
                mask_fully_matched[check_names(list(spanet_dict.keys())[j])][
                    mask_30[check_names(list(spanet_dict.keys())[j])]
                ][mask]
            ) / len(
                mask_fully_matched[check_names(list(spanet_dict.keys())[j])][
                    mask_30[check_names(list(spanet_dict.keys())[j])]
                ][mask]
            )
            total_eff_spanet = eff_spanet * frac_fully_matched
            unc_total_eff_spanet = sqrt(
                (total_eff_spanet * (1 - total_eff_spanet))
                / len(
                    mask_fully_matched[check_names(list(spanet_dict.keys())[j])][
                        mask_30[check_names(list(spanet_dict.keys())[j])]
                    ][mask]
                )
            )
            diff_eff_spanet_mask30[j].append(eff_spanet)
            unc_diff_eff_spanet_mask30[j].append(unc_eff_spanet)
            total_diff_eff_spanet_mask30[j].append(total_eff_spanet)
            unc_total_diff_eff_spanet_mask30[j].append(unc_total_eff_spanet)

    print("Plotting differential efficiencies")
    plot_diff_eff(
        mhh_bins,
        # HERE
        (
            diff_eff_run2_mask30
            if not args.klambda
            else diff_eff_run2_mask30[: -len(kl_values_true)]
        ),
        (
            unc_diff_eff_run2_mask30
            if not args.klambda
            else unc_diff_eff_run2_mask30[: -len(kl_values_true)]
        ),
        true_dict,
        (
            diff_eff_spanet_mask30
            if not args.klambda
            else diff_eff_spanet_mask30[: -len(kl_values_spanet)]
        ),
        (
            unc_diff_eff_spanet_mask30
            if not args.klambda
            else unc_diff_eff_spanet_mask30[: -len(kl_values_spanet)]
        ),
        spanet_dict,
        plot_dir,
        "diff_eff_mask30",
    )
    plot_diff_eff(
        mhh_bins,
        (
            total_diff_eff_run2_mask30
            if not args.klambda
            else total_diff_eff_run2_mask30[: -len(kl_values_true)]
        ),
        (
            unc_total_diff_eff_run2_mask30
            if not args.klambda
            else unc_total_diff_eff_run2_mask30[: -len(kl_values_true)]
        ),
        true_dict,
        (
            total_diff_eff_spanet_mask30
            if not args.klambda
            else total_diff_eff_spanet_mask30[: -len(kl_values_spanet)]
        ),
        (
            unc_total_diff_eff_spanet_mask30
            if not args.klambda
            else unc_total_diff_eff_spanet_mask30[: -len(kl_values_spanet)]
        ),
        spanet_dict,
        plot_dir,
        "total_diff_eff_mask30",
    )


# All events

true_higgs_fully_matched = [
    best_reco_higgs(
        jet[i][mask_fully_matched[i]],
        idx_true_fully_matched[i],
    )
    for i in range(len(jet))
]

print("true_higgs_fully_matched", true_higgs_fully_matched)

true_hh_fully_matched = [
    true_higgs_fully_matched[i][:, 0] + true_higgs_fully_matched[i][:, 1]
    for i in range(len(true_higgs_fully_matched))
]
if not args.data:

    # print("Plotting true higgs")
    # plot_true_higgs(true_higgs_fully_matched, mh_bins, 1, plot_dir)
    # plot_true_higgs(true_higgs_fully_matched, mh_bins, 2, plot_dir)

    diff_eff_spanet = []
    unc_diff_eff_spanet = []
    total_diff_eff_spanet = []
    unc_total_diff_eff_spanet = []

    for j in range(len(list(spanet_dict.keys()))):
        diff_eff_spanet.append([])
        unc_diff_eff_spanet.append([])
        total_diff_eff_spanet.append([])
        unc_total_diff_eff_spanet.append([])
        for i in range(1, len(mhh_bins)):
            mask = (
                true_hh_fully_matched[check_names(list(spanet_dict.keys())[j])].mass
                > mhh_bins[i - 1]
            ) & (
                true_hh_fully_matched[check_names(list(spanet_dict.keys())[j])].mass
                < mhh_bins[i]
            )
            eff_spanet = ak.sum(correctly_fully_matched_spanet[j][mask]) / ak.count(
                correctly_fully_matched_spanet[j][mask]
            )
            unc_eff_spanet = sqrt(
                eff_spanet
                * (1 - eff_spanet)
                / ak.count(correctly_fully_matched_spanet[j][mask])
            )
            frac_fully_matched = ak.sum(
                mask_fully_matched[check_names(list(spanet_dict.keys())[j])][mask]
            ) / len(mask_fully_matched[check_names(list(spanet_dict.keys())[j])][mask])

            total_eff_spanet = eff_spanet * frac_fully_matched
            unc_total_eff_spanet = sqrt(
                (total_eff_spanet * (1 - total_eff_spanet))
                / len(
                    mask_fully_matched[check_names(list(spanet_dict.keys())[j])][mask]
                )
            )
            diff_eff_spanet[j].append(eff_spanet)
            unc_diff_eff_spanet[j].append(unc_eff_spanet)
            total_diff_eff_spanet[j].append(total_eff_spanet)
            unc_total_diff_eff_spanet[j].append(unc_total_eff_spanet)

    print("Plotting differential efficiencies")
    plot_diff_eff(
        mhh_bins,
        None,
        None,
        None,
        (
            diff_eff_spanet
            if not args.klambda
            else diff_eff_spanet[: -len(kl_values_spanet)]
        ),
        (
            unc_diff_eff_spanet
            if not args.klambda
            else unc_diff_eff_spanet[: -len(kl_values_spanet)]
        ),
        spanet_dict,
        plot_dir,
        "diff_eff_spanet",
    )
    plot_diff_eff(
        mhh_bins,
        None,
        None,
        None,
        (
            total_diff_eff_spanet
            if not args.klambda
            else total_diff_eff_spanet[: -len(kl_values_spanet)]
        ),
        (
            unc_total_diff_eff_spanet
            if not args.klambda
            else unc_total_diff_eff_spanet[: -len(kl_values_spanet)]
        ),
        spanet_dict,
        plot_dir,
        "total_diff_eff_spanet",
    )

    print("Plotting mhh")
    plot_mhh(
        mhh_bins,
        true_hh_fully_matched[0].mass,
        plot_dir,
        "mhh_fully_matched",
    )

# Reconstruction of the Higgs boson candidates with the predicted pairings

spanet_higgs_fully_matched_mask30 = [
    best_reco_higgs(
        jet_fully_matched_mask30[check_names(list(spanet_dict.keys())[i])],
        idx_spanet_pred_fully_matched_mask30[i],
    )
    for i in range(len(idx_spanet_pred_fully_matched_mask30))
]
run2_higgs_fully_matched_mask30 = [
    best_reco_higgs(j, idx)
    for j, idx in zip(jet_fully_matched_mask30, idx_run2_pred_fully_matched_mask30)
]

print("Plotting higgs 1d mask30")
for bins, name in zip([mh_bins, mh_bins_peak], ["", "_peak"]):
    for number in [1, 2]:
        plot_histos_1d(
            bins,
            [true[:, number - 1].mass for true in true_higgs_fully_matched_mask30],
            [run2[:, number - 1].mass for run2 in run2_higgs_fully_matched_mask30],
            [higgs[:, number - 1].mass for higgs in spanet_higgs_fully_matched_mask30],
            list(spanet_dict.keys()),
            list(true_dict.keys()),
            number,
            name=name + "_mask30",
            plot_dir=plot_dir,
        )

print("Plotting higgs 2d mask30")
# 2D histograms of the mass of the higgs1 and higgs2
labels_list = []
for sn, label in zip(
    [higgs for higgs in spanet_higgs_fully_matched_mask30], list(spanet_dict.keys())
):
    plot_histos_2d(
        mh_bins_2d[check_names(label)], sn, label, "SPANet_mask30", plot_dir=plot_dir
    )
    if check_names(label) in labels_list:
        continue

    if "data" not in label:
        plot_histos_2d(
            mh_bins_2d[check_names(label)],
            true_higgs_fully_matched_mask30[check_names(label)],
            list(true_dict.keys())[check_names(label)],
            "True_mask30",
            plot_dir=plot_dir,
        )
    plot_histos_2d(
        mh_bins_2d[check_names(label)],
        run2_higgs_fully_matched_mask30[check_names(label)],
        list(true_dict.keys())[check_names(label)],
        "Run2_mask30",
        plot_dir=plot_dir,
    )
    labels_list.append(check_names(label))


spanet_higgs_fully_matched = [
    best_reco_higgs(
        jet[check_names(list(spanet_dict.keys())[i])][
            mask_fully_matched[check_names(list(spanet_dict.keys())[i])]
        ],
        idx_spanet_pred_fully_matched[i],
    )
    for i in range(len(idx_spanet_pred_fully_matched))
]

print("Plotting higgs 1d all events")
for bins, name in zip([mh_bins, mh_bins_peak], ["", "_peak"]):
    for number in [1, 2]:
        plot_histos_1d(
            bins,
            [true[:, number - 1].mass for true in true_higgs_fully_matched],
            None,
            [higgs[:, number - 1].mass for higgs in spanet_higgs_fully_matched],
            list(spanet_dict.keys()),
            list(true_dict.keys()),
            number,
            name=name + "_all",
            plot_dir=plot_dir,
        )

print("Plotting higgs 2d all events")
# 2D histograms of the mass of the higgs1 and higgs2
labels_list = []
for sn, label in zip(
    [higgs for higgs in spanet_higgs_fully_matched], list(spanet_dict.keys())
):
    plot_histos_2d(
        mh_bins_2d[check_names(label)], sn, label, "SPANet", plot_dir=plot_dir
    )
    if check_names(label) in labels_list or "data" in label:
        continue
    plot_histos_2d(
        mh_bins_2d[check_names(label)],
        true_higgs_fully_matched[check_names(label)],
        list(true_dict.keys())[check_names(label)],
        "True",
        plot_dir=plot_dir,
    )
    labels_list.append(check_names(label))


# separate between high and low mhh spectrum
mask_hh_mass_400_mask30 = [
    (true_hh_fully_matched_mask30[i].mass > 400)
    & (true_hh_fully_matched_mask30[i].mass < 700)
    for i in range(len(true_hh_fully_matched_mask30))
]
#print("Plotting higgs 1d for high and low mhh mask30")
#for number in range(1, 3):
#    for bins, name in zip([mh_bins, mh_bins_peak], ["", "_peak"]):
#        for mask_mhh, name_mhh in zip(
#            [mask_hh_mass_400_mask30, [~m for m in mask_hh_mass_400_mask30]],
#            ["_mass400_700", "_mass0_400"],
#        ):
#            plot_histos_1d(
#                bins,
#                [
#                    true[mask][:, number - 1].mass
#                    for true, mask in zip(true_higgs_fully_matched_mask30, mask_mhh)
#                ],
#                [
#                    run2[mask][:, number - 1].mass
#                    for run2, mask in zip(run2_higgs_fully_matched_mask30, mask_mhh)
#                ],
#                [
#                    spanet_higgs_fully_matched_mask30[i][
#                        mask_mhh[check_names(list(spanet_dict.keys())[i])]
#                    ][:, number - 1].mass
#                    for i in range(len(spanet_higgs_fully_matched_mask30))
#                ],
#                list(spanet_dict.keys()),
#                list(true_dict.keys()),
#                number,
#                name + name_mhh + "_mask30",
#                plot_dir=plot_dir,
#            )


## all events
#mask_hh_mass_400 = [
#    (true_hh_fully_matched[i].mass > 400) & (true_hh_fully_matched[i].mass < 700)
#    for i in range(len(true_hh_fully_matched))
#]
#print("Plotting higgs 1d for high and low mhh all events")
#for number in range(1, 3):
#    for bins, name in zip([mh_bins, mh_bins_peak], ["", "_peak"]):
#        for mask_mhh, name_mhh in zip(
#            [mask_hh_mass_400, [~m for m in mask_hh_mass_400]],
#            ["_mass400_700", "_mass0_400"],
#        ):
#            plot_histos_1d(
#                bins,
#                [
#                    true[mask][:, number - 1].mass
#                    for true, mask in zip(true_higgs_fully_matched, mask_mhh)
#                ],
#                None,
#                [
#                    spanet_higgs_fully_matched[i][
#                        mask_mhh[check_names(list(spanet_dict.keys())[i])]
#                    ][:, number - 1].mass
#                    for i in range(len(spanet_higgs_fully_matched))
#                ],
#                list(spanet_dict.keys()),
#                list(true_dict.keys()),
#                number,
#                name + name_mhh + "_all",
#                plot_dir=plot_dir,
#            )
