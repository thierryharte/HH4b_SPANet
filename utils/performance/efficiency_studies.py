import awkward as ak
import numpy as np
import h5py
import vector
from math import sqrt
import matplotlib.pyplot as plt
import os

vector.register_numba()
vector.register_awkward()

import argparse

from efficiency_functions import (
    distance_func,
    reco_higgs,
    best_reco_higgs,
    plot_histos,
    check_names,
)

parser = argparse.ArgumentParser(
    description="Convert awkward ntuples in parquet files to h5 files."
)
parser.add_argument(
    "-it",
    "--input-true",
    nargs="*",
    default=[],
    help="Input file with true pairing",
)
parser.add_argument(
    "-ip",
    "--input-spanet-pred",
    nargs="*",
    default=[],
    help="Input files with predicted pairing",
)

args = parser.parse_args()

if args.input_spanet_pred:
    list_spanet_pred = args.input_spanet_pred
    labels_spanet_pred = [
        (f.split("_prediction_")[-1]).split(".h5")[-2] for f in list_spanet_pred
    ]
    spanet_dict = dict(zip(labels_spanet_pred, list_spanet_pred))
else:
    spanet_dir = "/eos/home-r/ramellar/out_prediction_files/"
    spanet_dict = {
        "4_jets": spanet_dir + "out_0_spanet_prediction_4jets.h5",
        # "5_jets":spanet_dir+ "out_1_spanet_prediction_5jets.h5",
        # "5_jets_btag_presel":spanet_dir+ "out_2_spanet_prediction_5jets_btagpresel.h5",
        "4_jets_5global": spanet_dir
        + "out_3_spanet_prediction_4jets_5global_9999pad.h5",
        "4_jets_5global_btagpresel": spanet_dir
        + "out_4_spanet_prediction_4jets_5global_9999pad_btagpresel.h5",
        "4_jets_5global_ATLAS": spanet_dir + "out_5_spanet_prediction_ATLAS.h5",
        "4_jets_5global_ptreg": spanet_dir
        + "out_7_spanet_prediction_4jets_5global_ptreg_klambda1.h5",
        # "4_jets_5global_ptreg_klambda0":spanet_dir + "out_7_spanet_prediction_4jets_5global_ptreg_klambda0.h5",
        # "4_jets_5global_ptreg_klambda2p45":spanet_dir + "out_7_spanet_prediction_4jets_5global_ptreg_klambda2p45.h5",
        # "4_jets_5global_ptreg_klambda5":spanet_dir + "out_7_spanet_prediction_4jets_5global_ptreg_klambda5.h5",
        "4_jets_5global_ATLAS_ptreg": spanet_dir
        + "out_9_spanet_prediction_4jets_5global_ATLAS_ptreg_klambda1.h5",
        # "4_jets_5global_ATLAS_ptreg_klambda0":spanet_dir + "out_9_spanet_prediction_4jets_5global_ATLAS_ptreg_klambda0.h5",
        # "4_jets_5global_ATLAS_ptreg_klambda2p45":spanet_dir + "out_9_spanet_prediction_4jets_5global_ATLAS_ptreg_klambda2p45.h5",
        # "4_jets_5global_ATLAS_ptreg_klambda5":spanet_dir + "out_9_spanet_prediction_4jets_5global_ATLAS_ptreg_klambda5.h5",
        "4_jets_5global_ATLAS_ptreg_cos_sin_phi": spanet_dir
        + "out_01_spanet_prediction_ATLAS_4jets_5global_ptreg_cos_sin_phi.h5",
        "4_jets_5global_ptreg_cos_sin_phi": spanet_dir
        + "out_01_spanet_prediction_4jets_5global_ptreg_cos_sin_phi.h5",
    }

if args.input_true:
    input_true = args.input_true
    labels_true = [(f.split("_prediction_")[-1]).split(".h5")[-2] for f in input_true]
    true_dict = dict(zip(labels_true, input_true))
else:
    true_dir = "/eos/home-r/ramellar/out_prediction_files/"
    true_dict = {
        "4_jets": true_dir + "output_JetGoodHiggs_test.h5",
        "5_jets": true_dir + "output_JetGood_test.h5",
        "klambda0": true_dir + "kl0_output_JetGoodHiggs_test.h5",
        "klambda2p45": true_dir + "kl2p45_output_JetGoodHiggs_test.h5",
        "klambda5": true_dir + "kl5_output_JetGoodHiggs_test.h5",
    }


plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# open files
df_true = [h5py.File(f, "r") for f in true_dict.values()]
df_spanet_pred = [h5py.File(f, "r") for f in spanet_dict.values()]


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

# Fully matched events
mask_fully_matched = [ak.all(ak.all(idx >= 0, axis=-1), axis=-1) for idx in idx_true]

idx_true_fully_matched = [idx[mask] for idx, mask in zip(idx_true, mask_fully_matched)]

idx_spanet_pred_fully_matched = [
    idx_spanet_pred[i][mask_fully_matched[check_names(list(spanet_dict.keys())[i])]]
    for i in range(len(idx_spanet_pred))
]

correctly_fully_matched_spanet = [
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
    for i in range(len(idx_spanet_pred_fully_matched))
]

# compute efficiencies for fully matched events
efficiencies_fully_matched_spanet = [
    ak.sum(correctly_fully_matched) / len(correctly_fully_matched)
    for correctly_fully_matched in correctly_fully_matched_spanet
]
frac_fully_matched = [ak.sum(mask) / len(mask) for mask in mask_fully_matched]
for label, frac in zip(list(true_dict.keys()), frac_fully_matched):
    print(f"Fraction of fully matched events for {label}: {frac:.4f}")
for label, eff in zip(list(spanet_dict.keys()), efficiencies_fully_matched_spanet):
    print("Efficiency fully matched for {}: {:.4f}".format(label, eff))

# do the same for partially matched events (only one higgs is matched)
mask_1h = [ak.any(idx == -1, axis=-1) == 1 for idx in idx_true]
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
for label, frac in zip(list(true_dict.keys()), frac_partially_matched_1h):
    print(f"Fraction of partially matched events for {label}: {frac:.4f}")
for label, eff in zip(list(spanet_dict.keys()), efficiencies_partially_matched_spanet):
    print("Efficiency partially matched for {}: {:.4f}".format(label, eff))

# compute number of events with 0 higgs matched
mask_0h = [ak.all(idx == -1, axis=-1) == 2 for idx in idx_true]


idx_true_unmatched = [idx[mask] for idx, mask in zip(idx_true, mask_0h)]
frac_unmatched = [ak.sum(mask) / len(mask) for mask in mask_0h]
for label, frac in zip(list(true_dict.keys()), frac_unmatched):
    print(f"Fraction of unmatched events for {label}: {frac:.4f}")


# load jet information
jet_ptPNetRegNeutrino = [df["INPUTS"]["Jet"]["ptPnetRegNeutrino"][()] for df in df_true]
jet_eta = [df["INPUTS"]["Jet"]["eta"][()] for df in df_true]
jet_phi = [df["INPUTS"]["Jet"]["phi"][()] for df in df_true]
jet_mass = [df["INPUTS"]["Jet"]["mass"][()] for df in df_true]

# create a LorentzVector for the jets
jet = [
    ak.zip(
        {
            "pt": ptPNetRegNeutrino,
            "eta": eta,
            "phi": phi,
            "mass": mass,
        },
        with_name="Momentum4D",
    )
    for ptPNetRegNeutrino, eta, phi, mass in zip(
        jet_ptPNetRegNeutrino, jet_eta, jet_phi, jet_mass
    )
]


# implement the Run2 pairing algorithm
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
    np.tile(comb_idx, (len(m), 1, 1, 1))[mask] for m, mask in zip(comb_idx, mask_30)
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

# compute efficiencies for fully matched events for Run2 pairing
idx_true_fully_matched_mask30 = [
    idx[m_30][m[m_30]] for idx, m, m_30 in zip(idx_true, mask_fully_matched, mask_30)
]
correctly_fully_matched_run2_mask30 = [
    ak.all(
        i[:, 0] == i2[:, 0],
        axis=1,
    )
    | ak.all(
        i[:, 0] == i2[:, 1],
        axis=1,
    )
    for i, i2 in zip(idx_true_fully_matched_mask30, idx_run2_pred_fully_matched_mask30)
]
efficiency_fully_matched_run2_mask30 = [
    ak.sum(corr) / len(corr) for corr in correctly_fully_matched_run2_mask30
]
for label, eff in zip(list(true_dict.keys()), efficiency_fully_matched_run2_mask30):
    print(f"Efficiency fully matched for {label} (DeltaR>30): {eff:.4f}")


# compute efficiencies for fully matched events for spanet
idx_spanet_pred_fully_matched_mask30 = [
    idx_spanet_pred[i][mask_30[check_names(list(spanet_dict.keys())[i])]][
        mask_fully_matched[check_names(list(spanet_dict.keys())[i])][
            mask_30[check_names(list(spanet_dict.keys())[i])]
        ]
    ]
    for i in range(len(idx_spanet_pred))
]
correctly_fully_matched_spanet_mask30 = [
    ak.all(
        idx_true_fully_matched_mask30[check_names(list(spanet_dict.keys())[i])][:, 0]
        == idx_spanet_pred_fully_matched_mask30[i][:, 0],
        axis=1,
    )
    | ak.all(
        idx_true_fully_matched_mask30[check_names(list(spanet_dict.keys())[i])][:, 0]
        == idx_spanet_pred_fully_matched_mask30[i][:, 1],
        axis=1,
    )
    for i in range(len(idx_spanet_pred_fully_matched_mask30))
]
efficiencies_fully_matched_spanet_mask30 = [
    ak.sum(correctly_fully_matched) / len(correctly_fully_matched)
    for correctly_fully_matched in correctly_fully_matched_spanet_mask30
]
for label, eff in zip(
    list(spanet_dict.keys()), efficiencies_fully_matched_spanet_mask30
):
    print(f"Efficiency fully matched for {label} (DeltaR>30): {eff:.4f}")

# Reconstruct the Higgs boson candidates with the four-vectors
# of the jets considering the true pairings, the spanet pairings
# and the run2 pairings
jet_fully_matched_mask30 = [
    j[m_30][m[m_30]] for j, m_30, m in zip(jet, mask_30, mask_fully_matched)
]
true_higgs_fully_matched_mask30 = [
    best_reco_higgs(j, idx)
    for j, idx in zip(jet_fully_matched_mask30, idx_true_fully_matched_mask30)
]
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

# for each event plot the mass of the higgs1 and higgs2
mh_bins = np.linspace(50, 200, 80)
plot_histos(
    mh_bins,
    true_higgs_fully_matched_mask30[0][:, 0].mass,
    run2_higgs_fully_matched_mask30[0][:, 0].mass,
    [higgs[:, 0].mass for higgs in spanet_higgs_fully_matched_mask30],
    list(spanet_dict.keys()),
    1,
)
plot_histos(
    mh_bins,
    true_higgs_fully_matched_mask30[0][:, 1].mass,
    run2_higgs_fully_matched_mask30[0][:, 1].mass,
    [higgs[:, 1].mass for higgs in spanet_higgs_fully_matched_mask30],
    list(spanet_dict.keys()),
    2,
)

true_hh_fully_matched_mask30 = [
    true_higgs_fully_matched_mask30[i][:, 0] + true_higgs_fully_matched_mask30[i][:, 1]
    for i in range(len(true_higgs_fully_matched_mask30))
]

diff_eff_run2_mask30 = []
unc_diff_eff_run2_mask30 = []
diff_eff_spanet_mask30 = []
unc_diff_eff_spanet_mask30 = []


mhh_bins = np.linspace(250, 700, 10)

for i in range(1, len(mhh_bins)):
    mask = (true_hh_fully_matched_mask30[0].mass > mhh_bins[i - 1]) & (
        true_hh_fully_matched_mask30[0].mass < mhh_bins[i]
    )
    eff_run2 = ak.sum(correctly_fully_matched_run2_mask30[0][mask]) / ak.count(
        correctly_fully_matched_run2_mask30[0][mask]
    )
    unc_eff_run2 = sqrt(
        eff_run2
        * (1 - eff_run2)
        / ak.count(correctly_fully_matched_run2_mask30[0][mask])
    )
    diff_eff_run2_mask30.append(eff_run2)
    unc_diff_eff_run2_mask30.append(unc_eff_run2)

for j in range(len(list(spanet_dict.keys()))):
    diff_eff_spanet_mask30.append([])
    unc_diff_eff_spanet_mask30.append([])
    for i in range(1, len(mhh_bins)):
        mask = (
            true_hh_fully_matched_mask30[check_names(list(spanet_dict.keys())[j])].mass
            > mhh_bins[i - 1]
        ) & (
            true_hh_fully_matched_mask30[check_names(list(spanet_dict.keys())[j])].mass
            < mhh_bins[i]
        )
        eff_spanet = ak.sum(correctly_fully_matched_spanet_mask30[j][mask]) / ak.count(
            correctly_fully_matched_spanet_mask30[j][mask]
        )
        unc_eff_spanet = sqrt(
            eff_spanet
            * (1 - eff_spanet)
            / ak.count(correctly_fully_matched_spanet_mask30[j][mask])
        )
        diff_eff_spanet_mask30[j].append(eff_spanet)
        unc_diff_eff_spanet_mask30[j].append(unc_eff_spanet)

fig, ax = plt.subplots(figsize=(10, 8))
plt.errorbar(
    0.5 * (mhh_bins[1:] + mhh_bins[:-1]),
    diff_eff_run2_mask30,
    yerr=unc_diff_eff_run2_mask30,
    label="Run2",
    color="red",
    marker="o",
)
for eff, unc_eff, label in zip(
    diff_eff_spanet_mask30, unc_diff_eff_spanet_mask30, list(spanet_dict.keys())
):
    plt.errorbar(
        0.5 * (mhh_bins[1:] + mhh_bins[:-1]),
        eff,
        yerr=unc_eff,
        label=f"SPANet {label}",
        marker="o",
    )

fig.legend()
plt.savefig(f"{plot_dir}/diff_eff_mask30.png")


# do the same for all events

true_higgs_fully_matched = [
    best_reco_higgs(j, idx)
    for j, idx in zip(jet[mask_fully_matched], idx_true_fully_matched)
]
true_hh_fully_matched = [
    true_higgs_fully_matched[i][:, 0] + true_higgs_fully_matched[i][:, 1]
    for i in range(len(true_higgs_fully_matched))
]


diff_eff_spanet = []
unc_diff_eff_spanet = []

for j in range(len(list(spanet_dict.keys()))):
    diff_eff_spanet.append([])
    unc_diff_eff_spanet.append([])
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
        diff_eff_spanet[j].append(eff_spanet)
        unc_diff_eff_spanet[j].append(unc_eff_spanet)


print(unc_diff_eff_spanet)
print(diff_eff_spanet)

fig, ax = plt.subplots(figsize=(10, 8))
for j in range(len(list(spanet_dict.keys()))):
    plt.errorbar(
        0.5 * (mhh_bins[1:] + mhh_bins[:-1]),
        diff_eff_spanet[j],
        yerr=unc_diff_eff_spanet[j],
        label=f"SPANet {list(spanet_dict.keys())[j]}",
        marker="o",
    )
fig.legend()
plt.savefig(f"{plot_dir}/diff_eff_spanet.png")


mask_hh_mass_400 = [
    (true_hh_fully_matched_mask30[i].mass > 400)
    & (true_hh_fully_matched_mask30[i].mass < 700)
    for i in range(len(true_hh_fully_matched_mask30))
]

plot_histos(
    mh_bins,
    true_higgs_fully_matched_mask30[mask_hh_mass_400[0]][:, 0].mass,
    run2_higgs_fully_matched_mask30[mask_hh_mass_400[0]][:, 0].mass,
    [
        spanet_higgs_fully_matched_mask30[i][
            mask_hh_mass_400[check_names(list(spanet_dict.keys())[i])]
        ][:, 0].mass
        for i in range(len(spanet_higgs_fully_matched_mask30))
    ],
    list(spanet_dict.keys()),
    1,
    "_mass400_700",
)
plot_histos(
    mh_bins,
    true_higgs_fully_matched_mask30[mask_hh_mass_400[0]][:, 1].mass,
    run2_higgs_fully_matched_mask30[mask_hh_mass_400[0]][:, 1].mass,
    [
        spanet_higgs_fully_matched_mask30[i][
            mask_hh_mass_400[check_names(list(spanet_dict.keys())[i])]
        ][:, 1].mass
        for i in range(len(spanet_higgs_fully_matched_mask30))
    ],
    list(spanet_dict.keys()),
    2,
    "_mass400_700",
)

# plot the same but for the anti mask
plot_histos(
    mh_bins,
    true_higgs_fully_matched_mask30[~mask_hh_mass_400[0]][:, 0].mass,
    run2_higgs_fully_matched_mask30[~mask_hh_mass_400[0]][:, 0].mass,
    [
        spanet_higgs_fully_matched_mask30[i][
            ~mask_hh_mass_400[check_names(list(spanet_dict.keys())[i])]
        ][:, 0].mass
        for i in range(len(spanet_higgs_fully_matched_mask30))
    ],
    list(spanet_dict.keys()),
    1,
    "_mass0_400",
)

plot_histos(
    mh_bins,
    true_higgs_fully_matched_mask30[~mask_hh_mass_400][:, 1].mass,
    run2_higgs_fully_matched_mask30[~mask_hh_mass_400][:, 1].mass,
    [
        spanet_higgs_fully_matched_mask30[i][
            ~mask_hh_mass_400[check_names(list(spanet_dict.keys())[i])]
        ][:, 1].mass
        for i in range(len(spanet_higgs_fully_matched_mask30))
    ],
    list(spanet_dict.keys()),
    2,
    "_mass0_400",
)
