import awkward as ak
import numpy as np
import h5py
import vector
from math import sqrt
import matplotlib.pyplot as plt

vector.register_numba()
vector.register_awkward()

import argparse

from pairing_functions import distance_func, reco_higgs, best_reco_higgs, plot_histos

parser = argparse.ArgumentParser(
    description="Convert awkward ntuples in parquet files to h5 files."
)
parser.add_argument(
    "-it", "--input-true", type=str, required=True, help="Input file with true pairing"
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
    labels_spanet_pred = [(f.split("_prediction_")[-1]).split(".h5")[-2] for f in list_spanet_pred]
    spanet_dict = dict(zip(labels_spanet_pred, list_spanet_pred))
else:
    spanet_dict={
        "example_name": "example_file.h5",
        "example_name2": "example_file2.h5",
    }

plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# open files
df_true = h5py.File(args.input_true, "r")
df_spanet_pred = [h5py.File(f, "r") for f in spanet_dict.values()]


idx_b1_true = df_true["TARGETS"]["h1"]["b1"][()]
idx_b2_true = df_true["TARGETS"]["h1"]["b2"][()]
idx_b3_true = df_true["TARGETS"]["h2"]["b3"][()]
idx_b4_true = df_true["TARGETS"]["h2"]["b4"][()]

idx_b1_spanet_pred = [df["TARGETS"]["h1"]["b1"][()] for df in df_spanet_pred]
idx_b2_spanet_pred = [df["TARGETS"]["h1"]["b2"][()] for df in df_spanet_pred]
idx_b3_spanet_pred = [df["TARGETS"]["h2"]["b3"][()] for df in df_spanet_pred]
idx_b4_spanet_pred = [df["TARGETS"]["h2"]["b4"][()] for df in df_spanet_pred]

idx_h1_true = ak.concatenate(
    (
        ak.unflatten(idx_b1_true, ak.ones_like(idx_b1_true)),
        ak.unflatten(idx_b2_true, ak.ones_like(idx_b2_true)),
    ),
    axis=1,
)
idx_h2_true = ak.concatenate(
    (
        ak.unflatten(idx_b3_true, ak.ones_like(idx_b3_true)),
        ak.unflatten(idx_b4_true, ak.ones_like(idx_b4_true)),
    ),
    axis=1,
)

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

idx_true = ak.concatenate(
    (
        ak.unflatten(idx_h1_true, ak.ones_like(idx_h1_true[:, 0])),
        ak.unflatten(idx_h2_true, ak.ones_like(idx_h2_true[:, 0])),
    ),
    axis=1,
)
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

# FUlly matched events
mask_fully_matched = ak.all(ak.all(idx_true >= 0, axis=-1), axis=-1)

idx_true_fully_matched = idx_true[mask_fully_matched]
idx_spanet_pred_fully_matched = [idx[mask_fully_matched] for idx in idx_spanet_pred]

correctly_fully_matched_spanet = [
    ak.all(idx_true_fully_matched[:, 0] == idx[:, 0], axis=1)
    | ak.all(idx_true_fully_matched[:, 0] == idx[:, 1], axis=1)
    for idx in idx_spanet_pred_fully_matched
]

# compute efficiencies for fully matched events
efficiencies_fully_matched_spanet = [
    ak.sum(correctly_fully_matched) / len(correctly_fully_matched)
    for correctly_fully_matched in correctly_fully_matched_spanet
]
frac_fully_matched = ak.sum(mask_fully_matched) / len(mask_fully_matched)
print("Fraction of fully matched events: {:.4f}".format(frac_fully_matched))
for label, eff in zip(list(spanet_dict.keys()), efficiencies_fully_matched_spanet):
    print("Efficiency fully matched for {}: {:.4f}".format(label, eff))

# do the same for partially matched events (only one higgs is matched)
mask_1h = ak.sum(ak.any(idx_true == -1, axis=-1), axis=-1) == 1
idx_true_partially_matched_1h = idx_true[mask_1h]
idx_spanet_pred_partially_matched_1h = [idx[mask_1h] for idx in idx_spanet_pred]

correctly_partially_matched_spanet = [
    (
        ak.all(
            idx_true_partially_matched_1h[:, 0] == idx[:, 0],
            axis=1,
        )
        | ak.all(
            idx_true_partially_matched_1h[:, 0] == idx[:, 1],
            axis=1,
        )
        | ak.all(
            idx_true_partially_matched_1h[:, 1] == idx[:, 0],
            axis=1,
        )
        | ak.all(
            idx_true_partially_matched_1h[:, 1] == idx[:, 1],
            axis=1,
        )
    )
    for idx in idx_spanet_pred_partially_matched_1h
]

efficiencies_partially_matched_spanet = [
    ak.sum(correctly_partially_matched_1h) / len(correctly_partially_matched_1h)
    for correctly_partially_matched_1h in correctly_partially_matched_spanet
]
frac_partially_matched_1h = ak.sum(mask_1h) / len(mask_1h)
print(
    "Fraction of partially matched events (1h): {:.4f}".format(
        frac_partially_matched_1h
    )
)
for label, eff in zip(list(spanet_dict.keys()), efficiencies_partially_matched_spanet):
    print("Efficiency partially matched for {}: {:.4f}".format(label, eff))

# compute number of events with 0 higgs matched
mask_0h = ak.sum(ak.any(idx_true == -1, axis=-1), axis=-1) == 2

idx_true_unmatched = idx_true[mask_0h]
idx_spanet_pred_unmatched = [idx[mask_0h] for idx in idx_spanet_pred]
frac_unmatched = ak.sum(mask_0h) / len(mask_0h)
print("Fraction of unmatched events: {:.4f}".format(frac_unmatched))

# load jet information
jet_ptPNetRegNeutrino = df_true["INPUTS"]["Jet"]["ptPnetRegNeutrino"][()]
jet_eta = df_true["INPUTS"]["Jet"]["eta"][()]
jet_phi = df_true["INPUTS"]["Jet"]["phi"][()]
jet_mass = df_true["INPUTS"]["Jet"]["mass"][()]

# create a LorentzVector for the jets
jet = ak.zip(
    {
        "pt": jet_ptPNetRegNeutrino,
        "eta": jet_eta,
        "phi": jet_phi,
        "mass": jet_mass,
    },
    with_name="Momentum4D",
)


# implement the Run2 pairing algorithm
comb_idx = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]

higgs_candidates_unflatten_order = reco_higgs(jet, comb_idx)
distance = distance_func(
    higgs_candidates_unflatten_order,
    1.04,
)
dist_order_idx = ak.argsort(distance, axis=1, ascending=True)
dist_order = distance[dist_order_idx]

# if the distance between the two best candidates is less than 30, we do not consider the event
min_idx = ak.where(dist_order[:, 1] - dist_order[:, 0] > 30, dist_order_idx[:, 0], -1)
mask_30 = min_idx != -1


comb_idx_mask30 = np.tile(comb_idx, (len(min_idx), 1, 1, 1))[mask_30]
min_idx_mask30 = min_idx[mask_30]
# given the min_idx, select the correct combination corresponding to the index
comb_idx_min_mask30 = comb_idx_mask30[np.arange(len(min_idx_mask30)), min_idx_mask30]
idx_run2_pred_fully_matched_mask30 = ak.Array(comb_idx_min_mask30)[
    mask_fully_matched[mask_30]
]

# compute efficiencies for fully matched events for Run2 pairing
idx_true_fully_matched_mask30 = idx_true[mask_30][mask_fully_matched[mask_30]]
correctly_fully_matched_run2_mask30 = ak.all(
    idx_true_fully_matched_mask30[:, 0] == idx_run2_pred_fully_matched_mask30[:, 0],
    axis=1,
) | ak.all(
    idx_true_fully_matched_mask30[:, 0] == idx_run2_pred_fully_matched_mask30[:, 1],
    axis=1,
)
efficiency_fully_matched_run2_mask30 = ak.sum(
    correctly_fully_matched_run2_mask30
) / len(correctly_fully_matched_run2_mask30)
print(
    f"Efficiency fully matched for Run2 (DeltaR>30): {efficiency_fully_matched_run2_mask30:.4f}"
)

# compute efficiencies for fully matched events for spanet
idx_spanet_pred_fully_matched_mask30 = [
    idx[mask_30][mask_fully_matched[mask_30]] for idx in idx_spanet_pred
]
correctly_fully_matched_spanet_mask30 = [
    ak.all(idx_true_fully_matched_mask30[:, 0] == idx[:, 0], axis=1)
    | ak.all(idx_true_fully_matched_mask30[:, 0] == idx[:, 1], axis=1)
    for idx in idx_spanet_pred_fully_matched_mask30
]
efficiencies_fully_matched_spanet_mask30 = [
    ak.sum(correctly_fully_matched) / len(correctly_fully_matched)
    for correctly_fully_matched in correctly_fully_matched_spanet_mask30
]
for label, eff in zip(list(spanet_dict.keys()), efficiencies_fully_matched_spanet_mask30):
    print(f"Efficiency fully matched for {label} (DeltaR>30): {eff:.4f}")

# Reconstruct the Higgs boson candidates with the four-vectors
# of the jets considering the true pairings, the spanet pairings
# and the run2 pairings
jet_fully_matched_mask30 = jet[mask_30][mask_fully_matched[mask_30]]
true_higgs_fully_matched_mask30 = best_reco_higgs(
    jet_fully_matched_mask30, idx_true_fully_matched_mask30
)
spanet_higgs_fully_matched_mask30 = [
    best_reco_higgs(jet_fully_matched_mask30, idx)
    for idx in idx_spanet_pred_fully_matched_mask30
]
run2_higgs_fully_matched_mask30 = best_reco_higgs(
    jet_fully_matched_mask30, idx_run2_pred_fully_matched_mask30
)

# for each event plot the mass of the higgs1 and higgs2
mh_bins = np.linspace(50, 200, 80)
plot_histos(
    mh_bins,
    true_higgs_fully_matched_mask30[:, 0].mass,
    run2_higgs_fully_matched_mask30[:, 0].mass,
    [higgs[:, 0].mass for higgs in spanet_higgs_fully_matched_mask30],
    list(spanet_dict.keys()),
    1,
)
plot_histos(
    mh_bins,
    true_higgs_fully_matched_mask30[:, 1].mass,
    run2_higgs_fully_matched_mask30[:, 1].mass,
    [higgs[:, 1].mass for higgs in spanet_higgs_fully_matched_mask30],
    list(spanet_dict.keys()),
    2,
)

true_hh_fully_matched_mask30 = (
    true_higgs_fully_matched_mask30[:, 0] + true_higgs_fully_matched_mask30[:, 1]
)

diff_eff_run2_mask30 = []
unc_diff_eff_run2_mask30 = []
diff_eff_spanet_mask30 = []
unc_diff_eff_spanet_mask30 = []


mhh_bins = np.linspace(250, 700, 10)

for i in range(1, len(mhh_bins)):
    mask = (true_hh_fully_matched_mask30.mass > mhh_bins[i - 1]) & (
        true_hh_fully_matched_mask30.mass < mhh_bins[i]
    )
    eff_run2 = ak.sum(correctly_fully_matched_run2_mask30[mask]) / ak.count(
        correctly_fully_matched_run2_mask30[mask]
    )
    unc_eff_run2 = sqrt(
        eff_run2 * (1 - eff_run2) / ak.count(correctly_fully_matched_run2_mask30[mask])
    )
    diff_eff_run2_mask30.append(eff_run2)
    unc_diff_eff_run2_mask30.append(unc_eff_run2)

for j in range(len(list(spanet_dict.keys()))):
    diff_eff_spanet_mask30.append([])
    unc_diff_eff_spanet_mask30.append([])
    for i in range(1, len(mhh_bins)):
        mask = (true_hh_fully_matched_mask30.mass > mhh_bins[i - 1]) & (
            true_hh_fully_matched_mask30.mass < mhh_bins[i]
        )
        eff_spanet = ak.sum(correctly_fully_matched_spanet_mask30[j][mask]) / ak.count(
            correctly_fully_matched_spanet_mask30[j][mask]
        )
        unc_eff_spanet = sqrt(
            eff_spanet * (1 - eff_spanet) / ak.count(correctly_fully_matched_spanet_mask30[j][mask])
        )
        diff_eff_spanet_mask30[j].append(eff_spanet)
        unc_diff_eff_spanet_mask30[j].append(unc_eff_spanet)

fig, ax = plt.subplots()
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

true_higgs_fully_matched = best_reco_higgs(
    jet[mask_fully_matched], idx_true_fully_matched
)
true_hh_fully_matched = true_higgs_fully_matched[:, 0] + true_higgs_fully_matched[:, 1]


diff_eff_spanet = []
unc_diff_eff_spanet = []

for j in range(len(list(spanet_dict.keys()))):
    diff_eff_spanet.append([])
    unc_diff_eff_spanet.append([])
    for i in range(1, len(mhh_bins)):
        mask = (true_hh_fully_matched.mass > mhh_bins[i - 1]) & (
            true_hh_fully_matched.mass < mhh_bins[i]
        )
        eff_spanet = ak.sum(correctly_fully_matched_spanet[j][mask]) / ak.count(
            correctly_fully_matched_spanet[j][mask]
        )
        unc_eff_spanet = sqrt(
            eff_spanet * (1 - eff_spanet) / ak.count(correctly_fully_matched_spanet[j][mask])
        )
        diff_eff_spanet[j].append(eff_spanet)
        unc_diff_eff_spanet[j].append(unc_eff_spanet)


print(unc_diff_eff_spanet)
print(diff_eff_spanet)

fig, ax = plt.subplots()
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


mask_hh_mass_400 = (true_hh_fully_matched_mask30.mass>400) & (true_hh_fully_matched_mask30.mass<700)

plot_histos(
    mh_bins,
    true_higgs_fully_matched_mask30[mask_hh_mass_400][:, 0].mass,
    run2_higgs_fully_matched_mask30[mask_hh_mass_400][:, 0].mass,
    [higgs[mask_hh_mass_400][:, 0].mass for higgs in spanet_higgs_fully_matched_mask30],
    list(spanet_dict.keys()),
    1,
    "_mass400_700"
)
plot_histos(
    mh_bins,
    true_higgs_fully_matched_mask30[mask_hh_mass_400][:, 1].mass,
    run2_higgs_fully_matched_mask30[mask_hh_mass_400][:, 1].mass,
    [higgs[mask_hh_mass_400][:, 1].mass for higgs in spanet_higgs_fully_matched_mask30],
    list(spanet_dict.keys()),
    2,
    "_mass400_700"
)

# plot the same but for the anti mask
plot_histos(
    mh_bins,
    true_higgs_fully_matched_mask30[~mask_hh_mass_400][:, 0].mass,
    run2_higgs_fully_matched_mask30[~mask_hh_mass_400][:, 0].mass,
    [higgs[~mask_hh_mass_400][:, 0].mass for higgs in spanet_higgs_fully_matched_mask30],
    list(spanet_dict.keys()),
    1,
    "_mass0_400"
)

plot_histos(
    mh_bins,
    true_higgs_fully_matched_mask30[~mask_hh_mass_400][:, 1].mass,
    run2_higgs_fully_matched_mask30[~mask_hh_mass_400][:, 1].mass,
    [higgs[~mask_hh_mass_400][:, 1].mass for higgs in spanet_higgs_fully_matched_mask30],
    list(spanet_dict.keys()),
    2,
    "_mass0_400"
)