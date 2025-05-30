import awkward as ak
import numpy as np
import h5py
import vector
from math import sqrt
import os
import argparse

from efficiency_functions import separate_klambda, plot_diff_eff, plot_diff_eff_klambda, plot_histos_1d, plot_histos_2d, plot_mhh, reco_higgs, distance_pt_func, best_reco_higgs
from efficiency_configuration import spanet_dict, true_dict, run2_dataset

vector.register_numba()
vector.register_awkward()

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


if args.data:
    # remove non data samples
    spanet_dict = {k: v for k, v in spanet_dict.items() if "data" in k}
else:
    spanet_dict = {k: v for k, v in spanet_dict.items() if "data" not in k}

print(spanet_dict)

mh_bins = np.linspace(0,300,150)
mh_bins_peak = np.linspace(100,140,20)
mh_bins_2d = (
    [np.linspace(50, 200, 80) for _ in range(3)]
    + [np.linspace(50, 200, 40) for _ in range(6)]
    + [np.linspace(0, 500, 50) for _ in range(2)]
    + [np.linspace(0, 500, 100) for _ in range(5)]
)

# For differential pairing efficiency by hh mass. (where the pairing efficiency is somehow added cumulative).
mhh_bins = np.linspace(250, 700, 10)

# Create plotting directory
plot_dir = args.plot_dir
os.makedirs(plot_dir, exist_ok=True)

# open files (all of the true and spanet files)
# df_true = [h5py.File(f, "r") for f in true_dict.values()]
# df_spanet_pred = [h5py.File(f, "r") for f in spanet_dict.values()["file"]]

# We are now filling a dictionary with one entry for each file.
# Then we extract all the information from the files, that we did before, but now targeted at the specific entries.
df_collection = {}
for model_name, file_dict in spanet_dict.items():
    spanetfile = h5py.File(file_dict["file"], "r")
    truefile = h5py.File(true_dict[file_dict["true"]]["name"], "r")
    truefile_klambda = true_dict[file_dict["true"]]["klambda"]

    idx_b1_true = truefile["TARGETS"]["h1"]["b1"][()]
    idx_b2_true = truefile["TARGETS"]["h1"]["b2"][()]
    idx_b3_true = truefile["TARGETS"]["h2"]["b3"][()]
    idx_b4_true = truefile["TARGETS"]["h2"]["b4"][()]

    idx_b1_spanet_pred = spanetfile["TARGETS"]["h1"]["b1"][()]
    idx_b2_spanet_pred = spanetfile["TARGETS"]["h1"]["b2"][()]
    idx_b3_spanet_pred = spanetfile["TARGETS"]["h2"]["b3"][()]
    idx_b4_spanet_pred = spanetfile["TARGETS"]["h2"]["b4"][()]

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

    idx_h1_spanet_pred = ak.concatenate(
            (
                ak.unflatten(idx_b1_spanet_pred, ak.ones_like(idx_b1_spanet_pred)),
                ak.unflatten(idx_b2_spanet_pred, ak.ones_like(idx_b2_spanet_pred)),
            ),
            axis=1,
        )
    idx_h2_spanet_pred = ak.concatenate(
            (
                ak.unflatten(idx_b3_spanet_pred, ak.ones_like(idx_b3_spanet_pred)),
                ak.unflatten(idx_b4_spanet_pred, ak.ones_like(idx_b4_spanet_pred)),
            ),
            axis=1,
        )

    idx_true = ak.concatenate(
            (
                ak.unflatten(idx_h1_true, ak.ones_like(idx_h1_true[:, 0])),
                ak.unflatten(idx_h2_true, ak.ones_like(idx_h2_true[:, 0])),
            ),
            axis=1,
        )
    idx_spanet_pred = ak.concatenate(
            (
                ak.unflatten(idx_h1_spanet_pred, ak.ones_like(idx_h1_spanet_pred[:, 0])),
                ak.unflatten(idx_h2_spanet_pred, ak.ones_like(idx_h2_spanet_pred[:, 0])),
            ),
            axis=1,
        )

    # load jet information
    jet_ptPNetRegNeutrino = truefile["INPUTS"]["Jet"]["ptPnetRegNeutrino"][()]
    jet_eta = truefile["INPUTS"]["Jet"]["eta"][()]
    jet_phi = truefile["INPUTS"]["Jet"]["phi"][()]
    jet_mass = truefile["INPUTS"]["Jet"]["mass"][()]

    jet_infos = [jet_ptPNetRegNeutrino, jet_eta, jet_phi, jet_mass]
    
    # These lists are to be expanded. Didn't think of a better way than to copy them here already
    # if not klambda, the two lists stay equal.
    alltrue_idx = [idx_true]
    allspanet_idx = [idx_spanet_pred]
    alljetinfos = [jet_infos]
    all_name_list = [model_name]

    if args.klambda and truefile_klambda != "none":
        print("Separating the klambdas")
        # separate the different klambdas
        (
            true_kl_idx_list,
            spanet_kl_idx_list,
            kl_values,
            jet_infos_separate_klambda,
        ) = separate_klambda(
            truefile, spanetfile, idx_true, idx_spanet_pred, true_dict, spanet_dict
        )

        #I got now indexes for true and spanet and different kl.
        #For this, I will add them to a list. If no kl is there, I will just make it a list of one element
        alltrue_idx.extend(true_kl_idx_list)
        allspanet_idx.extend(spanet_kl_idx_list)
        alljetinfos.extend(jet_infos_separate_klambda)
        all_name_list.extend(kl_values)
   
    # Fully matched events
    mask_fully_matched = [ak.all(ak.all(idx >= 0, axis=-1), axis=-1) for idx in alltrue_idx]
    alltrue_idx_fully_matched = [idx[mask] for idx, mask in zip(alltrue_idx, mask_fully_matched)]
    allspanet_idx_fully_matched = [idx[mask] for idx, mask in zip(allspanet_idx, mask_fully_matched)]

    print("Model name: ", model_name)
    print(file_dict["label"])
    print("alltrue_idx_fully_matched", [len(idx) for idx in alltrue_idx_fully_matched])
    print("allspanet_idx_fully_matched", [len(idx) for idx in allspanet_idx_fully_matched])
    print("allspanet_idx", [len(pred) for pred in allspanet_idx])
    print("mask_fully_matched", [len(mask) for mask in mask_fully_matched])

    if not args.data:
        correctly_fully_matched_spanet = [
            (   ak.all(true[:, 0] == spanet[:, 0], axis=1)
              | ak.all(true[:, 0] == spanet[:, 1], axis=1))
            & ( ak.all(true[:, 1] == spanet[:, 0], axis=1)
              | ak.all(true[:, 1] == spanet[:, 1], axis=1))
            for true, spanet in zip(alltrue_idx_fully_matched, allspanet_idx_fully_matched)
        ]

        # compute efficiencies for fully matched events
        efficiencies_fully_matched = [
            ak.sum(correctly_fully_matched) / len(correctly_fully_matched)
            for correctly_fully_matched in correctly_fully_matched_spanet
        ]
        print(
            "correctly_fully_matched_spanet",
            [len(c) for c in correctly_fully_matched_spanet],
        )
        frac_fully_matched = [ak.sum(mask) / len(mask) for mask in mask_fully_matched]
        total_efficiencies_fully_matched = [
            eff * frac
            for frac, eff in zip(frac_fully_matched, efficiencies_fully_matched)
        ]
        # Printing the values
        print("\n")
        for label, frac in zip([file_dict["true"]] + kl_values.tolist(), frac_fully_matched):
            print(f"Fraction of fully matched events for {label}: {frac:.3f}")

        print("\n")
        for label, eff in zip(all_name_list, efficiencies_fully_matched):
            print("Efficiency fully matched for {}: {:.3f}".format(label, eff))
        print("\n")

        for name, toteff in zip(all_name_list, total_efficiencies_fully_matched):
            print(
                "Total efficiency fully matched for {}: {:.3f}".format(name, toteff)
            )

        # do the same for partially matched events (only one higgs is matched)
        mask_1h = [
            ak.sum(ak.any(idx == -1, axis=-1) == 1, axis=-1) == 1 for idx in alltrue_idx
        ]
        idx_true_partially_matched_1h = [idx[mask] for idx, mask in zip(alltrue_idx, mask_1h)]
        idx_spanet_partially_matched_1h = [idx[mask] for idx, mask in zip(allspanet_idx, mask_1h)]

        correctly_partially_matched_spanet = [
            (   ak.all(true[:, 0] == spanet[:, 0], axis=1)
              | ak.all(true[:, 0] == spanet[:, 1], axis=1))
            & ( ak.all(true[:, 1] == spanet[:, 0], axis=1)
              | ak.all(true[:, 1] == spanet[:, 1], axis=1))
            for true, spanet in zip(idx_true_partially_matched_1h, idx_spanet_partially_matched_1h)
        ]
        efficiencies_partially_matched_spanet = [
            ak.sum(correctly_partially_matched_1h) / len(correctly_partially_matched_1h)
            for correctly_partially_matched_1h in correctly_partially_matched_spanet
        ]
        frac_partially_matched_1h = [ak.sum(mask) / len(mask) for mask in mask_1h]
        total_efficiencies_partially_matched_spanet = [
            eff * frac
            for frac, eff in zip(frac_partially_matched_1h, efficiencies_partially_matched_spanet)
        ]

        # Printing again the values
        print("\n")
        for label, frac in zip([file_dict["true"]] + kl_values.tolist(), frac_partially_matched_1h):
            print(f"Fraction of partially matched events for {label}: {frac:.3f}")
        print("\n")
        for label, eff in zip(all_name_list, efficiencies_partially_matched_spanet):
            print("Efficiency partially matched for {}: {:.3f}".format(label, eff))
        print("\n")
        for name, toteff in zip(all_name_list, total_efficiencies_partially_matched_spanet):
            print(
                "Total efficiency partially matched for {}: {:.3f}".format(name, toteff)
            )

        # compute number of events with 0 higgs matched
        mask_0h = [ak.sum(ak.any(idx == -1, axis=-1), axis=-1) == 2 for idx in alltrue_idx]

        idx_true_unmatched = [idx[mask] for idx, mask in zip(alltrue_idx, mask_0h)]
        frac_unmatched = [ak.sum(mask) / len(mask) for mask in mask_0h]
        print("\n")
        for label, frac in zip([file_dict["true"]] + kl_values.tolist(), frac_unmatched):
            print(f"Fraction of unmatched events for {label}: {frac:.3f}")

    #### The next part is for Run2 algorithm ####

    # create a LorentzVector for the jets
    jet = [
        ak.zip(
            {
                "pt": jet_i[0],
                "eta": jet_i[1],
                "phi": jet_i[2],
                "mass": jet_i[3],
            },
            with_name="Momentum4D",
        )
        for jet_i in alljetinfos
    ]

    print("len(jet)", len(jet), len(jet[0]))

    # implement the Run 2 pairing algorithm
    # TODO: extend to 5 jets cases (more comb idx)
    comb_idx = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]

    higgs_candidates_unflatten_order = [reco_higgs(j, comb_idx) for j in jet]
    distance = [
        distance_pt_func(
            higgs,
            1.04,
        )[0]
        for higgs in higgs_candidates_unflatten_order
    ]
    max_pt = [
        distance_pt_func(
            higgs,
            1.04,
        )[1]
        for higgs in higgs_candidates_unflatten_order
    ]
    dist_order_idx = [ak.argsort(d, axis=1, ascending=True) for d in distance]
    dist_order = [ak.sort(d, axis=1, ascending=True) for d in distance]

    pt_order_idx = [ak.argsort(pt, axis=1, ascending=False) for pt in max_pt]
    # if the distance between the two best candidates is less than 30, we do not consider the event
    min_idx = [
        ak.where(d[:, 1] - d[:, 0] > 30, d_idx[:, 0], pt_idx[:, 0])
        for d, d_idx, pt_idx in zip(dist_order, dist_order_idx, pt_order_idx)
    ]

    comb_idx = [
        np.tile(comb_idx, (len(m), 1, 1, 1)) for m in min_idx
    ]
    min_idx = [m for m in min_idx]
    # given the min_idx, select the correct combination corresponding to the index
    comb_idx_min = [
        comb[np.arange(len(m)), m] for comb, m in zip(comb_idx, min_idx)
    ]

    allrun2_idx_fully_matched = [
        ak.Array(comb)[m]
        for comb, m in zip(comb_idx_min, mask_fully_matched)
    ]

    if not args.data:
        # compute efficiencies for fully matched events for Run 2 pairing
        correctly_fully_matched_run2 = [
            (   ak.all(true[:, 0] == run2[:, 0], axis=1)
              | ak.all(true[:, 0] == run2[:, 1], axis=1))
            & ( ak.all(true[:, 1] == run2[:, 0], axis=1)
              | ak.all(true[:, 1] == run2[:, 1], axis=1))
            for true, run2 in zip(
                alltrue_idx_fully_matched, allrun2_idx_fully_matched
            )
        ]
        
        # Calculating run2 efficiencies
        efficiency_fully_matched_run2 = [
            ak.sum(corr) / len(corr) for corr in correctly_fully_matched_run2
        ]
        frac_fully_matched = [
            ak.sum(m) / len(m) for m in mask_fully_matched
        ]
        total_efficiency_fully_matched_run2 = [
            eff * frac
            for eff, frac in zip(frac_fully_matched, efficiency_fully_matched_run2)
        ]
        print("\n")
        for label, frac in zip([file_dict["true"]] + kl_values.tolist(), frac_fully_matched):
            print(f"Fraction of fully matched events for {label}: {frac:.3f}")

        print("\n")
        for label, eff in zip(all_name_list, efficiency_fully_matched_run2):
            print(f"Efficiency fully matched  for Run 2 {label}: {eff:.3f}")

        print("\n")
        for name, toteff in zip(all_name_list, total_efficiency_fully_matched_run2):
            print(
                "Total efficiency fully matched for Run 2 {}: {:.3f}".format(
                    name,
                    toteff,
                )
            )

    # Reconstruct the Higgs boson candidates with the efficiency_fully_matched_run2 = (
    # of the jets considering the true pairings, the spanet pairings
    # and the run2 pairings
    jet_fully_matched = [
        j[m] for j, m in zip(jet, mask_fully_matched)
    ]
    # Reconstruction of the Higgs boson candidates with the predicted/true pairings
    spanet_higgs_fully_matched = [
        best_reco_higgs(j, spanet_idx)
        for j, spanet_idx in zip(jet_fully_matched, allspanet_idx_fully_matched)
    ]
    run2_higgs_fully_matched = [
        best_reco_higgs(j, idx)
        for j, idx in zip(jet_fully_matched, allrun2_idx_fully_matched)
    ]
    true_higgs_fully_matched = [
        best_reco_higgs(j, idx)
        for j, idx in zip(jet_fully_matched, alltrue_idx_fully_matched)
    ]
    true_hh_fully_matched = [
        true_h_matched[:, 0] + true_h_matched[:, 1]
        for true_h_matched in true_higgs_fully_matched
    ]
    # I need the differential efficiency for the
    # True pairing
    # Spanet pairing
    # DHH pairing
    # All of these are just lists
    if not args.data:
        # Differential efficiency
        diff_eff_run2 = []
        unc_diff_eff_run2 = []
        total_diff_eff_run2 = []
        total_unc_diff_eff_run2 = []
        diff_eff_spanet = []
        unc_diff_eff_spanet = []
        total_diff_eff_spanet = []
        total_unc_diff_eff_spanet = []

        for true_hh, matched_spanet, matched_run2, mask_matched in zip(
                true_hh_fully_matched, correctly_fully_matched_spanet, correctly_fully_matched_run2, mask_fully_matched
                ):
            temp_diff_eff_run2 = []
            temp_unc_diff_eff_run2 = []
            temp_total_diff_eff_run2 = []
            temp_total_unc_diff_eff_run2 = []
            temp_diff_eff_spanet = []
            temp_unc_diff_eff_spanet = []
            temp_total_diff_eff_spanet = []
            temp_total_unc_diff_eff_spanet = []
            for i in range(1, len(mhh_bins)):
                mask = (true_hh.mass > mhh_bins[i - 1]) & (true_hh.mass < mhh_bins[i])
                eff_run2 = ak.sum(matched_run2[mask]) / ak.count(matched_run2[mask])
                unc_eff_run2 = sqrt(eff_run2 * (1 - eff_run2) / ak.count(matched_run2[mask]))

                frac_fully_matched = ak.sum(mask_matched[mask]) / len(mask_matched[mask])
                total_eff_run2 = eff_run2 * frac_fully_matched
                unc_total_eff_run2 = sqrt((total_eff_run2 * (1 - total_eff_run2)) / len(mask_matched[mask]))

                eff_spanet = ak.sum(matched_spanet[mask]) / ak.count(matched_spanet[mask])
                unc_eff_spanet = sqrt(eff_spanet * (1 - eff_spanet)/ ak.count(matched_spanet[mask]))
                frac_fully_matched = ak.sum(mask_matched[mask]) / len(mask_matched[mask])
                total_eff_spanet = eff_spanet * frac_fully_matched
                unc_total_eff_spanet = sqrt((total_eff_spanet * (1 - total_eff_spanet)) / len(mask_matched[mask]))

                temp_diff_eff_run2.append(eff_run2)
                temp_unc_diff_eff_run2.append(unc_eff_run2)
                temp_total_diff_eff_run2.append(total_eff_run2)
                temp_total_unc_diff_eff_run2.append(unc_total_eff_run2)
                temp_diff_eff_spanet.append(eff_spanet)
                temp_unc_diff_eff_spanet.append(unc_eff_spanet)
                temp_total_diff_eff_spanet.append(total_eff_spanet)
                temp_total_unc_diff_eff_spanet.append(unc_total_eff_spanet)
            # not so nice, but here we are filling the results from the different lists into a new list.
            # Remember, we iterate here through: [inclusive, *[single kls])
            diff_eff_run2.append(temp_diff_eff_run2)
            unc_diff_eff_run2.append(temp_unc_diff_eff_run2)
            total_diff_eff_run2.append(temp_total_diff_eff_run2)
            total_unc_diff_eff_run2.append(temp_total_unc_diff_eff_run2)
            diff_eff_spanet.append(temp_diff_eff_spanet)
            unc_diff_eff_spanet.append(temp_unc_diff_eff_spanet)
            total_diff_eff_spanet.append(temp_total_diff_eff_spanet)
            total_unc_diff_eff_spanet.append(temp_total_unc_diff_eff_spanet)



    #### Iteration over the different datasets over ####

    df_collection[model_name] = {
            "file_dict" : file_dict,
            "kl_values" : kl_values
            }
    if not args.data:
        df_collection[model_name] = df_collection[model_name] | {
                "efficiencies_fully_matched" : efficiencies_fully_matched,
                "total_efficiencies_fully_matched" : total_efficiencies_fully_matched,
                # Currently of all the spanet models, a run2 is created. We can improve this to only have one run2 model
                # The model to run is defined in the efficiendy_calibrations
                "efficiencies_fully_matched_run2" : efficiency_fully_matched_run2,
                "total_efficiencies_fully_matched_run2" : total_efficiency_fully_matched_run2,
                # Parameters for the diff_eff plots
                "diff_eff_run2": diff_eff_run2,
                "unc_diff_eff_run2": unc_diff_eff_run2,
                "total_diff_eff_run2": total_diff_eff_run2,
                "total_unc_diff_eff_run2": total_unc_diff_eff_run2,
                "diff_eff_spanet": diff_eff_spanet,
                "unc_diff_eff_spanet": unc_diff_eff_spanet,
                "total_diff_eff_spanet": total_diff_eff_spanet,
                "total_unc_diff_eff_spanet": total_unc_diff_eff_spanet,
                "true_hh_fully_matched":  true_hh_fully_matched,
                "spanet_higgs_fully_matched": spanet_higgs_fully_matched,
                "run2_higgs_fully_matched": run2_higgs_fully_matched,
                "true_higgs_fully_matched": true_higgs_fully_matched,
                }


## Plotting begins here
        # not data
if not args.data:
    print("\n")
    print("Plotting efficiencies fully matched for all klambda values")
    # We are adding the run2 of the chosen element (defined in efficiency_calibrations) as last element
    print("All datasets: ", df_collection.keys())
    print("Run2 set: ", run2_dataset)
    r2_model = df_collection[run2_dataset]
    print(df_collection[run2_dataset].keys())
    plot_diff_eff_klambda(
        [model["efficiencies_fully_matched"][1:] for model in df_collection.values()] + [r2_model["efficiencies_fully_matched_run2"][1:]],
        [model["kl_values"] for model in df_collection.values()] + [r2_model["kl_values"]],
        [model["file_dict"]["label"] for model in df_collection.values()] + [r"$D_{HH}$-method"],
        [model["file_dict"]["color"] for model in df_collection.values()] + ["yellowgreen"],
        "eff_fully_matched_allklambda",
        plot_dir,
    )
    plot_diff_eff_klambda(
        [model["total_efficiencies_fully_matched"][1:] for model in df_collection.values()] + [r2_model["total_efficiencies_fully_matched_run2"][1:]],
        [model["kl_values"] for model in df_collection.values()] + [r2_model["kl_values"]],
        [model["file_dict"]["label"] for model in df_collection.values()] + [r"$D_{HH}$-method"],
        [model["file_dict"]["color"] for model in df_collection.values()] + ["yellowgreen"],
        "tot_eff_fully_matched_allklambda",
        plot_dir,
    )

    print("Plotting differential efficiencies")
    plot_diff_eff(
        mhh_bins,
        [model["diff_eff_spanet"][0] for model in df_collection.values()] + [r2_model["diff_eff_run2"][0]],
        [model["unc_diff_eff_spanet"][0] for model in df_collection.values()] + [r2_model["unc_diff_eff_run2"][0]],
        [model["file_dict"]["label"] for model in df_collection.values()] + [r"$D_{HH}$-method"],
        [model["file_dict"]["color"] for model in df_collection.values()] + ["yellowgreen"],
        plot_dir,
        "diff_eff_spanet",
    )
    plot_diff_eff(
        mhh_bins,
        [model["total_diff_eff_spanet"][0] for model in df_collection.values()] + [r2_model["total_diff_eff_run2"][0]],
        [model["total_unc_diff_eff_spanet"][0] for model in df_collection.values()] + [r2_model["total_unc_diff_eff_run2"][0]],
        [model["file_dict"]["label"] for model in df_collection.values()] + [r"$D_{HH}$-method"],
        [model["file_dict"]["color"] for model in df_collection.values()] + ["yellowgreen"],
        plot_dir,
        "total_diff_eff_spanet",
    )

    print("Plotting mhh")
    plot_mhh(
        mhh_bins,
        list(df_collection.values())[0]["true_hh_fully_matched"][0].mass,
        plot_dir,
        "mhh_fully_matched",
    )

print("Plotting higgs 1d all events")
for bins, name in zip([mh_bins, mh_bins_peak], ["", "_peak"]):
    for number in [1, 2]:
        plot_histos_1d(
            bins,
            [model["spanet_higgs_fully_matched"][0][:, number - 1].mass for model in df_collection.values()], #spanet values
            r2_model["run2_higgs_fully_matched"][0][:, number - 1].mass, #run2 values
            r2_model["true_higgs_fully_matched"][0][:, number - 1].mass if not args.data else None, #true values
            [model["file_dict"]["label"] for model in df_collection.values()],
            [model["file_dict"]["color"] for model in df_collection.values()],
            number,
            name=name,
            plot_dir=plot_dir,
        )
