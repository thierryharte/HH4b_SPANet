import logging
from math import sqrt

import awkward
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import vector

logger = logging.getLogger(__name__)

vector.register_awkward()
vector.register_numba()

from efficiency_configuration import *

TRUE_PAIRING = False
H5_PADDING_VALUE = 9999.0


# TODO: Currently we only have the values for postEE!!!!
def get_region_mask(region, column_file, do_vbf_pairing):
    if region == "inclusive":
        return ak.ones_like(column_file["INPUTS"]["Jet"]["MASK"][:, 0])

    jet_btag = column_file["INPUTS"]["Jet"]["btagPNetB"]
    if region == "4b" or region == "4M":
        mask = (
            (jet_btag[:, 0] > 0.2605)
            & (jet_btag[:, 1] > 0.2605)
            & (jet_btag[:, 2] > 0.2605)
            & (jet_btag[:, 3] > 0.2605)
        )
    elif region == "3M":
        mask = (
            (jet_btag[:, 0] > 0.2605)
            & (jet_btag[:, 1] > 0.2605)
            & (jet_btag[:, 2] > 0.2605)
            & (jet_btag[:, 3] < 0.2605)
        )
    elif region == "2M":
        mask = (
            (jet_btag[:, 0] > 0.2605)
            & (jet_btag[:, 1] > 0.2605)
            & (jet_btag[:, 2] < 0.2605)
            & (jet_btag[:, 3] < 0.2605)
        )
    elif region == "3T1M":
        mask = (
            (jet_btag[:, 0] > 0.6915)
            & (jet_btag[:, 1] > 0.6915)
            & (jet_btag[:, 2] > 0.6915)
            & (jet_btag[:, 3] > 0.2605)
        )
    elif region == "3T1L":
        mask = (
            (jet_btag[:, 0] > 0.6915)
            & (jet_btag[:, 1] > 0.6915)
            & (jet_btag[:, 2] > 0.6915)
            & (jet_btag[:, 3] > 0.0499)
            & (jet_btag[:, 3] < 0.2605)
        )
    elif region == "vbf_presel" and do_vbf_pairing:
        mask = get_mask_vbf_region(column_file, 400, 3.5)
    elif region == "vbf_no_kin_cuts" and do_vbf_pairing:
        mask = get_mask_vbf_region(column_file, 0, 0)
    else:
        raise ValueError("Undefined region")
    return mask


def get_mask_vbf_region(column_file, mjj_cut, delta_eta_cut):
    jet_list = [
        get_jet_4vec(
            column_file, ak.ones_like(column_file["INPUTS"]["Jet"]["MASK"][:, 0])
        )
    ]
    jet_vbf = [j[:, 4:] for j in jet_list]
    idx_vbf_lead_mjj = get_lead_mjj_jet_idx(jet_vbf)[0]
    jet = jet_list[0]
    vbf_jets_max_mjj_0 = ak.unflatten(
        jet[
            ak.local_index(jet, axis=0),
            idx_vbf_lead_mjj[:, 0],
        ],
        1,
    )
    vbf_jets_max_mjj_1 = ak.unflatten(
        jet[
            ak.local_index(jet, axis=0),
            idx_vbf_lead_mjj[:, 1],
        ],
        1,
    )
    delta_eta = abs(vbf_jets_max_mjj_0.eta - vbf_jets_max_mjj_1.eta)
    mjj = (vbf_jets_max_mjj_0 + vbf_jets_max_mjj_1).mass

    mask = ak.flatten((mjj > mjj_cut) & (delta_eta > delta_eta_cut))
    mask = ak.where(ak.is_none(mask), False, mask)

    return mask


def get_class_mask(class_label, column_file):
    if class_label:
        try:
            class_array = column_file["CLASSIFICATIONS"]["EVENT"]["class"][()].astype(
                np.int64
            )
            mask = class_array == int(class_label)
            return mask
        except:
            logger.info(
                "The file doesn't contain a class array. Setting the mask for the class to True ..."
            )

    return ak.ones_like(column_file["INPUTS"]["Jet"]["MASK"][:, 0])


def get_lead_mjj_jet_idx(jet):
    # choose higgs jets as the two jets with the highest mjj that are not from higgs decay

    jet_combinations = [ak.combinations(j, 2) for j in jet]
    jet_combinations_mass = [(jc["0"] + jc["1"]).mass for jc in jet_combinations]
    jet_combinations_mass_max_idx = [
        ak.to_numpy(ak.argsort(jcm, axis=1, ascending=False)[:, 0])
        for jcm in jet_combinations_mass
    ]
    jets_max_mass = [
        jc[ak.local_index(jc, axis=0), jcm]
        for jc, jcm in zip(jet_combinations, jet_combinations_mass_max_idx)
    ]
    comb_idx_min_vbf = [
        ak.to_numpy(
            ak.concatenate(
                [
                    ak.unflatten(jmm["0"].index, 1),
                    ak.unflatten(jmm["1"].index, 1),
                ],
                axis=1,
            ),
            1,
        )
        for jmm in jets_max_mass
    ]

    return comb_idx_min_vbf


def get_jet_4vec(truefile, mask_true):
    # load jet information
    try:
        jet_pt = truefile["INPUTS"]["Jet"]["ptPnetRegNeutrino"][()][mask_true]
    except KeyError:
        logger.warning("Did not find ptPnetRegNeutrino, will try to load pt normal")
        jet_pt = truefile["INPUTS"]["Jet"]["pt"][()][mask_true]

    jet_eta = truefile["INPUTS"]["Jet"]["eta"][()][mask_true]
    jet_phi = truefile["INPUTS"]["Jet"]["phi"][()][mask_true]
    jet_mass = truefile["INPUTS"]["Jet"]["mass"][()][mask_true]

    jet_infos = [jet_pt, jet_eta, jet_phi, jet_mass]
    for i in range(len(jet_infos)):
        jet_infos[i] = ak.mask(jet_infos[i], jet_infos[i] != H5_PADDING_VALUE)

    # create a LorentzVector for the jets
    jet = ak.zip(
        {
            "pt": jet_infos[0],
            "eta": jet_infos[1],
            "phi": jet_infos[2],
            "mass": jet_infos[3],
            "index": ak.local_index(jet_infos[0], axis=1),
        },
        with_name="Momentum4D",
    )
    return jet


def check_double_assignment_vectorized(full_idx_list, label):
    # Stack indices for each event: shape (N_events, 4)
    idx_all = np.stack(full_idx_list, axis=1)
    # Mask for valid indices
    valid_mask = idx_all >= 0
    # Set invalid indices to a large negative number so they don't interfere
    idx_all_valid = np.where(valid_mask, idx_all, -9999)
    # For each event, sort the indices
    idx_sorted = np.sort(idx_all_valid, axis=1)
    # Compare adjacent indices for equality (double assignment)
    double_assigned = np.any(
        (idx_sorted[:, 1:] == idx_sorted[:, :-1]) & (idx_sorted[:, 1:] >= 0), axis=1
    )
    n_double = np.sum(double_assigned)
    if n_double > 0:
        rows = np.where(double_assigned)[0]
        logger.info(f"WARNING: {n_double} events in {label} have double-assigned jets!")
        logger.info(f"Rows with double assignments: {rows}")
    else:
        logger.info(f"OK: No double-assigned jets found in {label}.")


def load_jets_and_pairing(
    samplefile,
    label,
    allowed_idx_higgs=None,
    allowed_idx_vbf=None,
    higgs=True,
    vbf=False,
):
    full_idx_list = []
    if higgs:
        idx_b1 = samplefile["TARGETS"]["h1"]["b1"][()]
        idx_b2 = samplefile["TARGETS"]["h1"]["b2"][()]
        idx_b3 = samplefile["TARGETS"]["h2"]["b3"][()]
        idx_b4 = samplefile["TARGETS"]["h2"]["b4"][()]

        if allowed_idx_higgs is not None:
            # keep up to max_jets for the pairing
            idx_b1 = ak.where(
                ak.any(idx_b1[..., None] == allowed_idx_higgs, axis=-1), idx_b1, -1
            )
            idx_b2 = ak.where(
                ak.any(idx_b2[..., None] == allowed_idx_higgs, axis=-1), idx_b2, -1
            )
            idx_b3 = ak.where(
                ak.any(idx_b3[..., None] == allowed_idx_higgs, axis=-1), idx_b3, -1
            )
            idx_b4 = ak.where(
                ak.any(idx_b4[..., None] == allowed_idx_higgs, axis=-1), idx_b4, -1
            )

        idx_h1 = ak.concatenate(
            (
                ak.unflatten(idx_b1, ak.ones_like(idx_b1)),
                ak.unflatten(idx_b2, ak.ones_like(idx_b2)),
            ),
            axis=1,
        )
        idx_h2 = ak.concatenate(
            (
                ak.unflatten(idx_b3, ak.ones_like(idx_b3)),
                ak.unflatten(idx_b4, ak.ones_like(idx_b4)),
            ),
            axis=1,
        )
        full_idx_list += [
            ak.unflatten(idx_h1, ak.ones_like(idx_h1[:, 0])),
            ak.unflatten(idx_h2, ak.ones_like(idx_h2[:, 0])),
        ]

    if vbf and "vbf" in samplefile["TARGETS"].keys():
        # Include the VBF matching
        idx_q1 = samplefile["TARGETS"]["vbf"]["q1"][()]
        idx_q2 = samplefile["TARGETS"]["vbf"]["q2"][()]

        if allowed_idx_vbf is not None:
            # keep up to max_jets for the pairing
            idx_q1 = ak.where(
                ak.any(idx_q1[..., None] == allowed_idx_vbf, axis=-1), idx_q1, -1
            )
            idx_q2 = ak.where(
                ak.any(idx_q2[..., None] == allowed_idx_vbf, axis=-1), idx_q2, -1
            )

        idx_vbf = ak.concatenate(
            (
                ak.unflatten(idx_q1, ak.ones_like(idx_q1)),
                ak.unflatten(idx_q2, ak.ones_like(idx_q2)),
            ),
            axis=1,
        )
        full_idx_list.append(ak.unflatten(idx_vbf, ak.ones_like(idx_vbf[:, 0])))

    check_double_assignment_vectorized(full_idx_list, label)
    idx = ak.concatenate(
        tuple(full_idx_list),
        axis=1,
    )

    return idx


def calculate_efficiencies(
    true_idx,
    prediction_idx,
    mask,
    truename,
    kl_values,
    all_names,
    label,
    higgs=True,
    vbf=False,
):
    matching_eval_model = [ak.ones_like(true[:, 0, 0]) for true in true_idx]

    if higgs:
        # higgs 1 and higgs 2
        matching_eval_model_higgs = [
            (
                ak.all(true[:, 0] == prediction[:, 0], axis=1)
                | ak.all(true[:, 0] == prediction[:, 1], axis=1)
            )
            & (
                ak.all(true[:, 1] == prediction[:, 0], axis=1)
                | ak.all(true[:, 1] == prediction[:, 1], axis=1)
            )
            for true, prediction in zip(true_idx, prediction_idx)
        ]
        matching_eval_model = [
            matching_eval_model[i] & matching_eval_model_higgs[i]
            for i in range(len(matching_eval_model))
        ]

    if vbf:
        # if there are also the higgs idx, then the vbf is the third idx
        # otherwise is the first idx
        if higgs:
            idx_vbf = 2
        else:
            idx_vbf = 0

        # vbf
        matching_eval_model_vbf = [
            (
                ak.all(true[:, idx_vbf] == prediction[:, idx_vbf], axis=1)
                # check also if the idx are swapped (altought this shouldn't happen)
                | ak.all(true[:, idx_vbf, ::-1] == prediction[:, idx_vbf], axis=1)
            )
            for true, prediction in zip(true_idx, prediction_idx)
        ]
        matching_eval_model = [
            matching_eval_model[i] & matching_eval_model_vbf[i]
            for i in range(len(matching_eval_model))
        ]

    # compute efficiencies for events
    model_eff = [
        ak.sum(matching_eval_ds) / len(matching_eval_ds)
        for matching_eval_ds in matching_eval_model
    ]
    logger.info(f"matching_{label}_model: {[len(c) for c in matching_eval_model]}")
    fraction = [ak.sum(m) / len(m) for m in mask]
    total_model_eff = [eff * frac for frac, eff in zip(fraction, model_eff)]
    unc_model_eff = [
        sqrt(eff * (1 - eff) / len(matching_eval_ds))
        for eff, matching_eval_ds in zip(model_eff, matching_eval_model)
    ]
    unc_total_model_eff = [
        sqrt((total_eff * (1 - total_eff)) / len(matching_eval_ds))
        for total_eff, matching_eval_ds in zip(total_model_eff, matching_eval_model)
    ]
    # Printing the values
    logger.info("\n")
    for lab, frac in zip([truename] + kl_values.tolist(), fraction):
        logger.info(f"Fraction of {label} events for {lab}: {frac:.3f}")

    logger.info("\n")
    for lab, eff in zip(all_names, model_eff):
        logger.info(f"Efficiency {label} for {lab}: {eff:.3f}")
    logger.info("\n")
    for name, toteff in zip(all_names, unc_model_eff):
        logger.info(f"Efficiency uncertainty {label} for {name}: {toteff:.3f}")

    logger.info("\n")
    for lab, eff in zip(all_names, total_model_eff):
        logger.info(f"Total efficiency {label} for {lab}: {eff:.3f}")
    logger.info("\n")
    for name, toteff in zip(all_names, unc_total_model_eff):
        logger.info(f"Total efficiency uncertainty {label} for {name}: {toteff:.3f}")

    return (
        fraction,
        model_eff,
        total_model_eff,
        unc_model_eff,
        unc_total_model_eff,
        matching_eval_model,
    )


def distance_pt_func(higgs_pair, k):
    if len(higgs_pair[0, 0]) == 0:
        return np.array([])
    higgs1 = higgs_pair[:, :, 0]
    higgs2 = higgs_pair[:, :, 1]
    dist = abs(higgs1.mass - higgs2.mass * k) / sqrt(1 + k**2)
    max_pt = np.maximum(higgs1.pt, higgs2.pt)
    return dist, max_pt


def reco_higgs(jet_collection, idx_collection):
    if len(jet_collection) == 0:
        higgs_candidates_unflatten_order = ak.Array([[[], []], [[], []], [[], []]])
        return higgs_candidates_unflatten_order

    higgs_01 = ak.unflatten(
        jet_collection[:, idx_collection[0][0][0]]
        + jet_collection[:, idx_collection[0][0][1]],
        1,
    )
    higgs_23 = ak.unflatten(
        jet_collection[:, idx_collection[0][1][0]]
        + jet_collection[:, idx_collection[0][1][1]],
        1,
    )

    higgs_02 = ak.unflatten(
        jet_collection[:, idx_collection[1][0][0]]
        + jet_collection[:, idx_collection[1][0][1]],
        1,
    )
    higgs_13 = ak.unflatten(
        jet_collection[:, idx_collection[1][1][0]]
        + jet_collection[:, idx_collection[1][1][1]],
        1,
    )

    higgs_03 = ak.unflatten(
        jet_collection[:, idx_collection[2][0][0]]
        + jet_collection[:, idx_collection[2][0][1]],
        1,
    )
    higgs_12 = ak.unflatten(
        jet_collection[:, idx_collection[2][1][0]]
        + jet_collection[:, idx_collection[2][1][1]],
        1,
    )

    higgs_pair_0 = ak.concatenate([higgs_01, higgs_23], axis=1)
    higgs_pair_1 = ak.concatenate([higgs_02, higgs_13], axis=1)
    higgs_pair_2 = ak.concatenate([higgs_03, higgs_12], axis=1)

    higgs_candidates = ak.concatenate(
        [higgs_pair_0, higgs_pair_1, higgs_pair_2], axis=1
    )
    higgs_candidates_unflatten = ak.unflatten(higgs_candidates, 2, axis=1)

    # order the higgs candidates by pt
    higgs_candidates_unflatten_order_idx = ak.argsort(
        higgs_candidates_unflatten.pt, axis=2, ascending=False
    )
    higgs_candidates_unflatten_order = higgs_candidates_unflatten[
        higgs_candidates_unflatten_order_idx
    ]
    return higgs_candidates_unflatten_order


def best_reco_higgs(jet_collection, idx_collection, higgs=True):
    if higgs:
        higgs_1 = ak.unflatten(
            jet_collection[np.arange(len(idx_collection)), idx_collection[:, 0, 0]]
            + jet_collection[np.arange(len(idx_collection)), idx_collection[:, 0, 1]],
            1,
        )
        higgs_2 = ak.unflatten(
            jet_collection[np.arange(len(idx_collection)), idx_collection[:, 1, 0]]
            + jet_collection[np.arange(len(idx_collection)), idx_collection[:, 1, 1]],
            1,
        )

        higgs_pair = ak.concatenate([higgs_1, higgs_2], axis=1)

        # order the higgs candidates by pt
        higgs_candidates_unflatten_order_idx = ak.argsort(
            higgs_pair.pt, axis=1, ascending=False
        )
        higgs_candidates_unflatten_order = higgs_pair[
            higgs_candidates_unflatten_order_idx
        ]
    else:
        # if we don't reconstruct the higgs, put a dummy output
        higgs_candidates_unflatten_order = ak.ones_like(
            ak.concatenate(
                [
                    ak.unflatten(idx_collection[:, 0, 0], 1),
                    ak.unflatten(idx_collection[:, 0, 0], 1),
                ],
                axis=1,
            )
        )

    return higgs_candidates_unflatten_order


def run2_algorithm(jet, mask_fully_matched, higgs=True, vbf=False):
    comb_idx_min_list = []
    if higgs:
        # implement the Run 2 pairing algorithm
        comb_idx = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]

        higgs_candidates_unflatten_order = [reco_higgs(j, comb_idx) for j in jet]
        distance = [
            distance_pt_func(higgs, 1.04)[0]
            for higgs in higgs_candidates_unflatten_order
        ]
        max_pt = [
            distance_pt_func(higgs, 1.04)[1]
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

        comb_idx = [np.tile(comb_idx, (len(m), 1, 1, 1)) for m in min_idx]
        # given the min_idx, select the correct combination corresponding to the index
        comb_idx_min_higgs = [
            comb[np.arange(len(m)), m] for comb, m in zip(comb_idx, min_idx)
        ]
        comb_idx_min_list.append(comb_idx_min_higgs)

    if vbf:
        # choose higgs jets as the two jets with the highest mjj that are not from higgs decay
        jet_vbf = [j[:, 4:] for j in jet]
        comb_idx_min_vbf = get_lead_mjj_jet_idx(jet_vbf)
        comb_idx_min_vbf = [np.expand_dims(ci, 1) for ci in comb_idx_min_vbf]

        comb_idx_min_list.append(comb_idx_min_vbf)

    # combine the idx
    comb_idx_min_conc = ak.concatenate(comb_idx_min_list, axis=2)
    allrun2_idx_fully_matched = [
        ak.Array(comb)[m] for comb, m in zip(comb_idx_min_conc, mask_fully_matched)
    ]

    return allrun2_idx_fully_matched


def calculate_diff_efficiencies(matched, mask, mask_matched):
    """This function is designed to determine the differencial efficiencies by mhh_bins.
    Used to avoid repetition between run2 and spanet.
    :param matched: the events passing a mask for being fully matched
    :param mask: the mask determining the current bin
    :param mask_matched: masking array for an event being fully matched (not very efficient...)

    :return: efficiency, uncertainty of efficiency, total efficiency, uncertainty of total efficiency
    """
    eff = ak.sum(matched[mask]) / ak.count(matched[mask])
    unc_eff = sqrt(eff * (1 - eff) / ak.count(matched[mask]))
    frac_fully_matched = ak.sum(mask_matched[mask]) / len(mask_matched[mask])
    total_eff = eff * frac_fully_matched
    unc_total_eff = sqrt((total_eff * (1 - total_eff)) / len(mask_matched[mask]))
    return eff, unc_eff, total_eff, unc_total_eff


def plot_histos_1d(
    bins, spanet, run2, true, labels, color, num, name="", plot_dir="plots"
):
    # First create a big list with also the run2 and true values:
    values = spanet
    spanet_labels = labels
    spanet_color = color
    if isinstance(run2, awkward.highlevel.Array):
        values.append(run2)
        labels.append(r"$D_{HH}$-method")
        color.append("yellowgreen")
    if isinstance(
        true, awkward.highlevel.Array
    ):  # Meaning, if we have a "true" dataset
        values.append(true)
        labels.append("True pairing")
        color.append("black")

    # We want to add a ratio plot with run2 ratios. Basically to see the difference between the SPANet and Dhh pairings.
    if not isinstance(
        true, awkward.highlevel.Array
    ):  # Meaning, if we have a "true" dataset
        compare_run2 = True
    else:
        compare_run2 = False
    if compare_run2 and isinstance(run2, awkward.highlevel.Array):
        fig, (ax, ax_residuals) = plt.subplots(
            figsize=(6, 6), nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax_residuals.set_xlabel(
            r"Leading $m_{H}$ [GeV]" if num == 1 else r"Subleading $m_{H}$ [GeV]"
        )
    # This would be the normal ratio with comparison to true values
    elif isinstance(
        true, awkward.highlevel.Array
    ):  # Meaning, if we have a "true" dataset
        fig, (ax, ax_residuals) = plt.subplots(
            figsize=(6, 6), nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax_residuals.set_xlabel(
            r"Leading $m_{H}$ [GeV]" if num == 1 else r"Subleading $m_{H}$ [GeV]"
        )
    # Otherwise we would have baseline
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel(
            r"Leading $m_{H}$ [GeV]" if num == 1 else r"Subleading $m_{H}$ [GeV]"
        )
    ax.set_ylabel("Normalized events")

    for sn, label, color in zip(values, labels, color):
        # counts, edges = np.histogram(sn, bins=bins)
        # total_width = max(sn)-min(sn)
        # norm_counts = counts / (total_width * np.sum(counts))
        # ax.bar(edges[:-1], norm_counts, width=np.diff(edges), align="edge", alpha=0.7)
        ax.hist(
            sn,
            bins,
            label=label,
            histtype="step",
            linewidth=1,
            density=False,
            weights=np.repeat(1.0 / (len(sn) * np.diff(bins)[0]), len(sn)),
            color=color,
        )
    ax.grid(linestyle=":")
    if isinstance(true, awkward.highlevel.Array):
        true_hist = np.histogram(true, bins)
        true_norm = len(true)
    if isinstance(run2, awkward.highlevel.Array):
        run2_hist = np.histogram(run2, bins)
        run2_norm = len(run2)
    spanet_hists = [
        # np.histogram(spanet[i], bins, weights=np.ones_like(spanet[i]) / len(spanet[i]))
        np.histogram(span, bins)
        for span in spanet
    ]
    spanet_norm = [len(i) for i in spanet]
    ax.set_ylim(
        0,
        max(
            np.histogram(
                spanet[0],
                bins,
                density=False,
                weights=np.repeat(
                    1.0 / (len(spanet[0]) * np.diff(bins)[0]), len(spanet[0])
                ),
            )[0]
        )
        * (1.2 if "peak" not in name else 1.6),
    )
    print(f"Bin size of the plot: {np.diff(bins)[0]}")
    if "peak" not in name:
        ax.set_xlim(50, 300)
    if compare_run2 and isinstance(run2, awkward.highlevel.Array):
        # Plot the residuals with respect to Run2
        res_spanet_run2 = [
            (histo[0] / norm) / (run2_hist[0] / run2_norm)
            for histo, norm in zip(spanet_hists, spanet_norm)
        ]

        err_spanet = [
            np.sqrt(histo[0]) / norm for histo, norm in zip(spanet_hists, spanet_norm)
        ]
        err_run2 = np.sqrt(run2_hist[0]) / run2_norm
        res_spanet_run2_err = [
            np.sqrt(
                (err / (run2_hist[0]) / run2_norm) ** 2
                + ((err_run2 * (histo[0] / norm)) / (run2_hist[0] / run2_norm) ** 2)
                ** 2
            )
            for histo, norm, err in zip(spanet_hists, spanet_norm, err_spanet)
        ]
        for sn, label, sn_err, col in zip(
            res_spanet_run2, spanet_labels, res_spanet_run2_err, spanet_color
        ):
            ax_residuals.errorbar(
                run2_hist[1][:-1],
                sn,
                yerr=sn_err,
                marker=".",
                # markersize=1,
                label=label,
                linestyle="None",
                color=col,
            )

        # plot zero line
        ax_residuals.set_ylim(0, 3)
        ax_residuals.axhline(1, color="black", linewidth=1)
        ax_residuals.set_ylabel("SPANet / $D_{HH}$")

        ax_residuals.grid()

    elif isinstance(
        true, awkward.highlevel.Array
    ):  # Meaning, if we have a "true" dataset
        # Plot the residuals with respect to Run2
        res_spanet_true = [
            (histo[0] / norm) / (true_hist[0] / true_norm)
            for histo, norm in zip(spanet_hists, spanet_norm)
        ]

        err_spanet = [
            np.sqrt(histo[0]) / norm for histo, norm in zip(spanet_hists, spanet_norm)
        ]
        err_run2 = np.sqrt(run2_hist[0]) / run2_norm
        err_true = np.sqrt(true_hist[0]) / true_norm

        res_spanet_true_err = [
            np.sqrt(
                (err / (true_hist[0]) / true_norm) ** 2
                + ((err_true * (histo[0] / norm)) / (true_hist[0] / true_norm) ** 2)
                ** 2
            )
            for histo, norm, err in zip(spanet_hists, spanet_norm, err_spanet)
        ]
        for sn, label, sn_err, col in zip(
            res_spanet_true, spanet_labels, res_spanet_true_err, spanet_color
        ):
            ax_residuals.errorbar(
                run2_hist[1][:-1],
                sn,
                yerr=sn_err,
                marker=".",
                # markersize=1,
                label=label,
                linestyle="None",
                color=col,
            )

        # plot zero line
        ax_residuals.set_ylim(0, 3)
        ax_residuals.axhline(1, color="black", linewidth=1)
        ax_residuals.set_ylabel("SPANet / true pairing")

        ax_residuals.grid()

    # elif true: # meaning true pairing file
    #    # plot the residuals respect to true
    #    residuals_run2 = [
    #        (r[0] / np.sum(r[0])) / (t[0] / np.sum(t[0]))
    #        for r, t in zip(run2_hist, true_hist)
    #    ]
    #    logger.info(len(spanet_labels))
    #    logger.info(len(spanet_hists[0][0]))
    #    logger.info(len(spanet_hists[0][0]))

    #    residuals_spanet = [
    #        (spanet_hists[i][0] / np.sum(spanet_hists[i][0]))
    #        / (
    #            true_hist[check_names(spanet_labels[i])][0]
    #            / np.sum(true_hist[check_names(spanet_labels[i])][0])
    #        )
    #        for i in range(len(spanet_labels))
    #    ]

    #    residual_run2_err = (
    #        [
    #            np.sqrt(
    #                (np.sqrt(r[0]) / t[0]) ** 2 + ((np.sqrt(t[0]) * r[0]) / t[0] ** 2) ** 2
    #            )
    #            * (np.sum(t[0]) / np.sum(r[0]))
    #            for r, t in zip(run2_hist, true_hist)
    #        ]
    #        if run2
    #        else []
    #    )

    #  # residual_spanet_err = [
    #  #     np.sqrt(sn[0]) / true_hist[check_names(label)][0]
    #  #     for sn, label in zip(spanet_hists, spanet_labels)
    #  # ]
    #    residual_spanet_err = [
    #        np.sqrt(
    #            (np.sqrt(sn[0]) / true_hist[check_names(label)][0]) ** 2
    #            + (
    #                (np.sqrt(true_hist[check_names(label)][0]) * sn[0])
    #                / true_hist[check_names(label)][0] ** 2
    #            )
    #            ** 2
    #        )
    #        * (np.sum(true_hist[check_names(label)][0]) / np.sum(sn[0]))
    #        for sn, label in zip(spanet_hists, spanet_labels)
    #    ]

    #    for sn, label, sn_err in zip(
    #        residuals_spanet, spanet_labels, residual_spanet_err
    #    ):
    #        ax_residuals.errorbar(
    #            true_hist[check_names(label)][1][:-1],
    #            sn,
    #            yerr=sn_err,
    #            marker=".",
    #            # markersize=1,
    #            label=f"{names_dict[label]}" if label in names_dict else label,
    #            linestyle="None",
    #        )
    #        if check_names(label) in labels_list:
    #            continue
    #        if run2 and len(labels_list) == 0:
    #            which_run2 = (
    #                check_names(label) if ("data" in label or "klambda" in label) else 0
    #            )
    #            ax_residuals.errorbar(
    #                true_hist[check_names(label)][1][:-1],
    #                residuals_run2[which_run2],
    #                yerr=residual_run2_err[which_run2],
    #                marker=".",
    #                # markersize=1,
    #                label=f"D$_{{{HH}}}$ - method", #({true_labels[which_run2]})",
    #                color="red",
    #                linestyle="None",
    #            )
    #        labels_list.append(check_names(label))

    #    # plot zero line
    #    ax_residuals.axhline(1, color="black", linewidth=1)
    #    ax_residuals.set_xlabel(
    #        r"Leading $m_{H}$ [GeV]" if num == 1 else r"Subleading $m_{H}$ [GeV]"
    #    )
    #    ax_residuals.set_ylabel("Predicted / True")

    #    ax_residuals.grid()

    ax.legend(loc="upper right", frameon=False)

    hep.cms.label(year="2022", com="13.6", label="Private Work", ax=ax, data=True)
    plt.savefig(f"{plot_dir}/higgs_mass_{num}{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_dir}/higgs_mass_{num}{name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_dir}/higgs_mass_{num}{name}.svg", bbox_inches="tight")
    plt.close()


def plot_mhh(bins, mhh, plot_dir="plots", name="mhh"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(
        mhh,
        bins,
        label="mhh",
        histtype="step",
        linewidth=1,
        density=True,
    )
    ax.set_xlabel(r"$m_{HH}$ [GeV]")
    ax.set_ylabel("Normalized events")
    ax.grid()

    hep.cms.label(
        year="2022",
        com="13.6",
        label="Private Work",
        ax=ax,
    )
    plt.savefig(f"{plot_dir}/{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")
    plt.savefig(f"{plot_dir}/{name}.svg", dpi=300, bbox_inches="tight")
    plt.close()


def plot_histos_2d(mh_bins, higgs, label, name, plot_dir="plots"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist2d(
        np.array(higgs[:, 0].mass),
        np.array(higgs[:, 1].mass),
        bins=[mh_bins, mh_bins],
        label=f"{name} {label}",
        density=True,
        # yellow and green colors
        cmap=plt.cm.GnBu,
        # alpha=0.5,
    )
    cmap = mpl.cm.get_cmap("GnBu")
    norm = mpl.colors.Normalize(vmin=0, vmax=1.0)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax).set_label(
        "Normalized counts", loc="center", fontsize=10
    )
    # plot two lines at 125 GeV
    ax.plot([mh_bins, mh_bins], [120, 120], color="black")
    ax.plot([125, 125], [mh_bins, mh_bins], color="black")
    # draw a circle at 125 GeV or radius 5 GeV
    circle = plt.Circle((125, 120), 30, color="black", fill=False)
    ax.add_artist(circle)

    ax.set_xlabel(r"Leading $m_{H}$ [GeV]")
    ax.set_ylabel(r"Subleading $m_{H}$ [GeV]")
    ax.grid()
    # TODO: change title
    # ax.set_title(f"2D Higgs Mass {name} {label}", pad=20)

    hep.cms.label(year="2022", com="13.6", label="Private Work", ax=ax, data=True)
    plt.savefig(
        f"{plot_dir}/higgs_mass_2d_{name}_{label}.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        f"{plot_dir}/higgs_mass_2d_{name}_{label}.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(f"{plot_dir}/higgs_mass_2d_{name}_{label}.svg", bbox_inches="tight")
    plt.close()


def plot_diff_eff(
    mhh_bins,
    efficiency,
    unc_efficiency,
    labels,
    color,
    plot_dir,
    file_name,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    labels_list = []
    for eff, unc_eff, label, col in zip(efficiency, unc_efficiency, labels, color):
        ax.errorbar(
            0.5 * (mhh_bins[:-1] + mhh_bins[1:]),
            eff,
            yerr=unc_eff,
            label=label,
            marker="o",
            color=col,
        )

    ax.legend(frameon=False, loc="lower right")
    ax.set_xlabel(r"$m_{HH}$ [GeV]")
    ax.set_ylabel(file_name)
    ax.grid(linestyle=":")
    hep.cms.label(
        year="2022",
        com="13.6",
        label="Private Work",
        ax=ax,
    )
    plt.savefig(f"{plot_dir}/{file_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_dir}/{file_name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_dir}/{file_name}.svg", bbox_inches="tight")
    plt.close()


def plot_true_higgs(true_higgs_fully_matched, mh_bins, num, plot_dir="plots"):

    fig, (ax, ax_ratio) = plt.subplots(
        2, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax.hist(
        true_higgs_fully_matched[0][:, num - 1].mass,
        bins=mh_bins,
        histtype="step",
        label="True 4 jets",
        color="blue",
    )
    ax.hist(
        true_higgs_fully_matched[1][:, num - 1].mass,
        bins=mh_bins,
        histtype="step",
        label="True 5 jets",
        color="red",
    )
    hist_4 = np.histogram(true_higgs_fully_matched[0][:, num - 1].mass, bins=mh_bins)
    hist_5 = np.histogram(true_higgs_fully_matched[1][:, num - 1].mass, bins=mh_bins)

    hist_4_norm = hist_4[0] / np.sum(hist_4[0])
    hist_5_norm = hist_5[0] / np.sum(hist_5[0])

    ax_ratio.plot(
        mh_bins[:-1],
        hist_5[0] / hist_4[0],
        label="5 jets / 4 jets",
        color="green",
        linestyle=None,
        marker=".",
    )
    ax_ratio.plot(
        mh_bins[:-1],
        hist_5_norm / hist_4_norm,
        label="5 jets / 4 jets (norm)",
        color="purple",
        linestyle=None,
        marker=".",
    )
    ax_ratio.grid()
    ax.grid()

    hep.cms.label(
        year="2022",
        com="13.6",
        label="Private Work",
        ax=ax,
    )

    ax.legend()
    ax_ratio.legend()
    ax_ratio.set_xlabel(
        r"Leading $m_{H}$ [GeV]" if num == 1 else r"Subleading $m_{H}$ [GeV]"
    )
    ax.set_ylabel("Events")

    plt.savefig(f"{plot_dir}/true_higgs_mass_{num}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_dir}/true_higgs_mass_{num}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_dir}/true_higgs_mass_{num}.svg", bbox_inches="tight")
    plt.close()


def separate_klambda(
    jet, df_true, df_spanet_pred, idx_true, idx_spanet_pred, mask_region
):
    logger.info(f"jet {len(jet)}, {len(jet[0])}")

    try:
        kl_array_true = df_true["INPUTS"]["Event"]["kl"][()][mask_region]
    except KeyError:
        print("Did not find Event/kl in kl_array_true, will try EVENT/kl")
        kl_array_true = df_true["INPUTS"]["EVENT"]["kl"][()][mask_region]
    try:
        kl_array_spanet = df_spanet_pred["INPUTS"]["Event"]["kl"][()][mask_region]
    except KeyError:
        print("Did not find Event/kl in kl_array_spanet, will try EVENT/kl")
        kl_array_spanet = df_spanet_pred["INPUTS"]["EVENT"]["kl"][()][mask_region]
    logger.info(f"kl_arrays {kl_array_spanet}")

    # for each kl_array, separate the array based on the kl value
    # and create a list of arrays with the same kl value
    # true_separate_klambda = []
    # spanet_separate_klambda = []

    kl_unique_true = np.unique(kl_array_true)

    # Filling two dictionaries with the events of the respective kl values
    # One for the true one and one for the spanet one
    jet_separate_klambda = []

    true_kl_idx_list = []
    kl_values = []
    for kl in kl_unique_true:
        mask = kl_array_true == kl
        logger.info(f"mask {mask}")
        logger.info(f"kl_array_true[mask] {kl_array_true[mask]}")
        logger.info(f"kl {kl}")
        true_kl_idx_list.append(idx_true[mask])
        kl_values.append(kl)
        temp_jet = []
        for j, jet_info in enumerate(jet):
            temp_jet.append(jet_info[mask])
        jet_separate_klambda.append(temp_jet)

    if idx_spanet_pred is not None:
        kl_unique_spanet = np.unique(kl_array_spanet)
        spanet_kl_idx_list = []
        for kl in kl_unique_spanet:
            mask = kl_array_spanet == kl
            logger.info(f"mask {mask}")
            logger.info(f"kl_array_spanet[mask] {kl_array_spanet[mask]}")
            logger.info(f"kl {kl}")
            spanet_kl_idx_list.append(idx_spanet_pred[mask])
            assert kl in kl_values
    else:
        spanet_kl_idx_list = None

    # keep only two decimal in the kl_values
    kl_values = np.round(kl_values, 2)

    return (
        true_kl_idx_list,
        spanet_kl_idx_list,
        kl_values,
        jet_separate_klambda,
    )


def plot_diff_eff_klambda(
    effs, unc_effs, kl_values, labels, color, name, plot_dir="plots"
):
    # split the arrays depending on how many times the first kl value appears
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, kls, eff, unc_eff, col in zip(labels, kl_values, effs, unc_effs, color):
        ax.errorbar(
            kls,
            eff,
            yerr=unc_eff,
            linestyle="-",
            label=label,
            marker=".",
            color=col,
        )
    ax.legend(frameon=False, loc="lower left")
    ax.set_xlabel(r"$\kappa_{\lambda}$")
    ax.set_ylabel(name)
    ax.grid(linestyle=":")
    hep.cms.label(
        year="2022",
        com="13.6",
        label="Private Work",
        ax=ax,
    )
    plt.savefig(f"{plot_dir}/{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_dir}/{name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_dir}/{name}.svg", dpi=300, bbox_inches="tight")
    plt.close()


def add_fields(collection, fields=None, four_vec="PtEtaPhiMLorentzVector"):
    if fields == None:
        fields = list(collection.fields)
    for field in ["pt", "eta", "phi", "mass"]:
        if field not in fields:
            fields.append(field)
    if four_vec == "PtEtaPhiMLorentzVector":
        fields_dict = {field: getattr(collection, field) for field in fields}
        collection = ak.zip(
            fields_dict,
            with_name="PtEtaPhiMLorentzVector",
        )
    elif four_vec == "Momentum4D":
        fields = ["pt", "eta", "phi", "mass"]
        fields_dict = {field: getattr(collection, field) for field in fields}
        collection = ak.zip(
            fields_dict,
            with_name="Momentum4D",
        )
    else:
        for field in fields:
            collection = ak.with_field(collection, getattr(collection, field), field)

    return collection
