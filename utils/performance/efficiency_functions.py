from math import sqrt
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

vector.register_awkward()
vector.register_numba()


def check_names(name):
    if "klambda0" in name:
        return 3
    elif "klambda2p45" in name:
        return 4
    elif "klambda5" in name:
        return 5
    elif "5_jets_btag_presel" in name:
        return 2
    elif "4_jets" in name:
        return 0
    elif "5_jets" in name:
        return 1
    else:
        raise ValueError(f"Name {name} not recognized")


def distance_func(higgs_pair, k):
    higgs1 = higgs_pair[:, :, 0]
    higgs2 = higgs_pair[:, :, 1]
    dist = abs(higgs1.mass - higgs2.mass * k) / sqrt(1 + k**2)
    return dist


def reco_higgs(jet_collection, idx_collection):
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


def best_reco_higgs(jet_collection, idx_collection):
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
    higgs_candidates_unflatten_order = higgs_pair[higgs_candidates_unflatten_order_idx]
    return higgs_candidates_unflatten_order


def plot_histos(
    bins, true, run2, spanet, spanet_labels, num, name="", plot_dir="plots"
):
    fig, (ax, ax_residuals) = plt.subplots(
        figsize=(10, 8), nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    # plt.xlabel(r""
    plt.ylabel("Normalized events")

    # ax.hist(
    #     true,
    #     bins,
    #     label=f"True Higgs{num}Mass",
    #     color="black",
    #     histtype="step",
    #     linewidth=2,
    #     density=True,
    # )
    # ax.hist(
    #     run2,
    #     bins,
    #     label=f"Run2 RecoHiggs{num}Mass",
    #     color="red",
    #     histtype="step",
    #     linewidth=2,
    #     density=True,
    # )
    labels_list = []
    for sn, label in zip(spanet, spanet_labels):
        ax.hist(
            sn,
            bins,
            label=f"SPANet {label}",
            histtype="step",
            linewidth=2,
            density=True,
        )
        if check_names(label) in labels_list:
            continue
        ax.hist(
            true[check_names(label)],
            bins,
            label=f"True {label}",
            histtype="step",
            linewidth=2,
            density=True,
        )
        ax.hist(
            run2[check_names(label)],
            bins,
            label=f"Run2 {label}",
            histtype="step",
            linewidth=2,
            density=True,
        )
        labels_list.append(check_names(label))
    ax.grid()
    true_hist = [np.histogram(t, bins) for t in true]
    run2_hist = [np.histogram(r, bins) for r in run2]
    spanet_hists = [np.histogram(sn, bins) for sn in spanet]
    # plot error bars
    # ax.errorbar(
    #     0.5 * (bins[1:] + bins[:-1]),
    #     true_hist[0],
    #     yerr=np.sqrt(true_hist[0]),
    #     fmt="none",
    #     color="blue",
    # )
    # ax.errorbar(
    #     0.5 * (bins[1:] + bins[:-1]),
    #     run2_hist[0],
    #     yerr=np.sqrt(run2_hist[0]),
    #     fmt="none",
    #     color="red",
    # )
    # ax.errorbar(
    #     0.5 * (bins[1:] + bins[:-1]),
    #     spanet_hist[0],
    #     yerr=np.sqrt(spanet_hist[0]),
    #     fmt="none",
    #     color="green",
    # )

    # plot the residuals respect to true
    residuals_run2 = [r[0] / t[0] for r, t in zip(run2_hist, true_hist)]
    residuals_spanet = [
        spanet_hists[i][0] / true_hist[check_names(spanet_labels[i])][0]
        for i in range(len(spanet_labels))
    ]
    residual_run2_err = [np.sqrt(r[0]) / t[0] for r, t in zip(run2_hist, true_hist)]
    residual_spanet_err = [
        np.sqrt(sn[0]) / true_hist[check_names(label)][0]
        for sn, label in zip(spanet_hists, spanet_labels)
    ]
    # ax_residuals.errorbar(
    #     true_hist[1][:-1],
    #     residuals_run2,
    #     yerr=residual_run2_err,
    #     marker=".",
    #     color="red",
    #     label="Run2",
    #     fmt="none",
    # )
    labels_list = []
    for sn, label, sn_err in zip(residuals_spanet, spanet_labels, residual_spanet_err):
        ax_residuals.errorbar(
            true_hist[1][:-1],
            sn,
            yerr=sn_err,
            marker=".",
            label=f"SPANet {label}",
            fmt="none",
        )
        if check_names(label) in labels_list:
            continue
        ax_residuals.errorbar(
            true_hist[1][:-1],
            residuals_run2[check_names(label)],
            yerr=residual_run2_err[check_names(label)],
            marker=".",
            label=f"Run2 {label}",
            fmt="none",
        )
        labels_list.append(check_names(label))

    # plot zero line
    ax_residuals.axhline(1, color="black", linewidth=1)

    ax_residuals.grid()

    ax.legend(loc="upper right")

    hep.cms.label(
        year="2022",
        com="13.6",
        label=f"Private Work",
        ax=ax,
    )
    plt.savefig(f"{plot_dir}/higgs_mass_{num}{name}.png")
    # plt.show()
