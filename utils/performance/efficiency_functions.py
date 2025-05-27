from math import sqrt
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib as mpl

vector.register_awkward()
vector.register_numba()

from efficiency_configuration import *

k_lambda = [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.45, 3.0, 3.5, 4.0, 5.0]
TRUE_PAIRING = False

def check_names(name):
    # to be updated everytime you add a new item in the true dictionary
    # UPDATE_HERE indicates where to change the function
    lenght_true_dict = 25  # UPDATE_HERE to the new lenght of the true_dict
    if "klambda0" in name and "4_jets" in name:
        return 3
    elif "klambda2p45" in name and "4_jets" in name:
        return 4
    elif "klambda5" in name and "4_jets" in name:
        return 5
    elif "klambda0" in name and "5_jets" in name:
        return 6
    elif "klambda2p45" in name and "5_jets" in name:
        return 7
    elif "klambda5" in name and "5_jets" in name:
        return 8
    elif "5_jets_data" in name and "oldCutsEval" in name:
        return 11
    elif "5_jets_data" in name and "newCutsEval" in name:
        return 12
    elif "4_jets_data" in name:
        return 9
    elif "5_jets_data" in name:
        return 10
    elif (
        "allklambda" in name
        and "5_jets" in name
        and "oldCutsEval" in name
        and "newkl" in name
    ):
        for kl in k_lambda:
            if f"{kl}" in name:
                return lenght_true_dict + k_lambda.index(kl) + len(k_lambda) * 2
        return 15
    elif (
        "allklambda" in name
        and "5_jets" in name
        and "newCutsEval" in name
        and "newkl" in name
    ):
        for kl in k_lambda:
            if f"{kl}" in name:
                return lenght_true_dict + k_lambda.index(kl) + len(k_lambda) * 3
        return 16
    elif (
        "allklambda" in name
        and "4_jets" in name
        and "newCutsEval" in name
        and "newkl" in name
    ):
        for kl in k_lambda:
            if f"{kl}" in name:
                return lenght_true_dict + k_lambda.index(kl) + len(k_lambda) * 4
        return 17
    #UPDATE_HERE adding a new if statement
    elif "5_jets_pt_true_btag_allklambda" in name:
        for kl in k_lambda:
            if f"{kl}" in name:
                return lenght_true_dict + k_lambda.index(kl) + len(k_lambda) * 7
        return 21
    elif "5_jets_pt_true_vary_loose_btag_wide" in name:
        for kl in k_lambda:
            if f"{kl}" in name:
                return lenght_true_dict + k_lambda.index(kl) + len(k_lambda) * 9
        return 23
    elif "5_jets_pt_true_vary_loose_btag_01_10" in name:
        for kl in k_lambda:
            if f"{kl}" in name:
                return lenght_true_dict + k_lambda.index(kl) + len(k_lambda) * 10
        return 24
    elif "5_jets_pt_true_vary_loose_btag" in name:
        for kl in k_lambda:
            if f"{kl}" in name:
                return lenght_true_dict + k_lambda.index(kl) + len(k_lambda) * 8
        return 22
    elif '5_jets_pt_data' in name:
        return 20
    elif '5_jets_pt' in name:
        for kl in k_lambda:
            if f"{kl}" in name:
                return lenght_true_dict + k_lambda.index(kl) + len(k_lambda) * 5
        return 18
    elif '4_jets_pt' in name:
        for kl in k_lambda:
            if f"{kl}" in name:
                return lenght_true_dict + k_lambda.index(kl) + len(k_lambda) * 6
        return 19
    elif "allklambda" in name and "4_jets" in name:
        for kl in k_lambda:
            if f"{kl}" in name:
                return lenght_true_dict + k_lambda.index(kl)
        return 13
    elif "allklambda" in name and "5_jets" in name:
        for kl in k_lambda:
            if f"{kl}" in name:
                return lenght_true_dict + k_lambda.index(kl) + len(k_lambda)
        return 14
    elif "5_jets_btag_presel" in name:
        return 2
    elif "4_jets" in name:
        return 0
    elif "5_jets" in name:
        return 1
    else:
        raise ValueError(f"Name {name} not recognized")


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


def plot_histos_1d(
    bins, true, run2, spanet, spanet_labels, true_labels, num, name="", plot_dir="plots"
):
    #We want to add a ratio plot with run2 ratios. Basically to see the difference between the SPANet and Dhh pairings.
    compare_run2 = True
    if compare_run2 and run2:
        fig, (ax, ax_residuals) = plt.subplots(
            figsize=(6, 6), nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    #Otherwise we would have baseline
    elif any(["data" in label for label in spanet_labels]) or not TRUE_PAIRING:
        fig, ax = plt.subplots(
            figsize=(6, 6),
        )
        ax.set_xlabel(
            r"Leading $m_{H}$ [GeV]" if num == 1 else r"Subleading $m_{H}$ [GeV]"
        )
    #This would be the normal ratio with comparison to true values
    else:
        fig, (ax, ax_residuals) = plt.subplots(
            figsize=(6, 6), nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    ax.set_ylabel("Normalized events")

    labels_list = []
    for sn, label in zip(spanet, spanet_labels):
        if label[-4:] != "_1.0" and "." in label[-3]:
            continue
        #counts, edges = np.histogram(sn, bins=bins)
        #total_width = max(sn)-min(sn)
        #norm_counts = counts / (total_width * np.sum(counts))
        #ax.bar(edges[:-1], norm_counts, width=np.diff(edges), align="edge", alpha=0.7)
        ax.hist(
            sn,
            bins,
            label=f"{names_dict[label] if label in names_dict else label}",
            histtype="step",
            linewidth=1,
            density=False,
            weights=np.repeat(1.0/(len(sn)*np.diff(bins)[0]), len(sn)),
            color=color_dict[label] if label in color_dict else None,
        )
        if check_names(label) in labels_list:
            continue
        if "data" not in label and TRUE_PAIRING:  # and len(labels_list) == 0:
            #counts, edges = np.histogram(true[check_names(label)], bins=bins)
            #total_width = max(true[check_names(label)])-min(true[check_names(label)])
            #norm_counts = counts / (total_width * np.sum(counts))
            #ax.bar(edges[:-1], norm_counts, width=np.diff(edges), align="edge", alpha=0.7)
            ax.hist(
                true[check_names(label)],
                bins,
                label=f"True ({true_labels[check_names(label)]})",
                histtype="step",
                linewidth=1,
                density=False,
                weights=np.repeat(1.0/(len(true[check_names(label)]*np.diff(bins)[0]), len(true[check_names(label)]))),
                color="black" if len(labels_list) == 0 else "purple",
            )
        if run2 and len(labels_list) == 0:
            which_run2 = (
                check_names(label) if ("data" in label or "klambda" in label) else 0
            )
            print("plotting run2")
            print(which_run2)
            #counts, edges = np.histogram(run2[which_run2], bins=bins)
            #total_width = max(run2[which_run2])-min(run2[which_run2])
            #norm_counts = counts / (total_width * np.sum(counts))
            #ax.bar(edges[:-1], norm_counts, width=np.diff(edges), align="edge", alpha=0.7)
            ax.hist(
                run2[which_run2],
                bins,
                #label=names_dict[true_labels[which_run2]] if true_labels[which_run2] in names_dict else "D$_{{{HH}}}$ - method", #+f"({true_labels[which_run2]})",
                label="D$_{{{HH}}}$ - method", #+f"({true_labels[which_run2]})",
                histtype="step",
                linewidth=1,
                density=False,
                weights=np.repeat(1.0/(len(run2[which_run2])*np.diff(bins)[0]), len(run2[which_run2])),
                color="yellowgreen",
            )
        labels_list.append(check_names(label))
    ax.grid(linestyle=":")
    # true_hist = [np.histogram(true[i], bins, weights=np.ones_like(true[i])/len(true[i])) for i in range(len(true))]
    true_hist = [np.histogram(true[i], bins) for i in range(len(true))]
    true_norm = [len(i) for i in true]
    # print("true_hist", true_hist)
    run2_hist = (
        # [np.histogram(run2[i], bins, weights=np.ones_like(run2[i])/len(run2[i])) for i in range(len(run2))] if run2 else []
        [np.histogram(run2[i], bins) for i in range(len(run2))] if run2 else []
    )
    run2_norm = [len(i) for i in run2] if run2 else []
    spanet_hists = [
        #np.histogram(spanet[i], bins, weights=np.ones_like(spanet[i]) / len(spanet[i]))
        np.histogram(spanet[i], bins)
        for i in range(len(spanet))
    ]
    spanet_norm = [len(i) for i in spanet]
    print(run2_norm)
    print(spanet_norm)
    print("spanet_hists")
    print(len(spanet_hists))
    ax.set_ylim(
        0,
        max(
            np.histogram(
                spanet[0],
                # true[check_names(spanet_labels[0])],
                bins,
                density=False,
                weights=np.repeat(1.0/(len(spanet[0])*np.diff(bins)[0]), len(spanet[0])),
            )[0]
        )
        * (1.8 if "peak" not in name else 1.6),
    )
    labels_list = []
    if compare_run2 and run2:
        # Plot the residuals with respect to Run2
        print(len(spanet_labels))
        print(len(spanet_hists[0][0]))
        print(len(spanet_hists[0][0]))

        res_spanet_run2 = [
            (spanet_hists[i][0]/spanet_norm[i])
            / (
                run2_hist[which_run2][0] / run2_norm[which_run2]
            )
            for i in range(len(spanet_labels))
        ]
        
        err_spanet = [np.sqrt(spanet_hists[i][0])/spanet_norm[i] for i in range(len(spanet_hists))]
        err_run2 = [np.sqrt(run2_hist[i][0])/run2_norm[i] for i in range(len(run2_hist))]
        res_spanet_run2_err = [
            np.sqrt(
                (err_spanet[i] / (run2_hist[which_run2][0])/run2_norm[which_run2]) ** 2
                + (
                    (err_run2[which_run2] * (spanet_hists[i][0]/spanet_norm[i]))
                    / (run2_hist[which_run2][0]/run2_norm[which_run2]) ** 2
                )
                ** 2
            )
            for i in range(len(spanet_hists))
        ]
        for sn, label, sn_err in zip(
            res_spanet_run2, spanet_labels, res_spanet_run2_err
        ):
            ax_residuals.errorbar(
                run2_hist[which_run2][1][:-1],
                sn,
                yerr=sn_err,
                marker=".",
                # markersize=1,
                label=f"{names_dict[label]}" if label in names_dict else label,
                linestyle="None",
                color=color_dict[label] if label in color_dict else None,
            )
            if check_names(label) in labels_list:
                continue
            labels_list.append(check_names(label))

        # plot zero line
        ax_residuals.axhline(1, color="black", linewidth=1)
        ax_residuals.set_xlabel(
            r"Leading $m_{H}$ [GeV]" if num == 1 else r"Subleading $m_{H}$ [GeV]"
        )
        ax_residuals.set_ylabel("SPANet / $D_{HH}$")

        ax_residuals.grid()
    elif not any(["data" in label for label in spanet_labels]) and TRUE_PAIRING:
    # plot the residuals respect to true
        residuals_run2 = [
            (r[0] / np.sum(r[0])) / (t[0] / np.sum(t[0]))
            for r, t in zip(run2_hist, true_hist)
        ]
        print(len(spanet_labels))
        print(len(spanet_hists[0][0]))
        print(len(spanet_hists[0][0]))

        residuals_spanet = [
            (spanet_hists[i][0] / np.sum(spanet_hists[i][0]))
            / (
                true_hist[check_names(spanet_labels[i])][0]
                / np.sum(true_hist[check_names(spanet_labels[i])][0])
            )
            for i in range(len(spanet_labels))
        ]


        residual_run2_err = (
            [
                np.sqrt(
                    (np.sqrt(r[0]) / t[0]) ** 2 + ((np.sqrt(t[0]) * r[0]) / t[0] ** 2) ** 2
                )
                * (np.sum(t[0]) / np.sum(r[0]))
                for r, t in zip(run2_hist, true_hist)
            ]
            if run2
            else []
        )

      # residual_spanet_err = [
      #     np.sqrt(sn[0]) / true_hist[check_names(label)][0]
      #     for sn, label in zip(spanet_hists, spanet_labels)
      # ]
        residual_spanet_err = [
            np.sqrt(
                (np.sqrt(sn[0]) / true_hist[check_names(label)][0]) ** 2
                + (
                    (np.sqrt(true_hist[check_names(label)][0]) * sn[0])
                    / true_hist[check_names(label)][0] ** 2
                )
                ** 2
            )
            * (np.sum(true_hist[check_names(label)][0]) / np.sum(sn[0]))
            for sn, label in zip(spanet_hists, spanet_labels)
        ]

        for sn, label, sn_err in zip(
            residuals_spanet, spanet_labels, residual_spanet_err
        ):
            ax_residuals.errorbar(
                true_hist[check_names(label)][1][:-1],
                sn,
                yerr=sn_err,
                marker=".",
                # markersize=1,
                label=f"{names_dict[label]}" if label in names_dict else label,
                linestyle="None",
            )
            if check_names(label) in labels_list:
                continue
            if run2 and len(labels_list) == 0:
                which_run2 = (
                    check_names(label) if ("data" in label or "klambda" in label) else 0
                )
                ax_residuals.errorbar(
                    true_hist[check_names(label)][1][:-1],
                    residuals_run2[which_run2],
                    yerr=residual_run2_err[which_run2],
                    marker=".",
                    # markersize=1,
                    label=f"D$_{{{HH}}}$ - method", #({true_labels[which_run2]})",
                    color="red",
                    linestyle="None",
                )
            labels_list.append(check_names(label))

        # plot zero line
        ax_residuals.axhline(1, color="black", linewidth=1)
        ax_residuals.set_xlabel(
            r"Leading $m_{H}$ [GeV]" if num == 1 else r"Subleading $m_{H}$ [GeV]"
        )
        ax_residuals.set_ylabel("Predicted / True")

        ax_residuals.grid()

    ax.legend(loc="upper right", frameon=False)

    hep.cms.label(
        year="2022",
        com="13.6",
        label=f"Private Work",
        ax=ax,
        data=True
    )
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

    hep.cms.label(
        year="2022",
        com="13.6",
        label=f"Private Work",
        ax=ax,
        data=True
    )
    plt.savefig(
        f"{plot_dir}/higgs_mass_2d_{name}_{label}.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        f"{plot_dir}/higgs_mass_2d_{name}_{label}.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        f"{plot_dir}/higgs_mass_2d_{name}_{label}.svg", bbox_inches="tight"
    )
    plt.close()


def plot_diff_eff(
    mhh_bins,
    run2,
    unc_run2,
    true_dict,
    spanet,
    unc_spanet,
    spanet_dict,
    plot_dir,
    file_name,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    labels_list = []
    for eff, unc_eff, label in zip(spanet, unc_spanet, list(spanet_dict.keys())):
        ax.errorbar(
            0.5 * (mhh_bins + mhh_bins),
            eff,
            yerr=unc_eff,
            label=f"{names_dict[label] if label in names_dict else label}",
            marker="o",
            color=color_dict[label] if label in color_dict else None,
        )

        if check_names(label) in labels_list or not run2 or len(labels_list) > 0:
            continue
        which_run2 = (
            check_names(label) if ("data" in label or "klambda" in label) else 0
        )
        ax.errorbar(
            0.5 * (mhh_bins + mhh_bins),
            run2[which_run2],
            yerr=unc_run2[which_run2],
            label=r"$D_{HH}$-method",
            marker="o",
            color="yellowgreen",
        )
        labels_list.append(check_names(label))
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlabel(r"$m_{HH}$ [GeV]")
    ax.set_ylabel(names_dict[file_name])
    ax.grid(linestyle=":")
    hep.cms.label(
        year="2022",
        com="13.6",
        label=f"Private Work",
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
        label=f"Private Work",
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
    df_true, df_spanet_pred, idx_true, idx_spanet_pred, true_dict, spanet_dict
):
    # check if "allklambda" is in the key of the true_dict
    true_mask = [True if "allklambda" in key else False for key in true_dict.keys()]
    allkl_names_true = [key for key in true_dict.keys() if "allklambda" in key]
    # mask the list df_true with the true_mask
    true_allklambda = []
    idx_true_allklambda = []
    for df, mask, idx in zip(df_true, true_mask, idx_true):
        if mask:
            true_allklambda.append(df)
            idx_true_allklambda.append(idx)

    # load jet information
    jet_ptPNetRegNeutrino = [
        df["INPUTS"]["Jet"]["ptPnetRegNeutrino"][()] for df in true_allklambda
    ]
    jet_eta = [df["INPUTS"]["Jet"]["eta"][()] for df in true_allklambda]
    jet_phi = [df["INPUTS"]["Jet"]["phi"][()] for df in true_allklambda]
    jet_mass = [df["INPUTS"]["Jet"]["mass"][()] for df in true_allklambda]

    jet_infos = [jet_ptPNetRegNeutrino, jet_eta, jet_phi, jet_mass]
    print("jet_infos", len(jet_infos), len(jet_infos[0]))

    kl_arrays_true = [df["INPUTS"]["Event"]["kl"][()] for df in true_allklambda]

    spanet_mask = [True if "allklambda" in key else False for key in spanet_dict.keys()]
    allkl_names_spanet = [key for key in spanet_dict.keys() if "allklambda" in key]

    spanet_allklambda = []
    idx_spanet_allklambda = []

    for df, mask, idx in zip(df_spanet_pred, spanet_mask, idx_spanet_pred):
        if mask:
            spanet_allklambda.append(df)
            idx_spanet_allklambda.append(idx)

    print(true_allklambda, len(true_allklambda))
    print(spanet_allklambda, len(spanet_allklambda))

    kl_arrays_spanet = [df["INPUTS"]["Event"]["kl"][()] for df in spanet_allklambda]
    print("kl_arrays", kl_arrays_spanet)

    # for each kl_array, separate the array based on the kl value
    # and create a list of arrays with the same kl value
    true_separate_klambda = []
    spanet_separate_klambda = []
    jet_infos_separate_klambda = [[] for _ in range(len(jet_infos))]

    kl_values_true = []
    for i, kl_array in enumerate(kl_arrays_true):
        kl_unique = np.unique(kl_array)
        # kl_unique=np.array(k_lambda)
        kl_values_true += kl_unique.tolist()

        for kl in kl_unique:
            mask = kl_array == kl
            print("mask", mask)
            print("kl_array[mask]", kl_array[mask])
            true_separate_klambda.append(idx_true_allklambda[i][mask])
            for j, jet_info in enumerate(jet_infos):
                jet_infos_separate_klambda[j].append(jet_info[i][mask])

    kl_values_spanet = []
    for i, kl_array in enumerate(kl_arrays_spanet):
        kl_unique = np.unique(kl_array)
        kl_values_spanet += kl_unique.tolist()

        for kl in kl_unique:
            mask = kl_array == kl
            print("mask", mask)
            print("kl_array[mask]", kl_array[mask])
            spanet_separate_klambda.append(idx_spanet_allklambda[i][mask])

   # print(true_separate_klambda, len(true_separate_klambda))
   # print(spanet_separate_klambda, len(spanet_separate_klambda))
   # print(len(jet_infos_separate_klambda), len(jet_infos_separate_klambda[0]))

    # keep only two decimal in the kl_values
    kl_values_true = np.round(kl_values_true, 2).tolist()
    kl_values_spanet = np.round(kl_values_spanet, 2).tolist()

    # kl_values_true = k_lambda * 3
    # kl_values_spanet = k_lambda * 3

    # remove the allklambda from the list
    # idx_true = [idx for i, idx in enumerate(idx_true) if i not in all_kl_idx]
    # idx_spanet_pred = [idx for i, idx in enumerate(idx_spanet_pred) if i not in all_kl_idx]
    # print(all_kl_idx)
    idx_true.extend(true_separate_klambda)
    idx_spanet_pred.extend(spanet_separate_klambda)

    spanet_dict_add = {}
    true_dict_add = {}
    for key in spanet_dict.keys():
        if "allklambda" in key:
            for kl in kl_values_true:
                if f"{key}_{kl}" not in spanet_dict:
                    spanet_dict_add[f"{key}_{kl}"] = spanet_dict[key]
    for key in true_dict.keys():
        if "allklambda" in key:
            for kl in kl_values_true:
                if f"{key}_{kl}" not in true_dict:
                    true_dict_add[f"{key}_{kl}"] = true_dict[key]

    spanet_dict.update(spanet_dict_add)
    true_dict.update(true_dict_add)

    print("idx_true", len(idx_true))
    print("idx_spanet_pred", len(idx_spanet_pred))
    print("kl_values", kl_values_true)
    print(allkl_names_spanet)
    print(allkl_names_true)
    print("spanet_dict", spanet_dict, len(spanet_dict))
    print("true_dict", true_dict, len(true_dict))

    return (
        idx_true,
        idx_spanet_pred,
        true_dict,
        spanet_dict,
        jet_infos_separate_klambda,
        kl_values_true,
        kl_values_spanet,
        allkl_names_true,
        allkl_names_spanet,
    )


def plot_diff_eff_klambda(eff, kl_values, allkl_names, name, plot_dir="plots"):
    # split the arrays depending on how many times the first kl value appears
    kl_values_split = np.split(np.array(kl_values), kl_values.count(kl_values[0]))
    eff_split = np.split(np.array(eff), kl_values.count(kl_values[0]))
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (net_name, kls) in enumerate(zip(allkl_names, kl_values_split)):
        print(f"net_name = {net_name}")
        ax.plot(
            kls,
            eff_split[i],
            label=names_dict[net_name] if net_name in names_dict else net_name,
            linestyle="-",
            marker="o",
            color=color_dict[net_name] if net_name in color_dict else None,
        )

    ax.legend(frameon=False, loc="lower left")
    ax.set_xlabel(r"$\kappa_{\lambda}$")
    ax.set_ylabel(names_dict[name] if name in names_dict else name)
    ax.grid(linestyle=":")
    hep.cms.label(
        year="2022",
        com="13.6",
        label=f"Private Work",
        ax=ax,
    )
    plt.savefig(f"{plot_dir}/{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_dir}/{name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{plot_dir}/{name}.svg", dpi=300, bbox_inches="tight")
    plt.close()

def add_fields(collection, fields=None, four_vec="PtEtaPhiMLorentzVector"):
    if fields==None:
        fields= list(collection.fields)
    for field in ["pt", "eta", "phi", "mass"]:
        if field not in fields:
            fields.append(field)
    if four_vec=="PtEtaPhiMLorentzVector":
        fields_dict = {field: getattr(collection, field) for field in fields}
        collection = ak.zip(
            fields_dict,
            with_name="PtEtaPhiMLorentzVector",
        )
    elif four_vec=="Momentum4D":
        fields=["pt", "eta", "phi", "mass"]
        fields_dict = {field: getattr(collection, field) for field in fields }
        collection = ak.zip(
            fields_dict,
            with_name="Momentum4D",
        )
    else:
        for field in fields:
            collection = ak.with_field(collection, getattr(collection, field), field)

    return collection
