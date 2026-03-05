"""Generate ROC curves to compare performance of different classifiers."""
import argparse
import logging
import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import vector
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

# from matplotlib.colors import LogNorm
# import mplhep as hep
# hep.style.use(hep.style.ROOT)
from roc_configuration import spanet_dict, true_dict

vector.register_numba()
vector.register_awkward()
matplotlib.rcParams["figure.dpi"] = 300


def setup_logging(logpath):
    """Create logger for nicer logger.infoing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s",
        datefmt="%d-%b-%y %H-%M-%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"{logpath}/logger_output.log", mode="a", encoding="utf-8"
            ),
        ],
    )


def roc_curve_compare_weights(
    spanet_dict, plot_dir, fpr_cutoff, no_weights, log=False
):
    """Plot ROC curves from a dictionary fed into this function."""
    assert fpr_cutoff > 1e-6 and fpr_cutoff < 1e-1
    # Creating the figures:
    fig_zoom, ax_zoom = build_fig_ax("ROC curve zoomed in", [0, 0.4], [1e-6, 1e-1], log)
    fig_full, ax_full = build_fig_ax("ROC curve", [0, 1], [1e-6, 1], log)

    for model_name, model_dict in spanet_dict.items():
        spanetfile = h5py.File(model_dict["file"], "r")
        truefile = h5py.File(true_dict[model_dict["true"]]["name"], "r")
        # do the full chain per dataset
        logger.info(f"Loading new file {model_name}")
        for key, val in model_dict.items():
            logger.info(f"{key}: {val}")
        spanet_class = spanetfile["CLASSIFICATIONS"]["EVENT"]["class"][:, 1][()]
        true_class = truefile["CLASSIFICATIONS"]["EVENT"]["class"][()]
        weights = truefile["WEIGHTS"]["weight"][()]

        if no_weights:
            fpr, tpr, threshold = roc_curve(true_class, spanet_class)
        else:
            fpr, tpr, threshold = roc_curve(true_class, spanet_class, sample_weight=weights)
        logger.debug(f"fpr: {fpr}")
        logger.debug(f"tpr: {tpr}")
        logger.debug(f"thresholds: {threshold}")

        # Calculating the cutoff
        fpr_zoom = fpr[fpr <= fpr_cutoff]
        tpr_zoom = tpr[0:len(fpr_zoom)]

        auc_score_zoom = auc(fpr_zoom, tpr_zoom)
        auc_score = auc(fpr, tpr)

        logger.info(f"AUC(fpr[0,{fpr_cutoff}]): {auc_score_zoom}")
        logger.info(f"AUC total: {auc_score}")

        # Add to figures
        ax_zoom.plot(tpr, fpr, label=f"Model {model_dict['label']} | AUC(fpr[0,{fpr_cutoff:.0e}])={auc_score_zoom:.2e}", color=model_dict["color"])
        ax_full.plot(tpr, fpr, label=f"Model {model_dict['label']} | AUC(fpr[0,1])={auc_score:.2e}", color=model_dict["color"])

        for fig, ax, figname in zip([fig_zoom, fig_full], [ax_zoom, ax_full], ["ROC_curve_zoomed", "ROC_curve"]):
            ax.legend(fontsize="small")
            os.makedirs(f"{plot_dir}/roc_curves", exist_ok=True)
            for suffix in ["pdf", "svg", "png"]:
                fig.savefig(f"{plot_dir}/roc_curves/{figname}{'_log' if log else ''}.{suffix}", dpi=300, bbox_inches="tight")


def build_fig_ax(title, x_lims=[0, 1], y_lims=[1e-6, 1], log=False):
    """Build fig and ax objects to create plots."""
    assert len(x_lims) == len(y_lims) and len(x_lims) == 2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("True positive rate")
    ax.set_ylabel("False positive rate")
    if log:
        ax.set_yscale("log")
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    ax.grid(linestyle=":")
    if title:
        ax.set_title(f"{title}", fontsize="small")
    return fig, ax


def precision_recall_curve_function(
    spanet_dict, plot_dir, no_weights, log=False
):
    """Plot precision/recall curve from a dictionary fed into this function."""
    # Creating the figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    if log:
        ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision/Recall Curve", fontsize="small")
    ax.grid(linestyle=":")

    for model_name, model_dict in spanet_dict.items():
        spanetfile = h5py.File(model_dict["file"], "r")
        truefile = h5py.File(true_dict[model_dict["true"]]["name"], "r")
        # do the full chain per dataset
        logger.info(f"loading new file {model_name}")
        for key, val in model_dict.items():
            logger.info(f"{key}: {val}")
        spanet_class = spanetfile["CLASSIFICATIONS"]["EVENT"]["class"][:, 1][()]
        true_class = truefile["CLASSIFICATIONS"]["EVENT"]["class"][()]
        weights = truefile["WEIGHTS"]["weight"][()]

        if no_weights:
            precision, recall, threshold = precision_recall_curve(true_class, spanet_class)
            ap_score = average_precision_score(true_class, spanet_class)
        else:
            precision, recall, threshold = precision_recall_curve(true_class, spanet_class, sample_weight=weights)
            ap_score = average_precision_score(true_class, spanet_class, sample_weight=weights)

        logger.debug(f"precision: {precision}")
        logger.debug(f"recall: {recall}")
        logger.debug(f"thresholds: {threshold}")
        logger.info(f"Average Precision: {ap_score}")

        # Add to figures
        ax.plot(recall, precision, label=f"Model {model_dict['label']} | AP={ap_score:.2e}", color=model_dict["color"])

        ax.legend(fontsize="small")
        os.makedirs(f"{plot_dir}/precision_recall_curve", exist_ok=True)
        for suffix in ["pdf", "svg", "png"]:
            fig.savefig(f"{plot_dir}/precision_recall_curve/precision_recall_curve{'_log' if log else ''}.{suffix}", dpi=300, bbox_inches="tight")


def signal_background_hist(
    spanet_dict, plot_dir, no_weights, log=False
):
    """Plot background/signal histogram from a dictionary fed into this function."""
    # Creating the figure:
    for model_name, model_dict in spanet_dict.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel("SPANet Classification Score")
        ax.set_ylabel("Count")
        if log:
            ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_title(f"Background/Signal Histogram {model_dict['label']}", fontsize="small")
        ax.grid(linestyle=":")

        spanetfile = h5py.File(model_dict["file"], "r")
        truefile = h5py.File(true_dict[model_dict["true"]]["name"], "r")
        # do the full chain per dataset
        logger.info(f"loading new file {model_name}")
        for key, val in model_dict.items():
            logger.info(f"{key}: {val}")
        spanet_class = spanetfile["CLASSIFICATIONS"]["EVENT"]["class"][:, 1][()]
        true_class = truefile["CLASSIFICATIONS"]["EVENT"]["class"][()]
        weights = truefile["WEIGHTS"]["weight"][()]

        mask_background = (true_class == 0)

        # Add to figures
        ax.hist(spanet_class[mask_background], bins=50, alpha=0.5, label="Background", color="tab:blue", weights=weights[mask_background])
        ax.hist(spanet_class[~mask_background], bins=50, alpha=0.5, label="Signal for model", color="tab:orange", weights=weights[~mask_background])

        ax.legend(fontsize="small")
        os.makedirs(f"{plot_dir}/background_signal_hist", exist_ok=True)
        for suffix in ["pdf", "svg", "png"]:
            fig.savefig(f"{plot_dir}/background_signal_hist/background_signal_hist{'_'.join(model_dict['label'].split(' '))}{'_log' if log else ''}.{suffix}", dpi=300, bbox_inches="tight")


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-pd",
    "--plot-dir",
    default="./ROC_plots",
    help="Directory to save the plots",
)
arg_parser.add_argument(
    "-cut",
    "--fpr-cutoff",
    default=1e-2,
    help="In zoomed version, cutoff in fpr for AUC calculation",
)

arg_parser.add_argument(
    "-nw",
    "--no-weights",
    action="store_true",
    default=False,
    help="Compute the ROC with weights",
)
args = arg_parser.parse_args()

os.makedirs(args.plot_dir, exist_ok=True)
setup_logging(args.plot_dir)
logger = logging.getLogger(__name__)

logger.debug(spanet_dict)
logger.debug(true_dict)
# Load datatasets

roc_curve_compare_weights(spanet_dict, args.plot_dir, args.fpr_cutoff, args.no_weights)
precision_recall_curve_function(spanet_dict, args.plot_dir, args.no_weights)
signal_background_hist(spanet_dict, args.plot_dir, args.no_weights)
roc_curve_compare_weights(spanet_dict, args.plot_dir, args.fpr_cutoff, args.no_weights, log=True)
precision_recall_curve_function(spanet_dict, args.plot_dir, args.no_weights, log=True)
signal_background_hist(spanet_dict, args.plot_dir, args.no_weights, log=True)
