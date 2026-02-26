"""Generate ROC curves to compare performance of different classifiers."""
import argparse
import logging
import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import vector
from sklearn.metrics import auc, roc_auc_score, roc_curve

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
    spanet_dict, title, plot_dir, fpr_cutoff, no_weights
):
    """Plot ROC curves from a dictionary fed into this function."""
    assert fpr_cutoff > 1e-6 and fpr_cutoff < 1e-1
    # Creating the figures:
    fig_zoom, ax_zoom = build_fig_ax(f"{title} zoomed in", [0, 0.4], [1e-6, 1e-1])
    fig_full, ax_full = build_fig_ax(title, [0, 1], [1e-6, 1])

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
        ax_zoom.plot(tpr, fpr, label=f"{model_dict['label']} | AUC(fpr[0,{fpr_cutoff:.0e}])={auc_score_zoom:.2e}", color=model_dict["color"])
        ax_full.plot(tpr, fpr, label=f"{model_dict['label']} | AUC(fpr[0,1])={auc_score:.2e}", color=model_dict["color"])

        for fig, ax, figname in zip([fig_zoom, fig_full], [ax_zoom, ax_full], [f"{title}_zoomed" if title else "roc_plot_zoomed", title if title else "roc_plot"]):
            ax.legend(fontsize="small")
            for suffix in ["pdf", "svg", "png"]:
                if title:
                    fig.savefig(f"{plot_dir}/{figname}.{suffix}", dpi=300, bbox_inches="tight")
                else:
                    fig.savefig(f"{plot_dir}/{figname}.{suffix}", dpi=300, bbox_inches="tight")


def build_fig_ax(title, x_lims=[0, 1], y_lims=[1e-6, 1]):
    """Build fig and ax objects to create plots."""
    assert len(x_lims) == len(y_lims) and len(x_lims) == 2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("True positive rate")
    ax.set_ylabel("False positive rate")
    ax.set_yscale("log")
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    ax.grid(linestyle=":")
    if title:
        ax.set_title(f"{title}", fontsize="small")
    return fig, ax


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-l", "--title", type=str, default="", help="Title of the plot"
)
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

roc_curve_compare_weights(spanet_dict, args.title, args.plot_dir, args.fpr_cutoff, args.no_weights)
