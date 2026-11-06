"""
08_figures_test_set.py
======================
Step 8 of the analysis pipeline. Figures 1 and 2 (participant flow and held-out
test set).

Manuscript
----------
  Figure 1    fig1_participant_flow      participant flow and the patient-level
                                         development/test split
  Figure 2    fig2_roc_calibration       (A) ROC curves; (B) loess calibration
                                         curves
All written to results_strict_benchmarks/figures/ as TIFF, PDF and PNG.

Imports the styles and save() helper of 07_tables_and_figures.py (so that
module's set-up code runs first) and the counts and box helpers of
participant_flow.py.

Inputs
------
  results_strict_benchmarks/test_predictions_all_models.csv,
  A_test_benchmarks.csv                                           (03)
  data_strict/processed_episodes.csv (via participant_flow.py)    (01)
  results_strict_benchmarks/patient_cluster_bootstrap/
  run_metadata.json (number of resamples, via participant_flow.py) (06)

Usage
-----
  python 08_figures_test_set.py
"""

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from sklearn.metrics import roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tables_and_figures", ROOT / "07_tables_and_figures.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
js = F.js
RES, FIG = F.RES, F.FIG


def fig2_roc_calibration():
    te = pd.read_csv(RES / "test_predictions_all_models.csv")
    bench = pd.read_csv(RES / "A_test_benchmarks.csv").set_index("model")
    y = te["y_true"].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.9))
    for key, label, style in F.MODEL_STYLE:
        fpr, tpr, _ = roc_curve(y, te[f"prob_{key}"].values)
        ax1.plot(fpr, tpr, label=f"{F.SHORT[key]}, {bench.loc[key, 'auroc']:.3f}", **style)
    ax1.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.7, zorder=0)
    ax1.set_xlabel("1 − specificity"); ax1.set_ylabel("Sensitivity")
    ax1.set_title("Discrimination", loc="left")
    ax1.legend(loc="lower right", frameon=True, handlelength=2.2, borderpad=0.3, labelspacing=0.25)
    for key, label, style in F.MODEL_STYLE:
        p = te[f"prob_{key}"].values
        sm_ = lowess(y, p, frac=0.75, it=0, return_sorted=True)
        ax2.plot(sm_[:, 0], sm_[:, 1], label=F.SHORT[key], **style)
    ax2.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.7, zorder=0)
    ax2.set_xlabel("Predicted probability of poor outcome"); ax2.set_ylabel("Observed proportion (loess)")
    ax2.set_title("Calibration", loc="left")
    ax2.legend(loc="upper left", frameon=True, handlelength=2.2, borderpad=0.3, labelspacing=0.25)
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
    js.label_panels([ax1, ax2], x=-0.2, y=1.02)
    fig.tight_layout(w_pad=1.5)
    F.save(fig, "fig2_roc_calibration", 7.0, 3.9)


def fig1_participant_flow():
    import participant_flow as mpf
    c = mpf.gather_counts()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.set_xlim(0, 10.4); ax.set_ylim(4.05, 12.1); ax.axis("off")
    cx, w = 3.9, 5.6

    def arrow(x0, y0, x1, y1, head=True):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>" if head else "-", mutation_scale=11,
                                     linewidth=1.1, color=mpf.ARROW_C, shrinkA=0, shrinkB=0))
    mpf.box(ax, cx, 11.2, w, 1.1, "Royal Victorian Eye and Ear Hospital endophthalmitis registry\n"
                                 f"{c['n_episodes']} episodes ({c['n_patients']} patients)")
    arrow(cx, 10.65, cx, 8.95)
    mpf.box(ax, 8.6, 9.8, 3.0, 0.95, f"Excluded: {c['n_excluded']} episodes with no\ndocumented final visual acuity",
            fc=mpf.EXCL_FC, ec=mpf.EXCL_EC)
    arrow(cx, 9.8, 8.6 - 1.5 - 0.03, 9.8)
    dmin = c["date_min"].strftime("%b %Y"); dmax = c["date_max"].strftime("%b %Y")
    mpf.box(ax, cx, 8.3, w, 1.25,
            f"Analysed cohort: {c['n_model']} episodes ({c['n_model_pts']} patients), {dmin} to {dmax}\n"
            f"Poor outcome {c['n_poor']} ({c['pct_poor']:.1f}%)   Good outcome {c['n_good']} ({c['pct_good']:.1f}%)",
            weight="bold")
    ax.text(cx + 0.2, 7.2, "Random patient-level split (80:20, fixed seed)", ha="left", va="center",
            style="italic", color="#4A5A70")
    lx, rx, bw = 2.05, 5.75, 3.5
    arrow(cx, 7.675, cx, 6.75, head=False)
    ax.plot([lx, rx], [6.75, 6.75], color=mpf.ARROW_C, lw=1.1)
    arrow(lx, 6.75, lx, 6.2); arrow(rx, 6.75, rx, 6.2)
    mpf.box(ax, lx, 5.25, bw, 1.8, f"Development set\n{c['n_dev']} episodes ({c['n_dev_pts']} patients)\n"
                                  f"{c['dev_poor']} poor outcomes\nHyperparameter tuning (grouped\n5-fold CV) "
                                  "and thresholds")
    mpf.box(ax, rx, 5.25, bw, 1.8, f"Held-out test set\n{c['n_tst']} episodes ({c['n_tst_pts']} patients)\n"
                                  f"{c['tst_poor']} poor outcomes\nEvaluated once with {c['n_boot']:,} bootstraps;\n"
                                  "all reported performance and tests")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    js.enforce_fontsizes(fig)
    from PIL import Image
    tif = FIG / "fig1_participant_flow.tif"
    fig.savefig(tif, dpi=F.DPI_LINEART, format="tiff", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    im = Image.open(tif)
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[-1]); im = bg
    im.save(tif, compression="tiff_lzw", dpi=(F.DPI_LINEART, F.DPI_LINEART))
    fig.savefig(FIG / "fig1_participant_flow.pdf", facecolor="white"); fig.savefig(FIG / "fig1_participant_flow.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("saved fig1_participant_flow")


if __name__ == "__main__":
    fig1_participant_flow(); fig2_roc_calibration()
