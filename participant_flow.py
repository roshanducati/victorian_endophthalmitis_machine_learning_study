"""
participant_flow.py
===================
Helper module for Figure 1 (drawn by 08_figures_test_set.py). gather_counts()
recomputes every box count from the processed data, so the diagram cannot drift
from the analysis:

  registry export -> episodes (repeat instances consolidated)
                  -> episodes without a documented final acuity excluded
                  -> analysed episodes
                  -> patient-level split (GroupShuffleSplit, seed 61, 20% test)
                     into development and held-out test episodes

The counts in the reported analysis are 1700 episodes (1640 patients), 400
excluded, 1300 analysed (1262 patients), 1038 development (1009 patients) and
262 test episodes (253 patients). The number of bootstrap resamples in the
test-set box (10 000) is read from the run metadata of
06_patient_cluster_bootstrap.py. box() and the colour constants are the drawing
helpers.

Not a pipeline step: nothing runs when this file is executed.
"""

import json
from pathlib import Path

import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import GroupShuffleSplit

import journal_style

PROCESSED = Path("data_strict/processed_episodes.csv")
RAW_FILE  = Path("data/raw/VictorianEndophthalm_DATA_2026-05-17_2131.xlsx")
BOOT_META = Path("results_strict_benchmarks/patient_cluster_bootstrap/run_metadata.json")
SPLIT_SEED = 61          # patient-level split seed (see 02_train.py)
TEST_SIZE  = 0.20

journal_style.apply()          # 8-pt type
FONT_PT = journal_style.BASE_FONT_PT


def gather_counts():
    """Recompute every box count from source so the figure stays in sync."""
    proc = pd.read_csv(PROCESSED)
    n_episodes   = len(proc)
    n_patients   = proc["rveeh_ur"].nunique()

    known = proc.dropna(subset=["poor_outcome"]).copy()
    n_excluded = n_episodes - len(known)
    n_model    = len(known)
    n_model_pts = known["rveeh_ur"].nunique()
    n_poor = int((known["poor_outcome"] == 1).sum())
    n_good = int((known["poor_outcome"] == 0).sum())

    # Patient-level split (identical to 02_train.py)
    groups = known["rveeh_ur"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=SPLIT_SEED)
    tr_idx, te_idx = next(gss.split(known, known["poor_outcome"], groups=groups))

    dev = known.iloc[tr_idx]
    tst = known.iloc[te_idx]

    # Admission-date span of the analysed cohort
    ad = pd.to_datetime(known["admission_date"], errors="coerce")
    date_min, date_max = ad.min(), ad.max()

    # Registry export raw arm sizes (for the top annotation only)
    try:
        raw = pd.read_excel(RAW_FILE, sheet_name=0)
        n_review = int((raw["redcap_event_name"] == "review_arm_1").sum())
    except Exception:
        n_review = None

    # Bootstrap resamples behind every reported interval (step 6)
    n_boot = json.loads(BOOT_META.read_text())["n_boot"]

    return dict(
        n_review=n_review,
        n_episodes=n_episodes, n_patients=n_patients,
        n_excluded=n_excluded,
        n_model=n_model, n_model_pts=n_model_pts,
        n_poor=n_poor, n_good=n_good,
        pct_poor=100 * n_poor / n_model, pct_good=100 * n_good / n_model,
        n_dev=len(dev), n_dev_pts=dev["rveeh_ur"].nunique(),
        dev_poor=int((dev["poor_outcome"] == 1).sum()),
        dev_good=int((dev["poor_outcome"] == 0).sum()),
        n_tst=len(tst), n_tst_pts=tst["rveeh_ur"].nunique(),
        tst_poor=int((tst["poor_outcome"] == 1).sum()),
        tst_good=int((tst["poor_outcome"] == 0).sum()),
        date_min=date_min, date_max=date_max,
        n_boot=n_boot,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
BOX_FC   = "#EEF3F8"
BOX_EC   = "#33507A"
EXCL_FC  = "#F7EEEE"
EXCL_EC  = "#9A4A4A"
ARROW_C  = "#33507A"


def box(ax, x, y, w, h, text, fc=BOX_FC, ec=BOX_EC, fontsize=FONT_PT, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.3, facecolor=fc, edgecolor=ec, mutation_aspect=1))
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color="#12233B", zorder=5)
