"""
evaluation_metrics.py
=====================
Shared statistical functions, imported by 03_benchmarks.py,
05_heldout_statistics.py and 07_tables_and_figures.py. The function bodies are
unchanged from the evaluation script of the original analysis.

  compute_ici          integrated calibration index (Austin and Steyerberg, Stat
                       Med 2019): mean absolute difference between a loess
                       calibration curve (span 0.75, no robustness iterations)
                       and the predicted probabilities
  bootstrap_ici        episode-bootstrap percentile interval for the ICI
  _delong_components,  DeLong test for two correlated AUROCs on the same
  delong_compare       episodes (DeLong et al, Biometrics 1988)
  make_table3          operating characteristics across probability thresholds
                       (Supporting Information Table S7)

Not a pipeline step: nothing runs when this file is executed.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

# Fixed RNG for bootstrap resampling. Separate from the data-split seed (61, in
# 02_train.py) and held constant so that evaluation is reproducible.
ESTIMATOR_SEED = 42
N_BOOTSTRAP  = 1000   # for CI estimation


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def compute_ici(y_true, y_score, frac=0.75):
    """Integrated Calibration Index (Austin & Steyerberg, Stat Med 2019).

    Fits a loess curve regressing observed outcomes on predicted probabilities,
    then returns the mean absolute difference between the smoothed curve and
    the 45-degree calibration line: ICI = mean(|loess(p_i) - p_i|).
    frac=0.75 matches the R default used in the original paper.
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    smoothed = sm_lowess(y_true, y_score, frac=frac, it=0, return_sorted=True)
    fn = interp1d(smoothed[:, 0], smoothed[:, 1],
                  bounds_error=False,
                  fill_value=(smoothed[0, 1], smoothed[-1, 1]))
    return float(np.mean(np.abs(fn(y_score) - y_score)))


def bootstrap_ici(y_true, y_score, n_boot=N_BOOTSTRAP, rng=None, frac=0.75):
    """Bootstrap 95% CI for ICI."""
    if rng is None:
        rng = np.random.default_rng(ESTIMATOR_SEED)
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(compute_ici(y_true[idx], y_score[idx], frac=frac))
        except Exception:
            continue
    scores = np.array(scores)
    return scores.mean(), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


# ─────────────────────────────────────────────────────────────────────────────
# DeLong test for correlated AUROCs
# ─────────────────────────────────────────────────────────────────────────────

def _delong_components(y_true, y_score):
    """Kernel (structural) components used by DeLong's test."""
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    V10 = np.array([
        np.mean(y_score[pos[i]] > y_score[neg]) +
        0.5 * np.mean(y_score[pos[i]] == y_score[neg])
        for i in range(len(pos))
    ])
    V01 = np.array([
        np.mean(y_score[pos] > y_score[neg[j]]) +
        0.5 * np.mean(y_score[pos] == y_score[neg[j]])
        for j in range(len(neg))
    ])
    return V10.mean(), V10, V01


def delong_compare(y_true, prob_a, prob_b):
    """
    DeLong's method for comparing two AUROCs on the same test set.

    Accounts for the covariance between the two ROC curves (i.e. the
    comparison is inherently paired because both models see the same cases).
    Returns (auc_a, auc_b, diff, z, p_two_sided).

    Reference: DeLong et al. (1988). Biometrics 44(3):837–845.
    """
    y_true = np.asarray(y_true, dtype=int)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1

    auc_a, V10_a, V01_a = _delong_components(y_true, prob_a)
    auc_b, V10_b, V01_b = _delong_components(y_true, prob_b)

    S10 = np.cov(np.vstack([V10_a, V10_b])) / n1
    S01 = np.cov(np.vstack([V01_a, V01_b])) / n0
    S   = S10 + S01

    L        = np.array([1.0, -1.0])
    var_diff = float(L @ S @ L)
    z        = (auc_a - auc_b) / np.sqrt(var_diff)
    p        = float(2 * stats.norm.sf(abs(z)))
    return auc_a, auc_b, auc_a - auc_b, z, p


# ─────────────────────────────────────────────────────────────────────────────
# Operating characteristics across thresholds (Table S7)
# ─────────────────────────────────────────────────────────────────────────────

def make_table3(y_true, probs, metrics, lead_name=None) -> pd.DataFrame:
    """Operating-point table: multiple thresholds for the lead model."""
    best_name = lead_name or metrics.sort_values("test_auroc", ascending=False).iloc[0]["model"]
    if best_name not in probs:
        return pd.DataFrame()
    y_score = probs[best_name]

    rows = []
    for thr in np.arange(0.10, 0.91, 0.05):
        y_pred = (y_score >= thr).astype(int)
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_true, y_pred,
                                                    labels=[0, 1]).ravel()
        sens = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else np.nan
        spec = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else np.nan
        ppv  = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else np.nan
        npv  = tn_f / (tn_f + fn_f) if (tn_f + fn_f) > 0 else np.nan
        rows.append({
            "Threshold":      round(thr, 2),
            "Sensitivity":    round(sens, 3),
            "Specificity":    round(spec, 3),
            "PPV":            round(ppv,  3),
            "NPV":            round(npv,  3),
            "TP": int(tp_f), "FP": int(fp_f),
            "TN": int(tn_f), "FN": int(fn_f),
        })
    return pd.DataFrame(rows)
