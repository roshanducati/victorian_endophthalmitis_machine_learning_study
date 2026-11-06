"""
03_benchmarks.py
================
Step 3 of the analysis pipeline. Comparator models and point estimates on the
held-out test set. Reuses the split, preprocessing, model spaces and tuning
protocol of 02_train.py and the ICI and DeLong functions of
evaluation_metrics.py, so nothing here can drift from the main models.

Manuscript
----------
Methods 2.4 and 2.5; Results 3.2 and 3.3; Table 2; Figure 2; Supporting
Information Tables S2 (logistic regression), S4 and S5.
Point estimates from this step are carried forward unchanged; every reported
interval and p value comes from 06_patient_cluster_bootstrap.py.

Models evaluated on the held-out test set (split seed 61)
  - XGBoost with microbiology and pre-culture: the fitted pipelines from
    02_train.py (their saved test predictions must be reproduced exactly).
  - Penalised logistic regression with microbiology and pre-culture: tuned on
    the development set with the identical protocol (100-candidate randomised
    search, five-fold patient-grouped cross-validation, AUROC refit, class
    weighting).
  - Presenting acuity alone: univariable logistic regression on logMAR (mean
    imputation, balanced class weights), fitted on the development set.
Each model's operating threshold is the Youden index of its in-sample
development predictions, applied unchanged to the test set (Table S5).

Outputs (results_strict_benchmarks/)
------------------------------------
  test_predictions_all_models.csv   test-set predictions of every model, with
                                    labels and presenting logMAR (no
                                    identifiers, but patient-level: never
                                    share or commit)
  A_test_benchmarks.csv             point estimates and thresholds per model
  A_auroc_comparisons.csv           DeLong comparisons (superseded in the
                                    manuscript by the patient-clustered bootstrap)
  models/LR_{micro,nomicro}_pipeline.joblib
  summary.json                      tuned logistic regression parameters

Usage
-----
  python 03_benchmarks.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("benchmarks")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_strict_benchmarks"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tr = _load("train02", "02_train.py")        # split, preprocessing, model spaces
ev = _load("evaluation_metrics", "evaluation_metrics.py")   # ICI, DeLong

tr.IN_FILE = ROOT / "data_strict" / "processed_episodes.csv"
SPLIT_SEED = tr.SPLIT_SEED                  # 61
SEED = tr.ESTIMATOR_SEED                    # 42
N_BOOT = 1000

NUM = tr.NUMERIC_FEATURES
CAT = tr.CATEGORICAL_FEATURES
NUM_NM = tr.NUMERIC_FEATURES_NOMICRO
CAT_NM = tr.CATEGORICAL_FEATURES_NOMICRO


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def boot_ci(y, s, fn=roc_auc_score, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap CI, same construction as 02_train._bootstrap_ci."""
    return tr._bootstrap_ci(np.asarray(y), np.asarray(s), fn, n_boot=n_boot, seed=seed)


def op_metrics(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    tp = int(((yhat == 1) & (y == 1)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    sd = lambda a, b: a / b if b else np.nan
    return dict(sens=sd(tp, tp + fn), spec=sd(tn, tn + fp), ppv=sd(tp, tp + fp),
                npv=sd(tn, tn + fn), tp=tp, fn=fn, tn=tn, fp=fp)


def summarise(label, y, p, thr=None, ci=True):
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    auc = roc_auc_score(y, p)
    lo, hi = boot_ci(y, p) if ci else (np.nan, np.nan)
    row = dict(model=label, n=len(y), n_poor=int(y.sum()),
               auroc=auc, auroc_lo=lo, auroc_hi=hi,
               auprc=average_precision_score(y, p),
               brier=brier_score_loss(y, p),
               ici=ev.compute_ici(y, p),
               mean_pred=float(p.mean()), obs_rate=float(y.mean()))
    if thr is not None:
        row["threshold"] = float(thr)
        row.update(op_metrics(y, (p >= thr).astype(int)))
    return row


def paired_diff(y, a, b, n_boot=N_BOOT, seed=SEED):
    """DeLong test plus paired bootstrap CI for AUROC(a) - AUROC(b)."""
    y = np.asarray(y)
    auc_a, auc_b, diff, z, p = ev.delong_compare(y, np.asarray(a), np.asarray(b))
    rng = np.random.default_rng(seed)
    n = len(y); diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return dict(auc_a=auc_a, auc_b=auc_b, diff=diff,
                diff_lo=float(np.percentile(diffs, 2.5)),
                diff_hi=float(np.percentile(diffs, 97.5)), z=z, p=p)


def train_youden(pipe, X, y):
    """Operating threshold from in-sample development predictions (canonical)."""
    p = pipe.predict_proba(X)[:, 1]
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


# ─────────────────────────────────────────────────────────────────────────────
# Model construction (canonical protocol)
# ─────────────────────────────────────────────────────────────────────────────

def tune(name, X, y, groups, num, cat, n_iter=tr.N_ITER_SEARCH):
    """Randomised search exactly as 02_train.tune_and_train (AUROC refit)."""
    spw = tr.imbalance_ratio(y)
    est, space = {n: (e, s) for n, e, s in tr.get_models_and_spaces(spw, [name])}[name]
    pipe = Pipeline([("pre", tr.make_preprocessor(num, cat)), ("clf", est)])
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="roc_auc",
                                refit=True, cv=StratifiedGroupKFold(n_splits=tr.CV_FOLDS),
                                n_jobs=-1, random_state=SEED)
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def va_only_pipeline():
    """Univariable logistic regression on presenting logMAR (mean-imputed)."""
    pre = ColumnTransformer([("num", Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler())]), ["pres_logmar"])], remainder="drop")
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(penalty=None, class_weight="balanced",
                                                max_iter=2000))])


# ─────────────────────────────────────────────────────────────────────────────
# Held-out test set
# ─────────────────────────────────────────────────────────────────────────────

def held_out_benchmarks(summary):
    log.info("== Held-out test-set benchmarks ==")
    Xtr, Xte, ytr, yte, dk, gtr, tri, tei = tr.load_and_split(NUM, CAT, SPLIT_SEED)
    Xtr_nm = Xtr[NUM_NM + CAT_NM]; Xte_nm = Xte[NUM_NM + CAT_NM]
    ytr_a = ytr.values; yte_a = yte.values

    # Canonical fitted pipelines
    xgb_m = joblib.load(ROOT / "results_strict/models/XGB_pipeline.joblib")
    xgb_n = joblib.load(ROOT / "results_strict_nomicro/models/XGB_pipeline.joblib")

    # Guard: the canonical predictions must reproduce exactly.
    saved = pd.read_csv(ROOT / "results_strict/test_predictions.csv")
    p_xgb_m = xgb_m.predict_proba(Xte)[:, 1]
    assert np.allclose(saved["prob_XGB"].values, p_xgb_m, atol=1e-6), "XGB drift"
    assert np.array_equal(saved["y_true"].values, yte_a), "label drift"

    t0 = time.time()
    lr_m, lr_m_params, lr_m_cv = tune("LR", Xtr, ytr, gtr, NUM, CAT)
    lr_n, lr_n_params, lr_n_cv = tune("LR", Xtr_nm, ytr, gtr, NUM_NM, CAT_NM)
    log.info("LR tuned in %.0fs  micro CV %.4f  pre-culture CV %.4f  params %s / %s",
             time.time() - t0, lr_m_cv, lr_n_cv, lr_m_params, lr_n_params)
    joblib.dump(lr_m, OUT / "models" / "LR_micro_pipeline.joblib")
    joblib.dump(lr_n, OUT / "models" / "LR_nomicro_pipeline.joblib")

    va = va_only_pipeline().fit(Xtr, ytr)

    models = {
        "XGB_micro": (xgb_m, Xtr, Xte),
        "XGB_nomicro": (xgb_n, Xtr_nm, Xte_nm),
        "LR_micro": (lr_m, Xtr, Xte),
        "LR_nomicro": (lr_n, Xtr_nm, Xte_nm),
        "VA_only": (va, Xtr, Xte),
    }
    probs, thr = {}, {}
    rows = []
    for name, (pipe, Xa, Xb) in models.items():
        probs[name] = pipe.predict_proba(Xb)[:, 1]
        thr[name] = train_youden(pipe, Xa, ytr_a)
        rows.append(summarise(name, yte_a, probs[name], thr[name]))
    bench = pd.DataFrame(rows)
    has_va = Xte["pres_logmar"].notna().values

    # Pairwise AUROC comparisons (DeLong + paired bootstrap)
    comps = []
    for a, b in [("XGB_micro", "XGB_nomicro"), ("XGB_micro", "LR_micro"),
                 ("XGB_nomicro", "LR_nomicro"), ("XGB_micro", "VA_only"),
                 ("XGB_nomicro", "VA_only"), ("LR_micro", "VA_only"),
                 ("LR_nomicro", "VA_only"), ("LR_micro", "LR_nomicro")]:
        d = paired_diff(yte_a, probs[a], probs[b])
        d.update(model_a=a, model_b=b)
        comps.append(d)
    comps = pd.DataFrame(comps)

    # Test-set predictions for the figures (no identifiers)
    pred = pd.DataFrame({"y_true": yte_a, **{f"prob_{k}": v for k, v in probs.items()},
                         "pres_logmar": Xte["pres_logmar"].values})
    pred.to_csv(OUT / "test_predictions_all_models.csv", index=False)

    bench.to_csv(OUT / "A_test_benchmarks.csv", index=False)
    comps.to_csv(OUT / "A_auroc_comparisons.csv", index=False)

    summary["A"] = dict(lr_micro_params={k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                         for k, v in lr_m_params.items()},
                        lr_nomicro_params={k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                           for k, v in lr_n_params.items()},
                        lr_micro_cv_auroc=lr_m_cv, lr_nomicro_cv_auroc=lr_n_cv,
                        test_va_missing=int((~has_va).sum()))
    log.info("\n%s", bench[["model", "auroc", "auroc_lo", "auroc_hi", "brier", "ici",
                            "threshold", "sens", "spec", "ppv", "npv"]]
             .round(3).to_string(index=False))
    log.info("\n%s", comps.round(4).to_string(index=False))


def main():
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    t0 = time.time()
    held_out_benchmarks(summary)
    summary["runtime_min"] = (time.time() - t0) / 60
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    log.info("Done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
