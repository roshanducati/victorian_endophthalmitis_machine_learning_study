"""
05_heldout_statistics.py
========================
Step 5 of the analysis pipeline. Point estimates and conventional tests on the
held-out test set (262 episodes): pairwise comparisons, calibration,
discrimination by sex and age, class weighting and follow-up.

Manuscript
----------
Supplies the point estimates and comparison lists that
06_patient_cluster_bootstrap.py re-evaluates, and the conventional tests that
the bootstrap supersedes. The manuscript reports the patient-clustered bootstrap
intervals and p values; the exceptions taken from this step are the subgroup
and follow-up group sizes. Supporting Information Tables S4, S6 and S8 to S11
combine this step with the bootstrap.

  AUROC, paired (same episodes) ... DeLong test; 95% CI of the difference from a
                                    paired episode bootstrap (1000 resamples,
                                    seed 42)
  AUROC, sex and age groups ...... z test with DeLong variances (two groups),
                                    chi-squared heterogeneity test (three groups)
  Brier score and ICI differences . paired episode bootstrap
  Class weighting ................ unweighted XGBoost refitted with the tuned
                                    hyperparameters (Table S10)
  Follow-up (all 1300 episodes) .. chi-squared test, at least 180 days versus
                                    less (Table S9)

Inputs
------
  results_strict_benchmarks/test_predictions_all_models.csv   (03_benchmarks.py)
  results_strict/models/XGB_pipeline.joblib                   (02_train.py)
  data_strict/processed_episodes.csv                          (01_preprocess.py)

Outputs (results_strict_benchmarks/heldout/, aggregate tables only)
-------------------------------------------------------------------
  auroc_comparisons.csv, calibration_by_model.csv,
  accuracy_calibration_comparisons.csv, subgroup_heterogeneity.csv,
  fairness_test.csv, class_weighting.csv, followup_180_days.csv, summary.json

Usage
-----
  python 05_heldout_statistics.py
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_strict_benchmarks"
OUT = RES / "heldout"
OUT.mkdir(parents=True, exist_ok=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tr = _load("train02", "02_train.py")
ev = _load("evaluation_metrics", "evaluation_metrics.py")
tr.IN_FILE = ROOT / "data_strict" / "processed_episodes.csv"
SEED, N_BOOT = 42, 1000
MODELS =["XGB_micro", "XGB_nomicro", "LR_micro", "LR_nomicro", "VA_only"]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def boot_idx(n, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, size=n) for _ in range(n_boot)]


def delong_var(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    auc, v10, v01 = ev._delong_components(y, s)
    return auc, np.var(v10, ddof=1) / y.sum() + np.var(v01, ddof=1) / (len(y) - y.sum())


def paired_auc(y, a, b):
    y = np.asarray(y)
    auc_a, auc_b, diff, z, p = ev.delong_compare(y, np.asarray(a), np.asarray(b))
    ds = []
    for idx in boot_idx(len(y)):
        if y[idx].min() == y[idx].max():
            continue
        ds.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return dict(auc_a=auc_a, auc_b=auc_b, diff=diff, lo=np.percentile(ds, 2.5),
                hi=np.percentile(ds, 97.5), z=z, p=p)


def boot_auc_ci(y, s):
    y = np.asarray(y); vals = []
    for idx in boot_idx(len(y)):
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def boot_diff_test(y, a, b, fn):
    """Paired bootstrap for fn(a) - fn(b): point, 95% CI, two-sided p from the bootstrap SE."""
    y = np.asarray(y)
    point = fn(y, a) - fn(y, b)
    ds = []
    for idx in boot_idx(len(y)):
        if y[idx].min() == y[idx].max():
            continue
        ds.append(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
    se = np.std(ds, ddof=1)
    z = point / se if se > 0 else np.nan
    return dict(diff=point, lo=np.percentile(ds, 2.5), hi=np.percentile(ds, 97.5), z=z,
                p=2 * stats.norm.sf(abs(z)))


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────────────────────────

def load_test():
    te = pd.read_csv(RES / "test_predictions_all_models.csv")
    Xtr, Xte, ytr, yte, dk, gtr, tri, tei = tr.load_and_split(tr.NUMERIC_FEATURES, tr.CATEGORICAL_FEATURES, 61)
    full = dk.iloc[tei].reset_index(drop=True)
    assert np.array_equal(full["poor_outcome"].astype(int).values, te["y_true"].values)
    for c in ["gender_female", "age_years"]:
        te[c] = full[c].values
    return te, (Xtr, Xte, ytr.astype(int), yte.astype(int))


def main():
    te, (Xtr, Xte, ytr, yte) = load_test()
    y = te["y_true"].values
    P = {m: te[f"prob_{m}"].values for m in MODELS}
    summary = {}

    # 1. pairwise AUROC comparisons ------------------------------------------------
    pairs = [(m, "VA_only") for m in MODELS if m != "VA_only"] + [
        ("XGB_micro", "XGB_nomicro"), ("LR_micro", "LR_nomicro"),
        ("XGB_micro", "LR_micro"), ("XGB_nomicro", "LR_nomicro")]
    rows = []
    for a, b in pairs:
        r = paired_auc(y, P[a], P[b]); r.update(model_a=a, model_b=b); rows.append(r)
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "auroc_comparisons.csv", index=False)

    # 2. calibration and overall accuracy per model --------------------------------
    rows = []
    for m in MODELS:
        ici = ev.compute_ici(y, P[m]); _, ici_lo, ici_hi = ev.bootstrap_ici(y, P[m], rng=np.random.default_rng(SEED))
        brier = brier_score_loss(y, P[m])
        bl = [brier_score_loss(y[i], P[m][i]) for i in boot_idx(len(y))]
        lo, hi = boot_auc_ci(y, P[m])
        rows.append(dict(model=m, auroc=roc_auc_score(y, P[m]), auroc_lo=lo, auroc_hi=hi, ici=ici,
                         ici_lo=ici_lo, ici_hi=ici_hi, brier=brier, brier_lo=np.percentile(bl, 2.5),
                         brier_hi=np.percentile(bl, 97.5)))
    cal = pd.DataFrame(rows)
    cal.to_csv(OUT / "calibration_by_model.csv", index=False)
    rows = []
    for a, b in [("XGB_micro", "LR_micro"), ("XGB_nomicro", "LR_nomicro"), ("XGB_micro", "VA_only"),
                 ("XGB_nomicro", "VA_only"), ("XGB_micro", "XGB_nomicro")]:
        for name, fn in [("Brier", brier_score_loss), ("ICI", ev.compute_ici)]:
            r = boot_diff_test(y, P[a], P[b], fn); r.update(model_a=a, model_b=b, metric=name); rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "accuracy_calibration_comparisons.csv", index=False)

    # 3. discrimination by sex and age (fairness; test set) -------------------------------
    # independent-group comparisons of discrimination
    def unpaired(mask_a, mask_b, mdl="XGB_micro"):
        a, va = delong_var(y[mask_a], P[mdl][mask_a]); b, vb = delong_var(y[mask_b], P[mdl][mask_b])
        z = (a - b) / np.sqrt(va + vb)
        return dict(auc_a=a, auc_b=b, diff=a - b, z=z, p=2 * stats.norm.sf(abs(z)))
    ind = []
    for label, ma, mb in [("Female vs male", te.gender_female.values == 1, te.gender_female.values == 0)]:
        for mdl in ["XGB_micro", "XGB_nomicro"]:
            r = unpaired(ma, mb, mdl); r.update(comparison=label, model=mdl); ind.append(r)
    # age heterogeneity (three groups)
    age = te.age_years.values
    groups = [age < 65, (age >= 65) & (age < 80), age >= 80]
    for mdl in ["XGB_micro", "XGB_nomicro"]:
        est = [delong_var(y[g], P[mdl][g]) for g in groups]
        w = np.array([1 / v for _, v in est]); a = np.array([x for x, _ in est])
        abar = np.sum(w * a) / w.sum(); Q = np.sum(w * (a - abar) ** 2)
        ind.append(dict(comparison="Age <65 vs 65-79 vs >=80 (heterogeneity)", model=mdl, auc_a=a[0], auc_b=a[2],
                        diff=np.nan, z=Q, p=stats.chi2.sf(Q, df=2)))
    pd.DataFrame(ind).to_csv(OUT / "subgroup_heterogeneity.csv", index=False)
    fair = []
    for label, g in [("Female", te.gender_female.values == 1), ("Male", te.gender_female.values == 0),
                     ("Age <65", groups[0]), ("Age 65 to 79", groups[1]), ("Age >=80", groups[2])]:
        lo, hi = boot_auc_ci(y[g], P["XGB_micro"][g])
        fair.append(dict(subgroup=label, n=int(g.sum()), n_poor=int(y[g].sum()),
                         auroc=roc_auc_score(y[g], P["XGB_micro"][g]), lo=lo, hi=hi))
    pd.DataFrame(fair).to_csv(OUT / "fairness_test.csv", index=False)

    # 4. class weighting (refit without weighting, tuned hyperparameters) ---------------
    num, cat = tr.NUMERIC_FEATURES, tr.CATEGORICAL_FEATURES
    spw = float((ytr == 0).sum()) / float((ytr == 1).sum())
    xgb_hp = {k: joblib.load(ROOT / "results_strict/models/XGB_pipeline.joblib").named_steps["clf"].get_params()[k]
              for k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                        "min_child_weight", "gamma", "reg_alpha", "reg_lambda"]}

    def refit(est):
        pipe = tr.Pipeline([("pre", tr.make_preprocessor(num, cat)), ("clf", est)])
        pipe.fit(Xtr, ytr)
        return pipe.predict_proba(Xte)[:, 1]
    arms = {
        "XGB_weighted": refit(xgb.XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1,
                                                scale_pos_weight=spw, **xgb_hp)),
        "XGB_unweighted": refit(xgb.XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1,
                                                  scale_pos_weight=1.0, **xgb_hp)),
    }
    assert np.allclose(arms["XGB_weighted"], P["XGB_micro"], atol=1e-6), "weighted XGB must reproduce canonical"
    rows = []
    for mdl in ["XGB"]:
        a, b = arms[f"{mdl}_unweighted"], arms[f"{mdl}_weighted"]
        r = paired_auc(y, a, b)
        rows.append(dict(model=mdl, metric="AUROC", weighted=r["auc_b"], unweighted=r["auc_a"], diff=r["diff"],
                         lo=r["lo"], hi=r["hi"], p=r["p"]))
        for name, fn in [("Brier", brier_score_loss), ("ICI", ev.compute_ici)]:
            r = boot_diff_test(y, a, b, fn)
            rows.append(dict(model=mdl, metric=name, weighted=fn(y, b), unweighted=fn(y, a), diff=r["diff"],
                             lo=r["lo"], hi=r["hi"], p=r["p"]))
    pd.DataFrame(rows).to_csv(OUT / "class_weighting.csv", index=False)

    # 5. follow-up (all 1300 episodes) ------------------------------------------------------
    df = pd.read_csv(tr.IN_FILE); d = df.dropna(subset=["poor_outcome"]).copy()
    fu = d.days_to_final_visit
    long_ = fu >= 180; short = (fu < 180) & fu.notna()
    rows = [dict(group=g, n=int(m.sum()), n_poor=int(d.poor_outcome[m].sum()),
                 rate=float(d.poor_outcome[m].mean())) for g, m in [("<180", short), (">=180", long_)]]
    ct = [[int(d.poor_outcome[long_].sum()), int((1 - d.poor_outcome[long_]).sum())],
          [int(d.poor_outcome[short].sum()), int((1 - d.poor_outcome[short]).sum())]]
    chi_p = stats.chi2_contingency(ct)[1]
    fu_tab = pd.DataFrame(rows); fu_tab["ge180_vs_lt180_p"] = chi_p
    fu_tab.to_csv(OUT / "followup_180_days.csv", index=False)
    summary["followup"] = dict(ge180_rate=float(d.poor_outcome[long_].mean()),
                               lt180_rate=float(d.poor_outcome[short].mean()), chi_p=chi_p,
                               n_missing_fu=int(fu.isna().sum()))

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 220)
    print(comp.round(4).to_string(index=False)); print()
    print(cal[["model", "auroc", "ici", "brier"]].round(3).to_string(index=False)); print()
    print(pd.read_csv(OUT / "accuracy_calibration_comparisons.csv").round(4).to_string(index=False)); print()
    print(pd.read_csv(OUT / "fairness_test.csv").round(3).to_string(index=False)); print()
    print(pd.read_csv(OUT / "subgroup_heterogeneity.csv").round(4).to_string(index=False)); print()
    print(pd.read_csv(OUT / "class_weighting.csv").round(4).to_string(index=False)); print()
    print(fu_tab.round(3).to_string(index=False)); print(summary["followup"])


if __name__ == "__main__":
    main()
