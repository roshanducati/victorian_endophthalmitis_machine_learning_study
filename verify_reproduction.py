"""
verify_reproduction.py
======================
Checks a reproduction against the published results. Run after
run_pipeline.sh (or run_pipeline.ps1) with the registry data in place.

Each check reads an output of the pipeline, formats it at the precision printed
in the manuscript or its Supporting Information, and compares it with the
published value. The script prints every check and exits with status 1 if any
check fails, or if an output file is missing.

The expected values cover the cohort and split, Table 1, Table 2, the numbers
quoted in the Results and Discussion, and key values from Supporting Information
Tables S2 to S11. With synthetic data (tests/synthetic_registry.py) the checks fail by
design; use the synthetic data only to confirm that the pipeline runs.

Usage
-----
  python verify_reproduction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BM = ROOT / "results_strict_benchmarks"
PC = BM / "patient_cluster_bootstrap"
HO = BM / "heldout"
RESULTS: list[tuple[bool, str, str, str]] = []


# ── formatting, as printed in the manuscript ────────────────────────────────
def f(x, d=3):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def pv(p):
    return "<0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")


def check(name, actual, expected):
    actual = str(actual).replace("−", "-")
    expected = str(expected).replace("−", "-")
    RESULTS.append((actual == expected, name, actual, expected))


def row(df, label, metric=None):
    s = df[df.label == label]
    if metric is not None:
        s = s[s.metric == metric]
    if len(s) != 1:
        raise KeyError(f"{label} / {metric}: {len(s)} rows")
    return s.iloc[0]


def ci(lo, hi, d=3, scale=1.0):
    return f"{f(scale * lo, d)} to {f(scale * hi, d)}"


def main():
    # ── cohort and split ─────────────────────────────────────────────────────
    proc = pd.read_csv(ROOT / "data_strict/processed_episodes.csv", dtype={"rveeh_ur": str})
    known = proc.dropna(subset=["poor_outcome"])
    check("registry episodes", len(proc), 1700)
    check("registry patients", proc.rveeh_ur.nunique(), 1640)
    check("analysed episodes", len(known), 1300)
    check("analysed patients", known.rveeh_ur.nunique(), 1262)
    check("poor outcomes", int(known.poor_outcome.sum()), 599)
    te = pd.read_csv(ROOT / "results_strict/test_predictions.csv", dtype={"rveeh_ur": str})
    check("test episodes", len(te), 262)
    check("test patients", te.rveeh_ur.nunique(), 253)
    check("test poor outcomes", int(te.y_true.sum()), 125)
    check("test episodes with presenting VA", int(te.pres_logmar.notna().sum()), 248)
    check("development episodes", len(known) - len(te), 1038)
    check("development patients", known.rveeh_ur.nunique() - te.rveeh_ur.nunique(), 1009)
    check("development poor outcomes", int(known.poor_outcome.sum() - te.y_true.sum()), 474)
    meta = json.loads((PC / "run_metadata.json").read_text())
    check("test patients with repeat episodes", meta["repeat_test_patients"], 7)
    check("bootstrap resamples", meta["n_boot"], 10000)

    # ── Table 1 ──────────────────────────────────────────────────────────────
    d = known
    check("Table 1 age median (IQR)",
          f"{d.age_years.median():.1f} ({d.age_years.quantile(.25):.1f}-{d.age_years.quantile(.75):.1f})",
          "74.9 (63.0-82.4)")
    check("Table 1 female n/N", f"{int((d.gender_female == 1).sum())}/{int(d.gender_female.notna().sum())}", "631/1296")
    check("Table 1 fundus not visible n/N",
          f"{int((d.fundus_visible == 0).sum())}/{int(d.fundus_visible.notna().sum())}", "837/1195")
    check("Table 1 RAPD n/N", f"{int((d.rapd_present == 1).sum())}/{int(d.rapd_present.notna().sum())}", "63/515")
    check("Table 1 diabetes n/N", f"{int((d.diabetes == 1).sum())}/{int(d.diabetes.notna().sum())}", "265/1076")
    check("Table 1 immune suppression n/N",
          f"{int((d.immune_suppressed == 1).sum())}/{int(d.immune_suppressed.notna().sum())}", "24/96")
    check("Table 1 culture positive", int((d.culture_positive == 1).sum()), 829)
    check("Table 1 HM or worse %", pct((d.pres_logmar >= 2.29).sum() / d.pres_logmar.notna().sum()), "68.9")
    check("Table 1 follow-up median (IQR)",
          f"{d.days_to_final_visit.median():.0f} ({d.days_to_final_visit.quantile(.25):.0f}-"
          f"{d.days_to_final_visit.quantile(.75):.0f})", "52 (18-151)")
    for cat, n in [("cons", 254), ("streptococcus", 179), ("staph_aureus", 91), ("enterococcus", 31),
                   ("gram_negative", 136), ("bacillus", 13), ("propionibacterium", 14), ("fungal", 64),
                   ("other_bacteria", 26), ("no_growth", 460), ("unknown", 32)]:
        check(f"Table 1 organism {cat}", int((d.organism_cat == cat).sum()), n)
    unk = d.organism_cat == "unknown"
    check("Table 1 footnote: culture positive, organism not identified", int((unk & (d.culture_positive == 1)).sum()), 21)
    check("Table 1 footnote: culture result not recorded", int((unk & (d.culture_positive == 0)).sum()), 11)
    for grp, n in [("biopsy", 770), ("vitrectomy", 283)]:
        check(f"Table 1 primary procedure {grp}", int((d.procedure_group == grp).sum()), n)

    # ── Table 2, Table S4 and the Results text ──────────────────────────────
    mp = pd.read_csv(PC / "model_performance.csv")
    exp_perf = {  # model: AUROC, AUROC CI, ICI, ICI CI
        "VA_only": ("0.780", "0.726 to 0.831", "0.104", "0.072 to 0.163"),
        "XGB_nomicro": ("0.817", "0.765 to 0.866", "0.053", "0.027 to 0.098"),
        "LR_nomicro": ("0.791", "0.736 to 0.844", "0.041", "0.019 to 0.091"),
        "XGB_micro": ("0.863", "0.817 to 0.905", "0.076", "0.047 to 0.113"),
        "LR_micro": ("0.842", "0.793 to 0.888", "0.043", "0.020 to 0.087"),
    }
    for m, (a, aci, i, ici_) in exp_perf.items():
        r = row(mp, m, "AUROC")
        check(f"{m} AUROC", f(r.point), a)
        check(f"{m} AUROC 95% CI", ci(r.patient_percentile_lo, r.patient_percentile_hi), aci)
        r = row(mp, m, "ICI")
        check(f"{m} ICI", f(r.point), i)
        check(f"{m} ICI 95% CI", ci(r.patient_percentile_lo, r.patient_percentile_hi), ici_)
    for m, brier in [("VA_only", "0.197"), ("XGB_nomicro", "0.175"), ("XGB_micro", "0.159"),
                     ("LR_nomicro", "0.188"), ("LR_micro", "0.165")]:
        check(f"{m} Brier score", f(row(mp, m, "Brier").point), brier)

    auc = pd.read_csv(PC / "AUROC_comparison.csv")
    exp_auc = {
        "XGB_nomicro - VA_only": ("0.037", "0.001 to 0.073", "0.043"),
        "XGB_micro - VA_only": ("0.082", "0.039 to 0.126", "<0.001"),
        "XGB_micro - XGB_nomicro": ("0.045", "0.015 to 0.076", "0.004"),
        "XGB_nomicro - LR_nomicro": ("0.026", "0.005 to 0.047", "0.014"),
        "XGB_micro - LR_micro": ("0.020", "0.005 to 0.036", "0.011"),
        "LR_micro - LR_nomicro": ("0.051", "0.017 to 0.085", "0.004"),
        "LR_nomicro - VA_only": ("0.011", "-0.031 to 0.053", "0.61"),
        "LR_micro - VA_only": ("0.062", "0.016 to 0.108", "0.008"),
    }
    for lab, (pt, c, p) in exp_auc.items():
        r = row(auc, lab)
        check(f"AUROC difference {lab}", f(r.point), pt)
        check(f"AUROC difference {lab} 95% CI", ci(r.patient_lo, r.patient_hi), c)
        check(f"AUROC difference {lab} p", pv(r.patient_p), p)
    prob = pd.read_csv(PC / "probability_comparison.csv")
    exp_ici = {
        "XGB_nomicro - VA_only": ("-0.051", "-0.089 to -0.014", "0.008"),
        "XGB_micro - VA_only": ("-0.028", "-0.071 to 0.015", "0.20"),
        "XGB_micro - XGB_nomicro": ("0.023", "-0.001 to 0.047", "0.062"),
        "XGB_nomicro - LR_nomicro": ("0.012", "-0.006 to 0.031", "0.20"),
        "XGB_micro - LR_micro": ("0.033", "0.017 to 0.049", "<0.001"),
    }
    for lab, (pt, c, p) in exp_ici.items():
        r = row(prob, lab, "ICI difference")
        check(f"ICI difference {lab}", f(r.point), pt)
        check(f"ICI difference {lab} 95% CI", ci(r.patient_lo, r.patient_hi), c)
        check(f"ICI difference {lab} p", pv(r.patient_p), p)

    # ── operating points at the development Youden thresholds (Results 3.2; Table S5) ──
    bench = pd.read_csv(BM / "A_test_benchmarks.csv").set_index("model")
    for m, thr in [("VA_only", "0.67"), ("XGB_nomicro", "0.49"), ("XGB_micro", "0.46"),
                   ("LR_nomicro", "0.55"), ("LR_micro", "0.59")]:
        check(f"{m} development Youden threshold", f(bench.loc[m, "threshold"], 2), thr)
    for m, s, sp, npv in [("XGB_micro", "81.6", "72.3", "81.1"), ("XGB_nomicro", "71.2", "72.3", "73.3"),
                          ("VA_only", "50.4", "92.7", "67.2")]:
        check(f"{m} sensitivity at Youden threshold", pct(row(mp, m, "sensitivity").point), s)
        check(f"{m} specificity at Youden threshold", pct(row(mp, m, "specificity").point), sp)
        check(f"{m} negative predictive value at Youden threshold", pct(row(mp, m, "npv").point), npv)

    # ── threshold table (Table S7; the 0.30 row) ─────────────────────────────
    s7 = pd.read_csv(BM / "tables" / "tableS7_thresholds_xgb_with_cultures.csv").set_index("Threshold")
    r30 = s7.loc[0.30]
    check("Table S7 threshold 0.30: negative predictive value", pct(r30.NPV), "92.9")
    check("Table S7 threshold 0.30: test episodes below the threshold, %",
          f"{100 * (r30.TN + r30.FN) / (r30.TP + r30.FP + r30.TN + r30.FN):.0f}", "21")

    # ── sex and age (Results 3.4; Table S11) ─────────────────────────────────
    fair = pd.read_csv(HO / "fairness_test.csv").set_index("subgroup")
    sp = pd.read_csv(PC / "subgroup_performance.csv")
    for grp, n, k, val, c in [("Female", 130, 56, "0.826", "0.749 to 0.897"), ("Male", 131, 68, "0.887", "0.827 to 0.938"),
                              ("Age <65", 64, 26, "0.849", "0.731 to 0.941"),
                              ("Age 65 to 79", 96, 37, "0.848", "0.766 to 0.917"),
                              ("Age >=80", 101, 61, "0.872", "0.798 to 0.935")]:
        check(f"{grp} episodes, poor outcomes", f"{int(fair.loc[grp, 'n'])}, {int(fair.loc[grp, 'n_poor'])}", f"{n}, {k}")
        r = row(sp, f"{grp}: XGB_micro")
        check(f"{grp} AUROC, with cultures", f(r.point), val)
        check(f"{grp} AUROC 95% CI, with cultures", ci(r.patient_percentile_lo, r.patient_percentile_hi), c)
    het = pd.read_csv(PC / "subgroup_heterogeneity.csv")
    age = pd.read_csv(PC / "age_joint_tests.csv").set_index("model")
    for m, p_sex, p_age in [("XGB_micro", "0.20", "0.88"), ("XGB_nomicro", "0.59", "0.98")]:
        check(f"female versus male p, {m}", pv(row(het, f"Female vs male: {m}").patient_p), p_sex)
        check(f"age groups p, {m}", pv(age.loc[m, "patient_p"]), p_age)

    # ── missing predictors (Table S8) ───────────────────────────────────────
    miss = pd.read_csv(PC / "missingness_performance.csv")
    mc = pd.read_csv(PC / "missingness_comparison.csv")
    check("AUROC without immune suppression and RAPD (XGBoost)", f(row(miss, "XGB: drop_immune_rapd").point), "0.870")
    r = row(mc, "XGB: full - drop_immune_rapd")
    check("change in AUROC without both predictors (XGBoost)", f(-r.point), "0.008")
    check("p, without both predictors (XGBoost)", pv(r.patient_p), "0.27")

    # ── follow-up (limitations; Table S9) ───────────────────────────────────
    fg = pd.read_csv(HO / "followup_180_days.csv").set_index("group")
    fr = pd.read_csv(PC / "followup_rate.csv")
    for grp, n, k, rate in [("<180", 948, 430, "45.4"), (">=180", 262, 115, "43.9")]:
        check(f"follow-up {grp} days: episodes, poor outcomes",
              f"{int(fg.loc[grp, 'n'])}, {int(fg.loc[grp, 'n_poor'])}", f"{n}, {k}")
        check(f"poor outcome, follow-up {grp} days, %", pct(row(fr, grp).point), rate)
    fc = pd.read_csv(PC / "followup_comparison.csv").iloc[0]
    check("difference, at least 180 days minus less, percentage points (95% CI)",
          f"{f(100 * fc.point, 1)} ({ci(fc.patient_lo, fc.patient_hi, 1, 100)})", "-1.5 (-8.3 to 5.3)")
    check("p, at least 180 days versus less", pv(fc.patient_p), "0.67")
    check("follow-up unrecorded, episodes", int(known.days_to_final_visit.isna().sum()), 90)

    # ── class weighting (Table S10) and tuning (Table S2) ──────────────────
    cw = pd.read_csv(PC / "class_weighting.csv")
    for lab, metric, p in [("XGB: unweighted - weighted", "AUROC difference", "0.53")]:
        check(f"class weighting {lab} {metric} p", pv(row(cw, lab, metric).patient_p), p)
    cv_m = pd.read_csv(ROOT / "results_strict/cv_results.csv").set_index("model")
    cv_n = pd.read_csv(ROOT / "results_strict_nomicro/cv_results.csv").set_index("model")
    s03a = json.loads((BM / "summary.json").read_text())["A"]
    check("tuning CV AUROC, XGBoost", f"{f(cv_m.loc['XGB', 'best_cv_auroc_mean'])} / {f(cv_n.loc['XGB', 'best_cv_auroc_mean'])}",
          "0.800 / 0.776")
    check("tuning CV AUROC, logistic regression", f"{f(s03a['lr_micro_cv_auroc'])} / {f(s03a['lr_nomicro_cv_auroc'])}",
          "0.797 / 0.774")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError) as exc:
        print(f"Missing or incomplete output: {exc}")
        sys.exit(1)
    n_fail = 0
    for ok, name, actual, expected in RESULTS:
        n_fail += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {actual}" + ("" if ok else f"  (published: {expected})"))
    print(f"\n{len(RESULTS) - n_fail} of {len(RESULTS)} checks passed.")
    sys.exit(1 if n_fail else 0)
