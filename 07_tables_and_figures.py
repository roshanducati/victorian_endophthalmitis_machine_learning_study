"""
07_tables_and_figures.py
========================
Step 7 of the analysis pipeline. Main tables, Figure 3 and Figure S1.

Manuscript
----------
  Table 1     results_strict_benchmarks/tables/table1_characteristics.csv
  Table 2     results_strict_benchmarks/tables/table2_discrimination_calibration.csv
              (formatted from the patient-clustered bootstrap of step 6)
  Table S7    results_strict_benchmarks/tables/tableS7_thresholds_xgb_with_cultures.csv
  Figure 3    results_strict_benchmarks/figures/fig3_shap.{tif,pdf,png}
              (exact TreeSHAP values from XGBoost, pred_contribs, on the log-odds
              scale; one-hot organism categories summed within each episode in
              panel A)
  Figure S1   results_strict_benchmarks/figures/figS1_decision_curves.{tif,pdf,png}
              (net benefit across threshold probabilities; descriptive)
  Also written: tables/shap_importance_aggregated.csv, tables/decision_curve_test.csv

Figures 1 and 2 are drawn by 08_figures_test_set.py, which imports the styles
and the save() helper defined here.

Artwork: TIFF (RGB, LZW, 1200 dpi), vector PDF and a 300-dpi PNG preview, with
8-point type (journal_style.py), at most 7.0 in (178 mm) wide.

Inputs
------
  data_strict/processed_episodes.csv                              (01)
  results_strict[_nomicro]/test_predictions.csv, test_metrics.csv,
  feature_columns.txt, models/XGB_pipeline.joblib                 (02)
  results_strict_benchmarks/test_predictions_all_models.csv       (03)
  results_strict_benchmarks/patient_cluster_bootstrap/*.csv       (06)

Usage
-----
  python 07_tables_and_figures.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu

import journal_style as js

js.apply()
ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_strict_benchmarks"
FIG = RES / "figures"
TAB = RES / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
DPI_LINEART = 1200

# Paul Tol colour-blind-safe palette
C_XGB = "#228833"
C_LR = "#4477AA"
C_VA = "#777777"

LP, HM = 2.69, 2.29


def save(fig, stem, w, h):
    """TIFF (RGB, LZW, 1200 dpi) + vector PDF + 300-dpi PNG preview.

    matplotlib writes RGBA TIFFs; alpha in print artwork is flattened onto
    white here so that the TIFF is plain RGB.
    """
    from PIL import Image
    fig.set_size_inches(w, h)
    js.enforce_fontsizes(fig)
    tif = FIG / f"{stem}.tif"
    fig.savefig(tif, dpi=DPI_LINEART, format="tiff", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    im = Image.open(tif)
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
        im = bg
    im.save(tif, compression="tiff_lzw", dpi=(DPI_LINEART, DPI_LINEART))
    fig.savefig(FIG / f"{stem}.pdf", format="pdf", facecolor="white")
    fig.savefig(FIG / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("saved", stem)


# ─────────────────────────────────────────────────────────────────────────────
# Table 1
# ─────────────────────────────────────────────────────────────────────────────
ORG_LABELS = [
    ("cons", "Coagulase-negative staphylococci"),
    ("streptococcus", "Streptococcus species"),
    ("staph_aureus", "Staphylococcus aureus"),
    ("enterococcus", "Enterococcus species"),
    ("gram_negative", "Gram-negative bacteria"),
    ("bacillus", "Bacillus species"),
    ("propionibacterium", "Cutibacterium (Propionibacterium) species"),
    ("fungal", "Fungi"),
    ("other_bacteria", "Other or atypical organism"),
    ("no_growth", "No growth"),
    ("unknown", "Organism not identified or culture result not recorded"),
]


def table1():
    df = pd.read_csv(ROOT / "data_strict/processed_episodes.csv")
    d = df.dropna(subset=["poor_outcome"]).copy()
    g0, g1 = d[d.poor_outcome == 0], d[d.poor_outcome == 1]
    rows = []

    def cont(label, col, dec=1):
        cells = []
        for g in (d, g0, g1):
            v = g[col].dropna()
            cells.append(f"{v.median():.{dec}f} ({v.quantile(.25):.{dec}f}–{v.quantile(.75):.{dec}f})")
        p = mannwhitneyu(g0[col].dropna(), g1[col].dropna()).pvalue
        rows.append([label, *cells, p, f"{d[col].notna().sum()}"])

    def binary(label, col, val=1.0):
        cells = []
        for g in (d, g0, g1):
            tot = g[col].notna().sum(); n = int((g[col] == val).sum())
            cells.append(f"{n}/{tot} ({100 * n / tot:.1f})")
        ct = np.array([[int((g0[col] == val).sum()), int((g0[col] != val).sum() - g0[col].isna().sum())],
                       [int((g1[col] == val).sum()), int((g1[col] != val).sum() - g1[col].isna().sum())]])
        p = chi2_contingency(ct, correction=True)[1]
        rows.append([label, *cells, p, f"{d[col].notna().sum()}"])

    def categorical(label, series_fn, cats):
        s_all = series_fn(d); s0 = series_fn(g0); s1 = series_fn(g1)
        ct = np.array([[int((s0 == c).sum()) for c, _ in cats], [int((s1 == c).sum()) for c, _ in cats]])
        keep = ct.sum(0) > 0
        p = chi2_contingency(ct[:, keep])[1]
        rows.append([label, "", "", "", p, f"{s_all.notna().sum()}"])
        for c, lab in cats:
            cells = []
            for s in (s_all, s0, s1):
                tot = s.notna().sum(); n = int((s == c).sum())
                cells.append(f"{n}/{tot} ({100 * n / tot:.1f})")
            rows.append(["    " + lab, *cells, np.nan, ""])

    rows.append(["Episodes, n", f"{len(d)}", f"{len(g0)}", f"{len(g1)}", np.nan, ""])
    cont("Age, years, median (IQR)", "age_years")
    binary("Female sex", "gender_female")
    cont("Presenting VA, logMAR, median (IQR)", "pres_logmar")

    def va_cat(g):
        v = g["pres_logmar"]
        out = pd.Series(np.where(v >= LP, "lp", np.where(v >= HM, "hm", "better")), index=g.index)
        return out.where(v.notna())
    categorical("Presenting VA category", va_cat,
                [("better", "Better than hand movements"), ("hm", "Hand movements"),
                 ("lp", "Light perception or no light perception")])
    rows.append(["Fundus not visible", *[f"{int((g.fundus_visible == 0).sum())}/{g.fundus_visible.notna().sum()} "
                                         f"({100 * (g.fundus_visible == 0).sum() / g.fundus_visible.notna().sum():.1f})"
                                         for g in (d, g0, g1)],
                 chi2_contingency(np.array([[int((g0.fundus_visible == 0).sum()), int((g0.fundus_visible == 1).sum())],
                                            [int((g1.fundus_visible == 0).sum()), int((g1.fundus_visible == 1).sum())]]))[1],
                 f"{d.fundus_visible.notna().sum()}"])
    binary("RAPD present", "rapd_present")
    binary("Diabetes mellitus", "diabetes")
    binary("Immune suppression", "immune_suppressed")
    rows.append(["Precipitating event*", "", "", "", np.nan, ""])
    for col, lab in [("etiol_cataract_sx", "Cataract or IOL surgery"), ("etiol_ivi", "Intravitreal injection"),
                     ("etiol_glaucoma_sx", "Glaucoma or drainage surgery"), ("etiol_endogenous", "Endogenous"),
                     ("etiol_corneal_ulcer", "Corneal ulcer"), ("etiol_trauma", "Penetrating eye injury"),
                     ("etiol_other", "Other ocular procedure")]:
        binary("    " + lab, col)
    binary("Culture positive", "culture_positive")
    categorical("Organism category", lambda g: g["organism_cat"], ORG_LABELS)

    def proc(g):
        return g["procedure_group"].replace({"biopsy": "tap", "vitrectomy": "ppv", "unknown": "unk"})
    categorical("Primary procedure (not a predictor)", proc,
                [("tap", "Vitreous tap and intravitreal antibiotics"), ("ppv", "Primary pars plana vitrectomy"),
                 ("unk", "Not recorded")])
    cont("Follow-up to final recorded VA, days, median (IQR)", "days_to_final_visit", dec=0)

    t = pd.DataFrame(rows, columns=["Characteristic", "All episodes", "Good outcome",
                                    "Poor outcome", "p", "n_recorded"])
    t.to_csv(TAB / "table1_characteristics.csv", index=False)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Model styles (shared with 08_figures_test_set.py)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_STYLE = [
    ("XGB_micro", "XGBoost, with cultures", dict(color=C_XGB, ls="-", lw=1.5)),
    ("XGB_nomicro", "XGBoost, pre-culture", dict(color=C_XGB, ls=(0, (4, 2)), lw=1.3)),
    ("LR_micro", "Logistic regression, with cultures", dict(color=C_LR, ls="-", lw=1.0)),
    ("VA_only", "Presenting VA alone", dict(color=C_VA, ls=(0, (1, 1.5)), lw=1.3)),
]
SHORT = {"XGB_micro": "XGBoost + cultures", "XGB_nomicro": "XGBoost pre-culture",
         "LR_micro": "Logistic + cultures", "VA_only": "Presenting VA"}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: predictor contributions (exact TreeSHAP from XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
PRETTY = {
    "age_years": "Age", "pres_logmar": "Presenting VA (logMAR)", "gender_female": "Female sex",
    "fundus_visible": "Fundus visible", "rapd_present": "RAPD present", "diabetes": "Diabetes",
    "immune_suppressed": "Immune suppression", "etiol_cataract_sx": "After cataract surgery",
    "etiol_glaucoma_sx": "After glaucoma surgery", "etiol_trauma": "Penetrating injury",
    "etiol_corneal_ulcer": "Corneal ulcer", "etiol_endogenous": "Endogenous",
    "etiol_ivi": "After intravitreal injection", "etiol_other": "After other procedure",
    "culture_positive": "Culture positive", "organism_cat": "Organism category",
}
ORG_PRETTY = {"cons": "CoNS", "streptococcus": "Streptococcus", "staph_aureus": "S. aureus",
              "enterococcus": "Enterococcus", "gram_negative": "Gram-negative", "bacillus": "Bacillus",
              "propionibacterium": "Cutibacterium", "fungal": "Fungus", "other_bacteria": "Other organism",
              "no_growth": "No growth", "unknown": "Organism unknown"}


def tree_shap(pipe_path, X):
    pipe = joblib.load(pipe_path)
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = pre.transform(X)
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    contrib = clf.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
    return contrib[:, :-1], Xt, names


def aggregate(sv, names):
    base = ["organism_cat" if n.startswith("organism_cat") else n for n in names]
    df = pd.DataFrame(sv, columns=names)
    agg = df.T.groupby(base).sum().T          # per-episode sum within each clinical predictor
    return agg.abs().mean().sort_values(ascending=False)


def fig3_shap():
    te = pd.read_csv(ROOT / "results_strict/test_predictions.csv")
    feat_m = (ROOT / "results_strict/feature_columns.txt").read_text().split("\n")
    feat_n = (ROOT / "results_strict_nomicro/feature_columns.txt").read_text().split("\n")
    sv_m, Xt_m, nm_m = tree_shap(ROOT / "results_strict/models/XGB_pipeline.joblib", te[feat_m])
    sv_n, Xt_n, nm_n = tree_shap(ROOT / "results_strict_nomicro/models/XGB_pipeline.joblib", te[feat_n])
    imp_m = aggregate(sv_m, nm_m); imp_n = aggregate(sv_n, nm_n)
    imp = pd.DataFrame({"with_cultures": imp_m, "pre_culture": imp_n}).fillna(0)
    imp = imp.sort_values("with_cultures", ascending=True)
    imp.to_csv(TAB / "shap_importance_aggregated.csv")

    fig = plt.figure(figsize=(7.0, 3.9))
    ax1 = fig.add_axes([0.205, 0.12, 0.225, 0.8])
    ax2 = fig.add_axes([0.655, 0.12, 0.235, 0.8])
    yy = np.arange(len(imp)); h = 0.38
    ax1.barh(yy + h / 2, imp["pre_culture"], height=h, color="white", edgecolor=C_XGB, lw=0.8, label="Pre-culture model")
    ax1.barh(yy - h / 2, imp["with_cultures"], height=h, color=C_XGB, edgecolor=C_XGB, lw=0.8, label="With-culture model")
    ax1.set_yticks(yy); ax1.set_yticklabels([PRETTY.get(i, i) for i in imp.index])
    ax1.set_ylim(-0.6, len(imp) - 0.4)
    ax1.set_xlabel("Mean |SHAP value| (log-odds)")
    ax1.legend(loc="lower right", frameon=True)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("Contribution by predictor", loc="left")

    # Beeswarm, with-culture model, top 12 encoded features
    order = np.argsort(np.abs(sv_m).mean(0))[::-1][:12]
    rng = np.random.default_rng(42)
    cmap = plt.get_cmap("coolwarm")
    for rank, j in enumerate(order[::-1]):
        vals = sv_m[:, j]
        x = Xt_m[:, j]
        col = stats.rankdata(x) / len(x)
        if len(np.unique(x)) <= 2:
            col = (x > x.min()).astype(float) * 0.9 + 0.05
        # density-scaled jitter for a swarm-like layout
        hist, edges = np.histogram(vals, bins=40)
        dens = hist[np.clip(np.digitize(vals, edges) - 1, 0, len(hist) - 1)] / hist.max()
        jit = (rng.random(len(vals)) - 0.5) * 0.75 * np.sqrt(dens)
        ax2.scatter(vals, rank + jit, c=cmap(col), s=3.0, lw=0, rasterized=True)
    labels = []
    for j in order[::-1]:
        n = nm_m[j]
        if n.startswith("organism_cat_"):
            labels.append("Organism: " + ORG_PRETTY.get(n.replace("organism_cat_", ""), n))
        else:
            labels.append(PRETTY.get(n, n))
    ax2.set_yticks(range(len(order))); ax2.set_yticklabels(labels)
    ax2.axvline(0, color="#999999", lw=0.6)
    ax2.set_xlabel("SHAP value (log-odds of poor outcome)")
    ax2.set_title("Direction of effect, with-culture model", loc="left")
    ax2.spines[["top", "right"]].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); sm.set_array([])
    cax = fig.add_axes([0.905, 0.3, 0.012, 0.42])
    cb = fig.colorbar(sm, cax=cax, ticks=[0.05, 0.95])
    cb.ax.set_yticklabels(["Low /\nabsent", "High /\npresent"])
    cb.outline.set_linewidth(0.5)
    ax2.set_ylim(-0.6, len(order) - 0.4)
    fig.text(0.01, 0.955, "A", fontweight="bold"); fig.text(0.445, 0.955, "B", fontweight="bold")
    save(fig, "fig3_shap", 7.0, 3.9)
    return imp


# ─────────────────────────────────────────────────────────────────────────────
# Figure S1: decision curves
# ─────────────────────────────────────────────────────────────────────────────

def sfig_dca():
    te = pd.read_csv(RES / "test_predictions_all_models.csv")
    y = te["y_true"].values; n = len(y); prev = y.mean()
    pts = np.round(np.arange(0.05, 0.81, 0.01), 2)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    rows = []
    for key, label, style in MODEL_STYLE:
        p = te[f"prob_{key}"].values
        nb = [((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t) for t in pts]
        ax.plot(pts, nb, label=label, **style)
        rows += [dict(model=key, threshold=t, net_benefit=v) for t, v in zip(pts, nb)]
    nb_all = prev - (1 - prev) * pts / (1 - pts)
    ax.plot(pts, nb_all, color="#999999", lw=0.9, ls=(0, (3, 2)), label="Treat all as high risk")
    ax.axhline(0, color="black", lw=0.7, label="Treat none as high risk")
    ax.set_ylim(-0.05, 0.5); ax.set_xlim(0.05, 0.8)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=True)
    pd.DataFrame(rows).to_csv(TAB / "decision_curve_test.csv", index=False)
    fig.tight_layout()
    save(fig, "figS1_decision_curves", 4.6, 3.6)


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 (formatted from the patient-clustered bootstrap) and Table S7
# ─────────────────────────────────────────────────────────────────────────────

PC = RES / "patient_cluster_bootstrap"
NAMES = {"XGB_micro": "XGBoost, with cultures", "XGB_nomicro": "XGBoost, pre-culture",
         "LR_micro": "Logistic regression, with cultures", "LR_nomicro": "Logistic regression, pre-culture",
         "VA_only": "Presenting VA alone"}


def _num(x, d=3):
    return f"{x:.{d}f}".replace("-", "−")


def _rng(lo, hi, d=3):
    z = lambda v: 0.0 if abs(v) < 0.5 * 10 ** -d else v
    return f"{_num(z(lo), d)} to {_num(z(hi), d)}"


def _p(p):
    return "<0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")


def _row(df, label, metric=None):
    s = df[df.label == label]
    if metric is not None:
        s = s[s.metric == metric]
    assert len(s) == 1, (label, metric, len(s))
    return s.iloc[0]


def table2():
    """Table 2. AUROC and ICI with patient-bootstrap percentile 95% CIs; all
    differences with bootstrap-SE (Wald) 95% CIs and p values. Brier scores and
    AUPRC are reported in Table S4 (model_performance.csv)."""
    mp = pd.read_csv(PC / "model_performance.csv")
    auc = pd.read_csv(PC / "AUROC_comparison.csv")
    prob = pd.read_csv(PC / "probability_comparison.csv")

    def est(r, d=3):
        return f"{_num(r.point, d)} ({_rng(r.patient_percentile_lo, r.patient_percentile_hi, d)})"

    def wald(r, d=3):
        z = 0.0 if abs(r.point) < 0.5 * 10 ** -d else r.point
        return f"{_num(z, d)} ({_rng(r.patient_lo, r.patient_hi, d)})"

    rows = []
    for group, models in (("At presentation", ["VA_only", "XGB_nomicro", "LR_nomicro"]),
                          ("When culture results are available", ["XGB_micro", "LR_micro"])):
        for m in models:
            rows.append(dict(panel="A", group=group, row=NAMES[m],
                             auroc=est(_row(mp, m, "AUROC")), ici=est(_row(mp, m, "ICI"))))
    for a, b, label in [("XGB_nomicro", "VA_only", "Pre-culture XGBoost vs presenting VA alone"),
                        ("XGB_micro", "VA_only", "XGBoost with cultures vs presenting VA alone"),
                        ("XGB_micro", "XGB_nomicro", "XGBoost with cultures vs pre-culture XGBoost"),
                        ("XGB_nomicro", "LR_nomicro", "XGBoost vs logistic regression, pre-culture"),
                        ("XGB_micro", "LR_micro", "XGBoost vs logistic regression, with cultures")]:
        ra = _row(auc, f"{a} - {b}")
        rc = _row(prob, f"{a} - {b}", "ICI difference")
        rows.append(dict(panel="B", group="", row=label,
                         auroc_difference=wald(ra), auroc_p=_p(ra.patient_p),
                         ici_difference=wald(rc), ici_p=_p(rc.patient_p)))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "table2_discrimination_calibration.csv", index=False)
    return t


def table_s7():
    """Table S7. Operating characteristics of the with-culture XGBoost model across
    probability thresholds in the held-out test set (descriptive)."""
    spec = importlib.util.spec_from_file_location("evaluation_metrics", ROOT / "evaluation_metrics.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    pred = pd.read_csv(ROOT / "results_strict/test_predictions.csv")
    metrics = pd.read_csv(ROOT / "results_strict/test_metrics.csv")
    t = ev.make_table3(pred["y_true"].values, {"XGB": pred["prob_XGB"].values}, metrics, lead_name="XGB")
    t.to_csv(TAB / "tableS7_thresholds_xgb_with_cultures.csv", index=False)
    return t


def main():
    sys.stdout.reconfigure(encoding="utf-8")    # the tables contain en dashes and minus signs
    t1 = table1()
    print(t1.to_string(index=False))
    print(table2().to_string(index=False))
    print(table_s7().to_string(index=False))
    imp = fig3_shap()
    print(imp.round(3))
    sfig_dca()


if __name__ == "__main__":
    main()
