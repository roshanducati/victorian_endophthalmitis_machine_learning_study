"""
synthetic_registry.py
=====================
Writes a synthetic REDCap export with the structure that 01_preprocess.py
expects, so that the whole pipeline can be run end to end without registry
data. Every value is simulated: no record corresponds to a real patient, and the
results produced from it have no clinical meaning.

The simulated export mimics the features of the real one that the code depends
on: a registration arm (one row per patient, with sex) and a review arm (one or
more episodes per patient, some with repeat instances); Snellen acuities stored
as Excel dates (patterns A and B) or as text; non-Snellen categories; sparse
recording of RAPD and immune suppression; free-text organisms; and an outcome
related to the predictors.

Usage (in a separate copy of the repository that holds no registry data)
------------------------------------------------------------------------
  python tests/synthetic_registry.py
  bash run_pipeline.sh          # or run_pipeline.ps1

The pipeline then runs end to end on simulated data. verify_reproduction.py
fails by design, because the simulated results differ from the published ones.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Snellen denominators and how the registry encodes them.
_PATTERN_B = {5: 5, 6: 6, 9: 9, 12: 12}                    # 2026-MM-06
_PATTERN_A = {15: 2015, 19: 2019, 24: 2024, 30: 2030, 36: 2036,
              48: 2048, 60: 1960, 75: 1975, 96: 1996}        # YYYY-06-01
_TEXT = ["6/7.5", "6/120", "6/150", "6/192", "6/240"]

ORGANISMS = [  # (free text, relative frequency, effect on log-odds of a poor outcome)
    ("Staph epidermidis", 0.27, -1.0),
    ("coagulase negative staphylococcus", 0.05, -1.0),
    ("Streptococcus pneumoniae", 0.10, 1.3),
    ("Strep mitis", 0.08, 1.2),
    ("Staphylococcus aureus", 0.11, 0.0),
    ("Enterococcus faecalis", 0.04, 2.0),
    ("Pseudomonas aeruginosa", 0.10, 1.2),
    ("Klebsiella pneumoniae", 0.05, 1.2),
    ("Bacillus cereus", 0.02, 1.0),
    ("Propionibacterium acnes", 0.02, -0.3),
    ("Candida albicans", 0.07, 0.0),
    ("Corynebacterium species", 0.03, 0.0),
    ("mixed skin flora", 0.03, 0.0),
]


def encode_va(logmar: float, rng: np.random.Generator):
    """Return a registry-style acuity value (date-encoded, text or category)."""
    if logmar >= 3.9:
        return "EVIS"
    if logmar >= 2.95:
        return "NLP"
    if logmar >= 2.6:
        return "LP"
    if logmar >= 2.1:
        return "HM"
    if logmar >= 1.7:
        return "CF"
    denom = 6 * 10 ** logmar
    options = sorted(list(_PATTERN_B) + list(_PATTERN_A) + [7.5, 120, 150, 192, 240])
    d = min(options, key=lambda o: abs(o - denom))
    if d in _PATTERN_B:
        return datetime(2026, _PATTERN_B[d], 6)
    if d in _PATTERN_A:
        return datetime(_PATTERN_A[d], 6, 1)
    return f"6/{d:g}"


def simulate(n_patients: int = 1640, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    org_text = [o[0] for o in ORGANISMS]
    org_p = np.array([o[1] for o in ORGANISMS]); org_p /= org_p.sum()
    org_eff = {o[0]: o[2] for o in ORGANISMS}
    start = datetime(1997, 7, 1)
    for pid in range(n_patients):
        ur = f"SYN{100000 + pid}"
        female = rng.random() < 0.49
        rows.append({"rveeh_ur": ur, "redcap_event_name": "registration_arm_1",
                     "redcap_repeat_instance": np.nan, "gender": 2 if female else 1})
        n_episodes = rng.choice([1, 2, 3], p=[0.975, 0.02, 0.005])
        base_age = rng.normal(73, 13)
        for e in range(n_episodes):
            adm = start + timedelta(days=int(rng.integers(0, 10400)))
            age = float(np.clip(base_age + e, 18, 99))
            # aetiology (non-exclusive flags)
            u = rng.random()
            aet = dict(cat=u < 0.31, ivi=0.31 <= u < 0.59, glauc=0.59 <= u < 0.71, endo=0.71 <= u < 0.82,
                       ulcer=0.82 <= u < 0.90, trauma=0.90 <= u < 0.96, other=u >= 0.96)
            if rng.random() < 0.03:
                aet["other"] = True
            pres = float(np.clip(rng.normal(2.1, 0.55), 0.0, 3.0))
            fundus = rng.random() < (0.15 if pres > 2.2 else 0.5)
            diabetes = rng.random() < 0.25
            immune = rng.random() < 0.25
            rapd = rng.random() < (0.2 if pres > 2.5 else 0.08)
            culture_pos = rng.random() < 0.64
            org = rng.choice(org_text, p=org_p) if culture_pos else ""
            lp = (-3.2 + 1.3 * pres + 0.02 * (age - 73) + 0.8 * (not fundus) + 0.6 * rapd + 0.3 * diabetes
                  + 1.8 * aet["ulcer"] + 0.6 * aet["glauc"] - 0.6 * aet["cat"]
                  + (org_eff.get(org, 0.0) if culture_pos else -0.4) + rng.normal(0, 0.6))
            poor = rng.random() < 1 / (1 + np.exp(-lp))
            final = (float(np.clip(rng.normal(2.3, 0.5), 1.05, 3.0)) if poor
                     else float(np.clip(rng.normal(0.45, 0.3), -0.1, 1.0)))
            if poor and rng.random() < 0.12:
                final = 4.0
            laterality = rng.choice(["RE", "LE", "BE", None], p=[0.48, 0.48, 0.02, 0.02])
            other_eye = float(np.clip(rng.normal(0.2, 0.2), -0.1, 1.0))
            pres_re, pres_le = (pres, other_eye) if laterality in ("RE", None, "BE") else (other_eye, pres)
            fin_re, fin_le = (final, other_eye) if laterality in ("RE", None, "BE") else (other_eye, final)
            has_final = rng.random() > 0.235           # about 400 of 1700 without a final acuity
            fu_days = int(np.clip(rng.lognormal(3.9, 1.2), 1, 3000))
            vit = rng.random() < (0.3 if poor else 0.2)
            proc_known = rng.random() > 0.19
            ep = {
                "rveeh_ur": ur, "redcap_event_name": "review_arm_1", "redcap_repeat_instance": 1,
                "admission_date": adm, "admission_age": round(age, 1),
                "affected_eye": laterality,
                "re_va_on_presentation": encode_va(pres_re, rng) if rng.random() > 0.06 else np.nan,
                "le_va_on_presentation": encode_va(pres_le, rng),
                "re_va_final": encode_va(fin_re, rng) if has_final else np.nan,
                "le_va_final": encode_va(fin_le, rng) if has_final else np.nan,
                "fundus_visible_yn": (1 if fundus else 0) if rng.random() > 0.08 else np.nan,
                "rapd_yn": (1 if rapd else 0) if rng.random() < 0.40 else np.nan,
                "diabetes_yn": (1 if diabetes else 0) if rng.random() > 0.17 else np.nan,
                "immune_suppressed_yn": (1 if immune else 0) if rng.random() < 0.07 else np.nan,
                "surgery_yn": 1 if (aet["cat"] or aet["glauc"]) else np.nan,
                "precipitating_factor___1": int(aet["cat"]), "precipitating_factor___2": int(aet["glauc"]),
                "precipitating_factor___3": int(aet["trauma"]), "precipitating_factor___4": int(aet["ulcer"]),
                "precipitating_factor___5": int(aet["endo"]), "precipitating_factor___7": int(aet["ivi"]),
                "precipitating_factor___98": int(aet["other"]),
                "growths_yn": (1 if culture_pos else 0) if rng.random() > 0.01 else np.nan,
                "microorganism": org,
                "intervention___v": (int(vit) if proc_known else np.nan),
                "intervention___b": (int(not vit) if proc_known else np.nan),
                "final_visit_date": adm + timedelta(days=fu_days) if (has_final and rng.random() > 0.07) else np.nan,
            }
            rows.append(ep)
            if rng.random() < 0.05:                    # an updated repeat instance of the same episode
                rows.append(dict(ep, redcap_repeat_instance=2))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("data/raw/VictorianEndophthalm_DATA_2026-05-17_2131.xlsx"))
    ap.add_argument("--patients", type=int, default=1640)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing file. Never use this where the real registry export is stored.")
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} already exists and may be the real registry export; refusing to overwrite it. "
                 "Generate synthetic data in a separate copy of the repository.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = simulate(args.patients, args.seed)
    df.to_excel(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, {df.rveeh_ur.nunique()} synthetic patients")


if __name__ == "__main__":
    main()
