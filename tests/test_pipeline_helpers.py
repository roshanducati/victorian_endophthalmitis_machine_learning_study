"""Data-free tests of the pipeline's helper functions.

Run from the repository root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


preprocess = load_script("preprocess", "01_preprocess.py")
train = load_script("train", "02_train.py")
metrics = load_script("evaluation_metrics", "evaluation_metrics.py")
bootstrap = load_script("bootstrap", "06_patient_cluster_bootstrap.py")


def episode(**overrides) -> dict:
    row = {
        "rveeh_ur": "synthetic", "admission_date": "2026-01-01", "affected_eye": "RE",
        "re_va_on_presentation": "6/60", "le_va_on_presentation": "6/6",
        "re_va_final": "6/12", "le_va_final": "6/6", "admission_age": 70, "gender_female": 1.0,
        "fundus_visible_yn": 0, "rapd_yn": 1, "diabetes_yn": 0, "immune_suppressed_yn": np.nan,
        "surgery_yn": 1, "precipitating_factor___1": 1, "precipitating_factor___2": 0,
        "precipitating_factor___3": 0, "precipitating_factor___4": 0, "precipitating_factor___5": 0,
        "precipitating_factor___7": 0, "precipitating_factor___98": 0, "growths_yn": 0,
        "microorganism": "", "intervention___v": 0, "intervention___b": 1,
        "final_visit_date": "2026-02-01",
    }
    row.update(overrides)
    return row


class VisualAcuityTests(unittest.TestCase):
    def test_snellen_and_date_encodings(self):
        self.assertAlmostEqual(preprocess.decode_va("6/60"), 1.0)
        self.assertAlmostEqual(preprocess.decode_va(datetime(1960, 6, 1)), 1.0)            # pattern A
        self.assertAlmostEqual(preprocess.decode_va(datetime(2026, 9, 6)), np.log10(9 / 6))  # pattern B
        self.assertAlmostEqual(preprocess.decode_va("6/7.5"), np.log10(7.5 / 6))
        self.assertTrue(np.isnan(preprocess.decode_va(datetime(2026, 3, 15))))            # neither pattern

    def test_non_snellen_categories(self):
        expected = {"CF": 1.85, "HM": 2.30, "LP": 2.70, "NLP": 3.00, "EVIS": 4.00}
        for code, value in expected.items():
            self.assertEqual(preprocess.decode_va(code), value)

    def test_recorded_eye_is_used(self):
        row = {"affected_eye": "LE", "re_va_final": "6/6", "le_va_final": "6/60"}
        self.assertAlmostEqual(preprocess.extract_eye_va(row, preprocess.VA_FINAL_COLS), 1.0)

    def test_worse_eye_for_bilateral_or_unspecified(self):
        for eye in ("BE", None):
            row = {"affected_eye": eye, "re_va_final": "6/12", "le_va_final": "6/60"}
            self.assertAlmostEqual(preprocess.extract_eye_va(row, preprocess.VA_FINAL_COLS), 1.0)
        row = {"affected_eye": None, "re_va_final": np.nan, "le_va_final": "6/12"}
        self.assertAlmostEqual(preprocess.extract_eye_va(row, preprocess.VA_FINAL_COLS), np.log10(2))


class FeatureTests(unittest.TestCase):
    def test_strict_outcome_boundary(self):
        eps = pd.DataFrame([episode(re_va_final="6/60"), episode(re_va_final="6/75"),
                            episode(re_va_final="EVIS")])
        feat = preprocess.build_features(eps, strict=True)
        self.assertEqual(feat["poor_outcome"].tolist(), [0.0, 1.0, 1.0])

    def test_missing_final_acuity_gives_missing_outcome(self):
        feat = preprocess.build_features(pd.DataFrame([episode(re_va_final=np.nan)]), strict=True)
        self.assertTrue(np.isnan(feat.loc[0, "poor_outcome"]))

    def test_culture_coding(self):
        eps = pd.DataFrame([
            episode(growths_yn=0, microorganism=""),                       # no growth
            episode(growths_yn=1, microorganism="Staph epidermidis"),      # organism grown
            episode(growths_yn=np.nan, microorganism=""),                  # not recorded
            episode(growths_yn=np.nan, microorganism="Enterococcus faecalis"),
            episode(growths_yn=1, microorganism="mixed skin flora"),       # grown, not identified
        ])
        feat = preprocess.build_features(eps, strict=True)
        self.assertEqual(feat["culture_positive"].tolist(), [0.0, 1.0, 0.0, 1.0, 1.0])
        self.assertEqual(feat["organism_cat"].tolist(),
                         ["no_growth", "cons", "unknown", "enterococcus", "unknown"])

    def test_organism_mapping(self):
        cases = {"Pseudomonas aeruginosa": "gram_negative", "coagulase negative Staphylococcus": "cons",
                 "Streptococcus pneumoniae": "streptococcus", "Staphylococcus aureus": "staph_aureus",
                 "Candida albicans": "fungal", "Propionibacterium acnes": "propionibacterium",
                 "": "unknown", "mixed skin flora": "unknown"}
        for text, category in cases.items():
            self.assertEqual(preprocess.normalise_organism(text), category, text)


class ModelSpecificationTests(unittest.TestCase):
    def test_model_spaces(self):
        names = [n for n, _, _ in train.get_models_and_spaces(1.0, ["LR", "XGB"])]
        self.assertEqual(names, ["LR", "XGB"])
        self.assertEqual(train.SPLIT_SEED, 61)
        self.assertEqual(train.ESTIMATOR_SEED, 42)
        self.assertEqual(train.N_ITER_SEARCH, 100)
        self.assertEqual(len(train.NUMERIC_FEATURES) + len(train.CATEGORICAL_FEATURES), 16)
        self.assertEqual(len(train.NUMERIC_FEATURES_NOMICRO) + len(train.CATEGORICAL_FEATURES_NOMICRO), 14)

    def test_preprocessor_imputes_and_scales(self):
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [0.0, 1.0, np.nan, 1.0],
                          "organism_cat": ["cons", np.nan, "cons", "no_growth"]})
        Xt = train.make_preprocessor(["a", "b"], ["organism_cat"]).fit_transform(X)
        self.assertEqual(Xt.shape, (4, 4))
        self.assertFalse(np.isnan(Xt).any())


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.p = rng.uniform(0.02, 0.98, 4000)
        self.y = (rng.uniform(size=self.p.size) < self.p).astype(int)

    def test_ici_is_small_when_calibrated(self):
        self.assertLess(metrics.compute_ici(self.y, self.p), 0.02)
        self.assertGreater(metrics.compute_ici(self.y, np.clip(self.p * 0.5, 0, 1)), 0.15)

    def test_delong_matches_auroc_and_is_antisymmetric(self):
        q = np.clip(self.p + np.random.default_rng(2).normal(0, 0.2, self.p.size), 0, 1)
        a, b, diff, z, p = metrics.delong_compare(self.y, self.p, q)
        self.assertAlmostEqual(a, roc_auc_score(self.y, self.p), places=10)
        self.assertAlmostEqual(b, roc_auc_score(self.y, q), places=10)
        a2, b2, diff2, z2, p2 = metrics.delong_compare(self.y, q, self.p)
        self.assertAlmostEqual(diff, -diff2, places=12)
        self.assertAlmostEqual(p, p2, places=12)

    def test_threshold_table_counts(self):
        t = metrics.make_table3(self.y, {"XGB": self.p}, pd.DataFrame({"model": ["XGB"], "test_auroc": [0.8]}), "XGB")
        self.assertTrue(((t.TP + t.FP + t.TN + t.FN) == self.y.size).all())

    def test_patient_cluster_bootstrap_self_checks(self):
        bootstrap.self_check()          # raises AssertionError on failure

    def test_weighted_rank_metrics_match_sklearn(self):
        groups = np.repeat(np.arange(self.y.size // 2), 2)
        w, idx = bootstrap.make_samples(groups, 20, 42, True)
        auc, _ = bootstrap.rank_metrics(self.y, self.p, w)
        for b in (0, 5, 19):
            self.assertAlmostEqual(auc[b], roc_auc_score(self.y[idx[b]], self.p[idx[b]]), places=10)


if __name__ == "__main__":
    unittest.main()
