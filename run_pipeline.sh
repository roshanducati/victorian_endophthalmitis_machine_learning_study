#!/usr/bin/env bash
# Reproduce every reported result, in order. Requires the registry export at
# data/raw/ (see DATA_SCHEMA.md). Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/9] Preprocess the registry export (strict outcome definition)"
python 01_preprocess.py --strict

echo "[2/9] Develop the models with microbiology"
python 02_train.py --strict

echo "[3/9] Develop the pre-culture models"
python 02_train.py --strict --nomicro

echo "[4/9] Comparators: logistic regression and presenting acuity alone"
python 03_benchmarks.py

echo "[5/9] Sensitivity analysis: predictors with high missingness"
python 04_sensitivity_missing_predictors.py

echo "[6/9] Held-out test-set statistics"
python 05_heldout_statistics.py

echo "[7/9] Patient-clustered bootstrap (10 000 resamples)"
python 06_patient_cluster_bootstrap.py --n-boot 10000

echo "[8/9] Tables, Figure 3 and Figure S1"
python 07_tables_and_figures.py

echo "[9/9] Figures 1 and 2"
python 08_figures_test_set.py

echo "Checking the reproduced results against the manuscript"
python verify_reproduction.py
