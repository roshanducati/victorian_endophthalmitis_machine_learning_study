# Predicting visual outcome after endophthalmitis

Analysis code for:

> Karri R, Hadoux X, Dawkins RCH, Allen PJ. Predicting visual outcomes after endophthalmitis across various aetiologies: development and internal validation of a machine learning model. *Clinical and Experimental Ophthalmology* (submitted).

The code reproduces every result in the manuscript and its Supporting Information from the Victorian Endophthalmitis Registry export. That covers preprocessing, the patient-level development/test split, model development, held-out evaluation, the patient-clustered bootstrap, and the tables and figures.

The repository contains no patient data, no derived patient-level tables, no fitted models and no results.

The models are research outputs. They were evaluated in a held-out test set drawn from the same registry (internal validation) and have not yet been externally validated. 

## The analysis in brief

- **Cohort:** 1300 episodes of endophthalmitis of any cause (1262 patients) managed at the Royal Victorian Eye and Ear Hospital, Melbourne, admitted July 1997 to January 2026.
- **Outcome:** a poor visual outcome, meaning final recorded visual acuity (VA) worse than 6/60, or evisceration or enucleation.
- **Two prediction times:**
  - presentation, using the pre-culture model (14 predictors);
  - the return of cultures, using the model with cultures (16 predictors: the same plus culture result and organism category).
- **Models:** XGBoost, with penalised logistic regression as a linear comparator.
- **Benchmark:** presenting VA alone (univariable logistic regression).
- **Development and evaluation:**
  - random patient-level 80:20 split (seed 61);
  - hyperparameters tuned by patient-grouped, stratified fivefold cross-validation within the development set;
  - a single evaluation in the held-out test set (262 episodes, 253 patients, 125 poor outcomes), including discrimination by sex and age.
- **Uncertainty:** a 10 000-resample patient-clustered bootstrap, with predictions and thresholds fixed.
  - Standalone estimates have percentile 95% CIs.
  - Differences between models have bootstrap standard error (Wald) intervals with matching p values.

## Pipeline

Run the scripts from the repository root in this order (`run_pipeline.sh` or `run_pipeline.ps1` does this for you):

| Step | Script | What it does | Manuscript |
|---|---|---|---|
| 1 | `01_preprocess.py --strict` | Builds one row per episode from the REDCap export: decodes acuity to logMAR, selects the affected eye, and derives the outcome and predictors. Writes the data-quality audits. | Methods 2.1 to 2.3; Table S1 |
| 2 | `02_train.py --strict` and `02_train.py --strict --nomicro` | Makes the patient-level split, tunes and fits XGBoost with and without microbiology, and compares the development and test sets. | Methods 2.4; Tables S2, S3 |
| 3 | `03_benchmarks.py` | Tunes logistic regression and fits the presenting-VA model. Evaluates every model in the held-out test set at its development Youden threshold. | Results 3.2 and 3.3; Tables S2, S5 |
| 4 | `04_sensitivity_missing_predictors.py` | Re-tunes XGBoost without immune suppression, and without immune suppression and RAPD. | Results 3.5; Table S8 |
| 5 | `05_heldout_statistics.py` | Point estimates and conventional tests: comparisons, calibration, discrimination by sex and age and class weighting in the test set; follow-up of at least 180 days versus less in all analysed episodes. | Feeds step 6 |
| 6 | `06_patient_cluster_bootstrap.py --n-boot 10000` | Patient-clustered bootstrap: every reported interval and p value. | Table 2; Results; Tables S4 to S6, S8 to S11 |
| 7 | `07_tables_and_figures.py` | Table 1, Table 2, Table S7, Figure 3 (SHAP) and Figure S1 (decision curves). | Methods 2.6 and Results 3.5 (predictor contributions); Tables 1, 2, S7; Figures 3, S1 |
| 8 | `08_figures_test_set.py` | Figure 1 (participant flow) and Figure 2 (ROC and calibration curves). | Figures 1, 2 |
| Check | `verify_reproduction.py` | Compares the regenerated results with the published values and exits with an error on any mismatch. | All of the above |

Helper modules:
- `evaluation_metrics.py`: integrated calibration index, DeLong test and threshold table;
- `participant_flow.py`: flow-diagram counts and boxes;
- `journal_style.py`: figure styling.

Plain-language notes on each script are in [`code_explanations/`](code_explanations).

## Where each result is written

Paths are relative to the repository root. `results_strict_benchmarks/` is abbreviated to `RB/`, and `RB/patient_cluster_bootstrap/` to `PCB/`.

| Manuscript item | Output |
|---|---|
| Table 1 | `RB/tables/table1_characteristics.csv` |
| Table 2 | `RB/tables/table2_discrimination_calibration.csv` (from `PCB/model_performance.csv`, `AUROC_comparison.csv`, `probability_comparison.csv`) |
| Figure 1 | `RB/figures/fig1_participant_flow.{tif,pdf,png}` |
| Figure 2 | `RB/figures/fig2_roc_calibration.{tif,pdf,png}` |
| Figure 3 | `RB/figures/fig3_shap.{tif,pdf,png}` |
| Table S1 | `data_strict/missingness_report.csv` |
| Table S2 | `results_strict/cv_results.csv`, `results_strict_nomicro/cv_results.csv`, `RB/summary.json` (logistic regression) |
| Table S3 | `data_strict/train_test_comparison_seed61.csv` |
| Table S4 | `PCB/model_performance.csv` |
| Table S5 | `RB/A_test_benchmarks.csv` (thresholds), `PCB/model_performance.csv` |
| Table S6 | `PCB/AUROC_comparison.csv`, `PCB/probability_comparison.csv` |
| Table S7 | `RB/tables/tableS7_thresholds_xgb_with_cultures.csv` |
| Table S8 | `results_strict_sensitivity/sensitivity_missingness.csv`, `PCB/missingness_performance.csv`, `PCB/missingness_comparison.csv` |
| Table S9 | `RB/heldout/followup_180_days.csv`, `PCB/followup_rate.csv`, `PCB/followup_comparison.csv` |
| Table S10 | `PCB/class_weighting.csv` (refits in `RB/heldout/class_weighting.csv`) |
| Table S11 | `RB/heldout/fairness_test.csv`, `PCB/subgroup_performance.csv`, `PCB/subgroup_heterogeneity.csv`, `PCB/age_joint_tests.csv` |
| Figure S1 | `RB/figures/figS1_decision_curves.{tif,pdf,png}` |

Several output folders contain patient-level files (`data_strict/` and the `test_predictions*.csv` files). All generated folders are excluded by `.gitignore` and must not be shared.

## Environment

The reported analysis used Python 3.11.15. Package versions are pinned in [`requirements.txt`](requirements.txt); scikit-learn 1.6.1 and XGBoost 3.2.0 must match exactly.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The full pipeline takes about 15 minutes on a laptop. Most of the time goes on the randomised searches in steps 2 and 4, and on the bootstrap in step 6.

## Data access

No data are distributed with this repository. The registry contains potentially identifiable clinical information and cannot be shared publicly. De-identified data may be made available on reasonable request to the corresponding author, subject to approval by the Royal Victorian Eye and Ear Hospital Human Research Ethics Committee and a data-sharing agreement.

Approved users should place the REDCap export at:

```text
data/raw/VictorianEndophthalm_DATA_2026-05-17_2131.xlsx
```

The fields used and the coding rules are described in [`DATA_SCHEMA.md`](DATA_SCHEMA.md).

## Reproducing and checking the results

```bash
bash run_pipeline.sh          # or, in PowerShell:  .\run_pipeline.ps1
```

The last step runs `verify_reproduction.py`. It formats 150 regenerated values at the precision printed in the manuscript and compares each with its published value:
- cohort and split counts;
- Table 1 and Table 2;
- the numbers in the Results and Discussion;
- key values from Tables S2 to S11.

It prints every check and exits with an error if any value differs.

## Running without data

To check that the code runs, simulate a registry export with the same structure. Do this in a separate copy of the repository that holds no registry data:

```bash
python tests/synthetic_registry.py      # writes a simulated export to data/raw/
bash run_pipeline.sh                    # runs end to end; the final check fails by design
python -m unittest discover -s tests -v # data-free tests of the helper functions
```

The simulated data have no clinical meaning.

## Fixed random values

| Value | Where | Use |
|---:|---|---|
| 61 | `SPLIT_SEED` | Patient-level `GroupShuffleSplit` into development and test sets |
| 42 | `ESTIMATOR_SEED` and script defaults | Estimators, randomised search, imputation, and every bootstrap |

## Methods implemented

- **Imputation, scaling and encoding** sit inside the scikit-learn pipelines, so they are refitted within every training fold and on the whole development set for test predictions.
  - Numeric and binary predictors: `IterativeImputer` (Bayesian ridge, 10 iterations, one completed data set), then `StandardScaler`.
  - Organism category: most-frequent imputation, then one-hot encoding.
- **Class weighting:** XGBoost uses `scale_pos_weight` (negatives / positives); logistic regression and the presenting-VA model use balanced class weights.
- **Calibration** is assessed with loess calibration curves (span 0.75) and summarised by the integrated calibration index.
- **Predictor contributions** are exact TreeSHAP values from XGBoost (`pred_contribs`).

## Access to the fitted models

For research use, the two XGBoost models (refitted to all 1300 episodes with the hyperparameters selected on the development set) are available through a password-protected web interface. Investigators at other centres can request access from the corresponding author: Dr Rosie C. H. Dawkins, Centre for Eye Research Australia (rosie.dawkins@post.harvard.edu). Access is for two purposes: external validation in their own patients, and early implementation studies, such as assessments of usability and workflow integration.

The interface is for research only. It is not a medical device, is not intended for clinical decision-making and does not store the data entered.

## Citation and licence

Citation: to be added on publication.

The code is released under the [MIT License](LICENSE). The licence confers no rights to the registry data.
