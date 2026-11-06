# `07_tables_and_figures.py`

Writes the main tables and two figures:

- **Table 1.** Cohort characteristics by outcome, with Mann-Whitney U tests and χ² tests, the latter with continuity correction for 2×2 tables.
- **Table 2.** AUROC and integrated calibration index for each model, and the paired comparisons, formatted from the step 6 bootstrap. Brier scores and AUPRC are in Table S4.
- **Table S7.** Sensitivity, specificity and predictive values of the with-culture model at fixed probability thresholds (descriptive).
- **Figure 3.** Predictor contributions from exact TreeSHAP values computed by XGBoost. Panel A sums the one-hot organism categories into a single predictor; panel B shows the direction of effect for the 12 most influential encoded features.
- **Figure S1.** Decision curves (descriptive).

Figures are saved as TIFF (1200 dpi), PDF and PNG with 8-point type. The module also defines the colours, line styles and `save()` helper that `08_figures_test_set.py` uses.
