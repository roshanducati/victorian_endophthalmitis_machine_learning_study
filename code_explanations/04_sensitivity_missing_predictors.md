# `04_sensitivity_missing_predictors.py`

Immune suppression and RAPD were unrecorded in 93% and 60% of episodes, so most of their values in the main models are imputed. Imputation of this kind assumes the values are missing at random, which cannot be checked from the data.

This script asks a simpler question: does discrimination depend on these two predictors at all? It re-develops the XGBoost model with cultures from scratch:
- without immune suppression;
- without immune suppression and RAPD.

Each version uses the same split and tuning protocol as the main models. Each reduced model is compared with the model that has all 16 predictors, in the same test episodes.

The version with all predictors reproduces the main model exactly, and step 6 checks this. The intervals and p values reported in Table S8 come from the patient-clustered bootstrap in step 6, which reuses the test-set predictions saved here.
