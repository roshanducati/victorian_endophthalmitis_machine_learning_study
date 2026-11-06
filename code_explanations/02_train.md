# `02_train.py`

Develops the XGBoost models, with microbiology (16 predictors) and, with `--nomicro`, without it (the pre-culture model, 14 predictors).

**Split.** Episodes with a known outcome are split once, by patient, into a development set (80%) and a held-out test set (20%), using `GroupShuffleSplit` with seed 61. No patient appears in both sets.

**Tuning.** For each model, 100 hyperparameter combinations are drawn at random (seed 42) and scored by five-fold cross-validation within the development set. The folds are stratified by outcome and grouped by patient. The combination with the highest cross-validated AUROC is refitted on the whole development set.

**Preprocessing.** Preprocessing lives inside each model pipeline, so it is learnt only from the data being fitted.
- Missing numeric and binary values are filled by chained-equations imputation (Bayesian ridge, 10 iterations, one completed data set) and then standardised.
- A missing organism category is filled with the most frequent category, and the categories are one-hot encoded.

**Class weighting.** XGBoost up-weights the smaller class (`scale_pos_weight`).

**Outputs.** The script saves:
- the fitted pipelines, the cross-validation summary and held-out metrics;
- the test-set predictions, which include identifiers for linkage and must stay private;
- with the full predictor set, a table comparing the development and test sets (Table S3).

It also defines the model spaces that later steps reuse, including the logistic regression tuned in step 3.
