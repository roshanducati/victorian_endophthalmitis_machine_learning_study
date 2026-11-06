# `03_benchmarks.py`

Adds the comparators, and evaluates every model once in the held-out test set.

**Comparator models.** The main analysis compares XGBoost with simpler approaches:
- **Penalised logistic regression**, tuned exactly like XGBoost: same split, same 100-candidate search, same grouped cross-validation.
- **Presenting acuity alone:** a logistic regression on presenting logMAR, fitted on the development set.

**Operating thresholds.** Each model's threshold is the Youden index of its predictions in the development set. It is applied unchanged in the test set to give the sensitivities and specificities in Table S5.

**Checks and outputs.** Before anything else, the script checks that the saved XGBoost test predictions from step 2 are reproduced exactly. Point estimates from this step are final; their intervals and p values come from step 6.
