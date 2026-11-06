# `05_heldout_statistics.py`

Computes the point estimates and the conventional tests in the 262 test episodes. It also fixes the list of comparisons that step 6 re-evaluates.

It covers:
- pairwise AUROC comparisons (DeLong test);
- the Brier score and the integrated calibration index;
- discrimination by sex and age, the analysis of model fairness;
- class weighting: XGBoost refitted without weights, keeping the tuned hyperparameters;
- the proportion with a poor outcome in episodes followed for at least 180 days and in those followed for less, in all 1300 analysed episodes.

These conventional intervals treat episodes as independent, and seven test patients contributed more than one episode. The manuscript therefore reports the patient-clustered bootstrap of step 6 instead. The two agree closely: for the principal comparisons, the p values differ by no more than 0.002.
