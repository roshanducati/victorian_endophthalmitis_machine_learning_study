# `06_patient_cluster_bootstrap.py`

Produces every interval and p value reported in the manuscript.

**Resampling.** The 253 test patients are resampled with replacement 10 000 times. Whenever a patient is drawn, all of that patient's episodes are included. This respects the dependence between repeated episodes in the same patient while keeping the episode as the unit of analysis. Every model is evaluated on the same resamples, so differences between models are paired.

**What stays fixed.** The models, their predictions and the operating thresholds are not refitted. The intervals therefore describe uncertainty in evaluating these models on this test sample, not uncertainty in developing them.

**Intervals and tests.**
- Standalone estimates (for example an AUROC, or the proportion with a poor outcome among episodes followed for at least 180 days) get percentile 95% intervals.
- AUROCs in women and men, and in three age groups, are compared with Wald tests that use the same bootstrap.
- Differences between models get intervals of the estimate plus or minus 1.96 bootstrap standard errors. Each has a matching Wald p value, so the interval and the test always agree.

**Built-in checks.** An ordinary episode bootstrap runs alongside as a control. The script asserts that:
- it reproduces the earlier conventional intervals exactly;
- every point estimate matches the earlier steps;
- the predictions match the saved models.

It records input file hashes and package versions in `run_metadata.json`.
