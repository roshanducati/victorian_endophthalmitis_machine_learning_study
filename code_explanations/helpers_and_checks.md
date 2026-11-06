# Helper modules and checks

- **`evaluation_metrics.py`**
  - The integrated calibration index: the average absolute gap between a loess calibration curve and the predicted risks.
  - Its bootstrap interval.
  - The DeLong test for two AUROCs measured on the same episodes.
  - The threshold table behind Table S7.
- **`participant_flow.py`:** recomputes the participant-flow counts from the processed data and supplies the box-drawing helper for Figure 1.
- **`journal_style.py`:** shared figure settings (8-point type, decimal points, panel labels).
- **`verify_reproduction.py`:** after a full run on the registry data, checks 150 regenerated values against the published ones. It prints each check and exits with an error if any differ.
- **`tests/synthetic_registry.py`:** writes a simulated registry export with the same structure as the real one, so the pipeline can be run end to end without data. Run it only in a copy of the repository that holds no registry data; the script refuses to overwrite an existing export. The simulated results have no clinical meaning.
- **`tests/test_pipeline_helpers.py`:** data-free tests of acuity decoding, eye selection, the outcome boundary, culture coding, organism mapping, the model specification and the statistical helpers.
