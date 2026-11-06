# `01_preprocess.py`

Turns the registry's REDCap export into an analysis table with one row per episode of endophthalmitis.

**Episodes.** Each patient has one registration record (which holds sex) and one or more review records. Review records for the same patient and admission date are updates of one episode; the latest is kept. A new admission date is a new episode.

**Visual acuity.** The registry stores Snellen acuities as Excel dates (for example, 1 June 1960 stands for 6/60 and 6 September 2026 for 6/9) or as text, and records non-Snellen categories as codes. All are converted to logMAR:
- counting fingers 1.85;
- hand movements 2.30;
- light perception 2.70;
- no light perception 3.00;
- loss of the eye 4.00.

**Affected eye.** The recorded affected eye is used. For bilateral episodes, or when the affected eye is not recorded, the eye with the worse acuity is used at each assessment.

**Outcome.** With `--strict`, a poor outcome is a final acuity worse than 6/60 or loss of the eye. Episodes without a final acuity have no outcome and are not modelled.

**Predictors.** Age, sex, presenting acuity, fundus visibility, RAPD, diabetes, immune suppression, seven precipitating events, culture result and organism category. The free-text organism is grouped by pattern matching. The primary procedure is kept only for description.

The script also writes audits that let the decoding be checked independently: acuity date patterns, repeat instances, precipitating-event codes and organism mapping. Nothing is imputed here. The output contains identifiers and must stay private.
