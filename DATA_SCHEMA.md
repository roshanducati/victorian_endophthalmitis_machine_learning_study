# Data schema

This repository contains no patient data. An approved user runs the pipeline by placing the REDCap export at:

```text
data/raw/VictorianEndophthalm_DATA_2026-05-17_2131.xlsx
```

If your export has a different file name, change `RAW_FILE` at the top of `01_preprocess.py` and `participant_flow.py`.

## Export structure

The export is a single REDCap workbook. The `redcap_event_name` column has two values:

- `registration_arm_1`: one row per patient, holding sex (`gender`: 1 male, 2 female);
- `review_arm_1`: one or more rows per patient, one per episode, distinguished by admission date and `redcap_repeat_instance`.

`01_preprocess.py` flattens the review arm to one row per episode, keyed on `rveeh_ur` (patient identifier) and `admission_date`:
- rows for the same patient and admission date are updates of one episode, and the highest repeat instance is kept;
- distinct admission dates for the same patient are separate episodes.

## Fields used

| Role | REDCap fields | Use |
|---|---|---|
| Patient identifier | `rveeh_ur` | Patient-level grouping of the development/test split, cross-validation and bootstrap only. Never a predictor. |
| Episode | `admission_date`, `redcap_repeat_instance` | Episode key; most recent instance retained |
| Sex | `gender` (registration arm) | Predictor (female, male) |
| Age | `admission_age` | Predictor (years) |
| Affected eye | `affected_eye` (RE, LE, BE or blank) | Selects the eye whose acuity is used (below) |
| Visual acuity | `re_va_on_presentation`, `le_va_on_presentation`, `re_va_final`, `le_va_final` | Presenting acuity (predictor, logMAR) and final acuity (outcome) |
| Clinical signs | `fundus_visible_yn`, `rapd_yn` | Predictors: fundus visible; relative afferent pupillary defect (RAPD) |
| Comorbidities | `diabetes_yn`, `immune_suppressed_yn` | Predictors |
| Prior surgery | `surgery_yn` | Descriptive only (shown in the registry only for surgical precipitating events) |
| Precipitating event | `precipitating_factor___1`, `___2`, `___3`, `___4`, `___5`, `___7`, `___98` | Seven non-exclusive predictors: cataract or intraocular lens surgery (1), glaucoma or drainage surgery (2), penetrating eye injury (3), corneal ulcer (4), endogenous (5), intravitreal injection (7), other ocular procedure (98) |
| Microbiology | `growths_yn`, `microorganism` (free text) | Culture result and organism category (below) |
| Primary procedure | `intervention___v` (vitrectomy), `intervention___b` (vitreous tap) | Descriptive only (Table 1). Never a predictor, because it is a treatment decision made after presentation. |
| Follow-up | `final_visit_date` | Days from admission to the final recorded acuity (descriptive and sensitivity analyses) |

Values outside {0, 1} in the yes/no fields are treated as missing.

## Visual acuity

Snellen acuities are stored either as Excel dates or as text, and are decoded to logMAR = log10(denominator / 6):

| Encoding | Rule | Example |
|---|---|---|
| Pattern A: day 1, month 6 | denominator = year modulo 100 | 1960-06-01 means 6/60 |
| Pattern B: day 6 | denominator = month | 2026-09-06 means 6/9 |
| Text | `6/X` | "6/120" |

| Category | logMAR |
|---|---|
| Counting fingers (CF) | 1.85 |
| Hand movements (HM) | 2.30 |
| Light perception (LP) | 2.70 |
| No light perception (NLP) | 3.00 |
| Evisceration or enucleation (EVIS) | 4.00 |

Date values that follow neither pattern are decoded as missing and reported in `va_date_audit.csv`. `va_mapping_table.csv` lists every recorded code with its logMAR value.

## Affected eye

When the affected eye is recorded as right (RE) or left (LE), that eye's acuity is used at presentation and at the final review. For bilateral episodes (BE) and episodes without a recorded affected eye, the eye with the worse acuity is used at each assessment, separately at presentation and at the final review. If only one eye has a recorded acuity, that eye is used.

## Outcome

With `--strict` (the definition used throughout the manuscript), a poor outcome is a final logMAR above 1.0 (worse than 6/60), or loss of the eye. A final acuity of exactly 6/60 counts as a good outcome. Episodes without a documented final acuity have a missing outcome and are excluded from modelling (400 of the 1700 registry episodes).

## Microbiology

- `culture_positive` is 1 when `growths_yn` records growth and 0 when it records no growth.
- When `growths_yn` is blank but `microorganism` names a specific organism, the episode is coded culture positive.
- When `growths_yn` is blank and no specific organism is named, the culture result is coded 0, not culture positive, and the organism category is `unknown`.
- `organism_cat` maps the free text to one of nine groups, with first-match-wins ordering from worst to best prognosis for polymicrobial entries: coagulase-negative staphylococci (`cons`), *Staphylococcus aureus*, *Streptococcus* species, *Enterococcus* species, *Cutibacterium (Propionibacterium)* species, *Bacillus* species, Gram-negative bacteria, fungi, and other or atypical organisms.
- `no_growth` and `unknown` are separate levels. `unknown` covers culture-positive episodes without an identifiable organism (for example "mixed skin flora") and episodes without a recorded culture result.
- `organism_mapping_audit.csv` lists every distinct free-text value with its category.

## Outputs of preprocessing

`data_strict/processed_episodes.csv` holds one row per episode with the identifiers, predictors, outcome and descriptive variables. It contains identifiers and is patient-level data, so it must never be shared or committed. The audits (`va_date_audit.csv`, `repeat_instance_audit.csv`, `precipitating_code_audit.csv`, `organism_mapping_audit.csv`) allow the decoding and feature derivation to be checked independently.
