# Checkpoint 1: data and tabular benchmark

Updated 20 October 2026. Main cohort: 6,225 adult first ICU stays, 705 deaths
(11.33%), alive and still in ICU at the 48-hour landmark. EHR inputs use only
[intime, intime + 48 hours). Five patient-separated outer folds and inner
validation sets are persisted. Hyperparameters are fixed; preprocessing is fitted
on inner training patients only. Validation selects thresholds and early stopping.

## Pooled out-of-fold results

Seed 0; five folds; 95% percentile intervals from 1,000 patient bootstrap draws,
conditional on fitted models. AUPRC is average precision. Count variants also
control explicit missingness indicators. Full metrics and confusion counts are
in tables/tabular_summary.csv.

| Inputs | Learner | Counts | AUROC (95% CI) | AUPRC |
|---|---|---|---|---:|
| EHR | Logistic regression | Without | 0.797 (0.780-0.814) | 0.387 |
| EHR | Logistic regression | With | 0.822 (0.806-0.837) | 0.414 |
| EHR | LightGBM | Without | 0.816 (0.799-0.832) | 0.405 |
| EHR | LightGBM | With | 0.824 (0.807-0.840) | 0.420 |
| EHR + CXR | Logistic regression | Without | 0.800 (0.783-0.816) | 0.388 |
| EHR + CXR | Logistic regression | With | 0.820 (0.804-0.835) | 0.409 |
| EHR + CXR | LightGBM | Without | 0.816 (0.798-0.832) | 0.389 |
| EHR + CXR | LightGBM | With | 0.815 (0.797-0.832) | 0.411 |
| CXR only | Logistic regression | Without | 0.606 (0.583-0.627) | 0.162 |
| CXR only | LightGBM | Without | 0.569 (0.547-0.590) | 0.145 |

All four full-minus-EHR AUROC and AUPRC paired intervals include zero. These
fixed models provide no statistically supported CXR gain. Images have no timing
metadata, so neither benefit nor lack of benefit establishes clean temporal
clinical evidence. First AP, otherwise first PA, is the primary selection rule;
the alternate same-view selection comparison is also inconclusive.

For LightGBM EHR without explicit counts, removing context, vitals or labs
reduces AUROC by 0.016 (paired CI 0.006-0.027), 0.043 (0.028-0.056), and 0.029
(0.017-0.042). Each contributes in this model. These estimates do not establish
a statistically certain ranking between vitals and labs.

## Diagnostics and limits

Shuffled-label AUROCs: logistic 0.483, LightGBM 0.540; both inside the fixed
0.45-0.55 band. All EHR AUROCs are below 0.90. Passing these checks does not prove
absence of leakage. Mask-only LightGBM scores 0.687 in the final cohort, above
the 0.65 warning level; S0 scores 0.805. Measurement patterns remain predictive.
Retain both count variants and the warning (CANARIES.md).

Final QC removed no additional eligible stays. GCS requires three valid
components within the same hour; ETT verbal responses are missing/unassessable.
Five folds contain 136, 158, 125, 141 and 145 deaths. No patient text enters the
main benchmark. Most available text records combine multiple studies and have
unknown timing; the corrected report audit is documented separately.

## Execution status

CPU acquisition, cohort/QC, E0/E1, frozen CXR extraction, paired bootstrap and
aggregate figures completed. GPU smoke 215030 is pending Priority. Fixed E2
training 215538 waits for that smoke test; no deep-model results exist yet.
E3/E4 fusion, E5 deep ablations and E6 diagnosis remain outstanding. No GPU SU
has been consumed by these pending jobs. A queue start time cannot be inferred
from pending Priority. Maximum requested smoke plus E2 allocation is 980 SU,
within the stage stop threshold of 21,000 SU.