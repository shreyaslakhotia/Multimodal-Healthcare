# Canary results — 20 October 2026

Sources: `$P/results/canaries.json` and the corresponding fold/OOF artifacts
under `$P/results/tabular` and `$P/results/oof`. CPU baseline job 215460 completed
in 2:39 after its 36-test gate. No prespecified hard stop was triggered.

## Shuffled training labels

| Model | Pooled OOF AUROC |
|---|---:|
| Logistic regression | 0.483 |
| LightGBM | 0.540 |

Both lie within the prespecified 0.45–0.55 band. Validation and test labels remain
unaltered, and each training partition is shuffled deterministically. Passing
this band is a diagnostic check, not proof that every form of leakage is absent.

## Measurement-pattern warning

| Cohort | Logistic AUROC | LightGBM AUROC |
|---|---:|---:|
| S0, no landmark restriction | 0.656 | 0.805 |
| Final landmark cohort | 0.574 | **0.687** |

**The final LightGBM result exceeds the prespecified 0.65 warning threshold.**
Masks and observed-hour counts remain predictive after LOS filtering. Care
intensity, measurement selection and missingness may encode illness severity.
This result alone cannot identify a residual timing leak. Comparisons between
S0 and the main cohort also change the patient population and are not a causal
estimate of the LOS filter's effect. Keep with/without-count arms separate.

## Other gates

All EHR-only AUROCs are below 0.90. Final cohort support is 6,225 stays and 705
deaths. Five outer folds contain 136, 158, 125, 141 and 145 deaths. Patient
partitions are persisted; training-only preprocessing, timestamp boundaries,
forbidden inputs, unknown categories and restart behavior are covered by tests.

Current EHR results and bootstrap intervals are in `tables/tabular_summary.csv`.
CXR timing remains unknown and cannot be validated by EHR-only canaries.
