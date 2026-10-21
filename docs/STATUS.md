# Project status

Last verified 21 October 2026 at 23:00 Singapore time. This is a snapshot;
use squeue/sacct for live state. [Run guide](RUNBOOK.md) · [Report index](../reports/README.md).

## Active submissions

| Job | Work | Last observed state | QOS |
|---|---|---|---|
| 217883 | E3 late average, frozen early fusion, joint fine-tuning | Pending Priority | override-limits-but-killable |
| 217932 | Values-only/mask-only vitals and labs, 60 fits | Pending Priority | override-limits-but-killable |
| 217933 | Combined mask and E3 evaluation | Dependency on both GPU jobs | ug CPU |

There is one E3 copy and one mask-control copy. Each GPU job uses one A6000,
checkpoints every epoch and resumes after requeue. No E3/mask-control result
exists yet. A start estimate is not a completion estimate.

Old jobs 217740, 217849, 217853 and 217854 were cancelled before allocation.
No record for 217741 was found under this account; it is not active. Do not
claim its cancellation without an accounting record.

## Completed work

- Acquisition/provenance, cohort construction, hourly QC and fixed folds:
  6,225 patients, 705 deaths.
- E0/E1, image-selection sensitivity, bootstrap comparisons and figures.
- E7 report audit; text excluded because timing is unknown.
- E2: 60 unimodal fits, 12 OOF artifacts and the accepted eight-page evaluation.
- Three-seed E1 LightGBM comparators and E2 outer-fold deltas. All E1 seeds
  produced identical predictions under the fixed deterministic settings.
- CXR layout, EHR-first E4 design and synthetic checks.
- Latest verified suite: 54 tests passed (CPU job 217940).

## Remaining work and interpretation

Review E3/mask results when the dependent evaluator finishes. Next train the
EHR-only concatenation comparator and primary E4a, followed by secondary CXR
E4a/E4b and the declared E5 analyses. E6 remains outstanding. Logistic stacking
is dropped. The optional 100-epoch context rerun was not launched under the
five-minute condition.

Charged GPU stage use: approximately 64.2 SU from smoke and E2 allocated seconds.
CPU/pending work adds zero; free-QOS work uses no charged SU but its allocated
runtime must still be recorded. Stage stop threshold: 21,000 SU.

Read [the fairness addendum](../reports/E2_FAIRNESS_ADDENDUM.md) alongside E2.
Differences below pair-specific resolution or with intervals including zero
are unresolved, not equal. CXR timing is unknown; the mask-only warning remains
active at AUROC 0.687.
