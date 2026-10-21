# Report index

Only aggregate findings belong here. Patient artifacts remain outside Git.
For live state, see [project status](../docs/STATUS.md). Older decisions and
superseded runs remain in [the execution log](LOG.md), not the operating guide.

## Findings

| Report | Scope |
|---|---|
| [Data](DATA_REPORT.md) | Source counts, provenance, units and timing |
| [Cohort](COHORT_REPORT.md) | Eligibility funnel, labels, QC and fixed folds |
| [Canaries](CANARIES.md) | Shuffled labels and active missingness warning |
| [First checkpoint](CHECKPOINT_1.md) | E1 tabular EHR/CXR results and uncertainty |
| [E2 evaluation](E2_EVALUATION.md) · [PDF](E2_EVALUATION.pdf) | Methods, seed variation, figures and comparisons |
| [E2 fairness addendum](E2_FAIRNESS_ADDENDUM.md) | Three-seed E1 references, outer-fold signs and resolution |
| [CXR layout](CXR_LAYOUT.md) | Structural identifiers, repeated studies and selection alternatives |
| [Report audit](REPORT_AUDIT.md) | Corrected V2 parsing and exclusion of text |

The cluster's dependent evaluator will create `E3_RESULTS.md` and
`MASK_CONTROLS.md` after training succeeds. They are not yet completed results.
Copy and review their aggregate outputs before committing them here.

## Tables and figures

| Files under tables/ | Contents |
|---|---|
| `tabular_summary.csv`, `modality_contrasts.csv` | Complete E1 scores and paired contrasts |
| `e2_summary.csv`, `e2_fold_metrics.csv`, `e2_learning_curves.csv` | E2 seed/ensemble scores and validation traces |
| `e2_comparisons.csv` | Original E2 conditional paired intervals |
| `e1_three_seed_summary.csv` | Fixed E1 three-seed scores and measured diversity |
| `e2_fair_comparisons.csv`, `e2_logistic_fold_comparisons.csv` | Paired deltas, all fold signs and resolution |
| `cxr_layout.json` | Aggregate CXR structure/selection counts |
| `report_outcome_language.csv` | Corrected section-wise fixed-lexicon rates |

`figures/tabular_*.png` contains E1 plots. `figures/e2/` contains discrimination,
confusion and training-trajectory figures. Seed spread and patient intervals
measure different uncertainty; consult captions and methods before comparison.

Superseded EHR-only initial table copies were removed from the current tree;
their values are covered by the complete E1 tables and remain recoverable in Git.
