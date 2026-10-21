# Code map

The package is `icu_mm` under `src/`. Data construction has a separate subpackage;
experiment modules keep stable paths and contents because fitted artifacts hash
their implementation. Cosmetic moves or formatting changes can invalidate those
locks. Reusable Slurm stages remain available for reproduction.

| Responsibility | Modules in src/icu_mm |
|---|---|
| Acquire and verify sources | `data/acquisition.py`, `data/inspection.py`, `data/source_quality.py` |
| Cohort, hourly data and eligibility | `data/cohort.py`, `data/hourly.py`, `data/features.py` |
| CXR cache and structural audit | `data/cxr.py`, `data/cxr_layout.py` |
| Context, normalization and fixed splits | `context.py`, `preprocessing.py`, `splits.py` |
| Leakage guards and metrics | `guards.py`, `metrics.py`, `bootstrap.py`, `fair_metrics.py` |
| E0/E1 learners and arms | `tabular.py`, `baselines.py`, `cxr_baselines.py`, `e1_ensembles.py` |
| E2 encoders and controls | `models.py`, `train_unimodal.py`, `mask_ablation.py` |
| E3 concatenation and training | `fusion.py`, `train_fusion.py` |
| E4 EHR-first architecture | `attention_design.py`; training integration remains outstanding |
| Tabular aggregation/figures | `aggregate.py`, `figures.py` |
| Deep/fairness evaluation | `evaluate_e2.py`, `evaluate_fusion.py`, `evaluate_masks.py`, `complete_fairness.py` |
| E7 text audit | `report_audit.py` |

## Execution flow

```text
Source acquisition -> structure/unit checks -> candidate cohort -> final QC/folds
  -> E0/E1 -> cached CXR -> tabular aggregation/figures
  -> E2 -> E2 evaluation -> E3 and mask controls -> paired evaluation
  -> EHR-only concatenation/E4 -> E5 -> secondary E6

Report audit is independent of deep training and excludes text from main inputs.
```

Enter through [Slurm scripts](../scripts/slurm/README.md), not an interactive
Python process on a login node. `python -m icu_mm.<module>` is what the relevant
allocation runs. Builders in [scripts/reporting](../scripts/reporting/README.md)
consume aggregate files locally; they never read patient predictions.

Tests are synthetic and cover partition isolation, train-only preprocessing,
timestamp boundaries, metric behavior, resume RNG, frozen gradients, missing
modality invariance and attention behavior. They do not replace review of real
cohort/QC diagnostics and canaries. The prototype `CrossAttentionFusion` remains
in `models.py` for its tested historical API; the revised EHR-first design is
`EHRFirstCrossAttention` in `attention_design.py`.
