# First-48-hour ICU mortality benchmark

Research code for a patient-separated MIMIC-IV/MIMIC-CXR benchmark of in-hospital
mortality among adults alive and still in ICU at the 48-hour landmark. The goal
is reliable baselines and modality comparisons, with explicit leakage checks.

**Status (20 October 2026):** acquisition, cohort/QC, E0/E1 tabular models,
frozen CXR features, paired bootstrap comparisons and figures are complete.
The cohort contains 6,225 patients and 705 deaths. See reports/CHECKPOINT_1.md
and reports/CANARIES.md for results and limitations. E2 is prepared and queued
behind the GPU smoke test; deep/fusion experiments remain outstanding.

**CXR limitation:** the metadata has no timestamps. The image arm is exploratory
and cannot establish a temporally leakage-controlled CXR benefit. Source paths
retain study identifiers; a timed variant requires user-supplied metadata.

## Repository

```text
src/icu_mm/
  data/           acquisition, source inspection, cohort construction
  preprocessing.py
  splits.py       fixed patient partitions
  guards.py       feature, time-window and landmark checks
  metrics.py      validation thresholds and locked-test scores
  bootstrap.py    paired patient uncertainty estimates
  tabular.py      fixed tabular baselines
  models.py       modality encoders and cross-attention prototype
scripts/slurm/    reproducible CPU/GPU batch entrypoints
tests/            synthetic correctness and isolation tests
configs/          configuration notes and verified QC specifications
docs/             protocol and operating instructions
reports/          aggregate findings and decisions
```

Source code is authored and committed locally, then mirrored to the cluster
using Git bundles. Raw/interim data, arrays, predictions, environments and
checkpoints live outside the checkout. No notebooks or one-off exploration
scripts are required for this workflow.

## Reproduce

Read `docs/PROTOCOL.md` and `docs/RUNBOOK.md` before running experiments. Setup,
tests, data processing and training run through Slurm; do not install packages
or process data on login nodes.

Patient rows, reports, images, image paths and predictions must not appear in
Git or online AI-service tool output. Export aggregate diagnostics only. The
user confirmed access authorization; the Kaggle uploader's license label
conflicts with the archive's included PhysioNet license, as documented in
`reports/DATA_REPORT.md`.
