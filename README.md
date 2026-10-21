# First-48-hour ICU mortality benchmark

Research code for a patient-separated MIMIC-IV/MIMIC-CXR benchmark of in-hospital
mortality among adults alive and still in ICU at the 48-hour landmark. The goal
is reliable baselines and modality comparisons, with explicit leakage checks.

**Status (21 October 2026):** acquisition, cohort/QC, E0/E1, report audit and
all 60 E2 unimodal fits are complete. See the [E2 evaluation](reports/E2_EVALUATION.md)
and its [PDF](reports/E2_EVALUATION.pdf) for methods, figures and paired comparisons.
The cohort contains 6,225 patients and 705 deaths. E3 early and joint fusion
is queued as job 217740 after 49 tests passed; the late-average reference is
included. E4-E6 remain outstanding. Missingness and untimed-CXR limitations
remain active; see reports/CANARIES.md.

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

Start with the [repository and cluster run guide](docs/RUNBOOK.md) for SSH,
storage/quota checks, job monitoring, reproduction and synchronization. Read the
[scientific protocol](docs/PROTOCOL.md) before running experiments. Setup,
tests, data processing and training run through Slurm; do not install packages
or process data on login nodes.

Patient rows, reports, images, image paths and predictions must not appear in
Git or online AI-service tool output. Export aggregate diagnostics only. The
user confirmed access authorization; the Kaggle uploader's license label
conflicts with the archive's included PhysioNet license, as documented in
`reports/DATA_REPORT.md`.
