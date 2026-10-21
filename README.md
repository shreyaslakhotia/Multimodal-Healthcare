# Multimodal ICU mortality benchmark

A research benchmark of in-hospital mortality among adults alive and still in
ICU at a 48-hour landmark. Compare clinical context, vital signs, laboratory
measurements and chest X-ray representations using fixed patient partitions,
training-only preprocessing and explicit leakage diagnostics.

The matched cohort contains **6,225 patients and 705 deaths**. EHR inputs are
restricted to the first 48 hours. **CXR timing is unknown**: image associations
are exploratory, and report text is excluded from the main benchmark. The
mask-only prediction warning remains active. This is retrospective research,
not a deployed clinical prediction system.

## Start here

| Need | Guide |
|---|---|
| Current jobs and next steps | [Project status](docs/STATUS.md) |
| SSH, monitoring, storage and Slurm | [Cluster run guide](docs/RUNBOOK.md) |
| Fixed methods and leakage rules | [Protocol](docs/PROTOCOL.md) |
| Source modules and entrypoints | [Code map](docs/CODE_MAP.md) |
| Results, tables and figures | [Report index](reports/README.md) |
| Contribution and reproducibility checks | [Contributing](CONTRIBUTING.md) |

## Experiment ladder

| Stage | Purpose | Evidence or specification |
|---|---|---|
| E0 | Leakage and measurement-pattern canaries | [Canary report](reports/CANARIES.md) |
| E1 | Fixed logistic and LightGBM baselines | [Checkpoint](reports/CHECKPOINT_1.md) |
| E2 | Unimodal encoders and mask controls | [Evaluation](reports/E2_EVALUATION.md), [PDF](reports/E2_EVALUATION.pdf), [fairness addendum](reports/E2_FAIRNESS_ADDENDUM.md) |
| E3 | Late average, frozen early fusion, joint fine-tuning | [Specification](docs/E3_EXPERIMENT.md) |
| E4 | EHR-first cross-attention comparison | [Design](docs/E4_DESIGN.md) |
| E5 | Leave-one-out, masking and permutation | [Declared scope](docs/E4_DESIGN.md#preserve-e5) |
| E6 | Secondary diagnosis task | Planned; not yet implemented |
| E7 | Radiology report audit | [Audit](reports/REPORT_AUDIT.md) |

The main RQ2 comparison is EHR-only cross-attention versus EHR-only joint
concatenation. CXR fusion is secondary. Differences below pair-specific
resolution or whose interval includes zero are **unresolved**, not equal.
Patient bootstrap intervals are conditional on fitted models; key comparisons
also retain all five outer-fold deltas.

## Repository layout

```text
configs/           fixed feature definitions and configuration notes
docs/              methods, experiment specifications and cluster operations
reports/           aggregate findings, tables, figures and execution history
scripts/slurm/     reproducible allocation-aware stage entrypoints
scripts/reporting/ local document builders using aggregate inputs only
src/icu_mm/        data pipeline, learners, training and evaluation modules
tests/             synthetic correctness, isolation and missingness tests
```

Python 3.11.12; PyTorch 2.8.0 with CUDA 12.8; torchvision 0.23.0. Dependencies
are pinned in [requirements.txt](requirements.txt) and [environment.yml](environment.yml).
The cluster setup script installs the pinned wheels and editable package.
Tests run in a CPU allocation before long stages; the latest verified suite
contains **54 passing tests**.

Raw data, identifiers, images, report text, predictions, embeddings, checkpoints,
credentials and environments stay outside Git. The user confirmed data access
authorization; retain the included PhysioNet license provenance despite the
Kaggle uploader's conflicting label. See the [data report](reports/DATA_REPORT.md).

Development is published on **`shreyas`**. Cluster source is synchronized
through Git bundles; branch publication does not submit jobs.
