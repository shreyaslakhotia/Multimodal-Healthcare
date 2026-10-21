# Contributing

Use the `shreyas` branch for this research stage. Commit small, reviewable changes;
do not push directly to main. Read the [protocol](docs/PROTOCOL.md),
[code map](docs/CODE_MAP.md) and [cluster guide](docs/RUNBOOK.md) first.

## Preserve reproducibility

- Keep patient partitions fixed and fit preprocessing on inner training only.
- Declare new comparisons/ablation settings before scoring their test sets.
- Do not edit code used by running jobs. Fitted caches use source signatures;
  cosmetic edits or moves to fingerprinted modules can invalidate resume.
- Use a versioned result namespace for a substantive correction. Preserve the
  previous result and document why it was superseded.
- Keep acquisition, inspection and QC stages as reusable research code. Remove
  disposable setup helpers and duplicate aggregate copies after consolidation.

## Validate and report

Review `git diff --check` and changed local links. Run relevant synthetic tests
through `scripts/slurm/check_cpu.sbatch` on the cluster. A compatible local
environment can use `python -m pytest -q`; never run the suite on a login node.
Long stage entrypoints repeat the test gate. Do not claim unexecuted tests pass.

Add meaningful tests for leakage boundaries, alignment, missingness or changed
training behavior. Document metric definitions and limitations. Conditional
patient intervals do not include retraining uncertainty; retain fold deltas.
Unresolved differences are not equality. Update the status page after verifying
real jobs, and keep historical events in the execution log.

## Publication boundary

Commit source, configuration, synthetic tests, methods and aggregate reports.
Never commit raw/interim data, identifiers, images, report text, paths containing
patient IDs, embeddings, predictions, checkpoints, caches or environments.
Credentials/private keys do not belong in source or logs. Review tracked files
and history before an initial push; ignore rules do not remove old Git objects.

The repository does not grant rights to MIMIC data or pretrained weights. Follow
the applicable original licenses and access agreements. Preserve source-data
provenance and the documented discrepancy in the Kaggle label.
