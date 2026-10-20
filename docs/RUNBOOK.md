# Research runbook

## Storage and environment

Active SSD: /projects/icumultimodal (100 GB assigned).
HDD originals: /projects/icumultimodalarchive/zips (200 GB assigned).
Checkout: $P/code. Environment: $P/envs/icumm. Raw inputs: $P/raw.
Interim patient artifacts: $P/data/interim. Outputs: $P/results and $P/reports.
Account allowances are 150 GB SSD and 300 GB HDD; assigned limits govern.
Project target modes are 2700 (private, setgid); inspect with stat -L.

Explicitly source /projects/icumultimodal/cacheconfig.sh in every job. It is
also sourced by .bashrc. Use job-specific scratch and validated cleanup targets.
No large files, environments or caches go in HOME or /tmp.

Python 3.11.12, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128. Dependencies are
pinned in requirements.txt; resolved pipfreeze.txt and condaexplicit.txt live
in $P/results/setup.

## Batch commands

Run only the needed stage, not every line together:

```bash
source /projects/icumultimodal/cacheconfig.sh
cd "$P/code"
sbatch scripts/slurm/setup_environment.sbatch  # one-time setup
sbatch scripts/slurm/check_cpu.sbatch
sbatch scripts/slurm/acquire_data.sbatch
sbatch scripts/slurm/inspect_data.sbatch
sbatch scripts/slurm/build_cohort.sbatch
sbatch scripts/slurm/build_features.sbatch
# Select ONE supported GPU model from live sinfo output:
sbatch --gres=gpu:l40:1 scripts/slurm/check_gpu.sbatch
```

Entry points run synthetic tests before long jobs. Acquisition and inspection
are reproducibility stages, not disposable exploration. Final cohort QC and observation eligibility are complete; outer folds are persisted. Do not regenerate partitions after seeing results.

## Monitor and resource limits

```bash
squeue --me
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode
tail -f /projects/icumultimodal/reports/JOB_LOG.out
```

Ctrl+C stops watching without cancelling jobs. Pending Priority has no reliable
ETA. Keep at most two running and five submitted jobs and two SSH sessions.
Never retry rejected submissions with variations. CPU jobs cost no SU. GPU
stage cap is 30,000 SU; stop at 21,000 stage SU. Track this stage's jobs, not
just monthly account usage. A five-minute L40 smoke can use at most 20 SU.

## Local source and synchronization

Local source: C:/Users/lakho/Desktop/NTU/Multimodal Healthcare Project/icubenchmark.
Dedicated SSH key: ~/.ssh/icubenchmark; explicit identity and BatchMode required.
Never read or print the private key. Non-secret shell scripts may be UTF-8/base64
encoded for Windows SSH quoting and piped to bash -l to initialize modules.
Never put secrets in command arguments.

Commit locally, transfer a Git bundle to $P/.tmp, then fast-forward the clean
cluster checkout. Do not edit remote code by hand. Do not change code used by
running experiments; record output revisions. Pending GPU job 215030 uses Slurm's
saved script and imports icu_mm.models, unchanged by repository reorganization.

## Data boundary

All processing runs in Slurm allocations. No recursive filesystem scans over
dataset/image directories. ZIP central-directory metadata is used for image
mapping. No patient rows, paths, images, report text, embeddings or predictions
enter Git or online AI-service output. Export aggregate diagnostics only.

The user confirmed access authorization after reviewing the uploader-label versus
included-license discrepancy. Preserve that provenance caveat.

## Completed stages

- Setup 215010 installed packages but hit an expected-array test shape bug.
  The correction passed in CPU job 215024 (22 tests).
- Acquisition 215029 completed in 1:55:32; nine IV tables and two CXR metadata
  CSVs verified; originals kept on HDD; 25 tests passed.
- Inspection 215035 completed in 15 seconds; 65,086 metadata patients, no time
  columns, 136/200 sampled paths matched archive members; 25 tests passed.
- GPU smoke 215030 remains queued at the latest check.
- Candidate cohort 215446 completed in 3:15 with 28 tests passed: 6,225
  stays and 705 deaths. Final QC and persisted folds subsequently completed
  in job 215456; no additional eligible stays were removed.


## Benchmark stages and current status

The following CPU stages are complete; rerun only for reproducibility or a
justified correction. Cached results are protected by implementation signatures.

```bash
sbatch scripts/slurm/run_baselines.sbatch
sbatch scripts/slurm/extract_cxr_cpu.sbatch
sbatch scripts/slurm/run_cxr_baselines.sbatch
sbatch scripts/slurm/aggregate_results.sbatch
sbatch scripts/slurm/render_figures.sbatch
sbatch scripts/slurm/verify_sources.sbatch
sbatch scripts/slurm/audit_reports.sbatch
```

Final main cohort: 6,225 / 705 deaths. Forty-eight tabular configurations and
44 paired contrasts are in reports/tables; ROC/PR, confusion and leave-one-out
figures are in reports/figures. The corrected report audit uses the versioned
results/report_audit/inline_headers_v2 directory; preserve superseded V1 outputs.

GPU smoke 215030 waits for Priority. E2 job 215538 is already submitted with
afterok dependencies on 215030 and completed source check 215533. Do not submit
a duplicate. It requests one L40 for four hours (at most 960 SU). Fixed E2 is
four encoders x five folds x three seeds, with per-epoch resume checkpoints and
inner-validation early stopping. GPU behavior remains unverified until smoke
executes. No E2 results exist yet. E3-E6 remain to be implemented/run.

For the currently queued work:

```bash
squeue --me
sacct -j 215030,215538 --format=JobID,JobName,State,Elapsed,ExitCode
# Once E2 starts:
tail -f /projects/icumultimodal/reports/icue2-215538.out
```

Pending Priority has no dependable remaining-time estimate. Never select the
ARM gh200 node to bypass the queue for this x86 environment. Total GPU SU for
these jobs is zero while pending. Consult results/setup/gpujobs.txt for the
stage job registry and account only actual allocated time after execution.