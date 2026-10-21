# Repository and cluster run guide

Permanent operating guide for this project. For dated results and job history,
see [the checkpoint](../reports/CHECKPOINT_1.md) and [execution log](../reports/LOG.md).
For scientific decisions, see [the protocol](PROTOCOL.md). Do not rerun completed
stages merely because they appear in the commands below.

## Connect and locate the project

Connect to NTUSECURE or the NTU VPN first. From local PowerShell:

```powershell
ssh -i C:\Users\lakho\.ssh\icubenchmark -o IdentitiesOnly=yes shreyas010@10.97.216.128
```

The following commands run inside the resulting cluster shell:

```bash
source /projects/icumultimodal/cacheconfig.sh
cd /projects/icumultimodal/code
```

| Location | Contents |
|---|---|
| /projects/icumultimodal | SSD project; 100 GB assigned quota |
| /projects/icumultimodalarchive | HDD project; 200 GB assigned quota |
| $P/code | Git source checkout |
| $P/raw | Required extracted source tables and CXR metadata |
| $P/data/interim | Private cohort, features and persisted patient partitions |
| $P/envs/icumm | Python environment |
| $P/.tmp | Caches and per-job scratch |
| $P/results | Private predictions, models, checkpoints and aggregate results |
| $P/reports | Batch logs and generated figures |
| /projects/icumultimodalarchive/zips | Original archives/backups |

The account allowances are 150 GB SSD and 300 GB HDD; the project-assigned
quotas above govern these folders. The project roots are symlinks; use stat -L
to inspect target permissions (2700: private, setgid). Nothing is backed up by
the cluster. Active repeated reads belong on SSD.

## Check storage usage

Run the interactive quota tool in an SSH login session:

```bash
storagemgr
```

It shows assigned quotas and usage for both projects. Press q to exit. This
requires an interactive login shell; an uninitialized remote command may not
find storagemgr. Do not use df to infer your quota: it reports filesystem-wide
capacity shared by the cluster.

For a fast noninteractive size check, use Ceph's directory byte counters:

```bash
for project in /projects/icumultimodal /projects/icumultimodalarchive; do
  bytes=$(getfattr --only-values -n ceph.dir.rbytes "$project" 2>/dev/null) || continue
  awk -v p="$project" -v b="$bytes" 'BEGIN {printf "%s: %.2f GB (%.2f GiB)\n", p, b/1e9, b/1073741824}'
done
```

This reads directory metadata, without walking the files. It measures current
recursive file bytes, including environment, caches, checkpoints and outputs;
it is not a cumulative network-download counter. Quota-manager usage may differ
in accounting or update timing. To inspect downloaded inputs separately:

```bash
getfattr --only-values -n ceph.dir.rbytes /projects/icumultimodal/raw
printf '\n'
getfattr --only-values -n ceph.dir.rbytes /projects/icumultimodalarchive/zips
printf '\n'
```

Those values are bytes. SSD extraction and HDD backups overlap, so their sum is
not unique source data. Acquisition manifests in $P/results/acquisition record
verified file sizes and SHA256 digests. Avoid recursive du/find over project
or dataset trees on login nodes. Use storagemgr if Ceph counters are unavailable.

## Monitor jobs and interpret waiting

```bash
squeue --me
squeue --start -j 215030,215538
sacct -j 215030,215538 --format=JobID,JobName,State,Elapsed,ExitCode
date '+%Y-%m-%d %H:%M:%S %Z (%z)'
```

Replace job IDs for future runs. Start estimates use the cluster's timezone
(currently UTC); Singapore is UTC+8. Estimated start times can move or be N/A.
Priority means waiting for scheduler resources/priority. Dependency means a
prerequisite must finish successfully. A completed dependency makes the next
job eligible; it does not reserve a GPU or guarantee immediate execution.

At the 20 October checkpoint, smoke 215030 is queued and E2 215538 already has
an afterok dependency on it and completed source check 215533. Do not submit
another copy. Current output paths are:

```bash
tail -f /projects/icumultimodal/reports/icusmoke-215030.out
tail -f /projects/icumultimodal/reports/icue2-215538.out
```

Logs appear only after their jobs start. Ctrl+C ends tail without cancelling
the job. Check every 30-60 minutes while pending; no terminal needs to remain
open. Batch jobs and their dependencies survive SSH disconnection. On failure,
inspect sacct and the matching log before resubmitting. ExitCode 0:0 together
with COMPLETED indicates scheduler-level success; also verify stage output.

Smoke has a five-minute limit. E2 requests four hours per allocation, which is
a limit rather than a measured completion estimate. E2 checkpoints each epoch;
a timeout requires a reviewed resume submission, not an assumption that all
60 encoder/fold/seed fits finished. After E2, evaluate the encoder results and
proceed to E3 fusion baselines, E4 cross-attention, E5 ablations/missingness and
E6 secondary diagnosis, within the protocol priorities and remaining budget.

## Environment, caches and resource rules

Every job explicitly sources /projects/icumultimodal/cacheconfig.sh, also sourced
by .bashrc. Set job-specific TMPDIR under $P/.tmp and validate cleanup targets.
No environments, downloads or large caches go in HOME or /tmp. XRV downloads
need explicit cache_dir=$P/.tmp/torch/xrv. Python is 3.11.12, torch 2.8.0+cu128,
torchvision 0.23.0+cu128; direct requirements are pinned. Resolved environment
records are $P/results/setup/pipfreeze.txt and condaexplicit.txt.

Login nodes are for light editing, metadata and submission. Installs, tests,
data processing and training run in Slurm. No recursive dataset scans. Request
exactly one supported GPU model, e.g. --gres=gpu:l40:1. Do not request GPU-job
CPU/memory overrides or exclusive nodes; do not use salloc or install toolchains.
The gh200 is ARM and incompatible with this x86 environment. Check live sinfo
rather than assuming GPU availability. Never retry a rejected job with variants.
Keep at most two SSH sessions, two running jobs and five submitted jobs.

CPU jobs consume no GPU SU. The stage cap is 30,000 SU, with a stop threshold
of 21,000 SU. Track actual allocated GPU time for this stage using the registry
$P/results/setup/gpujobs.txt, not monthly account use alone. Pending jobs consume
no GPU runtime. The requested L40 smoke and E2 maxima are 20 and 960 SU.

## Reproduce stages

Run only the stage needed, in dependency order; synthetic tests gate batch
entrypoints. These commands are examples, not a script to paste all at once.

```bash
source /projects/icumultimodal/cacheconfig.sh
cd "$P/code"
sbatch scripts/slurm/setup_environment.sbatch  # one-time setup
sbatch scripts/slurm/check_cpu.sbatch
sbatch scripts/slurm/acquire_data.sbatch
sbatch scripts/slurm/inspect_data.sbatch
sbatch scripts/slurm/build_cohort.sbatch
sbatch scripts/slurm/build_features.sbatch
sbatch scripts/slurm/run_baselines.sbatch
sbatch scripts/slurm/extract_cxr_cpu.sbatch
sbatch scripts/slurm/run_cxr_baselines.sbatch
sbatch scripts/slurm/aggregate_results.sbatch
sbatch scripts/slurm/render_figures.sbatch
sbatch scripts/slurm/verify_sources.sbatch
sbatch scripts/slurm/audit_reports.sbatch
```

For a future fresh GPU run, first choose a supported model from live sinfo:

```bash
sbatch --gres=gpu:l40:1 scripts/slurm/check_gpu.sbatch
# After that smoke succeeds, or with an explicit afterok dependency:
sbatch --gres=gpu:l40:1 scripts/slurm/train_unimodal.sbatch
```

Do not duplicate the currently submitted jobs. E2 uses fixed settings, four
modalities, five folds and three seeds, with inner-validation early stopping.
Final patient partitions are already fixed: do not regenerate after viewing
results. Signature locks protect fitted-model caches from silent source changes.
Corrected report auditing uses results/report_audit/inline_headers_v2; preserve
V1 for provenance and cite only corrected section results.

## Source organization and synchronization

Local authoritative repository:
C:/Users/lakho/Desktop/NTU/Multimodal Healthcare Project/icubenchmark.

Reusable Python is in src/icu_mm, batch entrypoints in scripts/slurm, fixed
specifications in configs, synthetic checks in tests, operating/scientific
instructions in docs and aggregate findings in reports. Keep completed
acquisition/inspection stages for reproducibility. Avoid disposable exploration
scripts and duplicate setup helpers in the source tree.

Commit locally, create a Git bundle, transfer it to $P/.tmp, then fetch and
fast-forward the clean cluster checkout. Verify git status is clean before and
after synchronization. Do not edit remote source by hand or change code used
by a running experiment. Record its revision in the job log. Use the dedicated
SSH identity explicitly; never print/read the private key or put secrets in
command arguments. Non-secret shell scripts may be base64 piped to bash -l
when necessary to avoid Windows SSH quoting errors.

## Data and scientific boundaries

Patient data, identifiers, report text, images, image paths, embeddings,
predictions and checkpoints stay on the cluster outside Git. Only aggregate
reports/figures enter Git or online AI-service output. The user confirmed data
access authorization; preserve the included PhysioNet license despite the
Kaggle uploader's conflicting MIT label. No API key was used for the completed
public acquisition; future restricted downloads may need credentials.

EHR inputs are limited to the first 48 hours. CXR and report timestamps are
unknown: image associations are exploratory; report text is excluded from the
main benchmark. Keep the mask-only warning and both count variants visible.
See the checkpoint, canary, source-data and report-audit reports for evidence.

Official references: [cluster rules](https://github.com/NTUEEECluster/docs/blob/main/skill.md),
[Slurm](https://github.com/NTUEEECluster/docs/blob/main/slurm-guide.md) and
[storage](https://github.com/NTUEEECluster/docs/blob/main/storage-guide.md).
## Update on 21 October 2026

GPU smoke 215030 passed on L40S in 19 seconds. E2 215538 completed all 60 fits in
15:44. E2 evaluation 217724 completed in 1:22; its preflight passed 46 tests.
Read reports/E2_EVALUATION.pdf or the Markdown report before interpreting E2.
E3 implementation validation 217739 passed 49 tests and dependency checks.

E3 training job 217740 is queued; do not submit a duplicate. It covers frozen
encoder early fusion, joint fine-tuning and a probability-average reference.
Settings and reporting are in docs/E3_EXPERIMENT.md. The original 20 October
queue examples above are historical. Current monitoring commands are:

```bash
squeue --me
squeue --start -j 217740
sacct -j 217740 --format=JobID,JobName,State,Elapsed,ExitCode
tail -f /projects/icumultimodal/reports/icue3-217740.out
```

The two-hour allocation is a runtime limit; actual duration is not measured yet.
Early/joint fusion are checkpointed each epoch. GPU smoke plus E2 consumed about
64.2 SU from allocated seconds; E3 can use at most 480 additional SU on one L40.
The PDF report and aggregate E2 tables/figures are committed; no patient artifacts
were copied locally. An E3 CPU evaluator is prepared to run after training.