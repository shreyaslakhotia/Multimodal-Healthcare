# Repository and cluster run guide

Permanent operating guide. Check [current status](STATUS.md) for active job IDs
and [the execution log](../reports/LOG.md) for history. Do not rerun completed
stages merely because they appear below.

## Connect

Use NTUSECURE or the NTU VPN, then connect from PowerShell:

```powershell
ssh -i C:\Users\lakho\.ssh\icubenchmark -o IdentitiesOnly=yes shreyas010@10.97.216.128
```

In the cluster shell:

```bash
source /projects/icumultimodal/cacheconfig.sh
cd "$P/code"
```

## Watch current work

After SSH, refresh the queue every 30 seconds:

```bash
watch -n 30 'squeue --me'
```

For accounting, including completed/failed jobs, and estimated starts:

```bash
sacct -j 217883,217932,217933 --format=JobID,JobName,State,Elapsed,ExitCode
squeue --start -j 217883,217932
```

Once training starts, its log appears:

```bash
tail -f /projects/icumultimodal/reports/icue3-217883.out
tail -f /projects/icumultimodal/reports/icumasks-217932.out
```

Ctrl+C stops watching without cancelling jobs. Batch jobs/dependencies survive
SSH disconnection. Priority means waiting for scheduling; Dependency means a
prerequisite has not succeeded. Start estimates can move or be N/A. Cluster
times are UTC; Singapore is UTC+8. A time limit is not an ETA. An empty queue
alone is not proof of success: verify COMPLETED, exit 0:0 and the completion
artifact. Inspect failures before resubmission. Replace IDs for future runs.

The active GPU jobs use `override-limits-but-killable`: no charged SU, but regular
jobs can preempt/requeue them. Checkpoints are saved every epoch. Keep one copy
of each experiment. Resource/QOS changes require cancellation and resubmission;
never retry a rejection to bypass policy. A completed dependency can age out
of the controller even when sacct retains its record; use a verified fresh gate
when a dependency is needed.

## Storage and caches

| Location | Purpose |
|---|---|
| `/projects/icumultimodal` | SSD project, 100 GB assigned |
| `/projects/icumultimodalarchive` | HDD project, 200 GB assigned |
| `$P/code` | Clean source checkout |
| `$P/raw` | Required extracted tables and CXR metadata |
| `$P/data/processed` | Private cohorts, fixed partitions and arrays |
| `$P/data/interim` | Private intermediate data |
| `$P/envs/icumm` | Pinned environment |
| `$P/.tmp` | SSD caches and job scratch |
| `$P/results` | Models, predictions, checkpoints and metrics |
| `$P/reports` | Slurm logs and generated reports/figures |
| `/projects/icumultimodalarchive/zips` | Original archives and backups |

Account allowances are 150 GB SSD and 300 GB HDD; assigned project limits govern.
Roots are symlinks; inspect targets with `stat -L` (2700: private, setgid).
Cluster storage is not backed up. Use SSD for repeated reads.

For quota usage, run `storagemgr` interactively; press q to exit. It may not be
available in an uninitialized shell. `df` shows shared filesystem capacity,
not your quota. For fast directory metadata without a file scan:

```bash
for p in /projects/icumultimodal /projects/icumultimodalarchive; do
  b=$(getfattr --only-values -n ceph.dir.rbytes "$p" 2>/dev/null) || continue
  awk -v p="$p" -v b="$b" 'BEGIN {printf "%s: %.2f GB\n", p, b/1e9}'
done
```

These are stored file bytes, including caches/environment/results, not cumulative
downloads. Query `$P/raw` or HDD `zips` similarly to isolate source copies.
Archives and extracted tables overlap. Counters can update asynchronously;
storagemgr is the quota reference. Stop at 90 GB SSD usage under this protocol.

Every job sources `$P/cacheconfig.sh`, also sourced by .bashrc, before installs
or downloads. All caches and TMPDIR point to SSD. Each job uses
`$P/.tmp/$SLURM_JOB_ID` and validates cleanup targets. XRV needs explicit
`cache_dir=$P/.tmp/torch/xrv`. Nothing large goes into HOME or `/tmp`.

## Execute a stage

The [entrypoint catalog](../scripts/slurm/README.md) lists prerequisites. Submit
only the required stage after checking live jobs; these are future-run examples:

```bash
source /projects/icumultimodal/cacheconfig.sh
cd "$P/code"
sbatch scripts/slurm/check_cpu.sbatch
sbatch --gres=gpu:a6000:1 scripts/slurm/check_gpu.sbatch
sbatch --qos=override-limits-but-killable --gres=gpu:a6000:1 scripts/slurm/train_fusion.sbatch
```

Do not duplicate current submissions. Use live sinfo to choose exactly one GPU
model. CPUs/RAM accompany the GPU; do not set GPU-job CPU/memory overrides. The
gh200 is ARM and incompatible with this environment. No sudo, salloc, exclusive
nodes or self-installed toolchains. Keep at most two SSH sessions, two running
jobs and five submitted jobs. Login nodes are for light editing, metadata and
submission. Installs, tests, processing and training run in Slurm. No recursive
du/find/grep over dataset trees; use manifests and ZIP central directories.

Environment: Python 3.11.12, torch 2.8.0+cu128, torchvision 0.23.0+cu128.
One-time setup installs requirements and the editable package. Resolved records:
`$P/results/setup/pipfreeze.txt` and `condaexplicit.txt`.

Normal GPU SU = GPUs × allocated minutes × model weight. CPU work is free.
The stage cap is 30,000 SU, with a mandatory stop at 21,000. Track
`$P/results/setup/gpujobs.txt` and sacct, not monthly use alone. Pending and
cancelled-before-start jobs add no runtime. Free-QOS runtime still consumes
resources; record it separately from charged SU.

## Synchronize and publish

Local authoritative repo:
`C:/Users/lakho/Desktop/NTU/Multimodal Healthcare Project/icubenchmark`.
Develop on `shreyas`; do not push directly to `main`.

```bash
git status --short
git diff --check
git add <reviewed-files>
git commit -m "Describe the change"
git push origin shreyas
```

For the cluster, create a bundle of HEAD, transfer to `$P/.tmp`, then fetch and
fast-forward the clean checkout. Its local branch name may differ; the commit
and signatures matter. Do not change training code during execution.
Fingerprint-protected modules/entrypoints keep stable paths and contents to
preserve cached results and resume behavior. Record source revisions in logs.

Use the dedicated SSH identity explicitly; never print its private key or put
credentials in command arguments. Non-secret scripts may be base64 piped to
`bash -l` to avoid Windows SSH quoting errors. No credentials go to GitHub.

## Data and reporting boundary

Export aggregate metrics/figures only. Identifiers, rows, images, report text,
image paths, embeddings, predictions and checkpoints stay on the cluster outside
Git and online AI-service output. Preserve provenance in the [data report](../reports/DATA_REPORT.md).

Final folds are fixed. Imputation, normalization, vocabularies, PCA and class
weights use inner training only. Validation selects checkpoints and thresholds;
test outcomes select neither. Do not regenerate folds after viewing results.
Unknown CXR timing and the mask warning remain active. Text stays excluded.
Do not fetch PhysioNet metadata without the user's instruction.

Official guides: [rules](https://github.com/NTUEEECluster/docs/blob/main/skill.md),
[Slurm](https://github.com/NTUEEECluster/docs/blob/main/slurm-guide.md),
[storage](https://github.com/NTUEEECluster/docs/blob/main/storage-guide.md).
