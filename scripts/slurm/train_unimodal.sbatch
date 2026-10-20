#!/bin/bash -l
#SBATCH --job-name=icue2
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --time=04:00:00
#SBATCH --output=/projects/icumultimodal/reports/icue2-%j.out
# Submit with exactly one supported GPU model via --gres.
set -euo pipefail
source /projects/icumultimodal/cacheconfig.sh
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$P/.tmp/matplotlib"
export TMPDIR="$P/.tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
cleanup() {
  local target
  target=$(readlink -f "$TMPDIR")
  case "$target" in "$P/.tmp/"[0-9]*) test "$target" = "$P/.tmp/$SLURM_JOB_ID" && rm -rf -- "$target";; esac
}
trap cleanup EXIT
cd "$P/code"
git rev-parse HEAD
"$P/envs/icumm/bin/python" -m pytest -q
"$P/envs/icumm/bin/python" -m icu_mm.train_unimodal
