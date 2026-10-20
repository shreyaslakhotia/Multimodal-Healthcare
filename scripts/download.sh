#!/bin/bash -l
#SBATCH --job-name=icudata
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/projects/icumultimodal/reports/icudata-%j.out
set -euo pipefail
source /projects/icumultimodal/cacheconfig.sh
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export TMPDIR="$P/.tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
cleanup() {
  local target
  target=$(readlink -f "$TMPDIR")
  case "$target" in "$P/.tmp/"[0-9]*) test "$target" = "$P/.tmp/$SLURM_JOB_ID" && rm -rf -- "$target";; esac
}
trap cleanup EXIT
cd "$P/code"
"$P/envs/icumm/bin/python" -m pytest -q
"$P/envs/icumm/bin/python" -m icu_mm.download
