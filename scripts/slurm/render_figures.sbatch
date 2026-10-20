#!/bin/bash -l
#SBATCH --job-name=icufigures
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/projects/icumultimodal/reports/icufigures-%j.out
set -euo pipefail
source /projects/icumultimodal/cacheconfig.sh
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
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
"$P/envs/icumm/bin/python" -m icu_mm.figures
