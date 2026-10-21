#!/bin/bash -l
#SBATCH --job-name=icue2eval
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/projects/icumultimodal/reports/icue2eval-%j.out
set -euo pipefail
source /projects/icumultimodal/cacheconfig.sh
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR="$P/.tmp/matplotlib"
export TMPDIR="$P/.tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
trap 'test "$(readlink -f "$TMPDIR")" = "$P/.tmp/$SLURM_JOB_ID" && rm -rf -- "$TMPDIR"' EXIT
cd "$P/code"
git rev-parse HEAD
"$P/envs/icumm/bin/python" -m pytest -q
"$P/envs/icumm/bin/python" -m icu_mm.evaluate_e2
