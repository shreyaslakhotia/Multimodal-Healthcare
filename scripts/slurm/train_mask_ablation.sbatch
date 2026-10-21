#!/bin/bash -l
#SBATCH --job-name=icumasks
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=override-limits-but-killable
#SBATCH --time=02:00:00
#SBATCH --output=/projects/icumultimodal/reports/icumasks-%j.out
set -euo pipefail
source /projects/icumultimodal/cacheconfig.sh
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$P/.tmp/matplotlib"
export TMPDIR="$P/.tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
trap 'test "$(readlink -f "$TMPDIR")" = "$P/.tmp/$SLURM_JOB_ID" && rm -rf -- "$TMPDIR"' EXIT
cd "$P/code"
git rev-parse HEAD
"$P/envs/icumm/bin/python" -m pytest -q
"$P/envs/icumm/bin/python" -m icu_mm.mask_ablation
