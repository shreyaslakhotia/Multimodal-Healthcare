#!/bin/bash -l
#SBATCH --job-name=icutests
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/projects/icumultimodal/reports/icutests-%j.out
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
git rev-parse HEAD
"$P/envs/icumm/bin/python" -m pytest -q
"$P/envs/icumm/bin/python" -m pip check
"$P/envs/icumm/bin/python" - <<'PY'
import json, subprocess
from pathlib import Path
import torch, torchvision, torchxrayvision, numpy, sklearn
info = {'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'compiled_cuda': torch.version.cuda, 'numpy': numpy.__version__,
        'sklearn': sklearn.__version__, 'status': 'CPU_ENVIRONMENT_AND_SYNTHETIC_TESTS_PASSED',
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
Path('/projects/icumultimodal/results/setup/environment.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info))
PY
# Environment/cache size accounting runs here on a CPU allocation, not login.
du -sb "$P/envs/icumm" "$P/.tmp/pip" "$P/.tmp/conda" > "$P/results/setup/storagebytes.txt"
touch "$P/results/setup/environmentready"
