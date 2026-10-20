#!/bin/bash -l
#SBATCH --job-name=icusetup
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/projects/icumultimodal/reports/icusetup-%j.out
set -euo pipefail
source /projects/icumultimodal/cacheconfig.sh
export CONDA_ENVS_PATH="$P/envs"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="$P/.tmp/matplotlib"
export TMPDIR="$P/.tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$P/reports" "$P/results/setup"
cleanup() {
  local target
  target=$(readlink -f "$TMPDIR")
  case "$target" in "$P/.tmp/"[0-9]*) test "$target" = "$P/.tmp/$SLURM_JOB_ID" && rm -rf -- "$target";; esac
}
trap cleanup EXIT
trap 'printf "Environment setup failed in job %s; inspect its log before retrying.\n" "$SLURM_JOB_ID" > "$P/reports/BLOCKED.md"' ERR
cd "$P/code"
module load Miniforge3/24.11.3-1
# Lightweight network probes run on this CPU compute node, never the login node.
python - <<'PY'
import urllib.request
for name, url in {
    "kaggle": "https://www.kaggle.com/api/v1/datasets/view/mangeshwagle/mimic-iv-2-1",
    "pypi": "https://pypi.org/simple/numpy/",
    "pytorch": "https://download.pytorch.org/whl/cu128/torch/",
    "weights_host": "https://github.com/mlmed/torchxrayvision/releases",
}.items():
    with urllib.request.urlopen(url, timeout=30) as response:
        print(f"NETWORK {name} {response.status}", flush=True)
PY
ENV="$P/envs/icumm"
if ! test -x "$ENV/bin/python"; then
  conda create -y --override-channels -c conda-forge --no-default-packages -p "$ENV" python=3.11.12 pip=25.1.1
fi
"$ENV/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$ENV/bin/python" -m pip install -r requirements.txt
"$ENV/bin/python" -m pip install --no-deps --no-build-isolation -e .
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze > "$P/results/setup/pipfreeze.txt"
conda list -p "$ENV" --explicit > "$P/results/setup/condaexplicit.txt"
git rev-parse HEAD > "$P/results/setup/codecommit.txt"
"$ENV/bin/python" -m pytest -q
"$ENV/bin/python" - <<'PY'
import json
import torch, torchxrayvision, numpy, sklearn
from pathlib import Path
info = {"torch": torch.__version__, "compiled_cuda": torch.version.cuda,
        "numpy": numpy.__version__, "sklearn": sklearn.__version__,
        "status": "CPU_ENVIRONMENT_READY_GPU_NOT_YET_VERIFIED"}
Path('/projects/icumultimodal/results/setup/environment.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info))
PY
touch "$P/results/setup/environmentready"
