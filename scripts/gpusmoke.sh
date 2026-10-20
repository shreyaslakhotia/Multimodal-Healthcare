#!/bin/bash -l
#SBATCH --job-name=icusmoke
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --time=00:05:00
#SBATCH --output=/projects/icumultimodal/reports/icusmoke-%j.out
# Submit with one explicit GPU model, selected from live capacity.
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
"$P/envs/icumm/bin/python" - <<'PY'
import json
import torch
from pathlib import Path
from icu_mm.models import CrossAttentionFusion
if not torch.cuda.is_available():
    raise RuntimeError('GPU unavailable in allocated GPU job')
torch.manual_seed(0)
model = CrossAttentionFusion(22, modality_dropout=.15).cuda().train()
inputs = {'context': torch.randn(4,22,device='cuda'),
          'vitals': torch.randn(4,48,16,device='cuda'),
          'labs': torch.randn(4,48,22,device='cuda'),
          'cxr': torch.randn(4,49,1024,device='cuda')}
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for step in range(5):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, torch.tensor([0.,1.,0.,1.],device='cuda'))
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite GPU smoke loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
torch.cuda.synchronize()
info = {'torch':torch.__version__, 'cuda':torch.version.cuda,
        'arch_list':torch.cuda.get_arch_list(), 'device':torch.cuda.get_device_name(),
        'bf16_supported':torch.cuda.is_bf16_supported(), 'steps':5, 'status':'GPU_SMOKE_PASSED'}
Path('/projects/icumultimodal/results/setup/gpu.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info))
PY
