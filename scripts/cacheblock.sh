# Source from ~/.bashrc after storage links and quotas are verified.
# Also source explicitly in every batch job before any install or download.
export P=/projects/icumultimodal
export TMPDIR="$P/.tmp"
export PIP_CACHE_DIR="$P/.tmp/pip"
export CONDA_PKGS_DIRS="$P/.tmp/conda"
export HF_HOME="$P/.tmp/hf"
export TRITON_CACHE_DIR="$P/.tmp/triton"
export XDG_CACHE_HOME="$P/.tmp/cache"
export TORCH_HOME="$P/.tmp/torch"
umask 077
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME"
