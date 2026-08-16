#!/bin/bash
#SBATCH --job-name=cosmos3-g1-env
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-env-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-env-%j.err

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework

export UV_CACHE_DIR=$REPO/cache/uv
export UV_PYTHON_INSTALL_DIR=$REPO/cache/uv-python
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

if ! command -v uv >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

cd "$CF"
# CUDA 13 driver (580.159.03) -> cu130 train group
uv sync --all-extras --group=cu130-train
.venv/bin/python -c "import torch, lerobot, pandas, pyarrow; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "ENV OK"
