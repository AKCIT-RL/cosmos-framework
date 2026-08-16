#!/bin/bash
#SBATCH --job-name=cosmos3-g1-smoke
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-smoke-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-smoke-%j.err

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework
# torchcodec dlopens CUDA NPP libs shipped as pip wheels
export LD_LIBRARY_PATH=$CF/.venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PATH=$HOME/.local/bin:$PATH  # uv (checkpoint_db shells out to 'uv run hf download')
export UV_CACHE_DIR=$REPO/cache/uv
export UV_PYTHON_INSTALL_DIR=$REPO/cache/uv-python

set -a; source "$REPO/secrets/psi0.env"; set +a
export HF_HOME=$REPO/cache/huggingface
export WANDB_DIR=$REPO/cache/wandb
export G1_ROOT=$REPO/data/G1ToteMix-cosmos3-v3-smoke
export BASE_CHECKPOINT_PATH=$CF/psi0_g1/checkpoints/Cosmos3-Nano-dcp
export WAN_VAE_PATH=$CF/psi0_g1/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=$CF/psi0_g1/outputs/train
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$WANDB_DIR" "$IMAGINAIRE_OUTPUT_ROOT"

cd "$CF"

# 1. Convert the base checkpoint to DCP once (CPU path inside the GPU job).
if [[ ! -d "$BASE_CHECKPOINT_PATH" ]]; then
    .venv/bin/python -m cosmos_framework.scripts.convert_model_to_dcp \
        -o "$BASE_CHECKPOINT_PATH" --checkpoint-path Cosmos3-Nano
fi

# 2. 2-GPU FSDP SFT smoke (40 iters) — measures VRAM/it-time and validates the arch.
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
.venv/bin/torchrun --nproc_per_node=2 --master_port=50017 \
    -m cosmos_framework.scripts.train \
    --sft-toml="$CF/psi0_g1/toml/action_policy_g1_nano_smoke.toml"
echo "SMOKE OK"
