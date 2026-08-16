#!/bin/bash
#SBATCH --job-name=cosmos3-g1-full1g
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-full1g-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-full1g-%j.err

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework

export LD_LIBRARY_PATH=$CF/.venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=$REPO/cache/uv
export UV_PYTHON_INSTALL_DIR=$REPO/cache/uv-python

set -a; source "$REPO/secrets/psi0.env"; set +a
export HF_HOME=$REPO/cache/huggingface
export WANDB_DIR=$REPO/cache/wandb
export G1_ROOT=$REPO/data/G1ToteMix-cosmos3-v3-full
export BASE_CHECKPOINT_PATH=$CF/psi0_g1/checkpoints/Cosmos3-Nano-dcp
export WAN_VAE_PATH=$CF/psi0_g1/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=$CF/psi0_g1/outputs/train
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$WANDB_DIR" "$IMAGINAIRE_OUTPUT_ROOT"

cd "$CF"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
.venv/bin/torchrun --nproc_per_node=1 --master_port=50021 \
    -m cosmos_framework.scripts.train \
    --sft-toml="$CF/psi0_g1/toml/action_policy_g1_nano_full_1gpu.toml"
echo "FULL SFT OK"
