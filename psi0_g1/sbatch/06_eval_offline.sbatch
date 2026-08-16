#!/bin/bash
#SBATCH --job-name=cosmos3-g1-evaloff
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-evaloff-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-evaloff-%j.err

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework

export LD_LIBRARY_PATH=$CF/.venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=$REPO/cache/uv
export UV_PYTHON_INSTALL_DIR=$REPO/cache/uv-python

set -a; source "$REPO/secrets/psi0.env"; set +a
export WANDB_PROJECT=cosmos3-g1  # projeto W&B dedicado ao Cosmos
export HF_HOME=$REPO/cache/huggingface
export WANDB_DIR=$REPO/cache/wandb
export G1_ROOT=$REPO/data/G1ToteMix-cosmos3-v3-full
export BASE_CHECKPOINT_PATH=$CF/psi0_g1/checkpoints/Cosmos3-Nano-dcp
export WAN_VAE_PATH=$CF/psi0_g1/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=$CF/psi0_g1/outputs/train
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export EVAL_CHECKPOINT_PATH=${EVAL_CHECKPOINT_PATH:-$CF/psi0_g1/outputs/train/cosmos3_action_g1/action_sft/action_policy_g1_nano_full_1gpu/checkpoints/iter_000020000}

cd "$CF"
export PYTHONPATH=$CF

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
.venv/bin/python psi0_g1/eval_offline_g1.py \
    --checkpoint-path "$EVAL_CHECKPOINT_PATH" \
    --num-windows 200 \
    --batch-size 4
echo "EVAL OFFLINE OK"
