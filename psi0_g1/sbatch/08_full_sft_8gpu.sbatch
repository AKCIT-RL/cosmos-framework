#!/bin/bash
#SBATCH --job-name=cosmos3-g1-8gpu
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-8gpu-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-8gpu-%j.err

# Recipe-style SFT (moe_gen unfrozen, FSDP shard 8). TOML selecionavel por env:
#   SFT_TOML=.../action_policy_g1_nano_smoke_8gpu.toml sbatch ...  (smoke)
#   sbatch ...                                                      (full)

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework
SFT_TOML=${SFT_TOML:-$CF/psi0_g1/toml/action_policy_g1_nano_full_8gpu.toml}
NPROC=${NPROC:-8}

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
export SLURM_LOG_DIR=$REPO/logs/cosmos3-g1-${SLURM_JOB_ID}
mkdir -p "$WANDB_DIR" "$IMAGINAIRE_OUTPUT_ROOT" "$SLURM_LOG_DIR"
rm -f "$SLURM_LOG_DIR/SIGUSR1_RECEIVED" "$SLURM_LOG_DIR/SIGUSR1_CHECKPOINT_SUCCEEDED"
trap 'touch "$SLURM_LOG_DIR/SIGUSR1_RECEIVED"' USR1

cd "$CF"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
.venv/bin/torchrun --nproc_per_node="$NPROC" --master_port=50022 \
    -m cosmos_framework.scripts.train \
    --sft-toml="$SFT_TOML" -- checkpoint.dcp_async_mode_enabled=true &
TRAIN_PID=$!
while true; do
    set +e
    wait "$TRAIN_PID"
    TRAIN_STATUS=$?
    set -e
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        break
    fi
done
(( TRAIN_STATUS == 0 ))
echo "SFT 8GPU OK ($SFT_TOML)"
