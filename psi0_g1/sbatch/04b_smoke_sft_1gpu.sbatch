#!/bin/bash
#SBATCH --job-name=cosmos3-g1-smoke1g
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-smoke1g-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-smoke1g-%j.err

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework

# Race guard vs the 2-GPU smoke: if it is running or already done, skip; else
# cancel its pending copy so we don't run the smoke twice.
if squeue -h -u "$USER" -n cosmos3-g1-smoke -t R | grep -q .; then
    echo "2-GPU smoke already running; exiting."; exit 0
fi
if sacct -n -X --name=cosmos3-g1-smoke --state=COMPLETED --starttime=today 2>/dev/null | grep -q COMPLETED; then
    echo "2-GPU smoke already completed today; exiting."; exit 0
fi
squeue -h -u "$USER" -n cosmos3-g1-smoke -t PD -o %i | xargs -r scancel

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

if [[ ! -d "$BASE_CHECKPOINT_PATH" ]]; then
    .venv/bin/python -m cosmos_framework.scripts.convert_model_to_dcp \
        -o "$BASE_CHECKPOINT_PATH" --checkpoint-path Cosmos3-Nano
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
.venv/bin/torchrun --nproc_per_node=1 --master_port=50019 \
    -m cosmos_framework.scripts.train \
    --sft-toml="$CF/psi0_g1/toml/action_policy_g1_nano_smoke_1gpu.toml"
echo "SMOKE OK"
