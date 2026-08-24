#!/bin/bash
#SBATCH --job-name=cosmos3-g1-upload-best10
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-upload-best10-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-upload-best10-%j.err

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework
: "${BEST_STATE_FILE:?Set BEST_STATE_FILE}"
: "${RESULT_RUN_ID:?Set RESULT_RUN_ID}"

export CHECKPOINT_ITERATION
CHECKPOINT_ITERATION=$("$CF/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["iteration"])' "$BEST_STATE_FILE")
export WANDB_RUN_ID=cosmos3-g1-best-checkpoint-simple10-20260824
export WANDB_RUN_NAME=cosmos3-g1-best-checkpoint-simple10
export WANDB_GROUP=simple_best_checkpoint
exec bash "$CF/psi0_g1/sbatch/12_upload_simple_videos.sbatch"