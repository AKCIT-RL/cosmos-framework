#!/bin/bash
#SBATCH --job-name=cosmos3-g1-select-best
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-select-best-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-select-best-%j.err

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"
: "${BEST_STATE_FILE:?Set BEST_STATE_FILE}"

"$CF/.venv/bin/python" "$CF/psi0_g1/select_best_checkpoint.py" \
    --checkpoint "$CHECKPOINT_PATH" \
    --metrics-root "$CF/psi0_g1/outputs/eval_offline" \
    --state-file "$BEST_STATE_FILE"