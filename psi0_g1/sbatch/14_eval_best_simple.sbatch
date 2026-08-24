#!/bin/bash
#SBATCH --job-name=cosmos3-g1-simple-best10
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-simple-best10-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-simple-best10-%j.err

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework
: "${BEST_STATE_FILE:?Set BEST_STATE_FILE}"
: "${RUN_ID:?Set RUN_ID}"

export COSMOS_CHECKPOINT_PATH
COSMOS_CHECKPOINT_PATH=$("$CF/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$BEST_STATE_FILE")
export COSMOS_CHUNK_LENGTH=16
export ACTION_EXEC_HORIZON=16
export EPISODES=10
exec bash "$CF/psi0_g1/sbatch/07_serve_and_eval_simple.sbatch"