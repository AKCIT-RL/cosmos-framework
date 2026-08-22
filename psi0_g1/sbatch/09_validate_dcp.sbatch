#!/bin/bash
#SBATCH --job-name=cosmos3-g1-dcp-check
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-dcp-check-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-dcp-check-%j.err

set -euo pipefail
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the checkpoint directory}"

for component in model optim scheduler trainer; do
    test -s "$CHECKPOINT_PATH/$component/.metadata"
    find "$CHECKPOINT_PATH/$component" -name '*.distcp' -type f -size +0c -print -quit | grep -q .
done

LATEST_FILE=$(dirname "$CHECKPOINT_PATH")/latest_checkpoint.txt
test "$(cat "$LATEST_FILE")" = "$(basename "$CHECKPOINT_PATH")"
echo "DCP VALID: $CHECKPOINT_PATH"