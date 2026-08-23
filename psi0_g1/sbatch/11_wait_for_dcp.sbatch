#!/bin/bash
#SBATCH --job-name=cosmos3-g1-wait-dcp
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=06:00:00
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-wait-dcp-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-wait-dcp-%j.err

set -euo pipefail
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the target checkpoint directory}"
: "${TRAIN_JOB_ID:?Set TRAIN_JOB_ID to the Slurm training job ID}"

while true; do
    state=$(sacct -n -X -j "$TRAIN_JOB_ID" --format=State | head -1 | tr -d ' ')
    valid=true
    for component in model optim scheduler trainer; do
        [[ -s "$CHECKPOINT_PATH/$component/.metadata" ]] || valid=false
        find "$CHECKPOINT_PATH/$component" -name '*.distcp' -type f -size +0c -print -quit 2>/dev/null | grep -q . || valid=false
    done
    latest_file=$(dirname "$CHECKPOINT_PATH")/latest_checkpoint.txt
    [[ -s "$latest_file" ]] || valid=false
    [[ "$(cat "$latest_file" 2>/dev/null || true)" == "$(basename "$CHECKPOINT_PATH")" ]] || valid=false

    if [[ "$valid" == true ]]; then
        touch "$CHECKPOINT_PATH/.keep"
        echo "DCP READY AND PRESERVED: $CHECKPOINT_PATH"
        exit 0
    fi
    if [[ -n "$state" && "$state" != "PENDING" && "$state" != "RUNNING" ]]; then
        echo "Training $TRAIN_JOB_ID finished with state $state before publishing $CHECKPOINT_PATH" >&2
        exit 1
    fi
    sleep 60
done