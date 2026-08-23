#!/bin/bash
#SBATCH --job-name=cosmos3-g1-dcp-prune
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=24:00:00
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-dcp-prune-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-dcp-prune-%j.err

set -euo pipefail
: "${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the checkpoints directory}"
: "${TRAIN_JOB_ID:?Set TRAIN_JOB_ID to the Slurm training job ID}"

while true; do
    state=$(sacct -n -X -j "$TRAIN_JOB_ID" --format=State | head -1 | tr -d ' ')
    latest_file=$CHECKPOINT_ROOT/latest_checkpoint.txt

    if [[ -s "$latest_file" ]]; then
        latest=$(cat "$latest_file")
        latest_path=$CHECKPOINT_ROOT/$latest
        valid=true
        for component in model optim scheduler trainer; do
            [[ -s "$latest_path/$component/.metadata" ]] || valid=false
            find "$latest_path/$component" -name '*.distcp' -type f -size +0c -print -quit | grep -q . || valid=false
        done
        if [[ "$valid" == true ]]; then
            latest_iteration=${latest#iter_}
            for checkpoint_path in "$CHECKPOINT_ROOT"/iter_*; do
                [[ -d "$checkpoint_path" ]] || continue
                checkpoint_iteration=${checkpoint_path##*/iter_}
                if [[ ! -e "$checkpoint_path/.keep" ]] && (( 10#$checkpoint_iteration < 10#$latest_iteration )); then
                    echo "Pruning $checkpoint_path"
                    rm -rf "$checkpoint_path"
                fi
            done
        fi
    fi

    if [[ -n "$state" && "$state" != "PENDING" && "$state" != "RUNNING" ]]; then
        echo "Training $TRAIN_JOB_ID finished with state $state"
        exit 0
    fi
    sleep 60
done