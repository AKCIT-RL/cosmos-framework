#!/bin/bash
#SBATCH --job-name=cosmos3-g1-video-upload
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-video-upload-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-video-upload-%j.err

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework
: "${RESULT_RUN_ID:?Set RESULT_RUN_ID to the SIMPLE RUN_ID}"
: "${WANDB_RUN_ID:?Set WANDB_RUN_ID}"
: "${WANDB_RUN_NAME:?Set WANDB_RUN_NAME}"

mapfile -t matches < <(find "$REPO/eval/results" -mindepth 2 -maxdepth 2 -type d -name "$RESULT_RUN_ID")
[[ ${#matches[@]} -eq 1 ]] || { echo "Expected one result for $RESULT_RUN_ID, found ${#matches[@]}" >&2; exit 1; }
RESULT_ROOT=${matches[0]}

set -a; source "$REPO/secrets/psi0.env"; set +a
export WANDB_PROJECT=cosmos3-g1
export WANDB_DIR=$REPO/cache/wandb

"$CF/.venv/bin/python" - "$RESULT_ROOT" "$WANDB_RUN_ID" "$WANDB_RUN_NAME" <<'PY'
import os
import sys
from pathlib import Path

import wandb

result_root = Path(sys.argv[1])
run_id = sys.argv[2]
run_name = sys.argv[3]
stats_path = result_root / "smoke" / "eval_stats.txt"
stats = {}
for line in stats_path.read_text().splitlines():
    if line.startswith("episode_"):
        key, value = line.split(":", 1)
        stats[int(key.removeprefix("episode_"))] = value.strip() == "True"

run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY"),
    project="cosmos3-g1",
    id=run_id,
    resume="allow",
    name=run_name,
    group="simple_intermediate_checkpoints",
    job_type="evaluation",
    config={"result_root": str(result_root), "checkpoint_iteration": 250},
)
table = wandb.Table(columns=["episode", "left_video", "right_video", "success"])
for episode, success in sorted(stats.items()):
    episode_dirs = list((result_root / "smoke").rglob(f"episode_{episode}"))
    if len(episode_dirs) != 1:
        raise RuntimeError(f"Expected one directory for episode {episode}, got {episode_dirs}")
    left = list(episode_dirs[0].glob("head_stereo_left_*.mp4"))
    right = list(episode_dirs[0].glob("head_stereo_right_*.mp4"))
    if len(left) != 1 or len(right) != 1:
        raise RuntimeError(f"Missing stereo videos for episode {episode}")
    table.add_data(episode, wandb.Video(str(left[0]), format="mp4"), wandb.Video(str(right[0]), format="mp4"), success)
run.log({"episodes": table})
run.summary["episodes"] = len(stats)
run.summary["successes"] = sum(stats.values())
run.summary["success_rate"] = sum(stats.values()) / len(stats)
run.summary["upload_complete"] = True
run.finish()
print(f"W&B VIDEO UPLOAD OK: {run_id}")
PY