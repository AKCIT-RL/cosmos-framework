#!/usr/bin/env python
import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the checkpoint with the lowest offline global L1.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = sorted(args.metrics_root.glob(f"{args.checkpoint.name}_*/metrics.json"), key=os.path.getmtime)
    matching = []
    for metrics_path in candidates:
        metrics = json.loads(metrics_path.read_text())
        if Path(metrics["checkpoint"]).resolve() == args.checkpoint.resolve():
            matching.append((metrics_path, metrics))
    if not matching:
        raise RuntimeError(f"No metrics found for {args.checkpoint}")

    metrics_path, metrics = matching[-1]
    score = float(metrics["l1_model"]["global"])
    baseline = float(metrics["l1_baseline_repeat_last"]["global"])
    previous = json.loads(args.state_file.read_text()) if args.state_file.exists() else None
    selected = {
        "checkpoint": str(args.checkpoint.resolve()),
        "iteration": int(args.checkpoint.name.removeprefix("iter_")),
        "l1_global": score,
        "baseline_l1_global": baseline,
        "metrics": str(metrics_path.resolve()),
    }

    if previous is not None and float(previous["l1_global"]) <= score:
        selected = previous
        args.checkpoint.joinpath(".keep").unlink(missing_ok=True)
    else:
        args.checkpoint.joinpath(".keep").touch()
        if previous is not None:
            Path(previous["checkpoint"]).joinpath(".keep").unlink(missing_ok=True)

    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.state_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(selected, indent=2) + "\n")
    temporary.replace(args.state_file)

    latest_file = args.checkpoint.parent / "latest_checkpoint.txt"
    latest_iteration = int(latest_file.read_text().strip().removeprefix("iter_"))
    selected_path = Path(selected["checkpoint"])
    for checkpoint_path in args.checkpoint.parent.glob("iter_*"):
        iteration = int(checkpoint_path.name.removeprefix("iter_"))
        if checkpoint_path.resolve() != selected_path.resolve() and iteration < latest_iteration:
            checkpoint_path.joinpath(".keep").unlink(missing_ok=True)
            shutil.rmtree(checkpoint_path)
            print(f"Pruned evaluated loser: {checkpoint_path}")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()