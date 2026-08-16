#!/usr/bin/env python3
"""Build a v3-lite LeRobot layout for the G1ToteMix dataset + action stats.

Re-layouts a LeRobot v2.1 dataset into the minimal "v3" structure that
``cosmos_framework``'s ``ActionBaseDataset``/``G1WholebodyLeRobotDataset`` read:

- ``meta/info.json``            (v3 path templates, native fps)
- ``meta/episodes/chunk-000/file-000.parquet``  (per-episode video/data indices)
- ``meta/tasks.parquet``        (task text + task_index)
- ``data/chunk-000/file-XXX.parquet``           (hardlinks of v2.1 episode parquets)
- ``videos/<key>/chunk-000/file-XXX.mp4``       (hardlinks of v2.1 episode videos)

No video re-encoding: one episode per video file, addressed via the per-episode
``videos/<key>/chunk_index``/``file_index`` columns. Also computes action
normalization stats (mean/std/min/max/q01/q99) over the FULL source dataset and
writes them to the framework's ``normalizer_stats/g1_wholebody_stats.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VIDEO_KEY = "observation.images.egocentric"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="v2.1 dataset root")
    ap.add_argument("--dst", required=True, help="output v3-lite root")
    ap.add_argument("--episodes", type=int, default=20, help="number of episodes (0 = all)")
    ap.add_argument("--stats-out", required=True, help="path for g1_wholebody_stats.json")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    info = json.loads((src / "meta" / "info.json").read_text())
    episodes_meta = [json.loads(line) for line in (src / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    tasks = [json.loads(line) for line in (src / "meta" / "tasks.jsonl").read_text().splitlines() if line]
    chunks_size = int(info.get("chunks_size", 1000))

    all_eps = sorted(int(e["episode_index"]) for e in episodes_meta)
    sel = all_eps if args.episodes <= 0 else all_eps[: args.episodes]
    sel_set = set(sel)
    ep_len = {int(e["episode_index"]): int(e["length"]) for e in episodes_meta}
    print(f"source episodes={len(all_eps)} selected={len(sel)}")

    # ---- stats over the FULL dataset (stable normalization regardless of subset)
    parts = []
    for ep in all_eps:
        p = src / info["data_path"].format(episode_chunk=ep // chunks_size, episode_index=ep)
        parts.append(np.asarray(pq.read_table(p, columns=["action"])["action"].to_pylist(), dtype=np.float32))
    actions = np.concatenate(parts, axis=0)
    print(f"stats over {actions.shape[0]} frames x {actions.shape[1]} dims")
    stats = {
        "metadata": {
            "embodiment_type": "g1_wholebody",
            "action_dim": int(actions.shape[1]),
            "source": str(src),
            "frames": int(actions.shape[0]),
        },
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "q01": np.quantile(actions, 0.01, axis=0).tolist(),
        "q99": np.quantile(actions, 0.99, axis=0).tolist(),
    }
    stats_out = Path(args.stats_out)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(stats, indent=2))
    print(f"wrote stats -> {stats_out}")

    # ---- data + video hardlinks (one v2.1 episode = one v3 file)
    rows = []
    for i, ep in enumerate(sel):
        src_parquet = src / info["data_path"].format(episode_chunk=ep // chunks_size, episode_index=ep)
        link_or_copy(src_parquet, dst / f"data/chunk-000/file-{i:03d}.parquet")
        src_video = src / info["video_path"].format(episode_chunk=ep // chunks_size, episode_index=ep)
        link_or_copy(src_video, dst / f"videos/{VIDEO_KEY}/chunk-000/file-{i:03d}.mp4")
        rows.append(
            {
                "episode_index": ep,
                "length": ep_len[ep],
                "data/chunk_index": 0,
                "data/file_index": i,
                f"videos/{VIDEO_KEY}/chunk_index": 0,
                f"videos/{VIDEO_KEY}/file_index": i,
                f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
            }
        )

    # ---- meta
    ep_dir = dst / "meta/episodes/chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ep_dir / "file-000.parquet", index=False)

    tasks_df = pd.DataFrame([{"task": t["task"], "task_index": int(t["task_index"])} for t in tasks])
    tasks_df.to_parquet(dst / "meta/tasks.parquet", index=False)

    out_info = dict(info)
    out_info["codebase_version"] = "v3.0"
    out_info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    out_info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    out_info["total_episodes"] = len(sel)
    out_info["total_frames"] = int(sum(ep_len[e] for e in sel))
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "meta/info.json").write_text(json.dumps(out_info, indent=2))
    print(f"wrote v3-lite dataset -> {dst} ({len(sel)} eps, {out_info['total_frames']} frames)")

    # ---- sanity: action width per selected episode
    w = np.asarray(
        pq.read_table(dst / "data/chunk-000/file-000.parquet", columns=["action"])["action"].to_pylist(),
        dtype=np.float32,
    ).shape[1]
    assert w == actions.shape[1], f"width mismatch {w} vs {actions.shape[1]}"
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
