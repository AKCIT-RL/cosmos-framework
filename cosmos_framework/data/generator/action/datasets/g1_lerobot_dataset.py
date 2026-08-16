# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unitree G1 wholebody LeRobot dataset (Psi0 G1ToteMix, single ego camera).

36D absolute joint-space actions straight from the ``action`` feature:
``[left_hand(7), right_hand(7), left_arm(7), right_arm(7), rpy(3), height(1),
torso_vx/vy/vyaw(3), target_yaw(1)]``. Single egocentric camera
(``observation.images.egocentric``), ``quantile``-normalized against bundled
stats. Reads a "v3-lite" LeRobot layout (see ``psi0_g1/prepare_g1_v3_subset.py``):
parquet + per-episode video files addressed through ``meta/episodes`` columns.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.utils.action_spec import (
    ActionSpec,
    Joint,
    Reserved,
    build_action_spec,
)
from cosmos_framework.utils import log

CameraMode = Literal["image"]

_ACTION_FEATURE = "action"
_IMAGE_FEATURE = "observation.images.egocentric"
_STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")
_NORMALIZERS_DIR = Path(__file__).parent.parent / "normalizer_stats"

_G1_ACTION_DIM = 36


class G1WholebodyLeRobotDataset(ActionBaseDataset):
    """Unitree G1 wholebody action-policy dataset (absolute joint-space, ego view)."""

    def __init__(
        self,
        root: str,
        fps: float = 50.0,
        chunk_length: int = 16,
        mode: str = "wam",
        tolerance_s: float = 1e-4,
        camera_mode: CameraMode = "image",
        image_size: int = 256,
        embodiment_type: str = "g1_wholebody",
        action_normalization: str | None = "quantile",
        action_stats_path: str | None = None,
        split: str = "train",
        val_ratio: float = 0.05,
        seed: int = 0,
        sample_stride: int = 1,
    ) -> None:
        if camera_mode != "image":
            raise ValueError(f"G1 dataset has a single egocentric camera; got camera_mode={camera_mode!r}.")
        split = split.lower().strip()
        if split not in {"train", "val", "valid", "validation", "eval", "test", "full"}:
            raise ValueError(f"Unsupported split={split!r}. Use train/val/full.")
        if chunk_length % 4 != 0:
            raise ValueError(f"chunk_length must be divisible by 4, got {chunk_length}.")

        super().__init__(
            root=root,
            domain_name=embodiment_type,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention="backward_framewise",  # unused for joint actions; satisfies the base assert
            tolerance_s=tolerance_s,
            viewpoint="ego_view",
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        info_fps = self._info.get("fps")
        if info_fps:
            if int(info_fps) != int(fps):
                log.info(f"Using dataset native fps={info_fps} for conditioning (requested {fps}).")
            self._fps = float(info_fps)
            self._dt = 1.0 / self._fps
        self._image_size = int(image_size)
        self._video_keys = [_IMAGE_FEATURE]
        self._stats_file = self._resolve_stats_file(action_stats_path)

        # Compact, lazy frame index (mirrors LIBEROLeRobotDataset): only the columns
        # the sample builder needs, ordered by global frame index (COW-shared by workers).
        index_parts, episode_parts, task_parts, ts_parts, action_parts = [], [], [], [], []
        for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path, columns=["index", "episode_index", "task_index", "timestamp", _ACTION_FEATURE])
            index_parts.append(table["index"].to_numpy())
            episode_parts.append(table["episode_index"].to_numpy())
            task_parts.append(table["task_index"].to_numpy())
            ts_parts.append(table["timestamp"].to_numpy())
            action_parts.append(np.asarray(table[_ACTION_FEATURE].to_pylist(), dtype=np.float32))
        if not index_parts:
            raise FileNotFoundError(f"No data parquet found under {self._root / 'data'}.")
        order = np.argsort(np.concatenate(index_parts).astype(np.int64), kind="stable")
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(ts_parts).astype(np.float64)[order]
        self._row_action = np.concatenate(action_parts, axis=0).astype(np.float32)[order]
        if self._row_action.shape[-1] != _G1_ACTION_DIM:
            raise ValueError(f"Expected {_G1_ACTION_DIM}D actions, got {self._row_action.shape[-1]}D.")

        assert np.all(np.diff(self._row_episode) >= 0), "episode_index not contiguous after sorting by frame index"
        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)

        keep = self._split_episode_ids(ep_vals.tolist(), split, val_ratio, seed)
        kept = np.array([int(v) in keep for v in ep_vals], dtype=bool)
        self._ep_vals = ep_vals.astype(np.int64)[kept]
        self._ep_starts = ep_starts.astype(np.int64)[kept]
        kept_counts = ep_counts.astype(np.int64)[kept]
        self._valid_cum = np.cumsum(np.maximum(0, kept_counts - self._chunk_length)).astype(np.int64)

        log.info(
            f"Loaded G1 dataset root={self._root} split={split!r} fps={self._fps} "
            f"kept_episodes={len(self._ep_vals)}/{len(ep_vals)} "
            f"valid_indices={int(self._valid_cum[-1]) if self._valid_cum.size else 0}"
        )

    # ---- spec / dims -------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return _G1_ACTION_DIM

    def _action_spec(self) -> ActionSpec:
        # 28 joint dims (frame-diff idle detection) + 8 torso/base dims (metadata only).
        return build_action_spec(Joint(28, label="joint"), Reserved(8, label="base"))

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZERS_DIR / "g1_wholebody_stats.json"

    def _resolve_stats_file(self, action_stats_path: str | None) -> Path:
        if action_stats_path:
            p = Path(action_stats_path)
            if not p.is_absolute():
                p = _NORMALIZERS_DIR / p.name
            if not p.exists():
                raise FileNotFoundError(f"action_stats_path not found: {action_stats_path!r}")
            return p
        p = self._stats_path()
        if not p.exists():
            raise FileNotFoundError(
                f"Bundled G1 stats not found at {p}. Run psi0_g1/prepare_g1_v3_subset.py or pass action_stats_path."
            )
        return p

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is None:
            raw = json.loads(self._stats_file.read_text())
            self._norm_stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items() if k in _STAT_KEYS}
        return self._norm_stats

    # ---- index helpers -----------------------------------------------------

    @staticmethod
    def _split_episode_ids(ep_ids: list[int], split: str, val_ratio: float, seed: int) -> set[int]:
        if split == "full":
            return set(int(v) for v in ep_ids)
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")
        n_val = max(1, int(round(len(ep_ids) * val_ratio)))
        rng = random.Random(seed)  # identical selection on every rank
        val = set(int(v) for v in rng.sample(list(ep_ids), n_val))
        if split == "train":
            return set(int(v) for v in ep_ids) - val
        return val

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in np.asarray(self._valid_cum).tolist():
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks

    # ---- sample build ------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        n = len(self)
        last_err: Exception | None = None
        for _attempt in range(8):
            try:
                return self._build_item(idx)
            except Exception as e:  # noqa: BLE001 — skip past undecodable frames
                last_err = e
                log.warning(f"G1: sample idx={idx} failed to load ({type(e).__name__}: {e}); resampling")
                if n > 0:
                    idx = random.randint(0, n - 1)
        raise RuntimeError(f"G1: failed to load a sample after 8 resamples; last error: {last_err}")

    def _build_item(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(self._ep_starts[ep]) + (idx - prev)
        episode_index = int(self._ep_vals[ep])
        episode = self._episodes[episode_index]

        stop = start + self._chunk_length + 1
        timestamps = [float(self._row_timestamp[j]) for j in range(start, stop)]
        video = self._load_video(episode, timestamps)

        # Absolute joint-space actions used as stored (normalization in _build_result).
        raw = self._row_action[start : start + self._chunk_length]  # [chunk, 36]
        action = torch.from_numpy(np.ascontiguousarray(raw)).float()

        task = self._tasks[int(self._row_task[start])]
        ai_caption = random.choice([p.strip() for p in task.split(" | ") if p.strip()] or [task])

        return self._build_result(mode=mode, video=video, action=action, ai_caption=ai_caption)

    def _load_video(self, episode: dict[str, Any], timestamps: list[float]) -> torch.Tensor:
        from lerobot.datasets.video_utils import decode_video_frames

        key = self._video_keys[0]
        from_ts = float(episode.get(f"videos/{key}/from_timestamp", 0.0))
        frames = decode_video_frames(
            self._video_path(episode, key),
            [from_ts + ts for ts in timestamps],
            self._tolerance_s,
            backend="pyav",  # torchcodec needs system FFmpeg libs, absent on the DGX nodes
        )  # [T, C, H, W] in [0, 1]
        return self._resize(frames)

    def _resize(self, frames: torch.Tensor) -> torch.Tensor:
        # 360x640 -> square resize (matches LIBERO recipe; revisit pad-to-square later).
        if frames.shape[-1] == self._image_size and frames.shape[-2] == self._image_size:
            return frames
        return F.interpolate(frames, size=(self._image_size, self._image_size), mode="bilinear", align_corners=False)
