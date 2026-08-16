#!/usr/bin/env python
"""Offline open-loop eval of the Cosmos3-Nano G1 wholebody action policy.

Loads the SFT DCP checkpoint (1 GPU) the same way as
``cosmos_framework/scripts/action_policy_server_libero.py`` (OmniInference), then
iterates validation windows of ``G1WholebodyLeRobotDataset`` (same split/val_ratio
as training), predicts action chunks from a single repeated observation frame
(deployment-mimicking), denormalizes (quantile) and reports L1 vs ground truth,
globally and per joint group, against a repeat-last-action baseline.

Joint groups (confirmed in data/G1ToteMix-psi0/meta/modality.json):
  hand = dims 0-13  (left_hand 0-6, right_hand 7-13)
  arm  = dims 14-27 (left_arm 14-20, right_arm 21-27)
  base = dims 28-35 (rpy 28-30, height 31, torso_vx/vy/vyaw 32-34, target_yaw 35)

Outputs metrics.json + per-window plots under psi0_g1/outputs/eval_offline/<run>/.
"""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cosmos_framework.data.generator.action.action_normalization import denormalize_action
from cosmos_framework.data.generator.action.datasets.g1_lerobot_dataset import G1WholebodyLeRobotDataset
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.utils.transforms import (
    build_sequence_plan_from_mode,
    find_closest_target_size,
    reflection_pad_to_target,
)
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    disable_runtime_ema_for_frozen_config,
    maybe_init_distributed,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution

_REPO = Path("/raid/user_marcospaulo/Psi0")
_CF = _REPO / "third_party" / "cosmos-framework"
_DEFAULT_CHECKPOINT = (
    _CF / "psi0_g1/outputs/train/cosmos3_action_g1/action_sft"
    / "action_policy_g1_nano_full_1gpu/checkpoints/iter_000000020000"
)

_DOMAIN_NAME = "g1_wholebody"
_RAW_ACTION_DIM = 36
_JOINT_GROUPS = {"hand": (0, 14), "arm": (14, 28), "base": (28, 36)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(os.environ.get("EVAL_CHECKPOINT_PATH", str(_DEFAULT_CHECKPOINT))),
        help="DCP checkpoint dir (iter_XXXXXXXX). Default: $EVAL_CHECKPOINT_PATH or the full-1gpu 20k iter.",
    )
    p.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Training config.yaml. Default: <checkpoint>/../../config.yaml (job output root).",
    )
    p.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("G1_ROOT", str(_REPO / "data/G1ToteMix-cosmos3-v3-full"))))
    p.add_argument("--output-dir", type=Path, default=_CF / "psi0_g1/outputs/eval_offline")
    p.add_argument("--num-windows", type=int, default=200, help="Validation windows to evaluate (>=200 for the report).")
    p.add_argument("--batch-size", type=int, default=4, help="Windows per diffusion forward.")
    p.add_argument("--chunk-length", type=int, default=16, help="Action chunk size (must match training).")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--val-ratio", type=float, default=0.05, help="Must match training split.")
    p.add_argument("--split-seed", type=int, default=0, help="Episode split seed (dataset default used in training).")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-plots", type=int, default=10)
    p.add_argument("--use-ema-weights", action="store_true", help="Load net_ema (training ran with EMA disabled; keep off).")
    return p.parse_args()


def load_model(args: argparse.Namespace):
    """Load OmniMoTModel from a DCP checkpoint exactly like the policy server."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; submit via Slurm (see sbatch/06_eval_offline.sbatch).")
    maybe_init_distributed()

    config_file = args.config_file or (args.checkpoint_path.parent.parent / "config.yaml")
    if not Path(args.checkpoint_path).exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint_path}")
    if not Path(config_file).exists():
        raise FileNotFoundError(f"training config.yaml not found: {config_file}")

    overrides = OmniSetupOverrides.model_validate(
        {
            "checkpoint_path": str(args.checkpoint_path),
            "config_file": str(config_file),
            "use_ema_weights": bool(args.use_ema_weights),
        }
    )
    overrides.output_dir = args.run_dir / "inference_setup"
    overrides.sampler = "unipc"
    setup_args = overrides.build_setup()
    init_output_dir(setup_args.output_dir)
    setup_args = disable_runtime_ema_for_frozen_config(setup_args)

    log.info(f"[eval-offline] loading model: config='{config_file}' checkpoint='{args.checkpoint_path}'")
    pipe = OmniInference.create(setup_args)
    model = pipe.model
    model.eval()
    return model


def build_json_prompt(
    formatter: ActionPromptJsonFormatter,
    caption: str,
    *,
    video: torch.Tensor,
    image_size: torch.Tensor,
    fps: int,
    chunk_length: int,
    max_action_dim: int,
) -> str:
    """Training-format JSON prompt (format_prompt_as_json=True), mirroring the policy server.

    idle_frames=0: unknown at deployment; the closed-loop server does the same.
    """
    data_dict = {
        "ai_caption": caption,
        "viewpoint": "ego_view",  # G1WholebodyLeRobotDataset viewpoint
        "video": video,
        "image_size": image_size,
        "conditioning_fps": torch.tensor(fps, dtype=torch.long),
        "mode": "wam",
        "action": torch.zeros((chunk_length, max_action_dim), dtype=torch.float32),
        "idle_frames": torch.tensor(0, dtype=torch.long),
    }
    formatted = formatter(data_dict)["ai_caption"]
    return json.dumps(formatted) if isinstance(formatted, dict) else str(formatted)


def window_gt_and_prev(ds: G1WholebodyLeRobotDataset, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw (unnormalized) GT chunk [T,36] and previous executed action [36] (index math mirrors _build_item)."""
    ep = int(np.searchsorted(ds._valid_cum, idx, side="right"))
    prev_cum = int(ds._valid_cum[ep - 1]) if ep > 0 else 0
    ep_first = int(ds._ep_starts[ep])
    start = ep_first + (idx - prev_cum)
    gt = np.asarray(ds._row_action[start : start + ds._chunk_length], dtype=np.float64)
    prev = np.asarray(ds._row_action[start - 1 if start > ep_first else start], dtype=np.float64)
    return gt, prev


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)  # ai_caption uses random.choice over " | " parts

    iter_tag = Path(args.checkpoint_path).name
    args.run_dir = args.output_dir / f"{iter_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    args.run_dir.mkdir(parents=True, exist_ok=True)

    # Raw actions (action_normalization=None) -> GT comes out unnormalized; the
    # model output is denormalized with the same bundled quantile stats used in training.
    ds = G1WholebodyLeRobotDataset(
        root=str(args.dataset_root),
        chunk_length=args.chunk_length,
        image_size=args.image_size,
        mode="wam",
        split="val",
        val_ratio=args.val_ratio,
        seed=args.split_seed,
        action_normalization=None,
    )
    n_avail = len(ds)
    if n_avail == 0:
        raise RuntimeError("Empty validation split.")
    n_windows = min(args.num_windows, n_avail)
    indices = np.unique(np.linspace(0, n_avail - 1, n_windows).astype(np.int64)).tolist()
    log.info(f"[eval-offline] val windows available={n_avail}, evaluating {len(indices)}")

    stats = {k: v.clone() for k, v in ds._load_norm_stats().items()}  # bundled g1_wholebody_stats.json
    fps = int(ds._fps)

    model = load_model(args)
    max_action_dim = int(getattr(model.config, "max_action_dim", 64))
    input_video_key = getattr(model, "input_video_key", None) or model.config.input_video_key
    formatter = ActionPromptJsonFormatter(caption_key="ai_caption")
    sequence_plan = build_sequence_plan_from_mode(
        mode="wam", video_length=args.chunk_length + 1, action_length=args.chunk_length, has_text=True
    )

    preds, gts, prevs, plot_meta = [], [], [], []
    infer_times = []
    t_frames = args.chunk_length + 1

    for batch_start in range(0, len(indices), args.batch_size):
        batch_idx = indices[batch_start : batch_start + args.batch_size]
        items, gt_batch, prev_batch = [], [], []
        for idx in batch_idx:
            try:
                item = ds._build_item(int(idx))  # no silent resampling -> keeps GT aligned
            except Exception as e:  # noqa: BLE001 -- skip undecodable windows
                log.warning(f"[eval-offline] window idx={idx} failed to load ({e}); skipped")
                continue
            gt, prev = window_gt_and_prev(ds, int(idx))
            items.append((int(idx), item))
            gt_batch.append(gt)
            prev_batch.append(prev)
        if not items:
            continue

        n = len(items)
        videos, prompts = [], []
        for _, item in items:
            frame0 = item["video"][:, :1]  # [3,1,H,W] uint8 -- only the current observation is known
            video = frame0.repeat(1, t_frames, 1, 1)
            h, w = video.shape[-2:]
            resolution = get_vision_data_resolution((h, w))
            target_w, target_h = find_closest_target_size(h, w, resolution)
            pad_dict = {"video": video}
            reflection_pad_to_target(pad_dict, ["video"], True, target_w, target_h)
            prompts.append(
                build_json_prompt(
                    formatter,
                    item["ai_caption"],
                    video=pad_dict["video"],
                    image_size=pad_dict["image_size"],
                    fps=fps,
                    chunk_length=args.chunk_length,
                    max_action_dim=max_action_dim,
                )
            )
            videos.append(pad_dict)

        action_t_d = torch.zeros((args.chunk_length, max_action_dim), dtype=torch.float32)
        batch = {
            input_video_key: [[v["video"]] for v in videos],
            **make_batched_action_processing_fields(
                ActionProcessingRecord(raw_action_dim=_RAW_ACTION_DIM, action_normalizer=None), batch_size=n
            ),
            "action": [[action_t_d] for _ in range(n)],
            "mode": ["wam"] * n,
            "ai_caption": prompts,
            "prompt": prompts,
            "conditioning_fps": [torch.tensor(fps, dtype=torch.long) for _ in range(n)],
            "image_size": torch.stack([v["image_size"] for v in videos]).to(device="cuda"),
            "domain_id": [torch.tensor(get_domain_id(_DOMAIN_NAME), dtype=torch.long) for _ in range(n)],
            "sequence_plan": [sequence_plan] * n,
        }

        t0 = time.monotonic()
        with torch.inference_mode():
            samples = model.generate_samples_from_batch(
                batch,
                guidance=args.guidance,
                seed=[args.seed] * n,
                num_steps=args.num_steps,
                has_negative_prompt=False,
            )
        infer_times.append((time.monotonic() - t0) / n)

        for i, (idx, item) in enumerate(items):
            pred = samples["action"][i].float().squeeze(0)[:, :_RAW_ACTION_DIM].cpu()  # [T,36] in [-1,1]
            pred = denormalize_action(pred, "quantile", stats).numpy().astype(np.float64)
            preds.append(pred)
            gts.append(gt_batch[i])
            prevs.append(prev_batch[i])
            if len(plot_meta) < args.num_plots:
                plot_meta.append((idx, item["ai_caption"], len(preds) - 1))
        done = len(preds)
        log.info(f"[eval-offline] {done}/{len(indices)} windows, s/window={infer_times[-1]:.2f}")

    if not preds:
        raise RuntimeError("No window evaluated successfully.")

    pred_arr = np.stack(preds)  # [N,T,36]
    gt_arr = np.stack(gts)
    base_arr = np.repeat(np.stack(prevs)[:, None, :], args.chunk_length, axis=1)  # repeat-last-action

    def l1_report(p: np.ndarray) -> dict:
        err = np.abs(p - gt_arr)
        return {
            "global": float(err.mean()),
            **{g: float(err[..., s:e].mean()) for g, (s, e) in _JOINT_GROUPS.items()},
            "per_dim": err.mean(axis=(0, 1)).tolist(),
        }

    metrics = {
        "checkpoint": str(args.checkpoint_path),
        "config_file": str(args.config_file or (args.checkpoint_path.parent.parent / "config.yaml")),
        "dataset_root": str(args.dataset_root),
        "split": {"name": "val", "val_ratio": args.val_ratio, "seed": args.split_seed},
        "num_windows": int(pred_arr.shape[0]),
        "chunk_length": args.chunk_length,
        "fps": fps,
        "sampling": {"guidance": args.guidance, "num_steps": args.num_steps, "seed": args.seed},
        "joint_groups": {g: list(r) for g, r in _JOINT_GROUPS.items()},
        "l1_model": l1_report(pred_arr),
        "l1_baseline_repeat_last": l1_report(base_arr),
        "mean_inference_s_per_window": float(np.mean(infer_times)),
    }
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"[eval-offline] L1 model={metrics['l1_model']['global']:.4f} "
             f"baseline={metrics['l1_baseline_repeat_last']['global']:.4f} -> {args.run_dir / 'metrics.json'}")

    plots_dir = args.run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    t_axis = np.arange(args.chunk_length) / fps
    for idx, caption, k in plot_meta:
        fig, axes = plt.subplots(len(_JOINT_GROUPS), 1, figsize=(10, 10), sharex=True)
        for ax, (g, (s, e)) in zip(axes, _JOINT_GROUPS.items()):
            for d in range(s, e):
                ax.plot(t_axis, gt_arr[k, :, d], color="C0", lw=0.8, alpha=0.7)
                ax.plot(t_axis, pred_arr[k, :, d], color="C1", lw=0.8, alpha=0.7, ls="--")
            ax.set_ylabel(f"{g} [{s}:{e}]")
            ax.grid(alpha=0.3)
        axes[0].plot([], [], color="C0", label="GT")
        axes[0].plot([], [], color="C1", ls="--", label="pred")
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("t (s)")
        fig.suptitle(f"window {idx}: {caption[:90]}", fontsize=9)
        fig.tight_layout()
        fig.savefig(plots_dir / f"window_{idx:06d}.png", dpi=120)
        plt.close(fig)
    log.info(f"[eval-offline] saved {len(plot_meta)} plots to {plots_dir}")


if __name__ == "__main__":
    main()
