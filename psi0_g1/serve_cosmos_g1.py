#!/usr/bin/env python
"""HTTP policy server: Cosmos3-Nano G1 wholebody for the SIMPLE closed-loop eval.

Speaks the exact protocol of ``serve_psi0`` (src/psi/deploy/psi0_serve_simple.py),
consumed by SIMPLE's ``Psi0DecoupledWbcAgent`` + ``HttpActionClient``
(third_party/SIMPLE/src/simple/baselines/{psi0_decoupled_wbc,client}.py):

- GET  /health -> {"status": "ok"}
- POST /act    -> request: {"image": {name: np HxWx3 uint8}, "instruction": str,
                  "history": {"reset": true?}, "state": {"states": np (1,32)},
                  "condition": {}, "gt_action": [], "dataset_name": str,
                  "timestamp": str}; numpy values arrive b64-encoded as
                  {"__numpy__": b64, "dtype": descr, "shape": [...]}.
                  response: {"action": np (Ta,36) float, "err": 0.0,
                  "traj_image": np (1,1,3) uint8} (same numpy encoding).

The client applies the returned actions directly in the Psi0 36D convention
(hand 0-13, arm 14-27, base/torso 28-35) -- identical to the G1 dataset action
space used in Cosmos3 SFT, so no joint reordering is needed here.

Model loading/prediction reuses the exact path of psi0_g1/eval_offline_g1.py
(OmniInference over the DCP checkpoint). The proprioceptive state sent by the
client is ignored: the Cosmos3 action policy conditions on video + text only.

Run only on a GPU node via Slurm (see sbatch/07_serve_and_eval_simple.sbatch).
"""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import json
import os
import time
from base64 import b64decode, b64encode
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from numpy.lib.format import descr_to_dtype, dtype_to_descr

from cosmos_framework.data.generator.action.action_normalization import denormalize_action
from cosmos_framework.data.generator.action.datasets.g1_lerobot_dataset import (
    _STAT_KEYS,
    G1WholebodyLeRobotDataset,
)
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


# --- SIMPLE wire format (mirrors third_party/SIMPLE/src/simple/baselines/client.py) ---

def _numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {"__numpy__": b64encode(data).decode(), "dtype": dtype_to_descr(o.dtype), "shape": o.shape}
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _numpy_deserialize(dct):
    if "__numpy__" in dct:
        arr = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return arr.reshape(shape) if (shape := dct["shape"]) else arr[0]
    return dct


def _convert(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {k: _convert(v, func) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert(item, func) for item in data]
    if isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("COSMOS_PORT", "22085")))
    p.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(os.environ.get("COSMOS_CHECKPOINT_PATH", str(_DEFAULT_CHECKPOINT))),
    )
    p.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Training config.yaml; default <checkpoint>/../../config.yaml.",
    )
    p.add_argument(
        "--chunk-length",
        type=int,
        default=int(os.environ.get("COSMOS_CHUNK_LENGTH", "24")),
        help="Generated action chunk (must be %%4==0). Training used 16; 24 extrapolates and "
        "must be validated open-loop first (fallback: 16).",
    )
    p.add_argument(
        "--action-exec-horizon",
        type=int,
        default=None,
        help="Steps returned to the client (<= chunk-length). Default: full chunk.",
    )
    p.add_argument("--image-size", type=int, default=256, help="Square resize, as in G1 training.")
    p.add_argument("--fps", type=int, default=50, help="conditioning_fps; dataset native fps.")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-ema-weights", action="store_true")
    p.add_argument("--no-warmup", action="store_true", help="Skip warmup inference before serving.")
    args = p.parse_args()
    if args.chunk_length % 4 != 0:
        p.error("--chunk-length must be divisible by 4")
    args.action_exec_horizon = args.action_exec_horizon or args.chunk_length
    if args.action_exec_horizon > args.chunk_length:
        p.error("--action-exec-horizon must be <= --chunk-length")
    return args


class CosmosG1Server:
    def __init__(self, args: argparse.Namespace):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; run via Slurm on a GPU node.")
        maybe_init_distributed()
        torch.manual_seed(args.seed)

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
        overrides.output_dir = Path(
            os.environ.get("COSMOS_SERVER_OUTPUT_DIR", str(_CF / "psi0_g1/outputs/serve"))
        ) / time.strftime("%Y%m%d_%H%M%S")
        overrides.sampler = "unipc"
        setup_args = overrides.build_setup()
        init_output_dir(setup_args.output_dir)
        setup_args = disable_runtime_ema_for_frozen_config(setup_args)

        log.info(f"[serve-g1] loading model: config='{config_file}' checkpoint='{args.checkpoint_path}'")
        pipe = OmniInference.create(setup_args)
        self.model = pipe.model
        self.model.eval()

        stats_file = G1WholebodyLeRobotDataset._stats_path()
        raw = json.loads(stats_file.read_text())
        self.stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items() if k in _STAT_KEYS}

        self.args = args
        self.Tp = args.chunk_length
        self.Ta = args.action_exec_horizon
        self.fps = args.fps
        self.max_action_dim = int(getattr(self.model.config, "max_action_dim", 64))
        self.input_video_key = getattr(self.model, "input_video_key", None) or self.model.config.input_video_key
        self.formatter = ActionPromptJsonFormatter(caption_key="ai_caption")
        self.sequence_plan = build_sequence_plan_from_mode(
            mode="wam", video_length=self.Tp + 1, action_length=self.Tp, has_text=True
        )
        self.count = 0
        log.info(
            f"[serve-g1] ready: chunk={self.Tp} exec_horizon={self.Ta} fps={self.fps} "
            f"image_size={args.image_size} guidance={args.guidance} steps={args.num_steps}"
        )

    # --- pre/post processing ------------------------------------------------

    def _frame_to_video(self, image: np.ndarray) -> torch.Tensor:
        """HxWx3 uint8 -> [3, Tp+1, S, S] uint8, single obs frame repeated (as eval_offline_g1)."""
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"expected HxWx3 image, got shape {image.shape}")
        t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        s = self.args.image_size
        t = F.interpolate(t, size=(s, s), mode="bilinear", align_corners=False)  # matches G1 dataset _resize
        frame = (t * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3)  # [3,1,S,S]
        return frame.repeat(1, self.Tp + 1, 1, 1)

    def _build_prompt(self, caption: str, video: torch.Tensor, image_size: torch.Tensor) -> str:
        data_dict = {
            "ai_caption": caption,
            "viewpoint": "ego_view",
            "video": video,
            "image_size": image_size,
            "conditioning_fps": torch.tensor(self.fps, dtype=torch.long),
            "mode": "wam",
            "action": torch.zeros((self.Tp, self.max_action_dim), dtype=torch.float32),
            "idle_frames": torch.tensor(0, dtype=torch.long),
        }
        formatted = self.formatter(data_dict)["ai_caption"]
        return json.dumps(formatted) if isinstance(formatted, dict) else str(formatted)

    def _predict_chunk(self, image: np.ndarray, instruction: str) -> np.ndarray:
        video = self._frame_to_video(image)
        h, w = video.shape[-2:]
        target_w, target_h = find_closest_target_size(h, w, get_vision_data_resolution((h, w)))
        pad_dict = {"video": video}
        reflection_pad_to_target(pad_dict, ["video"], True, target_w, target_h)
        prompt = self._build_prompt(instruction, pad_dict["video"], pad_dict["image_size"])

        batch = {
            self.input_video_key: [[pad_dict["video"]]],
            **make_batched_action_processing_fields(
                ActionProcessingRecord(raw_action_dim=_RAW_ACTION_DIM, action_normalizer=None), batch_size=1
            ),
            "action": [[torch.zeros((self.Tp, self.max_action_dim), dtype=torch.float32)]],
            "mode": ["wam"],
            "ai_caption": [prompt],
            "prompt": [prompt],
            "conditioning_fps": [torch.tensor(self.fps, dtype=torch.long)],
            "image_size": pad_dict["image_size"].unsqueeze(0).to(device="cuda"),
            "domain_id": [torch.tensor(get_domain_id(_DOMAIN_NAME), dtype=torch.long)],
            "sequence_plan": [self.sequence_plan],
        }
        with torch.inference_mode():
            samples = self.model.generate_samples_from_batch(
                batch,
                guidance=self.args.guidance,
                seed=[self.args.seed + self.count],
                num_steps=self.args.num_steps,
                has_negative_prompt=False,
            )
        pred = samples["action"][0].float().squeeze(0)[:, :_RAW_ACTION_DIM].cpu()  # [Tp,36] in [-1,1]
        return denormalize_action(pred, "quantile", self.stats).numpy().astype(np.float32)

    # --- HTTP ----------------------------------------------------------------

    def predict_action(self, payload: dict[str, Any]) -> JSONResponse:
        try:
            req = _convert(payload, _numpy_deserialize)
            image_dict = req["image"]
            instruction = req.get("instruction") or "pick up the blue tote from the shelf and bring it to the table."
            history = req.get("history") or {}
            if "reset" in history:
                log.info("[serve-g1] episode reset")
            image = np.asarray(next(iter(image_dict.values())), dtype=np.uint8)

            t0 = time.monotonic()
            pred = self._predict_chunk(image, instruction)[: self.Ta]  # (Ta,36)
            dt = time.monotonic() - t0
            self.count += 1
            log.info(
                f"[serve-g1] act #{self.count}: instruction='{instruction[:60]}' img={image.shape} "
                f"-> action{pred.shape} in {dt:.2f}s"
            )
            response = {
                "action": pred,
                "err": 0.0,
                "traj_image": np.zeros((1, 1, 3), dtype=np.uint8),  # same placeholder as serve_psi0
            }
            return JSONResponse(content=_convert(response, _numpy_serialize))
        except Exception as e:  # noqa: BLE001 -- report to client like serve_psi0
            import traceback

            log.warning(traceback.format_exc())
            return JSONResponse(content=f'{{"status": "{e}"}}')

    def run(self, host: str, port: int) -> None:
        app = FastAPI()
        app.post("/act")(self.predict_action)
        app.get("/health")(lambda: JSONResponse(content={"status": "ok"}))
        log.info(f"[serve-g1] listening on {host}:{port}")
        uvicorn.run(app, host=host, port=port)


def main() -> None:
    args = parse_args()
    server = CosmosG1Server(args)
    if not args.no_warmup:
        log.info("[serve-g1] warmup inference (dummy frame)...")
        t0 = time.monotonic()
        server._predict_chunk(
            np.zeros((360, 640, 3), dtype=np.uint8),
            "pick up the blue tote from the shelf and bring it to the table.",
        )
        log.info(f"[serve-g1] warmup done in {time.monotonic() - t0:.1f}s")
    server.run(args.host, args.port)


if __name__ == "__main__":
    main()
