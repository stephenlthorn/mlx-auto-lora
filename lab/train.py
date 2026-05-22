#!/usr/bin/env python3
"""train.py - LoRA fine-tune wrapper around `mlx_lm.lora`.

Translates lab/config.yaml into an mlx-lm LoRA YAML config, then runs training
with a hard wall-clock ceiling. On timeout it sends SIGINT to mlx-lm, which
saves whatever checkpoint it has, then exits successfully. The eval step
scores that partial checkpoint - being cut off mid-run is fine.

The bandit only mutates config.yaml. This file (and eval.py) shouldn't need
editing for experiments.

Usage:
    python lab/train.py --config lab/config.yaml --run-id <id> --max-wall-clock-min 60
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from lab.model_paths import resolve_model_path

ROOT = Path(os.environ.get("MLX_AUTO_LORA_ROOT", Path.home() / "mlx-auto-lora"))

# Default LoRA target keys for Qwen-style hybrid architectures. Attention
# projections only. If empty, mlx-lm picks defaults.
DEFAULT_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj",
    "self_attn.v_proj", "self_attn.o_proj",
]

# Map our shorthand target_modules to mlx-lm key suffixes.
_MODULE_TO_KEY = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def build_mlx_config(cfg: dict, run_id: str, adapter_dir: Path,
                     data_dir: Path) -> dict:
    lora = cfg["lora"]
    optim = cfg["optim"]
    data = cfg["data"]
    train = cfg["training"]

    model_path = resolve_model_path(
        cfg["base_model"], cfg.get("quantization", "mlx8"),
        explicit_path=cfg.get("model_path"))
    rank = int(lora["rank"])
    scale = float(lora["alpha_over_rank"])  # mlx-lm scale = alpha / rank

    keys = [_MODULE_TO_KEY[m] for m in lora.get("target_modules", [])
            if m in _MODULE_TO_KEY]
    if not keys:
        keys = DEFAULT_KEYS

    num_layers = lora.get("num_layers", 16)
    if num_layers == "all":
        num_layers = -1
    else:
        num_layers = int(num_layers)

    mlx_cfg = {
        "model": model_path,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "batch_size": int(train["batch_size"]),
        "iters": int(train["iters"]),
        "learning_rate": float(optim["lr"]),
        "max_seq_length": int(data["max_seq_length"]),
        "grad_checkpoint": bool(train.get("grad_checkpoint", True)),
        "seed": int(train.get("seed", 42)),
        "adapter_path": str(adapter_dir),
        "save_every": 50,
        "steps_per_report": 10,
        "steps_per_eval": 100,
        "lora_parameters": {
            "rank": rank,
            "scale": scale,
            "dropout": float(lora.get("dropout", 0.0)),
            "keys": keys,
        },
    }
    if optim.get("schedule") == "cosine":
        mlx_cfg["lr_schedule"] = {
            "name": "cosine_decay",
            "warmup": int(optim.get("warmup_steps", 0)),
            "arguments": [float(optim["lr"]), int(train["iters"])],
        }
    return mlx_cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "lab" / "config.yaml"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-wall-clock-min", type=float, default=60.0)
    ap.add_argument("--data", default=str(ROOT / "corpus"))
    ap.add_argument("--iters-override", type=int, default=None)
    ap.add_argument("--num-layers-override", type=int, default=None)
    ap.add_argument("--max-seq-override", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.iters_override is not None:
        cfg["training"]["iters"] = args.iters_override
    if args.num_layers_override is not None:
        cfg["lora"]["num_layers"] = args.num_layers_override
    if args.max_seq_override is not None:
        cfg["data"]["max_seq_length"] = args.max_seq_override

    adapter_dir = ROOT / "adapters" / args.run_id
    adapter_dir.mkdir(parents=True, exist_ok=True)

    mlx_cfg = build_mlx_config(cfg, args.run_id, adapter_dir, Path(args.data))
    mlx_cfg_path = adapter_dir / "mlx_lora_config.yaml"
    with open(mlx_cfg_path, "w") as fh:
        yaml.safe_dump(mlx_cfg, fh, sort_keys=False)

    print(f"[train] run_id={args.run_id}")
    print(f"[train] model={mlx_cfg['model']}")
    print(f"[train] iters={mlx_cfg['iters']} batch={mlx_cfg['batch_size']} "
          f"num_layers={mlx_cfg['num_layers']} rank={mlx_cfg['lora_parameters']['rank']}")
    print(f"[train] adapter -> {adapter_dir}")
    print(f"[train] wall-clock ceiling: {args.max_wall_clock_min} min")

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(mlx_cfg_path)]
    deadline = time.time() + args.max_wall_clock_min * 60.0
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=str(ROOT))

    rc: int | None = None
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if time.time() > deadline:
                print(f"[train] wall-clock ceiling hit at "
                      f"{(time.time()-start)/60:.1f} min - terminating, "
                      f"keeping checkpoint")
                proc.send_signal(signal.SIGINT)  # mlx-lm saves on interrupt
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                rc = 0  # treat a budget stop as success (checkpoint saved)
                break
            time.sleep(5)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=120)
        rc = 130

    elapsed = (time.time() - start) / 60.0
    adapter_files = (list(adapter_dir.glob("*adapters*.safetensors")) +
                     list(adapter_dir.glob("adapters.safetensors")))
    ok = bool(adapter_files)
    print(f"[train] done rc={rc} elapsed={elapsed:.1f}min adapter_saved={ok}")
    if not ok:
        print("[train] ERROR: no adapter file produced", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
