"""model_paths.py - resolve config base_model/quantization to a local model path.

Two modes:

1. `model_path:` set in config.yaml - returned as-is. Use this when your model
   lives outside the HuggingFace cache (e.g. you converted weights yourself).

2. `base_model:` + optional `quantization:` - we look up a HuggingFace repo ID
   in `_SHORTHAND` (or accept a full `org/name` ID directly), then point at the
   latest snapshot in `~/.cache/huggingface/hub`. If the model isn't downloaded
   yet, we return the HF ID directly so mlx-lm's `load()` will fetch it on the
   fly.

Add your favourite models to `_SHORTHAND` or just use the full HF repo ID.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))

# Shorthand `(base_model, quantization)` -> HuggingFace repo ID.
# Add your own. The bundled list is just enough to demo.
_SHORTHAND = {
    ("qwen3-30b-a3b", "mlx8"): "mlx-community/Qwen3-30B-A3B-Instruct-2507-8bit",
    ("qwen3-30b-a3b", "mlx4"): "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    ("qwen3-14b", "mlx8"): "mlx-community/Qwen3-14B-8bit",
    ("qwen3-14b", "mlx4"): "mlx-community/Qwen3-14B-4bit",
    ("qwen2.5-7b", "mlx8"): "mlx-community/Qwen2.5-7B-Instruct-8bit",
    ("qwen2.5-7b", "mlx4"): "mlx-community/Qwen2.5-7B-Instruct-4bit",
    ("llama3.1-8b", "mlx8"): "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
    ("llama3.1-8b", "mlx4"): "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
}


def resolve_model_path(base_model: str, quantization: str = "mlx8",
                       explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path

    repo_id = _SHORTHAND.get((base_model, quantization))
    if repo_id is None:
        # Allow direct HF repo IDs like "mlx-community/Foo-7B-4bit" through.
        if "/" in base_model:
            repo_id = base_model
        else:
            known = sorted(_SHORTHAND)
            raise ValueError(
                f"No shorthand for ({base_model!r}, {quantization!r}). "
                f"Either use a full HuggingFace repo ID or add an entry to "
                f"_SHORTHAND in lab/model_paths.py.\nKnown shorthands: {known}"
            )

    cache_dir = _HUB / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    snaps = sorted(glob.glob(str(cache_dir / "*")))
    if snaps:
        return snaps[-1]
    # Not cached locally yet - let mlx-lm fetch on the fly.
    return repo_id


if __name__ == "__main__":
    import sys
    bm = sys.argv[1] if len(sys.argv) > 1 else "qwen3-30b-a3b"
    q = sys.argv[2] if len(sys.argv) > 2 else "mlx8"
    print(resolve_model_path(bm, q))
