#!/usr/bin/env python3
"""prepare_data.py - build a training corpus from local source directories.

Reads `data.sources` from config.yaml (a list of {path, label, extensions}),
walks each directory, filters out generated/vendored noise, chunks each file
to roughly `max_seq_length` tokens worth of characters, and emits
train.jsonl / valid.jsonl in mlx-lm's `{"text": ...}` format.

This is intentionally tiny. If you have something more interesting (Apple
docs, Stack Overflow exports, deduped Common Crawl slices), wire it as a new
loader function and add it to `SLICE_BUILDERS`.

Usage:
    python lab/prepare_data.py --config lab/config.yaml --out corpus/
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("MLX_AUTO_LORA_ROOT", Path.home() / "mlx-auto-lora"))

# Directories we skip when walking source trees - generated, vendored, or noise.
SKIP_DIR_PARTS = {
    ".build", "DerivedData", "Pods", "Carthage", ".git", "node_modules",
    "fastlane", ".swiftpm", "build", "dist", "target", "venv", ".venv",
    "__pycache__", ".tox", ".mypy_cache", ".pytest_cache",
}
SKIP_FILE_HINTS = ("Test", "Tests", "Mock", "Generated", "+Generated")

CHARS_PER_TOKEN = 4  # rough heuristic for chunking to max_seq_length


def _iter_files(repo: Path, extensions: list[str]):
    """Yield files whose extension matches and aren't under a skip directory."""
    exts = {("." + e.lstrip(".")).lower() for e in extensions}
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if set(p.parts) & SKIP_DIR_PARTS:
            continue
        if any(h in p.stem for h in SKIP_FILE_HINTS):
            continue
        yield p


def _chunk_text(text: str, max_chars: int):
    """Yield chunks no larger than max_chars, splitting on blank lines so we
    don't cut mid-declaration where avoidable."""
    if len(text) <= max_chars:
        yield text
        return
    buf, size = [], 0
    for block in text.split("\n\n"):
        b = block + "\n\n"
        if size + len(b) > max_chars and buf:
            yield "".join(buf).rstrip()
            buf, size = [], 0
        buf.append(b)
        size += len(b)
        while size > max_chars:
            joined = "".join(buf)
            yield joined[:max_chars]
            rest = joined[max_chars:]
            buf, size = ([rest], len(rest)) if rest else ([], 0)
    if buf:
        yield "".join(buf).rstrip()


def build_local_repos(sources: list[dict], max_chars: int,
                      max_files_per_source: int | None) -> list[dict]:
    """Default loader: read source files from local directories."""
    examples: list[dict] = []
    for src in sources:
        label = src.get("label") or Path(src["path"]).name
        repo = Path(os.path.expanduser(src["path"]))
        if not repo.exists():
            print(f"  [warn] {label}: {repo} not found, skipping", file=sys.stderr)
            continue
        exts = src.get("extensions") or [".swift"]
        files = list(_iter_files(repo, exts))
        if max_files_per_source:
            files = files[:max_files_per_source]
        n = 0
        for f in files:
            try:
                source_text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if len(source_text.strip()) < 40:
                continue
            rel = f.relative_to(repo)
            for chunk in _chunk_text(source_text, max_chars):
                if len(chunk.strip()) < 40:
                    continue
                header = f"// Source: {label}\n// File: {rel}\n\n"
                examples.append({
                    "text": header + chunk, "_slice": "local_repos", "_label": label,
                })
                n += 1
        print(f"  {label}: {len(files)} files -> {n} examples")
    return examples


# Add your own slice loaders here. Each returns list[dict] with at least a
# "text" key. Wire them into SLICE_BUILDERS and reference by name in config
# `data.mix_weights`.
SLICE_BUILDERS = {
    "local_repos": build_local_repos,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "lab" / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "corpus"))
    ap.add_argument("--max-files-per-source", type=int, default=None)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    max_seq = int(cfg["data"]["max_seq_length"])
    max_chars = max_seq * CHARS_PER_TOKEN
    weights = cfg["data"].get("mix_weights", {"local_repos": 1.0})
    sources = cfg["data"].get("sources", [])

    if not sources:
        print(
            "ERROR: config.data.sources is empty. Add a list of {path, label, "
            "extensions} entries pointing at the directories you want to learn "
            "from.", file=sys.stderr,
        )
        return 1

    print(f"Building corpus (max_seq={max_seq}, ~{max_chars} chars/example)")
    all_examples: list[dict] = []
    for slice_name, w in weights.items():
        if w <= 0:
            continue
        builder = SLICE_BUILDERS.get(slice_name)
        if not builder:
            print(f"  [warn] no builder for slice '{slice_name}'", file=sys.stderr)
            continue
        print(f"slice '{slice_name}' (weight {w}):")
        ex = builder(sources, max_chars, args.max_files_per_source)
        all_examples.extend(ex)

    if not all_examples:
        print("ERROR: no examples produced. Check sources / mix_weights.", file=sys.stderr)
        return 1

    random.seed(args.seed)
    random.shuffle(all_examples)
    n_valid = max(1, int(len(all_examples) * args.valid_frac))
    valid = all_examples[:n_valid]
    train = all_examples[n_valid:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        path = out / f"{name}.jsonl"
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps({"text": r["text"]}) + "\n")
        print(f"wrote {path}  ({len(rows)} examples)")

    by_label: dict[str, int] = {}
    for r in all_examples:
        key = r.get("_label", r.get("_slice", "?"))
        by_label[key] = by_label.get(key, 0) + 1
    total_chars = sum(len(r["text"]) for r in all_examples)
    print("\n=== CORPUS STATS ===")
    print(f"total examples: {len(all_examples)}  (train {len(train)} / valid {len(valid)})")
    print(f"approx tokens:  {total_chars // CHARS_PER_TOKEN:,}")
    for k, v in sorted(by_label.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
