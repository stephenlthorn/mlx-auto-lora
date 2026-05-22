#!/usr/bin/env python3
"""eval.py - score a LoRA adapter against a held-out prompt set.

Three components:

    compile   hard syntax gate (e.g. swiftc -parse for Swift). Domain-specific.
    lint      soft style check (e.g. swiftlint).                Domain-specific.
    judge     LLM idiomaticity judge (Anthropic Claude if ANTHROPIC_API_KEY).

Each component is in [0, 1]. The composite is the weighted mean over whichever
components were measurable (so if you don't have swiftlint installed, the lint
component is dropped and weights are renormalized; the run is still scored).
This is deliberate: refusing to score when one helper is missing is brittle.

Domain pluggability: `eval.domain` in config.yaml picks a module under
lab/domains/. See lab/domains/swift.py for the reference shape.

Usage:
    python lab/eval.py \\
        --adapter adapters/<run_id> \\
        --out lab/evals/<run_id>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

# Make `lab.*` imports work both when this file is run directly and as a module.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from lab.model_paths import resolve_model_path
from lab import domains as _domains_pkg

ROOT = Path(os.environ.get("MLX_AUTO_LORA_ROOT", Path.home() / "mlx-auto-lora"))
WEIGHTS = {"compile": 0.4, "lint": 0.3, "judge": 0.3}


# ── LLM judge (Anthropic Claude only; add other providers here if desired) ──
def _judge_client(domain_module):
    """Return (kind, callable) for the idiomaticity judge, or (None, None)."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None, None
    try:
        import anthropic
    except ImportError:
        print("[eval] anthropic SDK not installed; judge disabled", file=sys.stderr)
        return None, None
    client = anthropic.Anthropic(api_key=key)
    system = getattr(domain_module, "JUDGE_SYSTEM",
                     "Score 0.0 to 1.0. Reply with ONLY a number.")
    model = os.environ.get("ANTHROPIC_JUDGE_MODEL", "claude-opus-4-7")

    def judge(prompt: str, code: str) -> float:
        msg = client.messages.create(
            model=model,
            max_tokens=8,
            system=system,
            messages=[{
                "role": "user",
                "content": f"PROMPT:\n{prompt}\n\nCODE:\n{code}\n\nScore:",
            }],
        )
        return _parse_score(msg.content[0].text)
    return f"anthropic:{model}", judge


def _parse_score(text: str) -> float:
    m = re.search(r"(\d*\.?\d+)", text or "")
    if not m:
        return 0.5
    v = float(m.group(1))
    return max(0.0, min(1.0, v if v <= 1.0 else v / 100.0))


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(ROOT / "lab" / "config.yaml"))
    ap.add_argument("--prompts", default=None,
                    help="Override prompt set (default: from config eval.prompts)")
    ap.add_argument("--n", type=int, default=None, help="Limit number of prompts")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    eval_cfg = cfg.get("eval", {})
    domain_name = eval_cfg.get("domain", "text")
    domain = _domains_pkg.load(domain_name)

    prompts_path = args.prompts or eval_cfg.get(
        "prompts", str(ROOT / "lab" / "evals" / "quick_prompts.jsonl"))
    prompts = [json.loads(line) for line in open(prompts_path) if line.strip()]
    if args.n:
        prompts = prompts[:args.n]

    model_path = resolve_model_path(
        cfg["base_model"], cfg.get("quantization", "mlx8"),
        explicit_path=cfg.get("model_path"))

    print(f"[eval] domain={domain_name} prompts={len(prompts)} adapter={args.adapter}")
    print(f"[eval] model={model_path}")

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path, adapter_path=args.adapter)
    sampler = make_sampler(
        temp=float(eval_cfg.get("temperature", 0.4)),
        top_p=float(eval_cfg.get("top_p", 0.95)),
    )
    max_tokens = int(eval_cfg.get("max_tokens", 1024))

    judge_kind, judge = _judge_client(domain)
    print(f"[eval] judge: {judge_kind or 'none (recorded null)'}")

    results = []
    t0 = time.time()
    for p in prompts:
        messages = [{"role": "user", "content": p["prompt"]}]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)

        out = generate(model, tokenizer, prompt=prompt_text,
                       max_tokens=max_tokens, sampler=sampler, verbose=False)
        code = domain.extract_code(out)
        comp = domain.compile(code)
        lnt = domain.lint(code)
        jscore = judge(p["prompt"], code) if judge else None
        results.append({
            "id": p["id"], "compile": comp, "lint": lnt,
            "judge": jscore, "chars": len(code),
        })
        print(f"  {p['id']}: compile={comp} lint={lnt} "
              f"judge={jscore if jscore is None else round(jscore, 2)}")

    # Aggregate, renormalizing weights over whichever components were measured.
    def _avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return (sum(vals) / len(vals)) if vals else None

    comp_avg = _avg("compile")
    lint_avg = _avg("lint")
    judge_avg = _avg("judge")
    measured = {"compile": comp_avg, "lint": lint_avg, "judge": judge_avg}
    present = {k: v for k, v in measured.items() if v is not None}
    wsum = sum(WEIGHTS[k] for k in present)
    composite = sum(WEIGHTS[k] * v for k, v in present.items()) / wsum if wsum else 0.0

    summary = {
        "adapter": args.adapter,
        "domain": domain_name,
        "n": len(prompts),
        "composite_score": round(composite, 4),
        "compile_pass": comp_avg,
        "lint_clean": lint_avg,
        "judge_score": judge_avg,
        "judge_kind": judge_kind,
        "measured_components": list(present.keys()),
        "wall_clock_s": round(time.time() - t0, 1),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\n[eval] composite={composite:.4f} "
          f"(compile={comp_avg} lint={lint_avg} judge={judge_avg}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
