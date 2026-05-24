#!/usr/bin/env python3
"""eval.py - score a LoRA adapter against a held-out prompt set.

Three signals, but they are NOT equal:

    compile   HARD GATE (e.g. swiftc -parse). A run that stops producing
              parseable code is a regression no matter how good it looks, so
              compile is a pass/fail gate (compile_gate_pass), NOT a score
              weight. (A base model already emits syntactically-valid code, so
              folding compile into the score just pins ~40% of the weight at
              1.0 and drowns the real signal.)
    lint      soft style check (e.g. swiftlint). Weighted only when available.
    judge     LLM idiomaticity judge (Anthropic Claude if ANTHROPIC_API_KEY).
              This is the component that actually discriminates, so it carries
              the composite.

composite = judge                         (lint unavailable)
          = 0.7*judge + 0.3*lint          (lint available)

Determinism + noise: generation is stochastic, so a single sample per prompt
cannot tell a real improvement from sampler noise. We seed Python + MLX and
draw `--samples` completions per prompt (deterministic per (prompt, sample)),
then report `composite_std` so the orchestrator can apply a noise floor before
keeping a "winner". The judge runs at temperature 0.

Domain pluggability: `eval.domain` in config.yaml picks a module under
lab/domains/. See lab/domains/swift.py for the reference shape.

Usage:
    python lab/eval.py \\
        --adapter adapters/<run_id> \\
        --out lab/evals/<run_id>.json [--seed 42] [--samples 3]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
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

# Compile is a GATE, not a score weight. The composite is judge-driven; lint
# contributes only when a linter is installed. Weights are renormalized over
# whichever of {judge, lint} were measurable.
WEIGHTS = {"judge": 0.7, "lint": 0.3}
COMPILE_GATE_THRESHOLD = float(os.environ.get("MLX_AUTO_LORA_COMPILE_GATE", "1.0"))


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
            temperature=0.0,  # deterministic grading
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


def _stdev(xs: list[float]) -> float:
    """Sample standard deviation; 0.0 for fewer than two points."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def _composite_from(judge_v, lint_v) -> float | None:
    """Judge-driven composite, renormalized over measured components.

    Compile is intentionally excluded (it's a gate, handled separately).
    """
    measured = {"judge": judge_v, "lint": lint_v}
    present = {k: v for k, v in measured.items() if v is not None}
    wsum = sum(WEIGHTS[k] for k in present)
    if not wsum:
        return None
    return sum(WEIGHTS[k] * v for k, v in present.items()) / wsum


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(ROOT / "lab" / "config.yaml"))
    ap.add_argument("--prompts", default=None,
                    help="Override prompt set (default: from config eval.prompts)")
    ap.add_argument("--n", type=int, default=None, help="Limit number of prompts")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base seed for reproducible generation")
    ap.add_argument("--samples", type=int, default=3,
                    help="Completions per prompt (averaged; drives variance)")
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

    samples = max(1, int(args.samples))
    print(f"[eval] domain={domain_name} prompts={len(prompts)} "
          f"samples={samples} seed={args.seed} adapter={args.adapter}")
    print(f"[eval] model={model_path}")

    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    # Seed everything up front for reproducibility.
    random.seed(args.seed)
    mx.random.seed(args.seed)

    model, tokenizer = load(model_path, adapter_path=args.adapter)
    sampler = make_sampler(
        temp=float(eval_cfg.get("temperature", 0.4)),
        top_p=float(eval_cfg.get("top_p", 0.95)),
    )
    max_tokens = int(eval_cfg.get("max_tokens", 1024))

    judge_kind, judge = _judge_client(domain)
    print(f"[eval] judge: {judge_kind or 'none (recorded null)'}")

    results = []
    all_judge_scores: list[float] = []
    all_compile_flags: list[int] = []
    per_prompt_composites: list[float] = []
    t0 = time.time()

    for p_idx, p in enumerate(prompts):
        messages = [{"role": "user", "content": p["prompt"]}]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)

        judge_samples: list[float] = []
        composite_samples: list[float] = []
        compile_samples: list[int] = []
        lint_samples: list[int] = []
        for s_idx in range(samples):
            # Deterministic, distinct seed per (prompt, sample).
            mx.random.seed(args.seed + s_idx * 10_000 + p_idx)
            out = generate(model, tokenizer, prompt=prompt_text,
                           max_tokens=max_tokens, sampler=sampler, verbose=False)
            code = domain.extract_code(out)
            comp = domain.compile(code)
            lnt = domain.lint(code)
            jscore = judge(p["prompt"], code) if judge else None

            if comp is not None:
                compile_samples.append(comp)
                all_compile_flags.append(comp)
            if lnt is not None:
                lint_samples.append(lnt)
            if jscore is not None:
                judge_samples.append(jscore)
                all_judge_scores.append(jscore)
            c = _composite_from(jscore, lnt)
            if c is not None:
                composite_samples.append(c)

        def _mean(xs):
            return (sum(xs) / len(xs)) if xs else None

        prompt_composite = _mean(composite_samples)
        if prompt_composite is not None:
            per_prompt_composites.append(prompt_composite)
        results.append({
            "id": p["id"],
            "compile_mean": _mean(compile_samples),
            "lint_mean": _mean(lint_samples),
            "judge_mean": _mean(judge_samples),
            "composite_mean": prompt_composite,
            "judge_samples": judge_samples,
            "composite_samples": composite_samples,
        })
        print(f"  {p['id']}: compile={_mean(compile_samples)} "
              f"lint={_mean(lint_samples)} "
              f"judge={None if not judge_samples else round(_mean(judge_samples), 2)}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    def _mean_or_none(xs):
        return (sum(xs) / len(xs)) if xs else None

    comp_avg = _mean_or_none(all_compile_flags)
    judge_avg = _mean_or_none(all_judge_scores)
    lint_vals = [r["lint_mean"] for r in results if r["lint_mean"] is not None]
    lint_avg = _mean_or_none(lint_vals)

    composite_avg = (sum(per_prompt_composites) / len(per_prompt_composites)
                     if per_prompt_composites else 0.0)
    composite_std = _stdev(per_prompt_composites)
    judge_std = _stdev(all_judge_scores)

    # Compile gate: vacuously True when no compiler is available (comp_avg None).
    compile_gate_pass = True if comp_avg is None else comp_avg >= COMPILE_GATE_THRESHOLD

    measured = [k for k, v in (("judge", judge_avg), ("lint", lint_avg))
                if v is not None]

    summary = {
        "adapter": args.adapter,
        "domain": domain_name,
        "n_prompts": len(prompts),
        "n_samples": samples,
        "composite_score": round(composite_avg, 4),
        "composite_std": round(composite_std, 4),
        "compile_gate_pass": compile_gate_pass,
        "compile_pass": None if comp_avg is None else round(comp_avg, 4),
        "lint_clean": None if lint_avg is None else round(lint_avg, 4),
        "judge_score": None if judge_avg is None else round(judge_avg, 4),
        "judge_std": round(judge_std, 4),
        "judge_kind": judge_kind,
        "measured_components": measured,
        "wall_clock_s": round(time.time() - t0, 1),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\n[eval] composite={composite_avg:.4f} ± {composite_std:.4f} "
          f"compile_gate={'PASS' if compile_gate_pass else 'FAIL'} "
          f"(compile={comp_avg} lint={lint_avg} judge={judge_avg}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
