"""domains/text.py - judge-only fallback domain.

Use when there is no compiler or linter for your target. The judge does
all the work. Don't expect tight discrimination - the LLM judge has high
variance compared to a hard compile gate. Good for prose, dialog, summarization,
or anything where "did this even parse" doesn't apply.

If you have an interpreter or AST checker, prefer writing a real domain
module (see swift.py for the shape) - even a permissive parse check
contributes more signal than judge-only.
"""
from __future__ import annotations

import re

JUDGE_SYSTEM = (
    "You are a careful editor scoring text on quality, clarity, and how well "
    "it answers the prompt. Score 0.0 to 1.0. Reply with ONLY a number."
)


def extract_code(text: str) -> str:
    """Strip <think>...</think> reasoning blocks; return the rest verbatim."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def compile(code: str) -> int | None:
    return None  # no compile gate for prose


def lint(code: str) -> int | None:
    return None  # no lint gate for prose
