"""domains/swift.py - Swift / SwiftUI / iOS eval domain.

Pairs well with held-out prompts that ask for small Swift snippets
(views, models, extensions). The `compile` gate is `swiftc -parse`, which
catches a huge class of confabulation cheaply without needing a build
environment. `lint` shells out to swiftlint if it's installed.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from shutil import which

JUDGE_SYSTEM = (
    "You are a senior iOS engineer grading Swift code for idiomatic quality. "
    "Score 0.0 to 1.0 on: correct modern Swift/SwiftUI idioms, naming, structure, "
    "and whether it directly answers the prompt. Reply with ONLY a number."
)


def extract_code(text: str) -> str:
    """Strip <think>...</think> reasoning, then take the first fenced code block
    (```swift ... ``` or ``` ... ```), else the whole text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)  # orphan close
    m = re.search(r"```(?:swift)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def compile(code: str) -> int | None:
    """1 if `swiftc -parse` accepts it, 0 otherwise, None if swiftc is missing."""
    if not which("swiftc"):
        return None
    if not code.strip():
        return 0
    with tempfile.NamedTemporaryFile("w", suffix=".swift", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(
            ["swiftc", "-parse", path], capture_output=True, timeout=60
        )
        return 1 if r.returncode == 0 else 0
    except Exception:
        return 0
    finally:
        os.unlink(path)


def lint(code: str) -> int | None:
    """1 if swiftlint finds no serious violations, 0 if it does, None if missing.

    `brew install swiftlint` to enable.
    """
    if not which("swiftlint"):
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".swift", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(
            ["swiftlint", "lint", "--quiet", "--path", path],
            capture_output=True,
            timeout=60,
            text=True,
        )
        out = (r.stdout + r.stderr).lower()
        return 0 if ("error" in out or r.returncode != 0) else 1
    except Exception:
        return None
    finally:
        os.unlink(path)
