"""Eval domain modules.

A domain is an opinion about what "good output" means for one corpus type.
Each domain module provides four things:

    JUDGE_SYSTEM: str
        System prompt for the LLM idiomaticity judge.

    def extract_code(text: str) -> str
        Pull the relevant span out of the model's raw output (e.g. strip
        <think> blocks, take the first fenced code block).

    def compile(code: str) -> int | None
        1 if `code` passes a hard syntax check, 0 if it doesn't, None if the
        check is unavailable on this machine. Used as the dominant fitness
        signal. Return None for domains where this doesn't apply.

    def lint(code: str) -> int | None
        Same shape, for a softer style/quality check.

See lab/domains/swift.py for the reference implementation and
lab/domains/text.py for a judge-only example.
"""
import importlib


def load(name: str):
    """Import lab.domains.<name> and return the module."""
    return importlib.import_module(f"lab.domains.{name}")
