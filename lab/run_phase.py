#!/usr/bin/env python3
"""run_phase.py - cron phase runner for mlx-auto-lora.

Cron is the loop. Each invocation does ONE phase and exits:
    EXPERIMENT | EVAL | REPORT | HEALTH | RETRAIN_GATE

EXPERIMENT (the autoresearch loop): acquire lock -> propose a hypothesis with a
UCB1 bandit (built on the best-known config, constrained to the measured memory
envelope) -> edit config.yaml -> optionally rebuild a cached corpus variant ->
train -> evaluate -> append results.tsv -> keep (git commit) or revert (git
checkout) -> notify -> release lock.

Keep/revert is deterministic and statistically honest:
  * compile is a hard GATE (a run that stops parsing is reverted outright);
  * a winner must beat the incumbent by more than the eval noise
    (score > best + max(MIN_DELTA, NOISE_K * composite_std/sqrt(n_prompts)));
  * runs cut off by the wall clock (completion_ratio < MIN_COMPLETION) are
    rejected as not-comparable rather than scored as equals.

The bandit actually learns: it scores arms by their observed composite (UCB1),
explores untried arms first, and with some probability compounds two changes on
different axes so interactions (e.g. rank + lr) can be found. When the search
stalls it escalates toward high-leverage untried axes (data mix, depth, MLP),
and after a hard ceiling of consecutive reverts it pauses and alerts.

The only LLM call is the eval idiomaticity judge (Anthropic Claude, in eval.py).

Usage:
    python lab/run_phase.py --phase EXPERIMENT [--hypothesis H4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from lab.notifier import from_env as notifier_from_env

ROOT = Path(os.environ.get("MLX_AUTO_LORA_ROOT", Path.home() / "mlx-auto-lora"))
LAB = ROOT / "lab"
CONFIG = LAB / "config.yaml"
RESULTS = LAB / "results.tsv"
STATE = ROOT / "state.json"
LOCK = ROOT / "lab.lock"
EXPLORED = LAB / "explored.json"
ENVELOPE = ROOT / "logs" / "mem_envelope.tsv"
KEEPERS = ROOT / "adapters" / "keepers"
CORPUS_VARIANTS = ROOT / "corpus_variants"

# results.tsv gained a completion_ratio column. Old rows without it parse fine
# (the field reads back as "").
RESULTS_HEADER = (
    "run_id\tconfig_hash\tcomposite_score\tcompile_pass\tlint_clean\t"
    "judge_score\twall_clock_s\tcompletion_ratio\tnotes"
)

# ── Keep/revert tunables (env-overridable) ───────────────────────────────────
# Keep only when score beats the incumbent by more than the measured noise.
NOISE_K = float(os.environ.get("MLX_AUTO_LORA_NOISE_K", "2.0"))
MIN_DELTA = float(os.environ.get("MLX_AUTO_LORA_MIN_DELTA", "0.005"))
# Reject runs that did not reach this fraction of their iter target.
MIN_COMPLETION = float(os.environ.get("MLX_AUTO_LORA_MIN_COMPLETION", "0.95"))
# Convergence handling.
REVERT_EXPLORE_THRESHOLD = int(os.environ.get("MLX_AUTO_LORA_REVERT_EXPLORE", "4"))
REVERT_PAUSE_THRESHOLD = int(os.environ.get("MLX_AUTO_LORA_REVERT_PAUSE", "10"))
# Cap per-source file count when rebuilding a corpus variant (bounds rebuild time).
MIX_MAX_FILES_PER_SOURCE = int(os.environ.get("MLX_AUTO_LORA_MIX_MAX_FILES", "2000"))

_NOTIFIER = notifier_from_env()


def notify(text: str, tag: str) -> bool:
    cfg_hash = config_hash()
    return _NOTIFIER.post(f"cfg: {cfg_hash}\n{text}", tag)


def config_hash() -> str:
    return hashlib.sha256(CONFIG.read_bytes()).hexdigest()[:8]


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {
        "best_run": None, "best_score": -1.0, "best_std": 0.0,
        "experiment_cron_paused": False, "consecutive_reverts": 0,
        "prod_score": -1.0,
    }


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=2))


def acquire_lock(phase: str) -> bool:
    if LOCK.exists():
        try:
            info = json.loads(LOCK.read_text())
            pid, ts = int(info.get("pid", -1)), float(info.get("ts", 0))
            alive = _proc_exists(pid)
            stale = (time.time() - ts) > 6 * 3600
            if alive and not stale:
                notify(
                    f"prior run still active (PID={pid}, phase={info.get('phase')})",
                    "SKIP",
                )
                return False
            print(f"[lock] removing stale lock (pid={pid} alive={alive} stale={stale})")
        except Exception:
            pass
        LOCK.unlink(missing_ok=True)
    LOCK.write_text(json.dumps({"pid": os.getpid(), "ts": time.time(), "phase": phase}))
    return True


def release_lock() -> None:
    LOCK.unlink(missing_ok=True)


def _proc_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# Peak-memory budget for UNATTENDED runs. The raw GPU limit on a 96 GB M2 Ultra
# is ~96 GB, but in real conditions (other servers + OS share unified memory)
# the effective training ceiling is lower. 70 GB gives genuine margin. Configs
# above this are excluded from the autonomous loop. Tune to your box.
PEAK_BUDGET_GB = float(os.environ.get("MLX_AUTO_LORA_PEAK_GB", "70"))
MAX_SEQ_AUTONOMOUS = int(os.environ.get("MLX_AUTO_LORA_MAX_SEQ", "2048"))


def load_envelope() -> dict:
    """Return safe bounds + the feasible (num_layers, seq) point set.

    Only points whose measured peak <= PEAK_BUDGET_GB are feasible. Callers
    must validate the JOINT (num_layers, seq) combo via is_safe() - independent
    axis maxes are not sufficient.
    """
    pts: list[tuple[int, int, float]] = []
    if ENVELOPE.exists():
        for line in ENVELOPE.read_text().splitlines()[1:]:
            p = line.split("\t")
            if len(p) >= 4 and p[2] == "OK":
                try:
                    pts.append((int(p[0]), int(p[1]), float(p[3])))
                except ValueError:
                    continue
    if not pts:
        pts = [(2, 1024, 33.0)]
    safe = [(nl, sq) for nl, sq, pk in pts
            if pk <= PEAK_BUDGET_GB and sq <= MAX_SEQ_AUTONOMOUS]
    if not safe:
        safe = [(2, 1024)]
    return {
        "max_num_layers": max(nl for nl, _ in safe),
        "max_seq": max(sq for _, sq in safe),
        "feasible": safe,
    }


def is_safe(num_layers: int, seq: int, feasible: list) -> bool:
    return any(fnl >= num_layers and fsq >= seq for fnl, fsq in feasible)


def load_explored() -> list:
    return json.loads(EXPLORED.read_text()) if EXPLORED.exists() else []


def save_explored(e: list) -> None:
    EXPLORED.write_text(json.dumps(e, indent=2))


# ── Data-mix blends ───────────────────────────────────────────────────────────
# A "mix" arm reweights the corpus across the data slices the user has
# configured in config.yaml `data.mix_weights`. With a single slice (the
# default), no mix arms are generated and the loop trains on the prebuilt
# corpus. With >=2 slices, blends become a first-class hypothesis (data mix is
# usually the single highest-leverage lever for LoRA quality).
def _configured_slices(cfg: dict) -> list[str]:
    return [k for k in (cfg.get("data", {}).get("mix_weights") or {})]


def _mix_blends(slices: list[str]) -> dict[str, dict]:
    """Return {short_key: weight_dict} blends over the configured slices.

    Only meaningful with >=2 slices; returns {} otherwise.
    """
    if len(slices) < 2:
        return {}
    a, b = slices[0], slices[1]  # primary vs the next slice
    return {
        f"{a[:2]}100":          {a: 1.0},
        f"{a[:2]}70{b[:2]}30":  {a: 0.7, b: 0.3},
        f"{a[:2]}50{b[:2]}50":  {a: 0.5, b: 0.5},
        f"{a[:2]}30{b[:2]}70":  {a: 0.3, b: 0.7},
    }


# ── UCB1 bandit ────────────────────────────────────────────────────────────────
def _ucb1_select(arms: list[tuple], explored: list,
                 prefer_axes: set | None = None) -> int:
    """UCB1 selection over arms. Each arm is (hid, axis, fn, value).

    score = mean_reward + sqrt(2 * ln(total_plays) / arm_plays)
    Untried arms get +inf (explore-first). Reward = observed composite score.
    When prefer_axes is set, matching axes are boosted (convergence escalation).
    Returns the index into `arms`.
    """
    records: dict[tuple, list[float]] = {}
    for entry in explored:
        key = (entry["axis"], str(entry["value"]))
        records.setdefault(key, []).append(float(entry.get("score", 0.0)))
    total_plays = sum(len(v) for v in records.values())

    best_idx, best_ucb = 0, -1.0
    for i, (hid, axis, fn, value) in enumerate(arms):
        history = records.get((axis, str(value)), [])
        if not history:
            ucb = 1e9 + (1e6 if prefer_axes and axis in prefer_axes else 0.0)
        else:
            n_arm = len(history)
            mean_r = sum(history) / n_arm
            bonus = math.sqrt(2.0 * math.log(max(total_plays, 1)) / n_arm)
            ucb = mean_r + bonus
            if prefer_axes and axis in prefer_axes:
                ucb *= 10.0
        if ucb > best_ucb:
            best_ucb, best_idx = ucb, i
    return best_idx


def propose_hypothesis(cfg: dict, env: dict, forced: str | None,
                       consecutive_reverts: int = 0
                       ) -> tuple[str, str, dict, list[dict]]:
    """Mutate cfg in place per one or two hypotheses using a UCB1 bandit.

    Proposals build on the current committed config (which IS the best-known
    config, since reverts restore config.yaml to the last kept state).

    Returns (hid, description, mutated_cfg, new_explored_entries). The caller
    appends new_explored_entries (one per axis changed) to explored.json with
    the run's score.
    """
    explored = load_explored()
    max_nl, max_seq = env["max_num_layers"], env["max_seq"]
    feasible = env["feasible"]

    def set_rank(c, v):    c["lora"]["rank"] = v
    def set_lr(c, v):      c["optim"]["lr"] = v
    def set_layers(c, v):  c["lora"]["num_layers"] = v
    def set_seq(c, v):     c["data"]["max_seq_length"] = v
    def set_dropout(c, v): c["lora"]["dropout"] = v
    def set_warmup(c, v):  c["optim"]["warmup_steps"] = v
    def set_alpha(c, v):   c["lora"]["alpha_over_rank"] = v

    def set_mlp(c, v):
        base = ["q_proj", "k_proj", "v_proj", "o_proj"]
        c["lora"]["target_modules"] = base + (
            ["gate_proj", "up_proj", "down_proj"] if v else [])

    mix_blends = _mix_blends(_configured_slices(cfg))

    def set_mix(c, v):
        # v is a short blend key; expand to a full weight dict over all slices.
        all_slices = _configured_slices(c)
        blend = mix_blends[v]
        c["data"]["mix_weights"] = {s: float(blend.get(s, 0.0)) for s in all_slices}

    menu = [
        ("H1-rank",    "rank",            set_rank,    [8, 16, 32, 64]),
        ("H4-lr",      "lr",              set_lr,      [3.0e-5, 5.0e-5, 1.0e-4, 2.0e-4]),
        ("H5-warmup",  "warmup_steps",    set_warmup,  [0, 50, 100, 200]),
        ("H8-seq",     "max_seq_length",  set_seq,     [s for s in [1024, 2048] if s <= max_seq]),
        ("H11-layers", "num_layers",      set_layers,  [n for n in [2, 4, 8] if n <= max_nl]),
        ("Hd-dropout", "dropout",         set_dropout, [0.0, 0.05, 0.1]),
        ("Ha-alpha",   "alpha_over_rank", set_alpha,   [1.0, 2.0, 4.0]),
        ("H3-mlp",     "mlp_targets",     set_mlp,     [False, True]),
    ]
    if mix_blends:
        menu.append(("Hm-mix", "mix", set_mix, list(mix_blends.keys())))

    if forced:
        restricted = [m for m in menu if m[0].lower().startswith(forced.lower())]
        if restricted:
            menu = restricted

    cur_nl = int(cfg["lora"]["num_layers"]) if cfg["lora"]["num_layers"] != "all" else 999
    cur_seq = int(cfg["data"]["max_seq_length"])

    def _combo_after(axis, v):
        nl = v if axis == "num_layers" else cur_nl
        sq = v if axis == "max_seq_length" else cur_seq
        return int(nl), int(sq)

    def _feasible_arm(axis, v):
        nl, sq = _combo_after(axis, v)
        return is_safe(nl, sq, feasible)

    all_arms = [
        (hid, axis, fn, v)
        for hid, axis, fn, values in menu
        for v in values
        if _feasible_arm(axis, v)
    ]
    if not all_arms:
        all_arms = [("Hnoop", "lr",
                     lambda c, x: c["optim"].__setitem__("lr", x),
                     cfg["optim"]["lr"])]

    prefer_axes: set | None = None
    if consecutive_reverts >= REVERT_EXPLORE_THRESHOLD:
        prefer_axes = {"mix", "num_layers", "mlp_targets"}
        print(f"[bandit] convergence escalation: preferring {prefer_axes} "
              f"(consecutive_reverts={consecutive_reverts})")

    rng = random.Random()

    primary_idx = _ucb1_select(all_arms, explored, prefer_axes)
    p_hid, p_axis, p_fn, p_val = all_arms[primary_idx]
    p_fn(cfg, p_val)
    new_entries = [{"axis": p_axis, "value": str(p_val)}]

    if not is_safe(int(cfg["lora"]["num_layers"]),
                   int(cfg["data"]["max_seq_length"]), feasible):
        cfg["lora"]["num_layers"] = env["max_num_layers"]
        cfg["data"]["max_seq_length"] = min(cur_seq, env["max_seq"])

    hid = p_hid
    desc = f"{p_hid}: set {p_axis} = {p_val}"

    # 30% chance to compound a second arm on a different axis (find interactions).
    if not forced and rng.random() < 0.3:
        cur_nl2 = int(cfg["lora"]["num_layers"]) if str(cfg["lora"]["num_layers"]) != "all" else 999
        cur_seq2 = int(cfg["data"]["max_seq_length"])

        def _still_safe(axis2, v2):
            nl = v2 if axis2 == "num_layers" else cur_nl2
            sq = v2 if axis2 == "max_seq_length" else cur_seq2
            return is_safe(int(nl), int(sq), feasible)

        second_arms = [a for a in all_arms
                       if a[1] != p_axis and _still_safe(a[1], a[3])]
        if second_arms:
            s_hid, s_axis, s_fn, s_val = second_arms[_ucb1_select(second_arms, explored, prefer_axes)]
            s_fn(cfg, s_val)
            if is_safe(int(cfg["lora"]["num_layers"]),
                       int(cfg["data"]["max_seq_length"]), feasible):
                hid = f"{p_hid}+{s_hid}"
                desc = f"{p_hid}: set {p_axis} = {p_val} | {s_hid}: set {s_axis} = {s_val}"
                new_entries.append({"axis": s_axis, "value": str(s_val)})
                print(f"[bandit] compound proposal: {desc}")
            else:
                print(f"[bandit] second arm {s_axis}={s_val} unsafe - dropping")

    return hid, desc, cfg, new_entries


# ── Data-mix corpus variant management ───────────────────────────────────────
def _default_corpus_dir() -> Path:
    return ROOT / "corpus"


def _mix_hash(cfg: dict) -> str:
    mix = cfg["data"].get("mix_weights", {})
    payload = json.dumps({
        "mix": {k: mix[k] for k in sorted(mix)},
        "sources": cfg["data"].get("sources", []),
        "seq": int(cfg["data"]["max_seq_length"]),
        "cap": MIX_MAX_FILES_PER_SOURCE,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def ensure_corpus_variant(cfg: dict) -> Path | None:
    """Build (or reuse) a cached corpus for the given mix; return its dir.

    Returns the default corpus dir for a pure single-slice mix (the prebuilt
    corpus already covers it). For a multi-slice blend it rebuilds via
    prepare_data.py into corpus_variants/<hash>/, reusing the cache on repeat.
    Returns None on failure so the caller can fall back to the default corpus.
    """
    slices = [k for k, w in (cfg["data"].get("mix_weights") or {}).items() if w > 0]
    if len(slices) <= 1:
        return _default_corpus_dir()

    variant_hash = _mix_hash(cfg)
    variant_dir = CORPUS_VARIANTS / variant_hash
    if (variant_dir / "train.jsonl").exists() and (variant_dir / "valid.jsonl").exists():
        print(f"[data] reusing cached corpus variant {variant_hash}")
        return variant_dir

    print(f"[data] building corpus variant {variant_hash} "
          f"(mix={cfg['data']['mix_weights']})")
    variant_dir.mkdir(parents=True, exist_ok=True)
    # Write a temp config carrying this mix so prepare_data.py picks it up.
    tmp_cfg = variant_dir / "_mix_config.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    try:
        result = subprocess.run(
            [sys.executable, str(LAB / "prepare_data.py"),
             "--config", str(tmp_cfg), "--out", str(variant_dir),
             "--max-files-per-source", str(MIX_MAX_FILES_PER_SOURCE)],
            cwd=str(ROOT), timeout=600)
        if result.returncode != 0:
            print(f"[data] prepare_data.py failed (rc={result.returncode}) - falling back")
            shutil.rmtree(variant_dir, ignore_errors=True)
            return None
    except subprocess.TimeoutExpired:
        print("[data] corpus build timed out - falling back")
        shutil.rmtree(variant_dir, ignore_errors=True)
        return None
    except Exception as e:
        print(f"[data] corpus build error: {e} - falling back")
        shutil.rmtree(variant_dir, ignore_errors=True)
        return None

    if not (variant_dir / "train.jsonl").exists():
        print("[data] corpus variant empty - falling back to default corpus")
        shutil.rmtree(variant_dir, ignore_errors=True)
        return None
    return variant_dir


def append_result(row: dict) -> None:
    if not RESULTS.exists():
        RESULTS.write_text(RESULTS_HEADER + "\n")
    line = "\t".join(str(row.get(k, "")) for k in
                     ["run_id", "config_hash", "composite_score", "compile_pass",
                      "lint_clean", "judge_score", "wall_clock_s",
                      "completion_ratio", "notes"])
    with open(RESULTS, "a") as f:
        f.write(line + "\n")


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(LAB), *args],
                          capture_output=True, text=True)


def phase_experiment(forced_hyp: str | None) -> int:
    state = load_state()
    if state.get("experiment_cron_paused"):
        print("[experiment] paused - exiting")
        return 0
    if forced_hyp is None and state.get("forced_hypothesis"):
        forced_hyp = state["forced_hypothesis"]
        print(f"[experiment] using forced_hypothesis from state: {forced_hyp}")
        state.pop("forced_hypothesis")
        save_state(state)
    if not acquire_lock("EXPERIMENT"):
        return 0
    consecutive_reverts = int(state.get("consecutive_reverts", 0))
    try:
        env = load_envelope()
        cfg = yaml.safe_load(CONFIG.read_text())
        hid, desc, cfg, new_entries = propose_hypothesis(
            cfg, env, forced_hyp, consecutive_reverts)
        CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cfg_hash = config_hash()
        run_id = datetime.now(timezone.utc).strftime("exp-%Y%m%dT%H%M%S")
        notify(f"hypothesis: {desc}\nrun: {run_id}", "EXPERIMENT")
        print(f"[experiment] {desc} run={run_id}")

        # Resolve the corpus for this run (cached variant when mix changed).
        corpus_dir = ensure_corpus_variant(cfg)
        if corpus_dir is None:
            print("[data] falling back to default corpus; reverting mix in config")
            cfg2 = yaml.safe_load(CONFIG.read_text())
            slices = _configured_slices(cfg2)
            if slices:
                cfg2["data"]["mix_weights"] = {slices[0]: 1.0}
            CONFIG.write_text(yaml.safe_dump(cfg2, sort_keys=False))
            desc += " [mix-fallback: variant build failed]"
            corpus_dir = _default_corpus_dir()

        budget = int(cfg["budget"]["max_wall_clock_min"])
        t0 = time.time()
        tr = subprocess.run(
            [sys.executable, str(LAB / "train.py"),
             "--config", str(CONFIG), "--run-id", run_id,
             "--max-wall-clock-min", str(budget),
             "--data", str(corpus_dir)],
            cwd=str(ROOT))
        if tr.returncode != 0:
            git("checkout", "--", "config.yaml")
            notify(f"train failed (rc={tr.returncode}) - reverted\n{desc}", "REVERT")
            state["consecutive_reverts"] = consecutive_reverts + 1
            save_state(state)
            return 1

        adapter = ROOT / "adapters" / run_id

        # ── Fix: reject truncated / under-trained runs (not comparable) ──────
        completion_ratio, wall_clock_truncated = 1.0, False
        meta_path = adapter / "train_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                completion_ratio = float(meta.get("completion_ratio", 1.0))
                wall_clock_truncated = bool(meta.get("wall_clock_truncated", False))
            except Exception as e:
                print(f"[experiment] warn: could not parse train_meta.json: {e}")

        if wall_clock_truncated or completion_ratio < MIN_COMPLETION:
            git("checkout", "--", "config.yaml")
            if adapter.exists():
                shutil.rmtree(adapter, ignore_errors=True)
            append_result({
                "run_id": run_id, "config_hash": cfg_hash,
                "composite_score": "", "compile_pass": "", "lint_clean": "",
                "judge_score": "", "wall_clock_s": round(time.time() - t0, 1),
                "completion_ratio": f"{completion_ratio:.2f}",
                "notes": desc + " [REJECTED: truncated]",
            })
            state["consecutive_reverts"] = consecutive_reverts + 1
            save_state(state)
            notify(
                f"{desc}\ntraining truncated "
                f"(completion={completion_ratio:.2f}) - not comparable, reverted",
                "REVERT")
            return 0

        out_json = LAB / "evals" / f"{run_id}.json"
        ev = subprocess.run(
            [sys.executable, str(LAB / "eval.py"),
             "--adapter", str(adapter),
             "--out", str(out_json), "--config", str(CONFIG)],
            cwd=str(ROOT))
        if ev.returncode != 0 or not out_json.exists():
            git("checkout", "--", "config.yaml")
            notify(f"eval failed - reverted\n{desc}", "REVERT")
            state["consecutive_reverts"] = consecutive_reverts + 1
            save_state(state)
            return 1

        res = json.loads(out_json.read_text())
        score = float(res["composite_score"])
        composite_std = float(res.get("composite_std", 0.0))
        compile_gate_pass = bool(res.get("compile_gate_pass", True))
        wall = round(time.time() - t0, 1)

        append_result({
            "run_id": run_id, "config_hash": cfg_hash,
            "composite_score": score,
            "compile_pass": res.get("compile_pass"),
            "lint_clean": res.get("lint_clean"),
            "judge_score": res.get("judge_score"),
            "wall_clock_s": wall,
            "completion_ratio": f"{completion_ratio:.2f}",
            "notes": desc,
        })
        explored_snapshot = load_explored()
        for entry in new_entries:
            explored_snapshot.append({
                "axis": entry["axis"], "value": entry["value"],
                "score": score, "run_id": run_id,
            })
        save_explored(explored_snapshot)

        # ── Fix: compile gate - non-parsing output is a regression ───────────
        if not compile_gate_pass:
            git("checkout", "--", "config.yaml")
            if adapter.exists():
                shutil.rmtree(adapter, ignore_errors=True)
            state["consecutive_reverts"] = consecutive_reverts + 1
            save_state(state)
            notify(
                f"{desc}\nscore {score:.4f} - compile gate FAILED "
                f"(generated code did not parse) - reverted",
                "REVERT")
            return 0

        # ── Fix: noise-floor keep/revert ─────────────────────────────────────
        # The noise basis is the STANDARD ERROR OF THE MEAN composite
        # (composite_std / sqrt(n_prompts)), not the raw per-prompt spread:
        # we are testing whether the *mean* composite is reliably higher, and
        # the mean's uncertainty shrinks with more prompts. Raw std would set
        # an unreachable bar (~0.2) and reject genuine small gains as noise.
        best_state_score = float(state.get("best_score", -1.0))
        n_prompts = int(res.get("n_prompts", 1)) or 1
        sem = composite_std / math.sqrt(n_prompts)
        noise_floor = max(MIN_DELTA, NOISE_K * sem)
        if score > best_state_score + noise_floor:
            KEEPERS.mkdir(parents=True, exist_ok=True)
            if adapter.exists():
                shutil.move(str(adapter), str(KEEPERS / run_id))
            git("add", "config.yaml", "results.tsv")
            git("commit", "-q", "-m", f"exp {run_id}: {desc} -> {score:.4f}")
            delta = score - best_state_score
            state.update({"best_run": run_id, "best_score": score,
                          "best_std": composite_std, "consecutive_reverts": 0})
            save_state(state)
            notify(
                f"{desc}\nscore {score:.4f} (+{delta:.4f}) - NEW BEST\n"
                f"noise_floor={noise_floor:.4f} (SEM={sem:.4f}, std={composite_std:.4f}, n={n_prompts})\n"
                f"judge={res.get('judge_score')}",
                "KEEP")
        else:
            git("checkout", "--", "config.yaml")
            git("add", "results.tsv")
            git("commit", "-q", "-m", f"exp {run_id}: reverted ({score:.4f})")
            if adapter.exists():
                shutil.rmtree(adapter, ignore_errors=True)
            state["consecutive_reverts"] = consecutive_reverts + 1
            save_state(state)
            notify(
                f"{desc}\nscore {score:.4f} (best {best_state_score:.4f}, "
                f"floor +{noise_floor:.4f}) - reverted",
                "REVERT")

            # ── Fix: convergence handling ────────────────────────────────────
            if state["consecutive_reverts"] >= REVERT_PAUSE_THRESHOLD:
                state["experiment_cron_paused"] = True
                save_state(state)
                notify(
                    f"search converged / stuck after "
                    f"{state['consecutive_reverts']} reverts - pausing "
                    f"autonomous loop; clear experiment_cron_paused to resume",
                    "DECISION")
        return 0
    finally:
        release_lock()


def phase_eval() -> int:
    state = load_state()
    best = state.get("best_run")
    if not best:
        notify("no keeper adapter yet", "EVAL")
        return 0
    adapter = KEEPERS / best
    out = LAB / "evals" / f"{best}_fulleval.json"
    subprocess.run(
        [sys.executable, str(LAB / "eval.py"),
         "--adapter", str(adapter), "--out", str(out), "--config", str(CONFIG)],
        cwd=str(ROOT))
    if out.exists():
        r = json.loads(out.read_text())
        sc = r["composite_score"]
        gate = r.get("compile_gate_pass", True)
        verdict = ("REGRESSION" if not gate else
                   ("SHIP" if sc >= 0.9 else ("ITERATE" if sc >= 0.7 else "REGRESSION")))
        if verdict == "SHIP":
            state["prod_score"] = sc
            save_state(state)
        notify(
            f"keeper {best}\ncomposite {sc:.4f} ± {r.get('composite_std', 0):.4f} "
            f"(compile_gate={'PASS' if gate else 'FAIL'} judge={r.get('judge_score')})\n"
            f"verdict: {verdict}",
            "EVAL")
    return 0


def phase_report() -> int:
    if not RESULTS.exists():
        notify("no experiments yet", "REPORT")
        return 0
    rows = [l.split("\t") for l in RESULTS.read_text().splitlines()[1:] if l.strip()]
    n = len(rows)
    scores = [(float(r[2]), r[0]) for r in rows if _isfloat(r[2])]
    best = max(scores) if scores else (0, "-")
    state = load_state()
    top3 = sorted(scores, reverse=True)[:3]
    body = (
        f"experiments: {n}\nbest: {best[0]:.4f} ({best[1]})\n"
        f"top3: " + ", ".join(f"{s:.3f}" for s, _ in top3) + "\n"
        f"consecutive reverts: {state.get('consecutive_reverts', 0)}"
    )
    notify(body, "REPORT")
    return 0


def phase_health() -> int:
    problems = []
    free_gb = shutil.disk_usage(str(ROOT)).free / 1e9
    if free_gb < 100:
        problems.append(f"disk low: {free_gb:.0f}GB free")
    if LOCK.exists():
        try:
            info = json.loads(LOCK.read_text())
            if (time.time() - float(info.get("ts", 0))) > 6 * 3600 \
                    and not _proc_exists(int(info.get("pid", -1))):
                LOCK.unlink(missing_ok=True)
                problems.append("cleared stale lock")
        except Exception:
            pass
    if problems:
        notify("\n".join(problems), "HEALTH")
    else:
        print("[health] clean")
    return 0


def phase_retrain_gate() -> int:
    state = load_state()
    best, prod = state.get("best_score", -1), state.get("prod_score", -1)
    if best > 0 and best >= prod * 1.03:
        notify(
            f"candidate {state.get('best_run')} score {best:.4f} vs prod {prod:.4f}\n"
            f"improvement >=3% - reply /lab_deploy <run_id> to promote",
            "DECISION")
    else:
        notify(f"no_change (best {best:.4f}, prod {prod:.4f})", "DECISION")
    return 0


def _isfloat(s) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


PHASES = {
    "EXPERIMENT": phase_experiment,
    "EVAL": phase_eval,
    "REPORT": phase_report,
    "HEALTH": phase_health,
    "RETRAIN_GATE": phase_retrain_gate,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=list(PHASES))
    ap.add_argument("--hypothesis", default=None, help="Force a hypothesis id, e.g. H4")
    args = ap.parse_args()
    if args.phase == "EXPERIMENT":
        return phase_experiment(args.hypothesis)
    return PHASES[args.phase]()


if __name__ == "__main__":
    raise SystemExit(main())
