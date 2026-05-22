#!/usr/bin/env python3
"""run_phase.py - cron phase runner for mlx-auto-lora.

Cron is the loop. Each invocation does ONE phase and exits:
    EXPERIMENT | EVAL | REPORT | HEALTH | RETRAIN_GATE

EXPERIMENT (the autoresearch loop): acquire lock -> propose ONE hypothesis
(bandit, constrained to the measured memory envelope) -> edit config.yaml ->
train -> evaluate -> append results.tsv -> keep (git commit) or revert (git
checkout) -> notify -> release lock.

Hypothesis selection and keep/revert are deterministic; no LLM is in the loop.
The only LLM call is the eval idiomaticity judge (Anthropic Claude, in eval.py).

Usage:
    python lab/run_phase.py --phase EXPERIMENT [--hypothesis H4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

RESULTS_HEADER = (
    "run_id\tconfig_hash\tcomposite_score\tcompile_pass\tlint_clean\t"
    "judge_score\twall_clock_s\tnotes"
)

_NOTIFIER = notifier_from_env()


def notify(text: str, tag: str) -> bool:
    """Post a status update through whichever notifier env vars selected."""
    cfg_hash = config_hash()
    return _NOTIFIER.post(f"cfg: {cfg_hash}\n{text}", tag)


def config_hash() -> str:
    return hashlib.sha256(CONFIG.read_bytes()).hexdigest()[:8]


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {
        "best_run": None, "best_score": -1.0,
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


def propose_hypothesis(cfg: dict, env: dict,
                       forced: str | None) -> tuple[str, str, dict]:
    """Mutate cfg in place per ONE hypothesis. Return (id, description, cfg)."""
    explored = load_explored()
    tried = {(e["axis"], str(e["value"])) for e in explored}
    max_nl, max_seq = env["max_num_layers"], env["max_seq"]

    def set_rank(c, v): c["lora"]["rank"] = v
    def set_lr(c, v): c["optim"]["lr"] = v
    def set_layers(c, v): c["lora"]["num_layers"] = v
    def set_seq(c, v): c["data"]["max_seq_length"] = v
    def set_dropout(c, v): c["lora"]["dropout"] = v
    def set_warmup(c, v): c["optim"]["warmup_steps"] = v
    def set_alpha(c, v): c["lora"]["alpha_over_rank"] = v

    def set_mlp(c, v):
        base = ["q_proj", "k_proj", "v_proj", "o_proj"]
        c["lora"]["target_modules"] = base + (
            ["gate_proj", "up_proj", "down_proj"] if v else [])

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

    if forced:
        menu = [m for m in menu if m[0].lower().startswith(forced.lower())] or menu

    cur_nl = int(cfg["lora"]["num_layers"]) if cfg["lora"]["num_layers"] != "all" else 999
    cur_seq = int(cfg["data"]["max_seq_length"])
    feasible = env["feasible"]

    def _combo_after(axis, v):
        nl = v if axis == "num_layers" else cur_nl
        sq = v if axis == "max_seq_length" else cur_seq
        return int(nl), int(sq)

    candidates = []
    for hid, axis, fn, values in menu:
        for v in values:
            if (axis, str(v)) in tried:
                continue
            nl, sq = _combo_after(axis, v)
            if not is_safe(nl, sq, feasible):
                continue
            candidates.append((hid, axis, fn, v))
    if not candidates:
        pool = []
        for hid, axis, fn, values in menu:
            for v in values:
                nl, sq = _combo_after(axis, v)
                if is_safe(nl, sq, feasible):
                    pool.append((hid, axis, fn, v))
        candidates = pool or [
            ("Hnoop", "lr",
             lambda c, x: c["optim"].__setitem__("lr", x),
             cfg["optim"]["lr"])
        ]

    hid, axis, fn, v = random.choice(candidates)
    fn(cfg, v)
    if not is_safe(int(cfg["lora"]["num_layers"]),
                   int(cfg["data"]["max_seq_length"]), feasible):
        cfg["lora"]["num_layers"] = env["max_num_layers"]
        cfg["data"]["max_seq_length"] = min(cur_seq, env["max_seq"])
    desc = f"{hid}: set {axis} = {v}"
    return hid, desc, cfg


def append_result(row: dict) -> None:
    if not RESULTS.exists():
        RESULTS.write_text(RESULTS_HEADER + "\n")
    line = "\t".join(str(row.get(k, "")) for k in
                     ["run_id", "config_hash", "composite_score", "compile_pass",
                      "lint_clean", "judge_score", "wall_clock_s", "notes"])
    with open(RESULTS, "a") as f:
        f.write(line + "\n")


def best_score_so_far() -> float:
    if not RESULTS.exists():
        return -1.0
    best = -1.0
    for line in RESULTS.read_text().splitlines()[1:]:
        try:
            best = max(best, float(line.split("\t")[2]))
        except (IndexError, ValueError):
            continue
    return best


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
    try:
        env = load_envelope()
        cfg = yaml.safe_load(CONFIG.read_text())
        hid, desc, cfg = propose_hypothesis(cfg, env, forced_hyp)
        CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cfg_hash = config_hash()
        run_id = datetime.now(timezone.utc).strftime("exp-%Y%m%dT%H%M%S")
        notify(f"hypothesis: {desc}\nrun: {run_id}", "EXPERIMENT")
        print(f"[experiment] {desc} run={run_id}")

        budget = int(cfg["budget"]["max_wall_clock_min"])
        t0 = time.time()
        tr = subprocess.run(
            [sys.executable, str(LAB / "train.py"),
             "--config", str(CONFIG), "--run-id", run_id,
             "--max-wall-clock-min", str(budget)],
            cwd=str(ROOT))
        if tr.returncode != 0:
            git("checkout", "--", "config.yaml")
            notify(f"train failed (rc={tr.returncode}) - reverted\n{desc}", "REVERT")
            return 1

        adapter = ROOT / "adapters" / run_id
        out_json = LAB / "evals" / f"{run_id}.json"
        ev = subprocess.run(
            [sys.executable, str(LAB / "eval.py"),
             "--adapter", str(adapter),
             "--out", str(out_json), "--config", str(CONFIG)],
            cwd=str(ROOT))
        if ev.returncode != 0 or not out_json.exists():
            git("checkout", "--", "config.yaml")
            notify(f"eval failed - reverted\n{desc}", "REVERT")
            return 1

        res = json.loads(out_json.read_text())
        score = float(res["composite_score"])
        wall = round(time.time() - t0, 1)
        append_result({
            "run_id": run_id, "config_hash": cfg_hash,
            "composite_score": score,
            "compile_pass": res.get("compile_pass"),
            "lint_clean": res.get("lint_clean"),
            "judge_score": res.get("judge_score"),
            "wall_clock_s": wall, "notes": desc,
        })
        save_explored(load_explored() + [{
            "axis": desc.split("set ")[1].split(" =")[0],
            "value": desc.split("= ")[1], "score": score, "run_id": run_id,
        }])

        best = best_score_so_far()
        is_best = score >= best
        if is_best and score > state.get("best_score", -1):
            KEEPERS.mkdir(parents=True, exist_ok=True)
            if adapter.exists():
                shutil.move(str(adapter), str(KEEPERS / run_id))
            git("add", "config.yaml", "results.tsv")
            git("commit", "-q", "-m", f"exp {run_id}: {desc} -> {score:.4f}")
            delta = score - state.get("best_score", 0)
            state.update({"best_run": run_id, "best_score": score,
                          "consecutive_reverts": 0})
            save_state(state)
            notify(
                f"{desc}\nscore {score:.4f} (+{delta:.4f}) - NEW BEST\n"
                f"compile={res.get('compile_pass')} judge={res.get('judge_score')}",
                "KEEP")
        else:
            git("checkout", "--", "config.yaml")
            git("add", "results.tsv")
            git("commit", "-q", "-m", f"exp {run_id}: reverted ({score:.4f})")
            if adapter.exists():
                shutil.rmtree(adapter, ignore_errors=True)
            state["consecutive_reverts"] = state.get("consecutive_reverts", 0) + 1
            save_state(state)
            notify(f"{desc}\nscore {score:.4f} (best {best:.4f}) - reverted",
                   "REVERT")
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
        verdict = ("SHIP" if r["composite_score"] >= 0.9
                   else ("ITERATE" if r["composite_score"] >= 0.7 else "REGRESSION"))
        notify(
            f"keeper {best}\ncomposite {r['composite_score']:.4f} "
            f"(compile={r.get('compile_pass')} judge={r.get('judge_score')})\n"
            f"verdict: {verdict}",
            "EVAL")
    return 0


def phase_report() -> int:
    if not RESULTS.exists():
        notify("no experiments yet", "REPORT")
        return 0
    rows = [l.split("\t") for l in RESULTS.read_text().splitlines()[1:] if l.strip()]
    n = len(rows)
    scores = [(float(r[2]), r[0], r[7] if len(r) > 7 else "")
              for r in rows if _isfloat(r[2])]
    best = max(scores) if scores else (0, "-", "-")
    state = load_state()
    top3 = sorted(scores, reverse=True)[:3]
    body = (
        f"experiments: {n}\nbest: {best[0]:.4f} ({best[1]})\n"
        f"top3: " + ", ".join(f"{s:.3f}" for s, _, _ in top3) + "\n"
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
