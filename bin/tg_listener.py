#!/usr/bin/env python3
"""tg_listener.py - optional Telegram command listener for mlx-auto-lora.

Polls the bot for messages in a specific chat / topic thread and translates
recognised commands into state.json edits. Lets you /lab_pause the cron from
your phone, /lab_status to peek at progress, /lab_hypothesis H4 to force the
next experiment, /lab_deploy to promote a candidate, etc.

This is OPTIONAL. The cron loop runs fine without it; you'll just have to ssh
in to pause or inspect.

Required env vars (set them in $MLX_AUTO_LORA_ROOT/.env):
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID         the chat to listen in
    TELEGRAM_THREAD_ID       (optional) restrict to one topic thread

Commands (in the configured chat/thread only):
    /lab_pause   /lab_freeze    - stop new EXPERIMENTs
    /lab_resume  /lab_thaw      - re-enable EXPERIMENTs
    /lab_status  /lab_results   - current state + last 3 results
    /lab_hypothesis <H>         - force next EXPERIMENT to hypothesis H
    /lab_hypothesis clear       - clear forced hypothesis
    /lab_deploy <run_id>        - copy adapter to keepers/ + set prod_score
    /lab_kill                   - pause + SIGINT live run (emergency)
    /lab_help

Run as a launchd agent (idempotent installer):
    bin/install_listener.sh

Or directly for testing:
    python bin/tg_listener.py --once
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("MLX_AUTO_LORA_ROOT", Path.home() / "mlx-auto-lora"))
STATE = ROOT / "state.json"
RESULTS = ROOT / "lab" / "results.tsv"
ADAPTERS = ROOT / "adapters"
KEEPERS = ROOT / "adapters" / "keepers"
LOCK = ROOT / "lab.lock"
OFFSET_FILE = ROOT / "logs" / "tg_offset.txt"
LOG = ROOT / "logs" / "tg_listener.log"

POLL_INTERVAL_S = 20
LONG_POLL_TIMEOUT = 15


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _required_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(
            f"{name} not set. Put it in $MLX_AUTO_LORA_ROOT/.env or export it.")
    return v


def _bot_token() -> str:
    return _required_env("TELEGRAM_BOT_TOKEN")


def _chat_id() -> str:
    return _required_env("TELEGRAM_CHAT_ID")


def _thread_id() -> str | None:
    return os.environ.get("TELEGRAM_THREAD_ID") or None


def _api(token: str, method: str, payload: dict | None = None,
         timeout: int = 30) -> dict:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = json.dumps(payload or {}).encode() if payload else None
    req = urllib.request.Request(
        url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _reply(token: str, chat_id: str, thread_id: str | None, text: str) -> None:
    payload = {"chat_id": chat_id, "text": text[:4096], "parse_mode": "Markdown"}
    if thread_id:
        payload["message_thread_id"] = int(thread_id)
    try:
        _api(token, "sendMessage", payload)
    except Exception:
        payload.pop("message_thread_id", None)
        try:
            _api(token, "sendMessage", payload)
        except Exception as e:
            _log(f"[reply-fail] {e}")


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {
        "best_run": None, "best_score": -1.0,
        "experiment_cron_paused": False, "consecutive_reverts": 0,
        "prod_score": -1.0,
    }


def _save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=2))


def _load_offset() -> int:
    if OFFSET_FILE.exists():
        try:
            return int(OFFSET_FILE.read_text().strip())
        except Exception:
            pass
    return 0


def _save_offset(offset: int) -> None:
    OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    OFFSET_FILE.write_text(str(offset))


def cmd_pause(_token: str, _args: list[str]) -> str:
    s = _load_state()
    s["experiment_cron_paused"] = True
    _save_state(s)
    return "PAUSED - no new EXPERIMENT runs will start."


def cmd_resume(_token: str, _args: list[str]) -> str:
    s = _load_state()
    s["experiment_cron_paused"] = False
    _save_state(s)
    return "RESUMED - EXPERIMENT cron re-enabled."


def cmd_status(_token: str, _args: list[str]) -> str:
    s = _load_state()
    lines = ["*mlx-auto-lora status*"]
    lines.append(f"best\\_run: `{s.get('best_run','-')}`")
    lines.append(f"best\\_score: `{s.get('best_score',-1.0):.4f}`")
    lines.append(f"prod\\_score: `{s.get('prod_score',-1.0):.4f}`")
    lines.append(f"paused: `{s.get('experiment_cron_paused',False)}`")
    lines.append(f"consecutive\\_reverts: `{s.get('consecutive_reverts',0)}`")
    if s.get("forced_hypothesis"):
        lines.append(f"forced\\_hypothesis: `{s['forced_hypothesis']}`")
    if RESULTS.exists():
        rows = [r for r in RESULTS.read_text().splitlines()
                if r and not r.startswith("run_id")]
        if rows:
            lines.append("\n*Last results:*")
            for row in rows[-3:]:
                parts = row.split("\t")
                if len(parts) >= 3:
                    lines.append(f"`{parts[0]}` -> `{parts[2]}`")
    if LOCK.exists():
        try:
            info = json.loads(LOCK.read_text())
            lines.append(f"\nlock held by PID `{info.get('pid')}` phase=`{info.get('phase')}`")
        except Exception:
            lines.append("\nlock file present (malformed)")
    return "\n".join(lines)


def cmd_hypothesis(_token: str, args: list[str]) -> str:
    s = _load_state()
    if not args:
        cur = s.get("forced_hypothesis")
        return (f"Current forced hypothesis: `{cur or 'none'}`. "
                f"Use `/lab_hypothesis <H>` or `/lab_hypothesis clear`.")
    val = args[0].strip().upper()
    if val == "CLEAR":
        s.pop("forced_hypothesis", None)
        _save_state(s)
        return "Forced hypothesis cleared - bandit will pick freely."
    if not val.startswith("H") or not val[1:].split("-")[0].isdigit():
        return f"Unknown hypothesis `{val}`. Expected format: `H4`, `H11`, etc."
    s["forced_hypothesis"] = val
    _save_state(s)
    return f"Next EXPERIMENT will be forced to `{val}`."


def cmd_deploy(_token: str, args: list[str]) -> str:
    if not args:
        return "Usage: `/lab_deploy <run_id>`"
    run_id = args[0].strip()
    src = ADAPTERS / run_id
    if not src.exists():
        src = KEEPERS / run_id  # might already be a keeper
    if not src.exists():
        return f"Adapter not found: `{run_id}`"
    KEEPERS.mkdir(parents=True, exist_ok=True)
    dst = KEEPERS / run_id
    if dst != src:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(str(src), str(dst))
    prod_score: float = -1.0
    if RESULTS.exists():
        for row in RESULTS.read_text().splitlines():
            if row.startswith(run_id + "\t"):
                try:
                    prod_score = float(row.split("\t")[2])
                except Exception:
                    pass
    s = _load_state()
    s["prod_score"] = prod_score
    _save_state(s)
    return (f"Deployed `{run_id}` -> `adapters/keepers/{run_id}`\n"
            f"prod\\_score set to `{prod_score:.4f}`")


def cmd_kill(_token: str, _args: list[str]) -> str:
    lines = ["KILL received - pausing and attempting to stop live run."]
    s = _load_state()
    s["experiment_cron_paused"] = True
    _save_state(s)
    lines.append("Paused.")
    if LOCK.exists():
        try:
            info = json.loads(LOCK.read_text())
            pid = int(info.get("pid", -1))
            if pid > 0:
                import subprocess
                r = subprocess.run(["kill", "-SIGINT", str(pid)],
                                   capture_output=True)
                lines.append(f"Sent SIGINT to PID {pid} (rc={r.returncode}).")
            else:
                lines.append("Lock present but no valid PID.")
        except Exception as e:
            lines.append(f"Could not parse lock: {e}")
    else:
        lines.append("No active lock found.")
    return "\n".join(lines)


def cmd_help(_token: str, _args: list[str]) -> str:
    return (
        "*mlx-auto-lora commands*\n"
        "/lab\\_pause  /lab\\_freeze   - stop new EXPERIMENTs\n"
        "/lab\\_resume /lab\\_thaw     - re-enable EXPERIMENTs\n"
        "/lab\\_status /lab\\_results  - state + last 3 results\n"
        "/lab\\_hypothesis H4         - force next EXPERIMENT to H4\n"
        "/lab\\_hypothesis clear      - let bandit pick freely\n"
        "/lab\\_deploy <run\\_id>      - promote adapter to keepers/\n"
        "/lab\\_kill                  - pause + SIGINT live run\n"
        "/lab\\_help"
    )


COMMANDS = {
    "/lab_pause": cmd_pause, "/lab_freeze": cmd_pause,
    "/lab_resume": cmd_resume, "/lab_thaw": cmd_resume,
    "/lab_status": cmd_status, "/lab_results": cmd_status,
    "/lab_hypothesis": cmd_hypothesis,
    "/lab_deploy": cmd_deploy,
    "/lab_kill": cmd_kill,
    "/lab_help": cmd_help,
}


def _is_project_chat(msg: dict, chat_id: str, thread_id: str | None) -> bool:
    cid = str(msg.get("chat", {}).get("id", ""))
    if cid != chat_id:
        return False
    if thread_id is None:
        return True
    tid = str(msg.get("message_thread_id", ""))
    return tid == thread_id


def _parse_command(text: str) -> tuple[str, list[str]]:
    parts = text.strip().split()
    if not parts:
        return ("", [])
    cmd = parts[0].lower().split("@")[0]
    return (cmd, parts[1:])


def handle_update(token: str, chat_id: str, thread_id: str | None,
                  update: dict) -> None:
    msg = update.get("message") or update.get("channel_post")
    if not msg:
        return
    if not _is_project_chat(msg, chat_id, thread_id):
        return
    text = (msg.get("text") or "").strip()
    if not text.startswith("/"):
        return
    cmd, args = _parse_command(text)
    handler = COMMANDS.get(cmd)
    if handler is None:
        return
    from_user = msg.get("from", {}).get("username", "?")
    _log(f"[cmd] {cmd} {args} from @{from_user}")
    try:
        reply_text = handler(token, args)
    except Exception as e:
        _log(f"[cmd-error] {e}")
        reply_text = f"Error: {e}"
    _reply(token, chat_id, thread_id, reply_text)


def run(once: bool = False) -> None:
    token = _bot_token()
    chat_id = _chat_id()
    thread_id = _thread_id()
    offset = _load_offset()
    _log(f"[start] listener running (chat={chat_id} thread={thread_id} offset={offset})")
    try:
        _reply(token, chat_id, thread_id,
               "mlx-auto-lora listener online. Send /lab_help for commands.")
    except Exception as e:
        _log(f"[startup-announce] {e}")

    while True:
        try:
            result = _api(token, "getUpdates", {
                "offset": offset,
                "timeout": LONG_POLL_TIMEOUT,
                "allowed_updates": ["message"],
            }, timeout=LONG_POLL_TIMEOUT + 10)
            updates = result.get("result", [])
            for upd in updates:
                handle_update(token, chat_id, thread_id, upd)
                offset = upd["update_id"] + 1
                _save_offset(offset)
        except urllib.error.URLError as e:
            _log(f"[poll-error] network: {e}")
        except Exception as e:
            _log(f"[poll-error] {e}")

        if once:
            break
        time.sleep(POLL_INTERVAL_S)


def main() -> None:
    ap = argparse.ArgumentParser(description="mlx-auto-lora Telegram listener")
    ap.add_argument("--once", action="store_true",
                    help="Single getUpdates pass (for testing)")
    args = ap.parse_args()

    def _shutdown(_sig, _frame):
        _log("[shutdown] signal received, exiting.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    run(once=args.once)


if __name__ == "__main__":
    main()
