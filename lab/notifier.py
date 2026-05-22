"""notifier.py - tiny pluggable notification layer.

Two backends:

    StdoutNotifier    prints to stdout. Default if no creds are set.
    TelegramNotifier  posts to a chat (optionally inside a topic thread).

Pick a backend by setting environment variables. If TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID are present, Telegram is used; otherwise stdout. No third
backend is built in - if you want Slack, write 30 lines of `post()`.

The notifier interface is one method:

    notifier.post(text: str, tag: str = "") -> bool
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request


class StdoutNotifier:
    """Default backend: print to stdout."""

    def post(self, text: str, tag: str = "") -> bool:
        prefix = f"[{tag}] " if tag else ""
        print(f"{prefix}{text}", flush=True)
        return True


class TelegramNotifier:
    """Telegram bot backend. Reads TELEGRAM_* env vars."""

    def __init__(self, token: str, chat_id: str, thread_id: str | None = None):
        self.token = token
        self.chat_id = chat_id
        self.thread_id = thread_id

    def post(self, text: str, tag: str = "") -> bool:
        body = (f"[{tag}]\n{text}" if tag else text)[:3900]
        # Try with thread_id first (Telegram topic threads); fall back to
        # plain chat if the thread is missing or the chat isn't a forum.
        payloads = [{"chat_id": self.chat_id, "text": body}]
        if self.thread_id:
            payloads.insert(0, {
                "chat_id": self.chat_id,
                "message_thread_id": int(self.thread_id),
                "text": body,
            })
        for payload in payloads:
            try:
                req = urllib.request.Request(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    data=json.dumps(payload).encode(),
                    headers={"content-type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if json.load(resp).get("ok"):
                        return True
            except Exception as e:
                last = e
                continue
        print(f"[notifier] telegram post failed: {last}", file=sys.stderr)
        return False


def from_env() -> StdoutNotifier | TelegramNotifier:
    """Pick a notifier based on environment variables."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    thread = os.environ.get("TELEGRAM_THREAD_ID") or None
    if token and chat:
        return TelegramNotifier(token, chat, thread)
    return StdoutNotifier()
