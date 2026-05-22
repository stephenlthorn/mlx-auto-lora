#!/usr/bin/env bash
# install_listener.sh - install the optional Telegram listener as a launchd agent.
# macOS only. Idempotent: unloads prior agent (if any) and installs fresh.
#
# Skip this if you don't use the Telegram remote-control bot - the cron loop
# works without it.
#
# Requires TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (and optionally TELEGRAM_THREAD_ID)
# in $MLX_AUTO_LORA_ROOT/.env.
set -euo pipefail
ROOT="${MLX_AUTO_LORA_ROOT:-$HOME/mlx-auto-lora}"
LABEL="com.mlx-auto-lora.tg-listener"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
VENV="$ROOT/.venv"
SCRIPT="$ROOT/bin/tg_listener.py"
LOG_OUT="$ROOT/logs/tg_listener.log"
LOG_ERR="$ROOT/logs/tg_listener.err"
PYTHON="$VENV/bin/python3"
ENV_FILE="$ROOT/.env"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: venv not found at $VENV"
  echo "Create one with:"
  echo "  python3 -m venv $VENV && $VENV/bin/pip install -r requirements.txt"
  exit 1
fi
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found. Copy .env.example to .env and set Telegram vars."
  exit 1
fi

mkdir -p "$ROOT/logs"

if launchctl list | grep -q "$LABEL" 2>/dev/null; then
  echo "Unloading existing $LABEL ..."
  launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null \
    || launchctl unload "$PLIST" 2>/dev/null || true
fi

# Build PLIST EnvironmentVariables block from .env. Each KEY=VAL becomes
# a <key>/<string> pair so the listener can read the secrets without
# sourcing a shell file.
env_block=""
while IFS= read -r line; do
  case "$line" in
    ''|\#*) continue ;;
  esac
  key=${line%%=*}
  val=${line#*=}
  # Strip surrounding quotes
  val=${val#\"}; val=${val%\"}
  val=${val#\'}; val=${val%\'}
  env_block+="    <key>${key}</key>"$'\n'
  env_block+="    <string>${val}</string>"$'\n'
done <"$ENV_FILE"

# Make sure MLX_AUTO_LORA_ROOT is set even if not in .env
case "$env_block" in
  *"<key>MLX_AUTO_LORA_ROOT</key>"*) ;;
  *) env_block+="    <key>MLX_AUTO_LORA_ROOT</key>"$'\n'
     env_block+="    <string>${ROOT}</string>"$'\n' ;;
esac

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON}</string>
    <string>${SCRIPT}</string>
  </array>
  <key>KeepAlive</key>
  <true/>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${LOG_OUT}</string>
  <key>StandardErrorPath</key>
  <string>${LOG_ERR}</string>
  <key>ThrottleInterval</key>
  <integer>30</integer>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key>
    <string>${HOME}</string>
    <key>PATH</key>
    <string>${VENV}/bin:/usr/local/bin:/usr/bin:/bin</string>
${env_block}  </dict>
</dict>
</plist>
EOF

echo "Installing $PLIST ..."
launchctl bootstrap "gui/$(id -u)" "$PLIST"
sleep 1
launchctl list | grep "$LABEL" \
  && echo "${LABEL} loaded" \
  || echo "WARN: agent may not be running - check $LOG_ERR"
