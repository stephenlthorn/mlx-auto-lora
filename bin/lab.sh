#!/usr/bin/env bash
# mlx-auto-lora cron entrypoint. Cron is the loop; this runs ONE phase and exits.
# Usage: lab.sh <PHASE> [extra args]
#
# Env:
#   MLX_AUTO_LORA_ROOT  defaults to $HOME/mlx-auto-lora
set -euo pipefail
ROOT="${MLX_AUTO_LORA_ROOT:-$HOME/mlx-auto-lora}"
cd "$ROOT"
# Load secrets / env (optional, gitignored).
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
# Activate venv if present.
if [ -d "$ROOT/.venv" ]; then
  # shellcheck disable=SC1091
  . "$ROOT/.venv/bin/activate"
fi
exec python lab/run_phase.py --phase "$1" "${@:2}"
