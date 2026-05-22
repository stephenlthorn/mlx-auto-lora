#!/usr/bin/env bash
# Install the mlx-auto-lora cron schedule. Idempotent: removes any prior block
# tagged MLX-AUTO-LORA and re-adds. Run AFTER a manual EXPERIMENT verifies green.
#
# Default schedule:
#   EXPERIMENT  every 90 min, 20:00-03:30 (6 runs/night)
#   EVAL        04:05 daily (full keeper eval)
#   REPORT      09:00 daily (digest)
#   HEALTH      every 15 min (silent unless problem)
#   RETRAIN_GATE Sun 22:00
#
# Tweak the cron lines below to your own schedule.
set -euo pipefail
ROOT="${MLX_AUTO_LORA_ROOT:-$HOME/mlx-auto-lora}"
J="$ROOT/bin/lab.sh"
LOG="$ROOT/logs/cron.log"
CAF="/usr/bin/caffeinate -s"  # macOS: keep machine awake during a run
TAG="MLX-AUTO-LORA"

mkdir -p "$(dirname "$LOG")"

existing=$(crontab -l 2>/dev/null || true)
current=$(printf '%s\n' "$existing" | sed "/# >>> ${TAG} >>>/,/# <<< ${TAG} <<</d")

block=$(cat <<EOF
# >>> ${TAG} >>>
# Autonomous LoRA loop. Cron is the loop.
*/15 * * * * ${CAF} ${J} HEALTH >> ${LOG} 2>&1
0 20,23,2 * * * ${CAF} ${J} EXPERIMENT >> ${LOG} 2>&1
30 21,0,3 * * * ${CAF} ${J} EXPERIMENT >> ${LOG} 2>&1
5 4 * * * ${CAF} ${J} EVAL >> ${LOG} 2>&1
0 9 * * * ${CAF} ${J} REPORT >> ${LOG} 2>&1
0 22 * * 0 ${CAF} ${J} RETRAIN_GATE >> ${LOG} 2>&1
# <<< ${TAG} <<<
EOF
)

printf '%s\n%s\n' "$current" "$block" | crontab -
echo "Installed ${TAG} cron block:"
crontab -l | sed -n "/# >>> ${TAG} >>>/,/# <<< ${TAG} <<</p"
