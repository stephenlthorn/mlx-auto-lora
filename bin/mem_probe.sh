#!/usr/bin/env bash
# Memory-envelope probe: run short 2-iter trainings across (num_layers, seq)
# combos and record peak GPU memory or OOM. Defines the bandit's feasible space.
#
# Output: $ROOT/logs/mem_envelope.tsv
#
# Run this once after install, with anything memory-hungry (Ollama, browser
# tabs, other servers) stopped. The bandit reads this file to pick safe
# (num_layers, max_seq_length) combos.
set -uo pipefail
ROOT="${MLX_AUTO_LORA_ROOT:-$HOME/mlx-auto-lora}"
cd "$ROOT"
OUT="logs/mem_envelope.tsv"
mkdir -p logs
printf "num_layers\tseq\tresult\tpeak_gb\n" > "$OUT"

# Default probe grid. Tweak for your machine: smaller for 32-48 GB, larger for
# 192+ GB. The bandit only learns what the probe sees.
PROBES=(
  "2 1024"
  "4 1024" "4 2048"
  "8 1024" "8 2048"
  "12 1024" "12 2048"
)

for combo in "${PROBES[@]}"; do
  nl=$(echo "$combo" | cut -d' ' -f1)
  sq=$(echo "$combo" | cut -d' ' -f2)
  rid="probe-nl${nl}-sq${sq}"
  echo "=== PROBE num_layers=$nl seq=$sq ==="
  log=$(.venv/bin/python lab/train.py --config lab/config.yaml --run-id "$rid" \
        --max-wall-clock-min 8 --iters-override 2 \
        --num-layers-override "$nl" --max-seq-override "$sq" 2>&1)
  peak=$(echo "$log" | grep -oE "Peak mem [0-9.]+ GB" | tail -1 | grep -oE "[0-9.]+")
  if echo "$log" | grep -q "OutOfMemory\|Insufficient Memory"; then
    printf "%s\t%s\tOOM\t-\n" "$nl" "$sq" >> "$OUT"
    echo "  -> OOM"
  elif [ -n "$peak" ]; then
    printf "%s\t%s\tOK\t%s\n" "$nl" "$sq" "$peak" >> "$OUT"
    echo "  -> OK peak=${peak}GB"
  else
    printf "%s\t%s\tUNKNOWN\t-\n" "$nl" "$sq" >> "$OUT"
    echo "  -> UNKNOWN (no peak/OOM found)"
  fi
  rm -rf "adapters/$rid"
done
echo "ENVELOPE_DONE"
cat "$OUT"
