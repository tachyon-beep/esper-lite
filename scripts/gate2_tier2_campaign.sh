#!/usr/bin/env bash
# GATE 2 Tier-2 paired campaign (pre-registration:
# docs/analysis/2026-07-02-gate2-learnability-probe-preregistration.md).
#
# 5 same-seed pairs (control vs retro-write injection), K rounds each, run
# across 2 local GPUs with 2 concurrent runs per GPU (4 lanes). Seeds 51-55
# are disjoint from the n=5 J-read seeds (41-45) by design.
#
# Usage: scripts/gate2_tier2_campaign.sh [K_ROUNDS] [DELTA_RAW]
set -euo pipefail

K="${1:-20}"
DELTA="${2:-2.0}"
SEEDS=(51 52 53 54 55)
ARMS=(control retro)
OUTDIR="telemetry/gate2_probe"
mkdir -p "$OUTDIR"

# Build the run list: pair-major so both arms of a seed start close together.
RUNS=()
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    RUNS+=("${arm}:${seed}")
  done
done

# 4 lanes: lane i uses GPU (i % 2). Each lane consumes runs i, i+4, i+8, ...
run_lane() {
  local lane="$1"
  local gpu=$((lane % 2))
  local i
  for ((i = lane; i < ${#RUNS[@]}; i += 4)); do
    IFS=':' read -r arm seed <<<"${RUNS[$i]}"
    local out="${OUTDIR}/tier2_${arm}_s${seed}.jsonl"
    local log="${OUTDIR}/tier2_${arm}_s${seed}.log"
    echo "[lane ${lane}] gpu=cuda:${gpu} arm=${arm} seed=${seed} K=${K} -> ${out}"
    uv run python scripts/gate2_learnability_probe.py \
      --arm "${arm}" --delta-raw "${DELTA}" --seed "${seed}" \
      --rounds "${K}" \
      --device "cuda:${gpu}" --gpu-preload \
      --out "${out}" >"${log}" 2>&1
    echo "[lane ${lane}] DONE arm=${arm} seed=${seed}"
  done
}

for lane in 0 1 2 3; do
  run_lane "${lane}" &
done
wait
echo "GATE 2 Tier-2 campaign complete: $(ls ${OUTDIR}/tier2_*.summary.json 2>/dev/null | wc -l)/10 summaries"
