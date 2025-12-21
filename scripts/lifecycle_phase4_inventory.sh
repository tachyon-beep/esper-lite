#!/usr/bin/env bash
set -euo pipefail

# Phase 4 risk-reduction helper: inventory all remaining "cull" touchpoints and
# classify them by subsystem so the CULL → PRUNE rename lands atomically.
#
# This script is intentionally read-only. It is meant to be run BEFORE wiring
# changes so we can see the full blast radius.
#
# Usage:
#   ./scripts/lifecycle_phase4_inventory.sh

echo "== Phase 4 Inventory: CULL / PRUNE rename blast radius =="
echo

echo "## 1) Core action schema (Leyline) — policy op surface"
echo "-- LifecycleOp / OP_* constants --"
rg -n "class LifecycleOp\\b|\\bCULL\\b|OP_CULL\\b|\\bis_cull\\b|OP_NAMES\\b" \
  src/esper/leyline/factored_actions.py || true
echo

echo "## 2) Policy masking (Tamiyo) — op validity + stage gating"
echo "-- Action masks (CULL validity / MIN_CULL_AGE) --"
rg -n "LifecycleOp\\.CULL|\\bCULL\\b|MIN_CULL_AGE\\b|_CULLABLE_STAGES\\b" \
  src/esper/tamiyo/policy/action_masks.py || true
echo

echo "## 3) Rewards (Simic) — intervention costs + shaping"
echo "-- Reward logic referencing CULL --"
rg -n "LifecycleOp\\.CULL|\\bCULL\\b|cull_cost\\b|INTERVENTION_COSTS\\b" \
  src/esper/simic/rewards/rewards.py || true
echo

echo "## 4) Training runtime (Simic) — action execution + bookkeeping"
echo "-- Vectorized training / helpers using OP_CULL or cull_seed --"
rg -n "OP_CULL\\b|LifecycleOp\\.CULL|\\bcull_seed\\b|\\.cull\\(" \
  src/esper/simic/training src/esper/simic/agent || true
echo

echo "## 5) Kasmina runtime — slot engine + host façade"
echo "-- SeedSlot.cull() definition + call sites --"
rg -n "def cull\\b|\\.cull\\(" src/esper/kasmina || true
echo

echo "## 6) Safety mechanisms (Tolaria) — governor forced removal"
echo "-- Governor rollback uses slot.cull(...) --"
rg -n "\\.cull\\(" src/esper/tolaria || true
echo

echo "## 7) Tests — expectations that will change under Phase 4"
echo "-- Tests referencing CULL/LifecycleOp.CULL/slot.cull() --"
rg -n "LifecycleOp\\.CULL|OP_CULL\\b|\\bcull_seed\\b|\\.cull\\(" tests || true
echo

echo "## 8) Docs — legacy terminology to scrub"
echo "-- Docs referencing 'cull' --"
rg -n "\\bcull\\w*\\b" docs README.md ROADMAP.md || true
echo

echo "Done."

