#!/usr/bin/env bash
set -euo pipefail

# Phase 1 risk-reduction helper: locate all lifecycle naming touch points
# without making any behavioral/code changes.
#
# Usage:
#   ./scripts/lifecycle_phase1_inventory.sh

echo "== Phase 1 Inventory: lifecycle names (src + tests + top-level docs) =="
echo

echo "-- PROBATIONARY (files) --"
rg -l "PROBATIONARY" src tests README.md ROADMAP.md || true
echo

echo "-- probationary (any case) (files) --"
rg -li "\\bprobationary\\b" src tests README.md ROADMAP.md || true
echo

echo "-- CULLED (files) --"
rg -l "CULLED" src tests README.md ROADMAP.md || true
echo

echo "-- culled (any case) (files) --"
rg -li "\\bculled\\b" src tests README.md ROADMAP.md || true
echo

echo "-- cull / culling / culled (any case) (files) --"
rg -li "\\bcull\\w*\\b" src tests README.md ROADMAP.md || true
echo

echo "-- cull( call sites (files) --"
rg -l "\\bcull\\(" src tests || true
echo

echo "-- LifecycleOp.CULL / OP_CULL / is_cull (files) --"
rg -l "LifecycleOp\\.CULL|OP_CULL|is_cull\\b" src tests || true
echo

echo "-- Telemetry SEED_CULLED (files) --"
rg -l "SEED_CULLED" src tests README.md ROADMAP.md || true
echo

echo "== Phase 1 Inventory: schema/serialization boundaries (key grep targets) =="
echo
echo "-- SeedStage enum + transitions --"
rg -n "class SeedStage\\b|VALID_TRANSITIONS\\b|PROBATIONARY\\s*=|CULLED\\s*=" src/esper/leyline/stages.py || true
echo

echo "-- LifecycleOp enum + lookup tables --"
rg -n "class LifecycleOp\\b|OP_NAMES\\b|OP_CULL\\b" src/esper/leyline/factored_actions.py || true
echo

echo "-- TelemetryEventType --"
rg -n "class TelemetryEventType\\b|SEED_CULLED\\b" src/esper/leyline/telemetry.py || true
echo

echo "-- SeedState serialization (stage/op ints) --"
rg -n "def to_dict\\(|def from_dict\\(|\"stage\":|SeedStage\\(" src/esper/kasmina/slot.py || true
echo

echo "-- Action tensor / head schemas (enum indices, head sizes) --"
rg -n "FactoredAction\\b|ActionHead\\b|LifecycleOp\\b|SeedStage\\b" src/esper/leyline/factored_actions.py src/esper/tamiyo/policy/action_masks.py src/esper/simic/training || true
echo

echo "-- Reward stage constants (PBRS potentials) --"
rg -n "STAGE_POTENTIALS\\b|STAGE_PROBATIONARY\\b|LifecycleOp\\.CULL" src/esper/simic/rewards/rewards.py || true
echo

echo "Done."
