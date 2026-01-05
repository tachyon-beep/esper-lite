# Phase 0 Experiments & Gates (A0 + C0)

Phase 0 is validated by ROI-style metrics and decision-quality signals, not by “accuracy only”.

## 1) Cohorts (minimum)

Run at least the following:

1) **Control:** baseline blueprint set (no `conv_ladder`, no internal ops).
2) **C0 only:** `conv_ladder` + internal ops available; default slot count (e.g., 3).
3) **A0 only:** more slots on deeper host (e.g., `cifar_scale` + 5 slots); baseline blueprints.
4) **A0 + C0:** deeper host + more slots + ladder seed + internal ops.

## 2) Smoke pack (contracts + telemetry sanity)

Goal: fail fast on shape/contract drift before spending GPU time.

- Use small counts:
  - `--rounds 2 --envs 2 --episode-length 20`
- Confirm:
  - training runs without shape errors
  - `last_action_op` one-hot width matches `NUM_OPS`
  - `SEED_INTERNAL_LEVEL_CHANGED` events appear when internal ops are sampled

## 3) Learning pack (Phase 0 signal check)

Recommended initial settings (CIFAR):

- Task/preset: `cifar_scale` (deeper host, 5 injection points)
- Slots: `r0c0 r0c1 r0c2 r0c3 r0c4`
- Seed limit: `--max-seeds 2` (keeps slot explosion controlled)
- Horizon: `--episode-length 150` (align with long-horizon baseline)
- Scale: `--rounds 50 --envs 8` (enough to observe action mix + entropy trends)

Example run command (adjust `--devices` to your GPU layout):

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --preset cifar_scale \
  --task cifar_scale \
  --slots r0c0 r0c1 r0c2 r0c3 r0c4 \
  --max-seeds 2 \
  --rounds 50 \
  --envs 8 \
  --episode-length 150
```

Telemetry/views to watch during this pack (in addition to end metrics):

- Lever usage: `SEED_INTERNAL_LEVEL_CHANGED` rate and thrash (level changes per 100 steps).
- Economy: effective seed params (rent proxy) vs reward/accuracy (ROI curve).
- Blueprint mix: `conv_heavy` selection frequency and (if derivable) active-param share.
- Decision quality: invalid-action rate, per-head entropy trends (especially `op` head).
- Safety: governor rollback frequency and trigger reasons.

## 4) Metrics to decide Phase 0 success

**Primary (ROI-style):**

- Accuracy ROI: `(final_acc - baseline_acc) / (added_active_params)` (or rent proxy).
- Blueprint mix shift: reduced selection frequency of `conv_heavy` in treatment.

**Decision-quality:**

- Internal ops used non-trivially when ladder is present (≥10% of non-WAIT ops).
- Invalid-action rate not materially worse than control.
- Entropy: no sustained collapse on op/slot heads.

**Safety/stability:**

- Governor rollback rate does not increase materially.
- Seed lifecycle remains stable (no unusual prune/fossilize churn).

## 5) Gate checklist to unlock Phase 1

Advance to Phase 1 if all are true:

- Internal levers are learnably used (usage + non-thrashing).
- Conv-heavy dominance measurably reduced without worse ROI.
- No systemic instability introduced (entropy collapse, governor events, throughput regressions).
