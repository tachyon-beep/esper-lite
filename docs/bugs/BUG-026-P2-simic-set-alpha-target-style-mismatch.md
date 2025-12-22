# BUG-026: `SET_ALPHA_TARGET` ignored `style` (alpha_algorithm mismatch)

- **Title:** `SET_ALPHA_TARGET` did not apply the action-selected alpha algorithm (`style_idx`), causing action/effect mismatch.
- **Severity:** P2 (major correctness/contract issue for action semantics + telemetry).
- **Status:** Fixed.

## Symptom

During `SET_ALPHA_TARGET`, Simic executed `SeedSlot.set_alpha_target(..., alpha_algorithm=None)`, leaving the seed’s existing `alpha_algorithm` unchanged (e.g., `GATE` after gated germination), even though the sampled `FactoredAction` included a `style` head that encodes the algorithm choice.

This made the recorded action (`style`) diverge from the actual effect whenever retargeting occurred on a seed whose existing algorithm differed from the default.

## Root Cause

The GerminationStyle refactor fused `(blend, alpha_algorithm)` into a single `style` head, but only GERMINATE used it:

- `compute_action_masks()` kept `style` effectively constant outside GERMINATE.
- The policy network additionally forced `style` to a single default whenever `op != GERMINATE`.
- The environment’s `SET_ALPHA_TARGET` execution path explicitly passed `alpha_algorithm=None`.

Net result: `style` was sampled/logged but not applied during `SET_ALPHA_TARGET`.

## Fix (Implemented)

Treat `style` as causally relevant for `SET_ALPHA_TARGET` (for algorithm selection), and apply it end-to-end:

- **Masks:** Open `style_mask` when either GERMINATE is possible or HOLD retargeting is possible.
  - `src/esper/tamiyo/policy/action_masks.py`
- **Policy:** Only force `style` to a default for ops where it is truly irrelevant (WAIT/PRUNE/FOSSILIZE/ADVANCE), not for `SET_ALPHA_TARGET`.
  - `src/esper/simic/agent/network.py`
- **Execution:** Pass `alpha_algorithm = STYLE_ALPHA_ALGORITHMS[style_idx]` into Kasmina on `SET_ALPHA_TARGET`.
  - `src/esper/simic/training/vectorized.py`
- **Credit assignment:** Mark `style` as causally relevant for `SET_ALPHA_TARGET` so PPO can learn it.
  - `src/esper/simic/agent/advantages.py`
  - `src/esper/simic/agent/ppo.py`

## Validation

- Masking test updated: `tests/tamiyo/policy/test_action_masks.py`
- Full suite passing: `PYTHONPATH=src .venv/bin/python -m pytest`

