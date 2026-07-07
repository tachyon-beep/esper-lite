# PDR-0047 — Escrow telescoping fix landed; Stage-2 still owner-gated

Date: 2026-07-08   Status: accepted
Author: Codex (GPT-5)   Owner sign-off: n/a (correctness fix; no push/freeze/A/B)
Related: PDR-0044, PDR-0046, epic esper-lite-f25b71c165, task esper-lite-3defe42928,
task esper-lite-e6382020d2.
Code commit: `89b50a16`.

## Context

The EV-stabilization Now bet had a correctness issue independent of the Stage-2
acceptance harness: `escrow_delta_clip` clipped the difference of the escrow potential.
When the clip bound, the telescoping cancellation no longer held, so the shaping term could
silently shift the policy optimum.

The fix is the Markovian augmentation path chosen by the tracker task: remove reward-path
delta clipping and expose the state needed by the critic to model the dynamic potential.

## The call

Treat escrow telescoping as fixed for the current branch:

- Contribution reward no longer clips `escrow_credit_target - escrow_credit_prev`.
- Obs V3 schema v2 exposes min-over-window stable validation accuracy as a base feature.
- Obs V3 schema v2 exposes per-slot escrow credit as a symlog-compressed slot feature.
- The non-blueprint observation contract moves from 116 to 120 dims for the default
  3-slot configuration; full network input moves from 128 to 132 dims after blueprint
  embeddings.
- PPO checkpoint compatibility is bumped to v4 because old checkpoints target the prior
  observation contract.

This does not change the Stage-2 product gate. The paired fresh-init HRA ON/OFF A/B still
requires owner freeze of the remaining thresholds before any ON run.

## Verification recorded

- Focused feature/reward/golden/checkpoint suites passed:
  - `tests/tamiyo/policy/test_features.py::test_batch_obs_to_features_exposes_escrow_markov_state`
  - escrow property, ledger, pure, and wiring suites
  - PPO feature, checkpoint, deterministic OFF-leg, and HRA ON-leg golden suites
- Full default `PYTHONPATH=src uv run pytest`: 5292 passed, 5 skipped, 449 deselected.
- `uv run ruff check` on touched Python files passed.
- `uv run python scripts/lint_leyline_types.py` passed.
- `git diff --check` passed.
- Full mypy now has no errors from the escrow feature change; it still reports existing
  non-touched errors in PPO metrics, Sanctum, preview, and vectorized data-loader wiring.
- Full defensive-pattern and GPU-sync lints still report existing non-touched violations.

## Reversal trigger

- If a future reward-mode comparison shows optimum drift after this change, reopen the
  reward-shaping invariance question before running Stage-2.
- If the added observation fields are unavailable in a real vectorized trainer path, fix the
  producer contract directly; do not add defensive fallbacks to feature extraction.
- If checkpoint migration becomes necessary, write an explicit migration plan. Do not load
  pre-v4 PPO checkpoints through compatibility shims.
