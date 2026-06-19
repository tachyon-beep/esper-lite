# Sprint 1+2 — Trust & Rules fixes (P0-1, P0-3, P1-1, P1-2, P1-7, P1-8)

**Source:** `docs/arch-analysis-2026-06-17-0037/06-architect-handover.md` (Sprint 1 = P0s, Sprint 2 = P1s)
**Branch:** `sprint12-trust-rules` (off `0.1.1` @ `b1935d39`)
**Date:** 2026-06-17
**Status:** ready → in progress

## Design + review provenance

Designed and reality-checked by a multi-agent workflow (`sprint12-design`):
`reviewed_by`: drl-expert (P0-1, P0-3, P1-2), axiom-python-engineering:python-code-reviewer
(P1-1, P1-8), axiom-python-engineering:refactoring-architect (P1-7),
axiom-planning:plan-review-reality (all, hallucination-hunt), axiom-planning:plan-review-synthesizer.
Verdicts: P1-2 = GO; P0-1/P0-3/P1-1/P1-7/P1-8 = GO_WITH_FIXES (corrections folded in below). No BLOCK.

## Implementation order (strict linear chain; resolves file overlaps by sequencing)

1. **P1-8** — delete dead code: `PerformanceBudgets`/`DEFAULT_BUDGETS` (leyline/telemetry.py:150-172
   + __init__ import @744-745 + __all__ @1087-1088) and `SlotConfigProtocol` (karn/contracts.py:76-109).
   *Fix:* trim resulting blank runs to ≤2; guard tests use `'Name' not in dir(module)` idiom.
2. **P1-2** — **DELETE** the dead `grad_norm_host`/`grad_norm_seed` obs fields (leyline/signals.py:52-53 +
   tests/strategies.py:268-269). Zero writers/readers; live gradient signal flows via
   `seed_gradient_norm_ratio` per-slot. **OBS dim unchanged (base stays 23).** No wiring-in.
3. **P1-1** — narrow silent `except Exception` in nissa/output.py (221,264,273,662,810) + scripts/train.py
   (1064 re-raise→logger, 1083); fix genuine bug-hiding (dropped-counter not incremented on emit failure;
   `flush()` returning True on broken queue; `training_wrapper` not setting `shutdown_event` on crash).
   *Fix:* read `_join_with_timeout` body first; no unused `as e` (F841); flush() caller audit
   (tests/integration/test_phase_profiler_integration.py:191).
4. **P1-7** — move `RewardComponentsTelemetry` + `ObservationStatsTelemetry` (the two dataclasses only,
   plus `_parse_bool_field`) into new `leyline/telemetry_contracts.py`; delete originating defs + late
   imports; update ~15 src + ~11 test call sites in the same commit. Excise `new_drip_state` (a simic
   type) from `RewardComponentsTelemetry` via a simic-local result return (not the telemetry bag).
   Keep `compute_observation_stats` in simic. Add `tests/leyline/test_import_graph.py` asserting
   leyline imports no domain module. Add mypy strict override for the new submodule.
5. **P0-3** — governor-rollback credit: **keep full-episode forfeit** (deliberate P1-ROLLBACK-TAIL
   decision; last-k rejected — no principled k, re-opens reward-hacking hole), but (a) replace literal
   `0.0` with leyline `ROLLBACK_FORFEIT_REWARD`, (b) add `rollback_count` + `rollback_steps_zeroed`
   buffer counters surfaced via the buffer snapshot/coordinator (NOT the governor event — steps_zeroed
   is unknown at governor emit time). `mark_terminal_with_penalty` returns `RollbackPenaltyResult`
   (simic-local). *Fix:* insert new fields AFTER existing rollback-attribution block (~line 185/305),
   zero counters in `reset()`, update 4 test files. Lands under today's Q-semantics (regressions green).
6. **P0-1** — **separate op-independent `V(s)` head** for baseline/GAE/EV; rename `value_head`→`q_head`,
   keep Q(s,op) as a small **detached** aux head (`q_aux_coef = 0.5*value_coef`, target = MC returns)
   so op-q telemetry stays meaningful. `self.values` semantics flip Q→V (GAE formula unchanged).
   *Fix:* also update crash sites ppo_agent.py:373 (`critic_params`), 1325-1330 (grad-norm `'value'` +
   `value_head`), lstm_bundle.py:209 (`output['value']`); optimizer param-group must cover BOTH heads;
   pin `EvalResult.q_value` position; document `ForwardResult.value` Q→V shift. **Checkpoint break is
   intended** (No-Legacy): add `VALUE_HEAD_SCHEMA_VERSION=2` to leyline; old checkpoints fail strict load.

## File-overlap resolution
- **leyline** (`__init__.py` __all__ + `telemetry.py`): P1-8 → P1-7 → P0-3 (each rebases onto cleaner file).
- **rollout_buffer.py**: P0-3 (§757-814 + new fields) and P0-1 (§509-557) edit **disjoint line ranges**;
  the only coupling is the semantic flip of `self.values`, so P0-3 lands first and P0-1 explicitly
  re-baselines P0-3's two GAE-returns regressions from Q→V.

## Merge gate (user decision 2026-06-17)
Build all 6; **merge items 1–5 to `0.1.1`**; **hold P0-1 committed-but-unmerged** on the branch until the
in-flight 200-ep K=4/K=1 EV experiment timing is decided (P0-1 breaks checkpoints and changes what EV measures).

## TDD discipline
Every item writes its named failing tests first, then implements to green. Full `tests/simic/` +
touched-area suites must pass before each commit. Adversarial multi-reviewer pass over the whole diff
before any merge.
