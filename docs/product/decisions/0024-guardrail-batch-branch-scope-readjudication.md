# PDR-0024 — External-review guardrail batch re-adjudicated at branch scope and FIXED

Date: 2026-07-04   Status: accepted (within grant; user-directed fix)
Author: Claude (agent)   Amends: PDR-0022 (hygiene-finding disposition only —
its core call, A/B validity preserved / scoring path unblocked, stands).
Related: commit e0b2e84d (fix); PDR-0023 (minimal-patch-set ruling, unaffected).

## Context

The external review's three P1 guardrail findings (mypy return-type errors,
leyline boundary violations, unwhitelisted GPU syncs) were dispositioned in
PDR-0022 as pre-existing baselines — evaluated, not fixed. The user re-raised
them via systematic-debugging with an explicit fix directive. Root-cause
investigation **overturned the scope classification**: both lint gates have
been repository law on `main` since January 2026 (e0f958fb, a58db3fb), and
every violating site was introduced by THIS branch (575d2560, 406aeb25,
68fca06d, a479d776, bb8d3c90 — Jun 25 to Jul 3). "Pre-existing" was true only
relative to the A/B launch commit 6dd80716; relative to `main` — the scope
that governs the eventual PR — the branch owns all of them.

One finding concealed a real defect: `allow_batch_shrink` was a **severed
feature** — the Stage-2 collapse commit (575d2560) landed the call-site
kwargs and a `gpu_preload_gather_shrink` train param without ever adding the
receiving parameter to `SharedGPUGatherBatchIterator.__init__`. Any run with
`experimental_gpu_preload_gather` would TypeError immediately. The flag had
zero wiring (no config, CLI, or test).

## The call

Fix all three gates on branch HEAD now (commit e0b2e84d):

1. **mypy:** explicit `return None` on every non-suppression path of
   `apply_proof_baseline_action_controls` (contract `SuppressSlotResult | None`
   kept — the telemetry split depends on it).
2. **Severed feature DELETED** (param + both kwargs) per no-legacy-code
   policy; it returns with its implementation if ever wanted.
3. **Leyline:** 8 justified `allow_hits` — every flagged type is
   domain-internal machinery (several carry live torch.Tensors, disqualifying
   them as serializable contracts); the cross-domain Shapley contract
   (`CommittedShapleyTopUpPayload`) already lives in leyline/telemetry.
4. **GPU-sync:** 7 justified `allow_hits` — 3 once-per-PPO-update adv-norm
   telemetry, 1 textual false positive (Generator.get_state() is CPU-resident),
   3 R1-harness experiment-only reductions (gated by rng_three_domain_split).

## Options rejected

- **Move the 8 types into leyline:** they are stateful machinery, not
  contracts; none has a cross-domain importer (verified); moving tensors into
  the contracts module invites import cycles.
- **Implement the shrink feature:** speculative — zero demand, zero wiring.
- **Leave red until after the A/B:** the fixes are HEAD-only and cannot touch
  the frozen worktrees (`ab-frozen-6dd80716`, `ab-on-relaunch`); deferral
  bought nothing and left CI red for the PR.

## Evidence

mypy 0 errors (222 files); leyline lint 0 violations; gpu-sync lint 130/130
allowed; 0 stale entries in either whitelist; 57 intervention/proof-baseline
tests green; import smoke clean. Frozen A/B worktrees untouched — nothing
here enters the ON relaunch patch-set (PDR-0023 §3 preserved).

## Reversal triggers

- A whitelisted sync appears in a profiler hot path (>1% of step time) → the
  entry converts to a batched/removed sync, not a broader waiver.
- A whitelisted type gains a cross-domain importer → it migrates to leyline
  (the whitelist entry is deleted, not extended).
- Any lint-gate regression on this branch fails CI — there is no drift path.
