# PDR-0039 — Stage-2 MAJOR-1 telemetry wrapper: architecture + pure layer (S1–S4) landed

Date: 2026-07-07   Status: accepted (within grant: spec / dispatch / build acceptance code for the active bet)
Author: Claude (Opus 4.8)   Owner sign-off: n/a (autonomous within grant; owner dispatched "build the wrapper" + confirmed grant this session)
Related: PDR-0037 (acceptance harness — scorer built, wrapper is its "remaining half"), PDR-0029 (MAJOR-1/-3 folded into Stage-2 acceptance), gate doc `docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md`, task esper-lite-2a4b56e719, epic esper-lite-f25b71c165. Commit `da2c5f00`.

## Context

PDR-0037 built the **pure scorer** (`stage2_acceptance.py`, 42 tests) but marked the gate
**NOT-FREEZABLE** until the telemetry wrapper exists — the wrapper owns §1 validity, §0
provenance, burn-in `W`, §8B floored-exclusion, §8E covariates, G3/G4, the §9 diagnostic
scalars, and the tier-gated call into `composite_verdict`. This session the owner confirmed the
authority grant and dispatched the wrapper build. Two architecture-gating checks (advisor-run)
reshaped the approach before any code:

- **Check 1 (config fields):** most §1/§10 config fields the validity gates read are NOT in
  telemetry — `actor_advantage_source` is emitted nowhere in `src/`; `hra_value_decomposition`,
  `reward_family`, `gamma`, `gae_lambda`, `per_head_advantage_norm`, `ev_return_variance_floor`
  are absent from the `runs` view (which exposes `seed`, `reward_mode`, `lr`, `clip_ratio`,
  `entropy_coef`, `task`, `n_envs`, `param_budget`, `host_params`).
- **Check 2 (byte-identity runnable):** the full `pytest` gate is broken (wedges on GPU
  `test_data_opt`/`test_dual_ab`), so the §9 emission's OFF-leg byte-identity claim needs a
  targeted GPU-free test — still to be verified before S7 relies on it.

## Options considered

1. **Add telemetry columns for every §1/§10 field, then read them.** Rejected as the primary
   path: it front-loads a training-code/emitter touch (blast radius) to trust a *self-reported*
   flag, on a broken test gate.
2. **Caller-supplied pairing, telemetry-VERIFIED.** The wrapper is told the pairing
   `{seed, run_dir_on, run_dir_off}` and verifies ON/OFF from the observable telemetry signature
   (`ev_main`/`ev_cf`/`ev_sum` are emitted only under `hra_value_decomposition=True`,
   `ppo_agent.py:751`) rather than a flag. Chosen — a validated invariant beats a trusted flag and
   shrinks the emission touch to just `actor_advantage_source` + the §9 scalars.

## The call

Build the wrapper in slices, **pure → impure, reversible bulk first** (TDD throughout). This
session landed the entire **pure layer (S1–S4)**; the impure tail (S5–S7) remains.

**Architecture decisions (durable — the working plan lives in ephemeral scratch):**
1. **Caller-supplied pairing, telemetry-verified** (option 2 above). §1 rejects an OFF arm that
   carries any `ev_*` and an ON arm missing them.
2. **`actor_advantage_source` → run-provenance constant** (`total_reconstructed`), emitted in the
   runs payload (S7); makes §0/§1 real, not a `.get()` mask. Constant defined local to the wrapper
   for now (torch-free); **promote to a torch-free leyline home at S7** when the emitter becomes a
   second consumer (`value_metrics.py` imports torch, so it is NOT that home).
3. **Split `calibrate_off` / `score` entrypoints enforce §10 freeze order by TYPE:** `calibrate_off`
   raises if handed an ON leg; `score` consumes already-frozen `FrozenThresholds` it cannot
   recompute. "No peeking" becomes mechanical, not conventional.
4. **§8B floored-exclusion is total-level only** (telemetry exposes total `ev_return_variance`, not
   per-stream) → the wrapper excludes `ev_return_variance <= 1.0`, rejects on arm asymmetry, and
   **emits the `ev_return_variance` distribution into the packet** so the per-stream-blindness
   caveat (the M1/MECH artifact) travels with the verdict.
5. **§9 scalars typed-unavailable** until emission — not a validity failure, not a defaulted value.

**Slice map** — DONE: S1 series primitives · S2 `leg_series` per-leg reduction · S3 `validate_pair`
§1 gates · S4 `calibrate_off`/`score` + §0 provenance. **80 tests green (38 wrapper + 42 scorer),
ruff + mypy clean, torch-free, no training-code touch.** REMAINING: **S5** duckdb reader
(dict→`UpdateRow` at boundary, fail-loud; `runs`/`episode_outcomes` readers for `RunMeta` +
G1/G2/G3/G4) · **S6** CLI + markdown packet (mirror `proof_packet` structure) · **S7** emission
(§9 scalars ON-leg-only near `ppo_agent.py:824` + `actor_advantage_source`; GPU-free byte-identity
OFF-leg test; **drl-expert + pytorch-expert review**). Module:
`src/esper/simic/telemetry/stage2_acceptance_packet.py`.

## Rationale

The reversible pure core is where the pre-registered statistics and validity live and where
PDR-0037's *emergent* failure mode (compounding near-misses) hides — so it is built first, test-first,
and kept torch-free/synthetic-testable. The single training-code touch (S7) is deliberately last and
small, so its blast radius is late and specialist-reviewable, on a broken test gate.

## Reversal trigger

- If S5–S7 show the pure-scorer / wrapper predicate split is unworkable, or the §8B
  floored-exclusion cannot be computed from emitted telemetry → revisit the harness design before
  freeze (echoes PDR-0037).
- If the §9 emission's **OFF-leg byte-identity cannot be verified by a runnable GPU-free test**
  (advisor check 2) → the emission approach is reworked before it lands; the gate does not freeze on
  an unverifiable guarantee.
- The gate doc remains **NOT-FREEZABLE until S7 lands and its tests encode the wrapper predicates**;
  no paired A/B before freeze (MAJOR-1 pre-registration).
