# Prereg ADDENDUM — GATE-fossil coalition admission + OFF@6dd80716 / ON@fix deviation (2026-07-04)

**Status:** Addendum to `docs/analysis/2026-07-03-shapley-ab-scoring-preregistration.md`.
Records a pre-registered **scope + provenance deviation**: the OFF arm ran at
`6dd80716`; the ON arm re-runs at the gate-fossil-admission fix commit (branch
`feat/phase-minus1-scale-falsifier`). Adversarial estimand review verdict: **SHIP**
the fix, with the licensing/documentation clauses below folded into scoring.
**Authority:** PDR-0017/0018/0019/0020; tracker esper-lite-fbeead4efc (ON-wave 5/5
crash); this addendum. Does NOT change any PDR ruling — it scopes the deviation.

## 0. What the fix does (one line)

The 6dd80716 ON wave crashed 5/5 because the coalition builder fail-loud rejected any
FOSSILIZED slot whose seed used the GATE alpha algorithm, and `GerminationStyle.GATED_GATE`
is an ordinary policy style (~1 of the style head's options). The fix **admits GATE
fossils into the 2^k coalition family**: they keep their trained per-sample gate for life,
and coalition masking composes `effective_alpha = amplitude × gate(x)` (the `blend_gate`
contract), so `alpha=0 ⇒ host-only bit-identical`, `alpha=1 ⇒ the deployed gated forward`.

## 1. Estimand preserved (Q1 — SOUND)

`v(S)` remains *terminal val-acc of coalition S with non-members masked to host-only*, now
also for GATE members. `blend_gate` gives `y = lerp(h, s, amplitude·gate(x))`, so the
DISABLED leg (`amplitude=0`) is host-only **regardless of the learned gate**, and the
member-present leg (`amplitude=1`) is exactly the deployed fossilized forward. Bit-identity
of both legs is pinned by the new kasmina tests (`..._gate` override-zero → host, and
`test_fossilized_gate_override_one_equals_live_forward`).

**No homogeneity assumption is violated.** `compute_committed_shapley_topup` operates purely
on **scalar** `coalition_accs[frozenset]` values (already val-set-averaged). The Shapley
efficiency axiom (`Σφ = v(C) − v(∅)`) holds for *any* set function, gated or not. Per-sample
gating lives *inside* the forward and is averaged into each scalar `v(S)`; it never enters
the credit math as a per-sample quantity. The gate's input-dependence does shift the *shape*
of `v(·)` (a gate can express coalition-conditional complementarity a scalar `α` cannot) —
but that is a richer valid characteristic function, not an estimand violation. The
off-manifold concern (masking an upstream member shifts a downstream member's input) is
**pre-existing for ADD/MULTIPLY** and not introduced by the fix.

## 2. Reward-hacking surface (Q2 — SOUND; legitimate credit, structurally bounded)

Admitting gated seeds does **not** open a proxy-farming channel:

- **The credited quantity is real accuracy.** `gap = max(0, φ − c_paid − τ)` pays synergy
  *excess* — the seed's coalition-averaged marginal minus its standalone marginal, both
  derived from terminal val-acc. You cannot inflate `φ − c_paid` without the seed genuinely
  adding more accuracy in coalition than alone. There is no proxy to Goodhart.
- **The G-clamp bounds total credit by the real deployed gain.**
  `top_up = raw · min(1, G/Σraw)` with `G = max(0, v(C) − v(∅))`. A gate's greater
  expressiveness lets it claim a larger *share* of a fixed real-gain pie — never more than
  the coalition actually produced. This is the strongest anti-farming armor and it is
  blend-algorithm-agnostic.
- **Zero estimator variance is preserved.** The gate is frozen at fossilization; the 2^k
  evaluation is exact. The module docstring's farm-resistance property ("nothing for a
  policy to farm") holds for a frozen gate exactly as for scalar α.
- **The gate is trained by task loss, not reward.** The policy controls op/style, not gate
  weights; it cannot steer the gate toward the reward proxy.
- **F2 guards apply identically.** cap=5.0, std_floor=0.25, ncap=3.0, τ=0.28 are magnitude
  bounds independent of blend algorithm. G1/G2/G4 fossilize-rate guards are unchanged.

**Verdict: legitimate credit, not a defect.** GATE seeds *can* earn more synergy excess
(more expressive blend) — that is correct crediting of real conditional complementarity.
**Scoring caveat (descriptive, not a gate):** because credit capacity is blend-style-
conditional, report the **gate vs non-gate split of paid credit**, and check whether any
positive Δcorr direction effect is disproportionately gate-carried before banking.

## 3. OFF-arm behavioral identity (Q3 — SOUND; provably unchanged) + mandated spot-check

The diff has exactly two source hunks in `vectorized_trainer.py`:

- **Hunk 1 (committed_slot_list admission):** entirely inside
  `if shapley_synergy_scale > 0.0 and epoch == max_epochs:` → **never executes in the OFF
  arm** (scale=0.0, verified in `config-3slot-3seed-baseline-shaped.json`: no shapley keys).
- **Hunk 2 (P4-FIX stage guard):** adds `and stage != FOSSILIZED` to the
  materialize-fresh-gated-schedule branch. This branch is only *reached* for a fossilized
  slot when that slot is override-keyed (`needs_override=True`). Both A/B configs have
  `drip_fraction=0` (no key ⇒ default 0.0), so `active_slot_list` **excludes fossilized
  slots** (trainer line 1121-1125), and no OFF-arm config (solo/solo_on/pair/all_off/
  committed/shapley) keys a fossilized slot. The only configs that key a fossilized slot are
  the `committed_shapley` configs — **scale>0, ON-arm only**. ∴ Hunk 2 is unreachable for
  fossilized slots in the OFF arm.

**Conclusion: OFF-arm behavior is bitwise-identical across 6dd80716 and the fix commit.**

**PyTorch adversarial review confirmation (2026-07-04, verdict SHIP):** state the deviation
as **two differently-gated edits**, not a blanket "scale>0-only" — hunk 1 is *structurally*
scale-gated; hunk 2 is *behaviorally* inert at scale=0, resting on (i) the drip=0 exclusion
above and (ii) the verified invariant that a fossilized GATE slot never has a None schedule
(load-bearing guard: `set_alpha_target` refuses FOSSILIZED at slot.py:1877 before the only
in-flight clear; the GATE forward raises on None, so no schedule-less GATE slot survives a
single forward). The step-counter hazard was refuted (`get_alpha_for_blend` never touches
`_current_step`; `step()` uncalled in the val path), and the autocast question is closed:
the val pass is fp32/no-autocast, and override-vs-live legs run in the same fused pass
through the same kernels, so leg equality holds under any autocast state. Hunk 2 now has
differential CI coverage
(`test_fused_pass_fails_loud_when_fossilized_gate_schedule_missing`): a fossilized GATE
slot with a forcibly-nulled schedule fails loud in the GATE forward with zero schedule
materialization — without the stage clause that run would complete silently on a junk gate.
The invariant Hunk 2 hardens ("fossilized GATE keeps its trained `alpha_schedule` for life")
is verified: set at `start_blending`, preserved through HOLDING (`_on_blending_complete`
skips the clear for GATE), untouched by the FOSSILIZED transition; only `prune` clears it and
fossils cannot be pruned. The Hunk-2 fail-loud is a genuine invariant guard, expected never
to fire.

**MANDATED (linchpin of the paired deviation):** before scoring, run **one OFF seed at the
fix commit** and assert its telemetry + final-acc are identical to the banked OFF@6dd80716
run (bitwise on the reward/decision stream). The argument above rests on `drip=0` and the
`active_slot_list` exclusion; a config drift would break it silently. This is a
**precondition to reading the paired A/B**, not nice-to-have.

## 4. `episode_idx` decode — cross-commit provenance (Q3, corrected)

Commit `bbc8c6bc` (the TOPUP `episode_idx` batch-index fix, esper-lite-e780981fe7) sits
between `6dd80716` and the fix commit. **Verified primary-source scope:** it is *entirely*
inside `build_committed_shapley_env_credits` — the TOPUP credits path, which runs **only
when scale>0**. It did **not** touch general per-episode telemetry encoding.

Therefore **there is no live cross-arm decode asymmetry**:
- OFF arm (scale=0): emits **no** `COMMITTED_SHAPLEY_TOPUP` events; its per-episode telemetry
  used the standard `episodes_completed + env_idx` convention, unchanged across both commits.
- ON arm (re-planned at the fix commit): emits TOPUP with the **correct** `episode_idx`.

**MANDATE:** the prereg's 2026-07-04 FLAGGED decode note (`episode_idx_true =
stored*12 + env_id`) applies **only to the crashed 6dd80716 ON runs** and is **retired for
the re-planned deviation**. Any scorer joining the re-run ON TOPUP events per-episode MUST
**NOT** apply the `stored*12+env_id` decode — the events are already correct; applying it
would double-corrupt the join. Criterion (v) advantage-at-credited-t_f reads on the re-run
ON arm use the raw `episode_idx` as emitted.

## 5. τ / gate-transfer (Q4 — recalibration folds it in, WITH a licensing gap)

τ=0.28 is a **PIN-E ADD-algorithm placebo floor** (held at HOLDING). Its transfer to GATE
members is **unvalidated**: a gate's input-dependence can shift a *null-player's* `φ − c_paid`
noise floor even at zero true synergy, because `gate(x)` responds differently to coalition-
conditioned inputs. The prereg's mandatory ON-run recalibration (P99 of `φ − c_paid` over
null-player slots; PDR-0017 trigger P99 > 2× placebo τ = 0.56 pp reopens the deadband)
**already catches an inflated tail** — P99 is an outlier statistic and fires on gate
inflation if gate null-players are present.

**The gap:** if the ON run has **zero/few fossilized GATE null-players**, the recalibration
is **silent on gate-transfer**, and any ON magnitude claim resting on gate credit is
**unlicensed**. MANDATES for the addendum:
- (a) Record explicitly that τ=0.28 is an ADD-placebo floor; GATE-transfer unvalidated.
- (b) Recalibrate over the population **including** fossilized GATE members; **report gate
  vs non-gate P99 stratified** (attribution — so a gate-specific inflation isn't masked by
  averaging with more-numerous ADD/MULTIPLY seeds).
- (c) **Unlicensed-if-absent clause:** if the ON run yields no fossilized GATE null-player,
  no magnitude claim involving gate credit is banked until a **GATE-placebo PIN-E leg** runs.
  (This is a magnitude-claim licensing gate only; it does not block the n=5 direction read.)

## 6. Test-coverage limitation (Q1 paying path — flag, not blocker)

The new tests pin the **non-paying** and **masking-equivalence** cases: the integration test
drives a **k=1** gate fossil (`φ ≡ c_paid ⇒ top_up=0`) and asserts `blend_create_calls==1`
(fused pass materialized nothing); the kasmina tests pin both masking legs bit-identically.
**CLOSED 2026-07-04:** the mandated k≥2 coverage now exists —
`test_scale_on_k2_all_gate_coalition_pays_through_live_seam` drives two serial all-GATE
lifecycles through the live trainer (slots r0c0+r0c1, STATIC_FINAL spacing), asserts the
full 2^2 coalition family is evaluated over both gated fossils (k=2, 4 coalition accs, no
silent exclusion), and delivers a nonzero credit at exactly both FOSSILIZE transitions
through the live normalizer seam, with `blend_create_calls==2` (fused pass materialized
nothing). Residual accepted limit: the credit magnitude is forced at the spy seam (the
existing harness pattern), so the G-clamp arithmetic under *organic* v(S) values remains
math-module-tested only (exhaustive unit suite) — the live seam it feeds is now fully
exercised at k=2.

## 7. Overall

**SHIP** the gate-fossil-admission fix. No code defect: estimand preserved, no new farming
channel, OFF-arm provably identical, invariant real and hardened. The deviation is
defensible **conditional on** folding §3 (OFF bitwise-replay spot-check — precondition),
§4 (retire the stale decode note for the re-run), §5 (τ gate-transfer licensing), and §6
(paying-path coverage) into the run sheet. Only §4 touches a scoring read; the rest are
licensing/documentation.
