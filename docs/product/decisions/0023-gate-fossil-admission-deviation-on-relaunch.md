# PDR-0023 — ON-arm GATE crash adjudication: fix (a) shipped, controlled prereg deviation, ON-only relaunch from the minimal patch-set

Date: 2026-07-04   Status: accepted (owner-ratified; execution gates listed §6)
Author: Claude (agent)   Owner ruling: consolidated response 2026-07-04
Related: PDR-0015 (same-commit rule), PDR-0019 (scoring rebase), PDR-0020 (F2
freeze), PDR-0021 (pre-A/B build + launch), PDR-0022 (external-review
disposition); tracker esper-lite-fbeead4efc (comments 108–111);
docs/analysis/2026-07-04-shapley-ab-scoring-prereg-ADDENDUM-gate-fossil-admission.md;
commits c2e500d3 (fix + gate-identity telemetry), 9ddf1542 (addendum).

## Context

The paired n=5 OFF/ON A/B launched at 6dd80716 (PDR-0021). All five ON runs
(shapley_synergy_scale=1.0) crashed within 5–15 minutes on a designed
fail-loud guard: the coalition builder rejected any FOSSILIZED slot whose
seed used AlphaAlgorithm.GATE, and GerminationStyle.GATED_GATE is an ordinary
policy germination style — the reward term's executable domain did not cover
structures the policy routinely builds. The OFF wave was unaffected
(s41/s42/s44/s45 complete and clean at 6dd80716; s43 OOM-crashed at 80% and
was rerun solo at the same frozen commit — see §5). This is a
**domain-completeness failure**, not an outcome-dependent defect: the guard
did its job, failing loud instead of corrupting state.

## The calls (owner-ruled)

1. **Fix (a) — admit GATE-fossilized slots into the coalition family.
   Mechanism = admission + invariant preservation, NOT save/restore.** A GATE
   seed keeps its trained `alpha_schedule` for life: fossilization never
   clears it, `set_alpha_target` refuses FOSSILIZED before the only in-flight
   clear path, and the GATE forward raises on a missing schedule — so a
   schedule-less GATE fossil cannot survive a single forward. Coalition
   masking composes amplitude × gate (the `force_alpha` contract): alpha 0.0
   is host-only and alpha 1.0 is the deployed fossilized forward, both pinned
   bit-identically by tests. The fused pass's schedule-materialization repair
   is scoped away from FOSSILIZED so the impossible missing-schedule state
   fails loud rather than silently minting an untrained gate. No
   save/restore machinery exists because there is nothing to restore — the
   pass never mutates an existing schedule (object identity + parameter
   immutability pinned by tests). This is a **crash-compatibility repair,
   not a reward-method change**: v(S) semantics are unchanged.

2. **Alternatives rejected.** (b) Masking GATED_GATE at scale>0: the ON arm
   would play a different action-space game than OFF — a design-level
   confound. (c) Excluding GATE fossils from the coalition: bakes a
   permanently uncreditable seed class into a term whose purpose is
   repairing under-crediting. Only (a) preserves the estimand.

3. **Pre-registration deviation accepted (PDR-0015 same-commit rule).**
   OFF arm stays banked at 6dd80716; the ON arm relaunches from the
   **minimal patch-set**, not broad HEAD:
   - base `6dd80716`
   - `+ f98b6b3e` (cherry-pick of bbc8c6bc — TOPUP episode_idx fix;
     TOPUP-path/scale>0-only, needed for correct ON scoring joins)
   - `+ d6514416` (cherry-pick of c2e500d3 — GATE-fossil admission +
     gate-identity TOPUP telemetry)
   Worktree: `.worktrees/ab-on-relaunch`. Both cherry-picks applied clean.
   Endpoints, thresholds, arm definitions, seed set, and OFF telemetry are
   unchanged. The arms are NOT same-commit and must never be labeled as
   such; compatibility rests on the OFF no-op proof (§6).

4. **Scoring-read mandates.** The `stored×12 + env_id` episode_idx decode
   applies ONLY to the discarded crashed 6dd80716 ON attempts (quarantined in
   `telemetry/shapley_ab_n5/gate_crashed_on_wave_6dd80716/`); it is
   **RETIRED for relaunched ON TOPUP events**, which stamp true ids — applying
   it would double-corrupt the criterion (v) join. Gate/non-gate
   stratification reads come from the new TOPUP payload fields
   (`alpha_algorithms`, `tau_used`) — emitted on the scale>0-only path so OFF
   telemetry stays byte-identical.

5. **No outcome-conditioned rerun clause.** The draft "if n=5 is borderline,
   rerun both arms" contingency is STRUCK as an unregistered fork. n=5 is a
   direction read; n=10 is the already-registered magnitude/banking path; an
   inconclusive n=5 is called inconclusive.

6. **Tau gate-transfer caveat (magnitude-only).** tau=0.28 is an ADD-placebo
   (PIN-E) provisional lower bound; transfer to GATE null-players is
   unvalidated. The mandatory ON-run P99 recalibration is stratified gate vs
   non-gate; if the ON population carries zero/few gated null-players, **no
   GATE-credit magnitude claim is banked** without a GATE-placebo PIN-E leg
   or sufficient in-run gated null-player evidence. Does not block the n=5
   direction read. (Addendum §5.)

7. **Population-validity gates carried forward.** Prior R1 showed these gates
   must remain explicit; the specific entropy-collapse diagnosis from R1 was
   retired as a telemetry artefact (PDR-0006). Decision-step (conditional)
   entropy health, the std<1.0 confound gate, G4 paid-event review, and the
   G1 fossilize-rate guard apply to the relaunched ON runs unchanged.

8. **OOM disposition (kept separate from the GATE RCA).** off_s43's OOM was
   operational: 3-per-16GiB-GPU packing versus morphogenetic in-run memory
   growth (~5 → ~6.9 GiB/proc). Mitigation is arm-symmetric and
   scheduling-level only (≤2 concurrent runs per GPU for all relaunch
   waves); no reward/optimization/code-path or endpoint change. The OFF set
   contains known packing heterogeneity (three packed runs, one solo rerun),
   classified under the pre-registered 6(b) bitwise-perturbation caveat.

## Attestation — pre-outcome repair

Some crashed ON attempts completed partial episodes before dying. **Before
the fix decision, no ON outcome metrics were examined or used** — only crash
tracebacks, failure lines, and event-type counts were inspected. All crashed
ON attempts are discarded and quarantined out of the scoring glob. The
repair was chosen from code structure and two adversarial reviews, not from
outcome data.

## Evidence

- **Adversarial reviews: both SHIP.** drl-expert (estimand preserved — the
  Shapley efficiency axiom holds for any scalar set function, no per-sample
  homogeneity assumption; no new Goodhart channel — the G-clamp bounds total
  credit by real deployed gain and gate weights train on task loss, not
  reward). pytorch-expert (all five invariants CONFIRMED; the step-counter
  hazard refuted — `get_alpha_for_blend` never touches `_current_step`;
  framing: hunk 1 structurally scale-gated, hunk 2 behaviorally inert via
  drip=0 + the schedule invariant). Tracker comments 110–111.
- **Tests (all green; 2407 across affected suites):** production-crash
  regression RED→GREEN through the real trainer; k=2 all-gate paying path
  through the live delivery seam; forced-None fail-loud differential pin
  (H1); schedule object-identity + parameter-immutability pins; GATE masking
  legs bit-identical (override 0 ≡ host; override 1 ≡ live fossil forward);
  no silent exclusion (GATE fossils appear in credit records);
  delivery/builder/leyline schema coverage for the new telemetry.
- **OFF no-op, CPU leg: PASSED.** Deterministic free-policy smoke at
  scale=0: two same-tree control runs bit-identical (194 events; sole
  volatile field is a wall-clock ratio, excluded); patch-set digest
  byte-identical to pure 6dd80716 —
  `d468d3e7ca82c2a325a45f7838cd447c693c4fc1b8a29d36953ce31f362a26a8`.

## Execution gates before the full ON relaunch (owner checklist §14)

- [x] PDR-0023 written (this document) with attestation, deviation,
      rejected alternatives, patch-set provenance, decode retirement, tau
      caveat, population-validity carry-forward.
- [x] Test battery green (see Evidence).
- [x] OFF no-op replay — CPU deterministic leg (byte-identical).
- [x] OFF no-op replay — GPU leg: PASSED under the frozen acceptance rule
      (tracker comment 118, fixed pre-result). Replay at the patch-set
      completed clean (2400/2400, zero TOPUP); strict identity broken
      (episode_reward identical 5/2400) with distribution preserved
      (final_acc +0.10pp). Same-commit control at pure 6dd80716 decorrelated
      HARDER (identical 3/2400; fossilize delta 5.6x the replay's;
      CF-matrix count +10%) — same-class stochastic drift, zero TOPUP, exact
      indexing. Auxiliary s43 packed-vs-solo replicate concurs. No
      replay-unique directional shift (trip-wire silent). RECORD LANGUAGE
      (binding): this establishes the GPU drift is NOT patch-specific — it
      does not establish bitwise no-op; the CPU byte-identity digest remains
      the stronger no-op proof; GPU stream identity is a low-sensitivity
      check under this co-tenancy/nondeterminism regime.
- [x] ON smoke at the patch-set (PASSED,
      `telemetry/relaunch_gates/on_smoke_s51/`, 8 batch-episodes,
      scale=1.0): exit 0, 96/96 episodes, 0 tracebacks / GATE errors / G1
      trips; 8 finite TOPUP events; gate-identity fields present with
      organic {ADD, GATE, MULTIPLY} coverage; true episode ids; 0 paid
      (expected at n=8).
- [x] OFF rerun completion (off_s43 solo at 6dd80716: 2400/2400
      EPISODE_OUTCOME, 200 PPO updates, clean terminal scoreboard, zero
      tracebacks, zero TOPUP events — OFF set now 5/5).

When all gates pass: relaunch ON-only — 5 seeds at the patch-set, ≤2 runs
per GPU, telemetry to `telemetry/shapley_ab_n5/on_s4*`. Do not rerun OFF.

## Reversal triggers

- The GPU OFF replay shows stream drift attributable to the patch-set (not
  to GPU nondeterminism under a same-commit control) → the deviation's
  no-op basis fails → full both-arm relaunch at the patch-set.
- The ON smoke crashes on the GATE path or emits non-finite/missing TOPUP
  fields → fix incomplete → back to code, no relaunch.
- Criterion (v) failing at scoring still reopens GATE 2 (PDR-0011 trigger,
  unchanged).
