# Plan Metadata
id: reward-redesign-methodology
title: "Methodology — examine, critique, and develop a new morphogenetic-controller reward function"
type: concept
created: 2026-06-24
updated: 2026-06-24
owner: Claude (multi-agent workflow + John)

urgency: high
value: >
  A disciplined, evidence-gated methodology for deciding whether the morphogenetic
  PPO reward needs a new design (vs. the in-flight critic-variance fix) and, if so,
  how to develop and validate one without trading one high-variance/misaligned
  reward for another. Adds the ALIGNMENT gate the EV-stabilization epic lacks.

complexity: L
risk: medium
risk_notes: >
  This is a methodology (no behavioural change yet). The risk it manages is
  strategic: shipping a reward that passes an EV-only gate while still encoding
  the wrong optimum (high reward, low committed accuracy, near-zero fossilization).
  Mitigated by a composite, pre-registered, fail-closed acceptance gate and a
  PBRS-invariance precondition.

depends_on:
  - ev-stabilization-stage0-instrumentation   # per-component return variance + per-stream EV are the shared measurement substrate
  - esper-lite-3defe42928                      # correctness PR (filigree): escrow sqrt/harmonic reconciliation + pbrs_weight telescoping test — Phase-2 precondition
soft_depends:
  - ev-stabilization-stage2-deshape-hra-head   # the critic-variance lever; runs in parallel, must not be conflated
blocks: []

status_notes: >
  Drafted 2026-06-24 from a multi-agent workflow (4 recon + 4 expert lenses +
  synthesis + adversarial drl-expert critic), then verified against current code.
  RE-VERIFIED 2026-06-24 by a SECOND independent workflow (6 lenses + synthesizer)
  + Karn re-measurement + direct code adjudication (verdict: `revise-first`).
  The four blocking conditions found are now RESOLVED at the document level:
  (1) corrected the sign-flipped `reward↔fossilize` (true +0.21, not −0.43 — a
  mislabeled prune correlation) and the GATE-4.3 calibration; (2) added Phase −1
  cheap-falsifier PRECONDITIONS (contribution gate + clip/ESCROW/scale A/B) feeding
  GATE 1; (3) re-grounded J's commitment unit (alpha-weighted on-output-path
  residency, code-grounded; fossilization is an irreversible slot-burn, not the
  only contribution path) + rate/hazard commitment metric + `max_seeds≥2`
  re-measure; (4) routed the verified per-step-SCALE root cause through the cheap
  levers first. Two forks: (1) EVIDENCE-GATED diagnosis first; (2) J = committed
  COUNTERFACTUAL contribution per parameter (host-drift excluded).
  RATIFIED 2026-06-24 by a FOURTH independent workflow (`wf_90a109d8-63b`: 6 adversarial
  lenses + per-finding skeptic + synthesis; verdict `GO_WITH_CONDITIONS`; all 11 blocking
  findings refuted; record docs/analysis/2026-06-24-reward-methodology-gonogo-review.md).
  Owner RATIFIED both flagged design decisions: commitment UNIT = counterfactual-WEIGHTED
  on-output-path residency (raw residency rejected); §9 stance = fossilize-rate is a
  DIAGNOSTIC, not a gate (re-open at `max_seeds≥2`). Sign-offs recorded for drl-expert /
  reward-function-reviewer / morphogenesis-reviewer / plan-review-reality; pytorch-expert +
  axiom-python-engineering still pending (implementation time). Not implementation-ready
  until Phase −1 falsifiers run and GATE 1 fires.
percent_complete: 0

reviewed_by:
  # 4th independent multi-agent review 2026-06-24 (`wf_90a109d8-63b`): GO_WITH_CONDITIONS,
  # all 11 blocking findings refuted. These lenses constitute SIGN-OFF for promotion:
  - drl-expert            # SIGNED OFF (GO_WITH_CONDITIONS) 2026-06-24 via wf_90a109d8-63b
  - yzmir-deep-rl:reward-function-reviewer   # SIGNED OFF (GO_WITH_CONDITIONS) 2026-06-24
  - yzmir-morphogenetic-rl:morphogenesis-reviewer  # SIGNED OFF (GO_WITH_CONDITIONS) 2026-06-24
  - axiom-planning:plan-review-reality       # SIGNED OFF (GO_WITH_CONDITIONS) 2026-06-24 (~20 citations verified exact)
  - pytorch-expert        # PENDING (only if redesign touches HRA/normalizer path)
  - axiom-python-engineering  # PENDING (implementation-time review)

---

# Reward-Redesign Methodology

## 0. Scope and the two resolved forks

This document is the **methodology** for examining, critiquing, and developing a
new reward function for the Tamiyo morphogenetic controller (recurrent,
factored-action, on-policy PPO; seeds germinate → train → blend → fossilize/prune).
It is **not** the new reward. It stops at the decision gates; the reward design
itself is downstream work, gated on Phase 0/1 evidence.

**Owner decisions (2026-06-24):**

1. **Evidence-gated diagnosis first.** We do not presume a redesign. Phase 0
   instruments, Phase 1 diagnoses, and **GATE 1** decides *variance-only* vs
   *alignment-redesign* vs *both* from re-measured evidence.
2. **Objective `J` = committed counterfactual contribution per parameter.**
   The yardstick for "aligned" is the **counterfactual** seed contribution
   (`val_acc − all_disabled_acc`), minus a parameter cost — **not** final host
   accuracy. Host-optimizer drift is explicitly excluded (it is the documented
   confound, `contribution.py:442-445`, `types.py:37`). λ (param-cost weight) and
   the gain-baseline are pinned in Phase 0; the *form* is locked.
   - **Commitment unit — RATIFIED (owner, 2026-06-24).** "Committed" is keyed
     to seeds **on the output path** (alpha-weighted residency), NOT exclusively to
     the **fossilize** event. **J is the counterfactual-WEIGHTED residency integral** —
     `J = Σ_t counterfactual_total_improvement(seed) · alpha_residency_t / params` —
     **NOT** raw `Σ alpha_residency` (raw residency is a **REJECTED** form: a high-alpha,
     zero-counterfactual parked seed would maximize it; the counterfactual MUST multiply
     in so a freeloader scores ≤ no-op — see §8 "park-the-freeloader"). Rationale
     (code-grounded): `can_germinate` needs an empty slot (`germinate.py:58`); FOSSILIZE
     does **not** free the slot/seed-cap (state stays non-None, stage=FOSSILIZED,
     unprunable — `host.py:986,994`, `slot.py:1603,1617`; only PRUNE→DORMANT nulls state
     at `slot.py:2616`); a fresh `MorphogeneticModel` is built per episode
     (`parallel_env_state.py:211-227`), so FOSSILIZED and high-alpha BLENDING seeds are
     discarded **identically** within the optimized objective, and `final_acc` already
     counts live BLENDING. So fossilization is an **irreversible slot-burn**, not the only
     contribution path; keying J solely to it is partly circular ("never fossilize ⇒ J≈0 ⇒
     misaligned" by construction). The *normative* "we value an irreversible committed
     graft" goal was **NOT adopted as a gate** (§9 ratified 2026-06-24: fossilize-rate is a
     **diagnostic**; re-open at `max_seeds≥2`).
   - **Commitment metric is a RATE/HAZARD, not a per-episode correlation.**
     `committed-J-per-fossilize` is undefined on ~91% of episodes (fossilize_count=0);
     use a Poisson fossilization rate or `P(fossilize>0)·E[J|fossilize>0]` and
     power-calc against the rare-event rate. Pin in Phase 0 with λ.
   - **UNIT ↔ METRIC bridge (RESOLVED 2026-06-24).**
     The alignment unit above (counterfactual-weighted on-output-path residency) and the
     fossilize **rate/hazard** measure *different* things, used in *different roles*:
     **(i) Alignment/J** is measured on the **counterfactual-weighted residency** unit —
     "is the seed actually contributing", fossilization NOT required; this is an
     **un-instrumented metric today** (`grep` finds no residency/on-output-path
     telemetry) and is a **Phase-0 instrumentation deliverable** (build it next to
     `RewardComponentsTelemetry`). **(ii) The fossilize rate/hazard** is — per the §9
     ratification (owner, 2026-06-24) — a **DIAGNOSTIC, NOT a gate**: the commitment leg
     reduces to the counterfactual-weighted residency contribution (folded into GATE-4.2),
     and fossilize-rate is measured/reported but does not gate. GATE-4.3 and the Phase-1
     commitment leg are accordingly **demoted to diagnostics**; re-open the
     fossilize-rate-as-gate question at `max_seeds≥2`, where committing no longer ends
     morphogenesis.

## 1. The reframing — two COUPLED problems sharing one term

The single most important finding. There are **two** distinct defects sharing one
root object (`bounded_attribution`), and only one is a reward problem:

- **Problem A — critic-fit variance (ALREADY OWNED).** EV-liftoff is achieved
  (median ~0.31) but volatile. The plan-of-record's **Stage 2** HRA decomposition
  (`docs/plans/ready/2026-06-23-ev-stabilization/stage2-deshape-hra-head.md`)
  splits the *value function* into `V_main` + `V_cf` while keeping
  `R_main + R_cf ≡ total_reward` exactly. By GAE linearity the policy optimum is
  **provably unchanged**. This is a critic reparam; it does **not** and is not
  meant to change the reward. **Do not reopen it.**

- **Problem B — reward optimum / unbounded farmable dense credit (the candidate for
  a new reward).** The live SHAPED run shows the top-reward policies farm
  `bounded_attribution` via **germinate-blend-PRUNE churn** + an **unbounded
  survival-time tail**: 17/180 episodes fossilize anything; top-reward episodes
  show 8–11 germinate / 7–11 prune and fossilize nothing (acc 45–50%, not the run's
  best); episode reward is extreme right-skew (max 558). **Correction (verified
  2026-06-24):** the earlier "commitment is penalized / `reward↔fossilize = −0.43`"
  framing was a transcription error — the true value on the cited run is
  **`reward↔fossilize = +0.21`** (POSITIVE; the −0.43 was the `reward↔prune` value
  mislabeled). The surviving defect is the **unbounded dense farmable tail**, not a
  fossilize penalty. A critic reparam (Stage 2) is **optimum-preserving** and
  therefore **cannot** fix this. A reward/scale change can — **but** the cheapest
  fix (clip/normalize the dense term; see §5 Phase −1) must be falsified first.

This methodology is about **Problem B**, measured against **J**, running **on top
of** the Stage-2/Stage-1 critic work (never bundled with it). **Coupling caveat
(verified):** `bounded_attribution` is simultaneously the high-CoV variance source
Stage 2 isolates (its `R_cf`) AND the farmed dense term — the two tracks are NOT
strictly orthogonal. Any attribution-scale change re-opens the Stage-2
`Cov(R_cf,R)/Var(R) > 0.40` gate (it may no longer fire); measure-between, do not
assert independence.

### 1.1 Why "aligned" must be measured against J, not host accuracy

The adversarial critic's load-bearing correction (verified): a reward that
maximizes `reward↔final_accuracy` can do so by **crediting host-optimizer drift**
— producing a "drift-farmer" that looks aligned while the seed contributes
nothing. Final host accuracy is **deliberately not** the objective in this
codebase. The alignment gate is therefore `reward↔J`, with **J = committed
counterfactual gain per param** (resolved fork #2).

## 2. Empirical evidence (hypothesis-generating, NOT gating)

From the live run `telemetry_2026-06-24_033049` (config-3slot-1seed-baseline-shaped,
SHAPED mode, 12 vec-envs, ~180 episodes). **Single-seed; treat as hypothesis-
generating only — Phase 0/1 RE-MEASURE on ≥5 paired seeds before GATE 1 fires.**

- Episode reward: extreme right-skew (max +156.9 on this window; +558 on the full
  300-ep run), median near 0 — the headline is the heavy tail, not the centre.
- Top reward episodes dominated by `bounded_attribution`, with **8–11 germinate /
  7–11 prune** (germinate-blend-PRUNE churn) and **zero** fossilizations; final
  accuracy 45–50% (NOT the run's best).
- Run-wide (re-measured 2026-06-24): `reward↔final_accuracy = +0.19`,
  `reward↔germinate = −0.44`, `reward↔prune = −0.43`, `accuracy↔germinate = −0.32`,
  and **`reward↔fossilize = +0.21`** (POSITIVE — corrects the earlier −0.43, which
  was the prune value mislabeled).
- **Window-sensitivity of `reward↔fossilize` (disclosed 2026-06-24, 4th review).** The
  `+0.21` is a tail-driven, window-specific Pearson on the **first-180-ep** window; on the
  **full 300-ep** run it collapses to **`+0.052`** (Spearman steadier: `+0.33` window /
  `+0.21` full; max reward +156.9 window / +558 full, skew +4.15 / +7.97 — one +558
  episode dominates). It stays **≥0 under every cut**, so "commitment is NOT penalized"
  holds — but `+0.21` is NOT a stable run-wide figure. This **vindicates** the pivot away
  from a correlation target to a **rate/hazard** diagnostic re-measured on ≥5 paired seeds
  at `max_seeds≥2`.
- Only 17/180 episodes fossilize anything (avg 0.094/episode) — but this is a
  `max_seeds=1` config, where committing **ends** morphogenesis (fossilize can't
  free the one-seed cap; can't be pruned), so "never commit / churn" is partly
  **structurally rational**. RE-MEASURE on `max_seeds ≥ 2` before GATE 1.
- Mechanism (ep140, re-checked against telemetry): `bounded_attribution = 0` early
  (seeds pruned at BLENDING before maturing; early accuracy credited to the HOST),
  then **two** BLENDING stints (inner-epoch ~37–70, then a prune, then a new seed
  ~103–118) accumulate ~12 reward/step — i.e. the episode return is a **survival-time
  attribution integral** sustained across germinate-blend-PRUNE churn, not a single
  continuously-parked seed.

## 3. Current-reward critique (prioritized, verified against code)

1. **(critical) Return is a survival-time attribution integral, not committed
   gain.** Dense `bounded_attribution` (`contribution.py:538-571`, added :641)
   dominates; the only final-tied term is `terminal_bonus = val_acc*0.05` (~2.5
   once), ~100× out-scaled. Argmax = "keep one mid-quality seed alive forever."
2. **(critical) The honest commit signal is out-scaled by the farmable tail.**
   Immediate fossilize bonus capped ~0.965 (`3.0*tanh(1/3)`, and *less* for <5
   epochs in HOLDING) < ~2 steps of farmed attribution; anti-churn guards are
   O(0.1–0.4) against an O(12/step × 80) harvest. **Mechanism correction (verified):**
   the "stay alive" force is the **dense unbounded `bounded_attribution` term**, NOT
   PBRS. PBRS increments are GERMINATED→TRAINING +1.0, TRAINING→BLENDING +1.5,
   **BLENDING→HOLDING +2.0 (largest)**, HOLDING→FOSSILIZED +0.5 (smallest); PBRS
   therefore pulls toward **HOLDING**, not BLENDING (the "largest BLENDING increment"
   reading inherited a stale `shaping.py:60` comment). FOSSILIZED-smallest (+0.5,
   `shaping.py:63`) is real and anti-farming, but it is not the inversion force.
3. **(critical) `bounded_attribution` is high-CoV (~1.9 capable / ~2.7 impaired)
   dense reward off an un-validated noisy counterfactual × two nonlinear
   discounts.** This is exactly the value-target variance the EV epic targets;
   reward composition and the critic regressand are one coupled object.
4. **(high) PBRS policy-invariance is broken in multiple places** — escrow clip
   (`escrow_delta_clip=2.0`), PRUNE potential forfeiture, action-dependent costs,
   **and (verified) the ASYMMETRIC `germination_discount` scaling only the germinate
   potential** (`contribution.py:789,1022`). **Correction (4th review, 2026-06-24):
   `pbrs_weight=0.3` is NOT a breaker** — a *uniform* scalar `w` applied to every
   `γΦ(s′)−Φ(s)` just rescales the potential to `w·Φ` and still telescopes
   (`Σ = w(γ^TΦ_T − Φ_0)`), optimum preserved; it was wrongly listed among the breakers.
   The real breakers are the asymmetric `germination_discount` (`:789`), the
   PRUNE-forfeiture redefinition (`:1022`), and the escrow clip. The "safe main stream"
   to de-shape onto is not actually optimum-preserving until those are repaired.
5. **(high) ESCROW uses `sqrt(progress·contribution)` (geometric mean) while
   SHAPED uses `harmonic`** (`contribution.py:523` vs `:548`) — geometric mean was
   declared removed for host-drift inflation but ESCROW never migrated. Any
   cross-mode A/B silently confounds the variance lever with the algebra.
6. **(high) No off-switch/no-op subtraction in the dense per-step credit** — the
   reward can reward riding host drift; `reward↔germinate = −0.44` is the
   signature of a reward that punishes acting.
7. **(medium) ~60 entangled tunables across 7 partly-divergent modes with no
   per-term return-variance accounting** (Stage 0 closes this).
8. **(low — PASS to defend) Governor independence is intact.** `governor.preflight_
   lifecycle_mutation` reads only structural/health facts; the only coupling is
   one-directional governor→reward. A new reward must **not** route controller
   policy/attribution into governor decisions.

> **Correction to the workflow's own output (verified):** the synthesis claimed
> `compute_scaffold_hindsight_credit` was "exported but unwired." It is **LIVE**
> at `fossilize.py:99` (capped at `MAX_HINDSIGHT_CREDIT`). Term enumeration MUST
> cover `handlers/`, not just `compute_contribution_reward`.

## 4. Methodology principles

- **Judge the reward against J (the objective), not the critic.** EV measures
  critic fit, not whether the return encodes the goal. The headline gate is
  **composite**: EV/variance AND `reward↔J` AND a commitment/fossilization target.
- **Instrument before touching any term.** No behavioural reward change before
  Stage-0 per-component variance shares (`Cov(R_i,R)/Var(R)`, sum=1) and per-stream
  EV land. All four research reports and the epic converge on instrument-first.
- **Diagnose scope from evidence.** GATE 1 decides variance-only vs redesign.
- **This is the EV-epic's problem, not greenfield.** Run on top of the landed
  op-marginal `V(s)` critic (P0-1, `6a27b8e3`); reuse Stage-0; do not reopen the
  critic baseline; do not bundle a reward edit with a critic edit.
- **Optimum-preservation must be PROVEN, not asserted.** A term stays in the
  de-shaped main stream only if it telescopes (incl. `pbrs_weight`) or its ablation
  does not regress committed J; else it re-enters as a Harutyunyan-2015 DPBA or an
  HRA head/baseline input — never as raw dense reward.
- **One lever at a time, measure-between, paired and both-substrate.**
- **Pre-register the gate, J, λ, and the hack list before any A/B.** Success
  cannot be redefined post-hoc.
- **Fail-closed evidence only.** ≥5 paired seeds (target 10), baseline-subtracted +
  param-normalized ROI, against the four control baselines + lockstep A/B.
  `--dual-ab` is smoke-only.
- **Characterize the measurement before paying for it densely.** The leave-one-out
  counterfactual's noise floor/CoV per stage must be measured; densely optimizing a
  high-CoV proxy is the canonical Goodhart trap.
- **Defend governor independence as a tested invariant.**

## 5. Phased methodology

### Phase −1 — Cheap falsifiers (PRECONDITIONS — run before any L-complexity work)
Two near-free experiments can MOOT or DOWN-SCOPE the entire redesign; they run
first (verification 2026-06-24).
- **Contribution gate (the cheapest kill).** Run the off_switch / static_final /
  fixed_schedule vs M paired A/B from
  `docs/analysis/2026-06-23-controller-contribution-gate-runsheet.md` (Path A is
  zero-code today; Path B is one thin driver) on baseline-subtracted committed J.
  Decision table feeds GATE 1: **M-not-beat-OFF → STOP_THEORY** (signal/reward/
  observability, not a reward-optimum redesign); **M-beats-OFF-not-FIX →
  REVISE_ALGORITHM** = the already-owned in-PPO reforms (Stage-2 de-shape /
  per-head adv-norm), NOT a from-scratch reward; **M-beats-both → CONTINUE**, which
  also *falsifies* "commitment-avoidance is a defect" (the controller contributes
  despite rare fossilization) and forces fresh justification for a redesign.
- **Cheap-lever A/B (the verified root cause is per-step SCALE).** SHAPED
  `bounded_attribution` is **un-clipped** (`contribution.py:561`) over a counterfactual
  in 0–100 *accuracy points* (`seed_contribution` max ~35.8) → O(6–16)/step, ~100×
  over terminal/fossilize; **681/1278 positive steps exceed 2.0 (full 300-ep run;
  278/526 on the first-180-ep window)**. ESCROW already clips
  at `escrow_delta_clip=2.0` (`contribution.py:532-536`). A/B status-quo SHAPED vs
  (a) SHAPED + per-step attribution clip / unit-normalization (`acc_delta/100`) and
  (b) ESCROW pay-at-commitment+clawback, on J / fossilization-rate / churn. (NB: a
  clip is itself an optimum-changing nonlinearity — not free, not orthogonal to
  Stage 2 — but far cheaper to falsify than the Phase-3 lattice.)
- **Re-measure on `max_seeds ≥ 2`** so "commitment avoidance" is separated from
  single-budget exhaustion (the cited evidence is `max_seeds=1`).
- **GATE −1 (admission to the redesign track):** the contribution gate returns
  CONTINUE-or-ambiguous AND the farmable tail + churn SURVIVE the clip/ESCROW/scale
  A/B on `max_seeds ≥ 2`. If either cheap lever resolves the pathology, STOP — the
  redesign is not the right lever.

### Phase 0 — Objective formalization, signal characterization, Stage-0 instrumentation
- **Pin J:** functional form locked (committed counterfactual gain per param);
  pin **λ** and the **gain-baseline** (recommend baseline-subtracted vs `off_switch`,
  param-normalized) as Phase-0 *outputs* with a worked numeric example on one
  logged episode. **J is the anchor for every downstream gate — it cannot be left
  vague** (critic gap fixed).
- Land Stage-0 telemetry (epic option A: carry `RewardComponentsTelemetry` arrays
  into the rollout buffer): per-term `Cov(R_i,R)/Var(R)`, per-term CoV, per-stream
  EV (`ev_main/ev_cf/ev_sum`), per-term correlation to FINAL committed J and
  `fossilize_count`.
- Characterize the leave-one-out counterfactual: per-stage noise floor/bias/CoV.
  **Scope this explicitly** (critic gap): either re-run LOO over K resampled val
  minibatches, or use the inert-seed placebo credit distribution as the noise-floor
  proxy. Name which, and whether it blocks GATE 0.
- Emit a committed-vs-uncommitted credit split.
- Enumerate every LIVE reward term across `compute_contribution_reward` AND
  `handlers/` AND BASIC/BASIC_PLUS/SIMPLIFIED (resolve scaffold-hindsight wiring —
  it is live). Freeze a reference control arm (mode + config hash + host + seeds).
- Add a host-drift placebo harness (inert seed → credit ~0).
- **GATE 0 (instrument-first, fail-closed):** J computable end-to-end on ≥5 paired
  seeds; Stage-0 telemetry reconciles (shares sum to 1); counterfactual noise floor
  reported. No candidate proceeds until this passes.

### Phase 1 — Diagnose and decide scope
- Evaluate the **variance** gate: `Cov(R_cf,R)/Var(R) > 0.40` on the capable host.
- Evaluate the **alignment** gate independently: `Corr(return, J)` (on the
  counterfactual-weighted residency J unit per §0) and — *as a DIAGNOSTIC per the §9
  ratification, not a gate* — the commitment **rate/hazard** metric (Poisson fossilization
  rate or `P(fossilize>0)·E[J|fossilize>0]` — NOT a per-episode `Corr(return,fossilize)`,
  degenerate on ~91% zero-fossilize episodes) **re-measured on ≥5 paired seeds, both
  substrates, at `max_seeds ≥ 2`** (NOT the single-seed `max_seeds=1` numbers).
- Run the host-drift placebo.
- **GATE 1 (scope fork) — consumes the Phase −1 verdict.** First apply the
  contribution-gate result: **STOP_THEORY → halt** (no redesign); **CONTINUE →** the
  commitment-avoidance premise is *falsified*, redesign needs fresh justification;
  **REVISE_ALGORITHM →** scope is bounded to the already-owned in-PPO reforms unless
  the alignment leg independently fails. Then, conditional on Phase −1 admission:
  (a) variance fires + alignment healthy → targeted Stage-2 de-shape (already owned).
  (b) alignment fails (`Corr(return,J)` on the counterfactual-weighted residency unit
  weak, or placebo leaks; the commitment **rate/hazard** is a diagnostic input here, not a
  gate threshold) → **reward redesign justified independent of variance**. (c) both →
  joint.
  (d) neither → re-scope and stop.

### Phase 2 — PBRS invariance audit & repair (precondition)
- Enumerate **ALL** modifiers on PBRS deltas: `pbrs_weight`, `germination_discount`,
  `epoch_progress_bonus` dwell ramp, PRUNE forfeit, escrow clip (critic gap fixed).
- Remove the escrow clip OR move escrow/unrealized-credit state INTO the observation
  (history-potentials) so the recurrent critic predicts the un-clipped delta.
- Reconcile ESCROW `sqrt` vs SHAPED `harmonic` algebra.
- Extend `test_pbrs_properties` to assert telescoping **including `pbrs_weight`**
  (the naive `Σ == γΦ_T − Φ_0` would falsely fail — assert against realized shaping)
  over clip-binding / prune / long-dwell episodes; wire as build-failing CI.
- **GATE 2 (correctness):** telescoping residual within tolerance with no clip
  binding inside the property test; algebra normalized; every non-telescoping term
  documented as an intentional optimum-shift.

### Phase 3 — Candidate generation (a structured lattice)
- Three orthogonal axes: **credit-timing** (pay-at-observation → pay-at-commitment
  / fossilize-gated escrow+clawback → pay-in-hindsight drip); **counterfactual-
  placement** (raw addend [reject] / PBRS potential / COCOA-HCA baseline INPUT /
  HRA head / Harutyunyan DPBA potential); **term-partition**.
- Every candidate is a **provable shaping of J** (argmax preserved on paper, or cf
  re-entered as DPBA).
- **Minimal-then-augment:** start from a terminal-anchored skeleton (strict PBRS +
  terminal committed J + rent), add the smallest dense shaping needed to recover
  sample efficiency, measuring each addition's variance cost.
- **Anti-hack-first:** red-team each draft against squat-one-seed, germinate-prune-
  churn, alpha-oscillate, noise-dweller **and the two NEW hacks the gate itself
  opens** (see §8): **fossilize-churn** and **terminal-loading variance-dilution**.
- **Off-switch-anchored:** every dense signal is a delta vs an explicit no-op
  counterfactual.
- **GATE 3 (admission):** provably a shaping of J AND neutralizes every enumerated
  hack in static analysis.

> **Plan-of-record boundary (critic-flagged):** Stage 2 has ALREADY decided the
> `R_main`/`R_cf` partition and REJECTED the drop-cf "strong de-shape" as primary
> (cf is constitutive). Candidate generation must consume that decision, not
> re-litigate it. Any divergence (e.g. a strong-de-shape arm) is flagged and gets
> drl-expert sign-off — it does not enter as open design space.

### Phase 4 — Selection & tournament (judge panel + paired both-substrate A/B)
- **Pre-register** the composite gate (thresholds + statistical test) and the hack
  list before any A/B.
- **Offline judge-panel pre-filter** on logged episodes (held-out `reward↔J`,
  projected fossilization incentive, adversarial Goodhart probes); discard dominated
  designs before spending training compute.
- **Paired both-substrate A/B:** one factor changed vs the frozen control, identical
  seeds/host/env/γ/action-costs, on capable (CoV~1.9) AND impaired (CoV~2.7,
  **measured not reconstructed** — critic gap) hosts, ≥5 (target 10) paired seeds.
- **Unit of analysis = per-seed paired delta** with env-clustered block bootstrap
  (NOT per-episode; 12 vec-envs under one drifting policy are non-i.i.d.).
- **Measure-between** stacked levers; ablation ledger; drop any lever that moves no
  pre-registered metric. Re-warm PopArt-lite after each regressand change.
- **GATE 4 (composite, fail-closed — the headline):** ALL must hold on held-out
  paired seeds, both substrates:
  1. **EV/variance:** median `ev_main ≥ 0.50` AND positive post-anneal slope AND
     inter-decile range below baseline; **AND post-candidate `Cov(R_attribution,
     R_main)/Var(R_main)` below a pre-registered ceiling** (so variance is *removed*,
     not diluted — critic gap fixed).
  2. **Alignment:** `reward↔J` Spearman ≥ pre-registered floor; `reward↔J-ROI` > 0.
  3. **Commitment (RATE/HAZARD — DIAGNOSTIC, not a gate; §9 ratified owner 2026-06-24).**
     The §9 "value an irreversible committed graft" stance was **NOT adopted as a gate**, so
     the commitment leg of GATE-4 is the **counterfactual-weighted residency-based** J
     contribution (folded into leg 4.2), and the fossilize **rate/hazard** is **measured and
     reported but does NOT gate**. Diagnostics to report: the fossilization rate (Poisson,
     or `P(fossilize>0)·E[J|fossilize>0]`) and **committed-J-per-fossilize** (NOT raw count —
     blocks fossilize-churn, §8), with churn-rate + slot-occupancy-time co-measured.
     **NOTE:** the old "`Corr(reward,fossilize)` flip from −0.43 to ≥0" target is dead —
     −0.43 was a mislabeled prune correlation; the true baseline is already +0.21
     (window-specific; +0.052 full-run), so that flip is vacuous. **Re-open the
     fossilize-rate-as-gate question at `max_seeds≥2`**, where committing no longer ends
     morphogenesis.
  4. **Non-regression:** committed J and param-ROI not worse than baseline.
  5. **Adversarial:** no enumerated hack out-scores the honest lifecycle policy.
  EV-liftoff ALONE is explicitly insufficient.

### Phase 5 — Validation, ablation, determinism, load-bearing-credit
- **Load-bearing check:** if de-shaping regresses committed J or fossilization on
  matched seeds, the counterfactual was load-bearing → reintroduce as DPBA
  (decay-to-zero), NOT raw dense reward, and re-measure. **Add a DPBA kill
  criterion** (advice-value head EV floor; decay reaches zero before episode end —
  critic gap fixed; DPBA of a high-CoV signal can relocate variance, not remove it).
- Confound/placebo: inert-seed credit ~0; negative control (shuffle counterfactual
  across seeds) collapses EV/alignment.
- Determinism/replay: distinct RNG generators per subsystem; bit-for-bit reward-
  stream reproduction on same-seed re-run; lockstep cohorts diverge only after the
  first differing reward.
- Build-failing CI: governor-independence, telescoping (incl. `pbrs_weight`),
  value-collapse (`value_loss>5.0 OR bellman_error>5.0`).
- **GATE 5 (robustness):** all confound/placebo/determinism pass; load-bearing
  verdict resolved; CI guards green.

### Phase 6 — Fail-closed evidence packet & promotion
- Assemble the packet: ≥5 (target per §9 power calc) paired seeds, baseline-
  subtracted + param-normalized ROI, both substrates, four control baselines +
  lockstep A/B, every pre-registered metric with CIs, ablation ledger.
- Demonstrate controller SKILL: beat `off_switch` AND `static_final` AND
  `fixed_schedule` on baseline-subtracted committed J.
- Route through the fail-closed verdict (`BLOCKED_*`/`CONTINUE`/`REVISE_ALGORITHM`/
  `STOP_THEORY`); record specialist sign-off in metadata.
- **GATE 6 (promotion):** `CONTINUE` with all conditions met; `--dual-ab` not
  citable.

## 6. Coordination with the EV-stabilization epic (non-negotiable)
- Runs **on top of** the landed op-marginal `V(s)` critic (P0-1, `6a27b8e3`). Do
  **not** reopen Q(s,op)-vs-V(s).
- **Reuses Stage-0 verbatim** (`esper-lite-3d67b09687`) as the shared measurement
  substrate; sequenced **after** Stage 0.
- The epic's `Cov(R_cf,R)/Var(R) > 0.40` is this methodology's GATE-1 variance
  branch. The **addition** this methodology makes is the **alignment gate**
  (`reward↔J` plus the **commitment rate/hazard** metric — NOT a per-episode
  `reward↔fossilize` correlation, which is degenerate on the ~91% zero-fossilize
  episodes), which can justify a reward redesign even if the 0.40 variance gate does
  NOT fire — the epic alone would miss it.
- **Distinct OBJECTIVES, one SHARED term (verified — do not over-read as
  orthogonal).** The two *objectives* are distinct: Stage 2 is optimum-preserving
  (variance-fit); this track is optimum-changing (alignment). But they are **not
  mechanism-orthogonal** — `bounded_attribution` is simultaneously Stage-2's `R_cf`
  variance source AND this track's farmed dense term (see §1). So: keep them in
  separate A/B arms (do not bundle a reward edit with a critic edit), AND
  **measure-between** — re-check the Stage-2 `Cov(R_cf,R)/Var(R) > 0.40` gate after
  any attribution-scale change here (a clip/pay-at-commitment reward could shrink the
  very variance Stage 2 exists to absorb, partly mooting it).

## 7. Risk register (top items)
- **EV-only victory** shipping a misaligned reward → composite fail-closed GATE 4.
- **De-shaping onto a non-invariant main stream** → Phase 2 precondition incl.
  `pbrs_weight`/`germination_discount`.
- **Per-episode significance on non-i.i.d. vec-envs** → per-seed paired deltas,
  block bootstrap, ≥5 (target 10) seeds; the +156.9 episode is a best-of-N artifact.
- **Cross-mode algebra confound** (`sqrt` vs `harmonic`) → normalize in Phase 2.
- **Dropping load-bearing credit** → mandatory Phase-5 check + DPBA fallback with a
  kill criterion.
- **Dense re-payment of an un-characterized high-CoV signal** → Phase-0 noise floor;
  disqualify dense payment if >X% below noise.
- **Bundling reward + critic edits** → one lever per arm, on the fixed critic.
- **Governor-as-policy regression** → build-failing CI invariant.
- **Goalpost-moving** → pre-register gate/J/λ/hack-list.

## 8. New reward-hacking surfaces this methodology must gate (critic-found)
- **Fossilize-churn:** a gate on fossilize *count*/correlation is gamed by rushing
  low-quality seeds GERMINATE→HOLDING→FOSSILIZE to harvest the bonus. → GATE 4
  Commitment leg measures **committed-J-per-fossilize**, not raw count.
- **Terminal-loading / variance-dilution:** a candidate clears aggregate EV by
  adding a large low-variance terminal term that **dilutes** (not removes) the
  high-CoV attribution share. → GATE 4 EV leg adds an explicit attribution-share
  ceiling.
- **Alignment-via-host-drift:** neutralized by locking J to the **counterfactual**
  (resolved fork #2), not host accuracy.
- **Proxy-path host-drift credit (4th review, 2026-06-24):** when the clean counterfactual
  is absent at BLENDING entry, the dense path pays `proxy_contribution_weight(0.3) ·
  improvement_since_stage_start` — a **host-wide, non-counterfactual** delta
  (`contribution.py:606-612`) — which is both a drift-credit leak and **churn fuel**
  (germinate→BLENDING→collect proxy→prune→repeat). → Phase-0 term enumeration MUST cover
  the `seed_contribution is None` proxy branch, and GATE-3's "every dense signal is a delta
  vs an explicit no-op" must exercise it.
- **Park-the-freeloader (4th review):** a high-alpha, zero/negative-counterfactual seed
  parked on the output path. → neutralized by the **counterfactual-weighted** residency J
  (§0): such a seed scores ≤ no-op because the counterfactual multiplies in.
- **Enumeration completeness (4th review):** Phase-0 "enumerate every LIVE term" MUST also
  cover `compute_scaffold_hindsight_credit` (LIVE, `training/handlers/fossilize.py:99`) and
  `_compute_synergy_bonus` (`contribution.py:699-703`); the live handlers are under
  `src/esper/simic/training/handlers/`, NOT `rewards/handlers/`.
- **Adversarial-probe coverage gap:** the enumerated hack list is necessary, not
  sufficient; keep a standing red-team for novel degenerate policies.

## 9. Ratifications

**RATIFIED 2026-06-24 (owner, via the 4th review `wf_90a109d8-63b`):**
- **Commitment unit = counterfactual-WEIGHTED on-output-path residency** (raw residency
  rejected; fossilize-only rejected as partly circular). See §0.
- **§9 normative stance: an irreversible committed graft is NOT a gate.** Fossilize-rate is
  a **diagnostic**; GATE-4.3 demotes accordingly; re-open the gate question at `max_seeds≥2`.
- **Correctness-fix ownership: a SEPARATE correctness PR** (filigree
  `esper-lite-3defe42928`) — escrow `sqrt`/`harmonic` reconciliation + `pbrs_weight`
  telescoping test — is now a formal `depends_on` of this effort (see metadata).

**Still open (owner, before any A/B):**
- **Replicate budget / seeds-per-arm.** Proof floor ≥5; morphogenesis variance + rare
  fossilization (~9%) → target 10. **Do a power calculation for the fossilization-rate
  DIAGNOSTIC specifically** (rare-event, wide CI) and set the candidate-count cap
  (recommend ≤5) + seeds-per-arm from that, not the ≥5 floor.
- **Acceptance thresholds.** Ratify the pre-registered numbers: EV target (median
  `ev_main ≥ 0.50` + positive slope + IDR below baseline), `reward↔J` Spearman floor,
  attribution-share ceiling. (Commitment is a diagnostic, not a gate.)
- **λ and gain-baseline for J.** Pinned in Phase 0 with a worked example.

## 10. Reviewers (required before promotion to `ready`)
- **drl-expert** + **yzmir-deep-rl:reward-function-reviewer** — J definition, the
  alignment gate, candidate lattice, DPBA boundary, plan-of-record consistency.
- **pytorch-expert** — any obs-expansion (escrow-into-obs), normalizer/value-head
  interactions if the redesign touches the HRA path.
- **axiom-python-engineering** — leyline single-source contracts, no defensive
  drift, 6-file telemetry chain.
- **axiom-planning:plan-review-reality** — verify every cited symbol/line.
- **yzmir-morphogenetic-rl:morphogenesis-reviewer** — governor independence,
  separate RNG, baselines-run discipline.

## Provenance
Produced by a multi-agent workflow (`reward-methodology`, run `wf_733dc425-ade`):
4 recon agents (reward-core, research, active-plans; lifecycle agent failed and was
covered manually), 4 expert lenses (reward-function-reviewer, drl-expert ×2,
morphogenesis-reviewer), a synthesis pass, and an adversarial drl-expert critic
(verdict: "strong skeleton, wrong north star, cracked evidence base" — all
disqualifying claims verified against code and corrected herein).

## Independent verification verdict (2026-06-24)

Re-verified by a SECOND independent multi-agent workflow (`verify-reward-methodology`,
6 expert lenses — drl-expert, morphogenesis-reviewer, leverage/sequencing,
reality-citations, plan-quality, + a synthesizer; reward-function & dynamic-arch
lenses died on stream/schema errors and are covered by the others) plus direct
Karn re-measurement of the cited run and direct code adjudication of the crux.

> **RE-VERIFICATION of this revision (THIRD independent workflow, 2026-06-24):** the
> revised artifact was put back through 3 adversarial lenses (drl-substance,
> consistency/regression-hunter, morphogenesis) + synthesizer. Verdict:
> **all four blockers B1–B4 GENUINELY resolved** — B1's +0.21 re-measured a third
> time (corr(reward,fossilize)=+0.214 on the 180-ep window); B2/B4 code-confirmed;
> B3 passes the acceptable-resolution test (no longer asserted as fact, flagged as a
> normative claim). It caught two *new* internal contradictions the edits had
> introduced — (i) the §0 commitment **unit** (residency) vs the GATE-4.3/Phase-1
> commitment **metric** (fossilize-rate) mismatch, and (ii) a §6 "orthogonal"
> straggler contradicting §1 — plus minor nits.
>
> **Closure edits applied, NOT independently re-verified (owner: please review).**
> The author then applied edits intended to close those two contradictions (§0
> unit↔metric bridge; GATE-4.3 + Phase-1 flagged conditional on §9; §6 reconciled;
> line-number/window/ep140 nits). These closure edits have **not** themselves been
> put through an independent verification pass — they are the author's reconciliation,
> not a verified finding. **Two of them are OWNER DESIGN DECISIONS, not facts:**
> (a) defining the commitment *unit* as alpha-weighted on-output-path residency, and
> (b) making the GATE-4.3 commitment leg *conditional* on a §9 "value irreversible
> commitment" ratification. Ratify, amend, or revert these as you see fit (all edits
> are git-reversible). Remaining work before promotion is empirical (Phase −1
> falsifiers, residency instrumentation, `max_seeds≥2` re-measure) + specialist
> sign-off.

> **FOURTH independent review + OWNER RATIFICATION (2026-06-24, `wf_90a109d8-63b`).** A
> fourth workflow (6 adversarial lenses — citation/reality, telemetry re-measurement,
> drl-substance, reward-function, morphogenesis, executability — + per-finding skeptic +
> synthesis) returned **`GO_WITH_CONDITIONS`**; **all 11 blocking findings raised were
> REFUTED** by independent skeptics. The two OWNER DESIGN DECISIONS above are now
> **RATIFIED**: (a) commitment unit = **counterfactual-weighted** residency (raw residency
> rejected); (b) §9 = fossilize-rate is a **DIAGNOSTIC, not a gate** (re-open at
> `max_seeds≥2`). Sign-offs recorded for drl-expert / reward-function-reviewer /
> morphogenesis-reviewer / plan-review-reality; pytorch-expert + axiom-python-engineering
> still pending (implementation time). New findings folded in: +0.21 is window-specific
> (collapses to +0.052 full-run); `pbrs_weight` is a uniform scalar that telescopes
> (removed from the §3 invariance-breaker list); proxy-path / synergy / hindsight terms
> added to §8 + the Phase-0 enumeration; §5 `526/278`→`278/526`; the escrow + telescoping
> correctness PR (`esper-lite-3defe42928`) attached as a `depends_on`. Still pending: the
> stale `shaping.py:31/47/60` comments (queued for the correctness PR), the remaining two
> sign-offs, and the empirical Phase −1 work. Full record:
> docs/analysis/2026-06-24-reward-methodology-gonogo-review.md.

**CURRENT VERDICT (2026-06-24, supersedes `revise-first`): `GO_WITH_CONDITIONS`** — promote
`concept`→`ready` once the two remaining specialist sign-offs land; the two owner
ratifications are applied above and the free correctness edits are done. The original
`revise-first` verdict and its findings are retained below as the record.

**VERDICT: `revise-first` — the north star and gate discipline are sound; the doc
was NOT promotion-ready as written.** The four blocking conditions below were found
by the verification and have now been **RESOLVED at the document level** in this
revision (§0 fork-#2 commitment unit, §1/§2/§3 statistic + mechanism, §5 Phase −1
preconditions, GATE 1 + GATE 4.3). What remains before promotion is the *empirical*
work the revised doc now correctly gates on (the Phase −1 falsifiers) plus formal
specialist sign-off — NOT further document surgery. Original findings, as the record:

1. **GATE-4.3 is calibrated on a transcription error and is non-discriminating.**
   The headline `reward↔fossilize = −0.43` (§1/§2/§4/§8) is FALSE. Re-measured on
   the exact cited run (`telemetry_2026-06-24_033049`, first-180-ep window — every
   other figure reproduced to 2 d.p.): true `reward↔fossilize = +0.21` (POSITIVE).
   The −0.43 is the `reward↔prune` value mislabeled. The "flip from ~−0.43 to ≥0"
   commitment leg is therefore ALREADY passed by the unfixed reward → the composite
   gate cannot discriminate a successful candidate on its commitment dimension.
   Additionally `committed-J-per-fossilize` is undefined on ~91% of episodes
   (fossilize_count=0) → the metric must be REDEFINED as a rate/hazard (Poisson
   rate, or P(fossilize>0)·E[J|fossilize>0]), not merely recalibrated. Correct §1
   and re-derive GATE-4.3 before pre-registering. The doc's own re-measure-≥5-seeds
   hedge softens but does not cure this (the threshold + motivating narrative are
   built on the wrong number).

2. **Strategic sequencing is inverted.** The off_switch-vs-M controller-contribution
   gate (`docs/analysis/2026-06-23-controller-contribution-gate-runsheet.md` §0,
   "goes first") is the cheapest experiment that can kill or down-scope the whole
   program: M-beats-both ⇒ CONTINUE *falsifies* "commitment-avoidance is a defect";
   M-beats-OFF-not-FIX ⇒ REVISE_ALGORITHM names the already-owned in-PPO reforms
   (de-shape value target / per-head adv-norm), NOT a from-scratch redesign. The
   methodology folds it into Phase 6 (exit) and GATE 1 never consumes the M-vs-FIX
   verdict. Move it upstream as Phase −1 / GATE-0-pre feeding GATE 1.

3. **The objective J is partly circular and never re-examined (locked as fork #2).**
   J = committed counterfactual contribution of FOSSILIZED seeds ⇒ "never fossilize
   ⇒ J≈0 ⇒ misaligned" is partly true-by-construction. Code adjudication (settling a
   lens split): `can_germinate` needs `state is None` (germinate.py:58); FOSSILIZE
   does NOT free the slot/seed-cap — `state` stays non-None with stage=FOSSILIZED
   (host.py:986,994; slot.py only nulls state on the PRUNE→…→DORMANT path at
   slot.py:2616) and FOSSILIZED is unprunable (slot.py:1603,1617); a fresh model is
   built per episode (parallel_env_state.py:211-227) so FOSSILIZED and high-alpha
   BLENDING seeds are discarded identically within the optimized objective, and
   `final_acc` already includes live BLENDING contribution. So *pruning*, not
   fossilizing, is the slot-freeing valve (matching the verified churn). Under the
   cited `max_seeds=1` config (config-3slot-1seed-baseline-shaped.json:40),
   committing ENDS morphogenesis, so "never commit" is partly structurally RATIONAL,
   confounding reward-shape with capacity. Fossilization-as-target is defensible only
   as a NORMATIVE project choice the doc asserts but does not ground. Resolution:
   ground it (or redefine commitment as alpha-weighted on-output-path residency) AND
   re-measure on `max_seeds ≥ 2` before GATE 1.

4. **GATE-1 omits the cheapest lever and prematurely authorizes an L-complexity
   redesign.** The verified root cause is per-step SCALE: SHAPED `bounded_attribution`
   is un-clipped (contribution.py:561) over a counterfactual in 0–100 accuracy POINTS
   (seed_contribution max ~35.8), giving O(6–16)/step that dwarfs terminal/fossilize
   ~100× (681/1278 positive steps exceed 2.0). ESCROW already clips at
   `escrow_delta_clip=2.0` (contribution.py:532-536) and is a pay-at-commitment mode.
   Add a pre-GATE-1 A/B of status-quo SHAPED vs (a) SHAPED+per-step clip / counterfactual
   unit-normalization and (b) ESCROW pay-at-commitment+clawback; the alignment-redesign
   branch opens only if the farmable tail + churn survive. (The clip is itself an
   optimum-changing nonlinearity — not free, not orthogonal to Stage 2 — but far
   cheaper to falsify than the lattice.)

**Should-fix — APPLIED in this revision:** critique #2's PBRS "stay-in-blending"
mechanism corrected (increments GERMINATED→TRAINING +1.0, TRAINING→BLENDING +1.5,
BLENDING→HOLDING **+2.0 (largest)**, HOLDING→FOSSILIZED +0.5; PBRS pulls toward
HOLDING; the dense attribution term is the stay-alive force; the "largest BLENDING
increment" claim inherited a stale shaping.py:60 comment); "two orthogonal problems"
re-stated as one coupled object (§1 coupling caveat); slot-occupancy-time + churn-rate
added as co-measured quantities (GATE 4.3).
**Should-fix — STILL PENDING (forward work):** split the live correctness bugs
(escrow `sqrt` vs SHAPED `harmonic`; PBRS telescoping incl. `pbrs_weight`) into a
standalone PR this effort depends on; quantify the DPBA kill criterion + LOO
noise-floor method + power bound + the four named baselines in Phase 0.

**Genuinely sound (keep):** J locked to the COUNTERFACTUAL (not host accuracy) —
the right anti-drift-farmer anchor; the composite fail-closed gate as a real ALIGNMENT
addition over the EV-only epic; instrument-before-touch (Stage-0 reuse); per-seed
paired deltas + env-clustered block bootstrap on non-i.i.d. vec-envs; the anti-hack
inventory; verified governor independence; ~20 code citations confirmed exact;
Stage-2 R_main/R_cf consistency. The doc's honest hedging of its single-seed evidence
is why the wrong correlation is "fix before pre-registering," not fatal.
