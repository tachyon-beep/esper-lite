# Entropy-Floor Population Audit (concept)

**Date:** 2026-07-10
**Status:** concept — post-freeze; audit runs after the Stage-2 packet read
**Origin:** three-round external review (owner-relayed, "Sol") of the PDR-0053→0055
penalty-schedule confound arc, each round verified against the codebase in-session.
All code citations below were checked on `feat/ev-stab-stage2-hra` @ `6898f863`.
**Relates to:** TIP epic `esper-lite-c62d4891b0` (front-of-programme candidate),
per-head advantage re-standardization `esper-lite-89983714fb`, PDR-0055 schedule fix.
**reviewed_by:** (pending — drl-expert review required before any objective change;
observability additions reviewed with the schedule fix)

---

## 1. The question

The entropy-floor penalty may be regularising a population that conflates **policy
uncertainty** with **decision density** (how often a head has a genuine choice).
The audit quantifies, per head, whether floor activity reflects genuine low
conditional entropy, sparse choice opportunity, or a structurally unattainable
target — before any redesign is licensed.

This is the objective-side sibling of the entropy-thermometer measurement fix:
the telemetry population was repaired (choice-conditional emission), but the
optimiser's population was never re-audited. Measurement bugs and objective bugs
are cousins; fixing one is a reason to audit the other.

## 2. Verified implementation facts (code, not hypothesis)

- Per-step head entropy is normalised to [0,1] by `log(num_valid)` — the max
  entropy of that step's **legal action set**. A single-legal-action step is
  defined as **exactly 0**, via a grad-disconnected `torch.where`
  (`src/esper/tamiyo/policy/action_masks.py:595-611`). Singleton steps therefore
  carry **no entropy gradient path**.
- The floor penalty averages per-step entropy over the **availability mask**
  (head's parent op was *valid*), not the causal mask (op was *chosen*):
  `compute_entropy_floor_penalty` (`src/esper/simic/agent/ppo_update.py:158-246`),
  masks from `compute_availability_masks` (`src/esper/leyline/causal_masks.py:66`).
  This is a documented DRL-expert anti-death-spiral design (2026-01), not an
  oversight. Heads with `n_available < 1` are skipped.
- The `op` head's availability mask is **all-ones** (`causal_masks.py:109`), and
  its causal mask is also all-ones (`causal_masks.py:55`).
- Floors are fixed constants: op 0.30, slot 0.15, blueprint 0.20, style 0.15,
  tempo 0.20, alpha_* 0.10 (`src/esper/leyline/__init__.py:241-250`); per-head
  coefficients at `:269-278`. The penalty is `coef · clamp(floor − H_avail, 0)²`.
- The floor population **includes forced steps** (no unforced gating in the
  penalty path) and **includes singleton steps as zeros**.
- Existing telemetry populations (`src/esper/simic/agent/ppo_agent.py:1387-1405`):
  - `head_X_learnable_fraction` = |causal ∩ choice(num_valid>1) ∩ unforced| /
    |all valid rollout steps|.
  - `choice_conditional_head_entropy` = mean entropy over that same triple-gated
    set (causal ∩ choice ∩ unforced).
  - `conditional_head_entropy` = mean over the causal mask.

**Consequence:** there are three gating differences between the telemetered
choice-conditional entropy and the optimiser's floor input — causal-vs-available,
unforced-vs-all, choice-vs-singleton-zeros. The existing scalars **cannot exactly
reconstruct the sparse-head floor population.**

## 3. Named populations (do not conflate)

| Name | Definition |
|---|---|
| all | all valid rollout steps |
| available | steps in the head's availability mask (op: all steps) |
| **H_avail** | optimiser's actual floor input: mean normalised entropy over *available*, singletons counted as 0, forced included |
| **H_choice_avail** | mean normalised entropy over *available ∩ num_valid>1* (NEW — must be emitted) |
| **H_choice_causal_unforced** | existing `choice_conditional_head_entropy` (narrower; NOT substitutable) |
| causally-active | steps in the causal mask — the actor-gradient population |

The exact identity (holds because singletons are exact zeros):

```text
H_avail = d_avail × H_choice_avail,   d_avail = n_choice_available / n_available
```

> The identity applies only when the choice-conditional mean is computed over
> choice-bearing steps within the same availability population. The existing
> `choice_conditional_head_entropy` uses a narrower causal, unforced population
> and cannot be substituted — doing so would recreate the exact
> population-substitution mistake this audit is designed to catch.

## 4. Feasibility and pressure algebra

Since `H_choice_avail ≤ 1`: `H_avail ≤ d_avail`. Therefore:

- `feasibility_margin = d_avail − floor` — **first audit row per head**. If
  `d_avail < floor` the floor is mathematically unattainable by any policy.
- `H_choice_required = floor / d_avail` (for `d_avail > 0`) — near 1 means the
  floor demands a near-uniform choice policy even when technically feasible.

Gradient structure (verified against the implementation; active penalty):

```text
∂L/∂h_i              = −2·λ_eff·shortfall / n_available     (per choice step)
∂L/∂H_choice_avail   = −2·λ_eff·d_avail·shortfall           (aggregate)
```

> Singleton zeros can activate or make the floor infeasible, but their effect on
> corrective gradient pressure is **non-monotonic**: they raise the shortfall
> while also diluting the mean's gradient. As `d_avail → 0` the loss stays
> positive while the corrective force vanishes. The audit therefore reports both
> shortfall and effective pressure:

```text
effective_floor_pressure  = 2 · λ_eff · d_avail · shortfall
per_choice_step_pressure  = 2 · λ_eff · shortfall / n_available
```

## 5. Five-case classification (per head, per update)

| Case | Signature | Reading |
|---|---|---|
| A. Genuine confidence/collapse | d healthy, H_choice_avail low, floor active | policy genuinely low-entropy where choice exists |
| B. Density activation | d low, H_choice_avail high, floor active | floor fires because choices are rare |
| C. Structural infeasibility | d_avail < floor | no policy can satisfy the floor |
| D. Population-mismatch pressure | forced/noncausal choice steps materially shift H_avail vs H_choice_causal_unforced | regulariser optimises a broader population than the actor gradient |
| E. Persistent-loss / weak-gradient | shortfall high, effective_floor_pressure low | chronic complaint, little corrective force |

Priors to test, not assume: blueprint/tempo likely inactive (choice-conditional
0.97–1.0 vs floor 0.20); slot most likely structurally pressured; op may clear
0.30 despite forced-WAIT zeros; style borderline; alpha heads unmeasured.

## 6. What existing telemetry can and cannot answer

- **Op `H_avail`: exact now.** Availability is all-ones, so the optimiser's op
  floor input equals the raw all-steps normalised-entropy mean already emitted.
- **Op `d`: lower-bounded now, exact only if forced∩choice = 0.**
  `head_op_learnable_fraction = P(choice ∩ unforced)`; the floor's d is
  `P(choice)`. Verify forced-choice incidence before calling d_op exact.
- **Sparse heads: bounds only** until the new emission lands
  (`availability_fraction` is not emitted; entropy means are over mismatched
  populations). State the imprecision in any pre-emission read.

## 7. Minimum new telemetry contract (per head, per PPO update)

Reconciliation set: `n_all`, `n_available`, `n_singleton_available`,
`n_choice_available`, `entropy_mean_available` (= logged floor input),
`entropy_mean_choice_available`, `availability_fraction`,
`choice_fraction_given_available`, `floor_value`, `floor_schedule_phase`,
`floor_schedule_multiplier`, `base_floor_coef`, `effective_floor_coef`,
`floor_shortfall`, `floor_penalty_loss`, `effective_floor_pressure`.

Population decomposition: `n_choice_causal_unforced`,
`n_choice_noncausal_unforced`, `n_choice_forced`, matching entropy means,
`causal_active_fraction`, `usable_actor_sample_count`.

Telemetry reconciliation assertion (scar-suite candidate):

```text
entropy_mean_available ≈ choice_fraction_given_available × entropy_mean_choice_available
```

No backward hooks for per-head gradient norms initially — analytic pressure plus
actual loss contribution suffices for the first audit.

## 8. Sequencing discipline — three separate tracks

1. **Schedule correction** (PDR-0055, post-freeze, drl-reviewed): absolute
   successful-update breakpoints pinned to the 200-round shape. Tests: 0–199
   effective-coefficient identity between 200- and 600-round configs; off-by-one
   at k=49/50/150/151/199/200; skips keyed to successful `train_steps`; resume
   preserves schedule state; k≥200 behaviour explicitly defined.
2. **Observability addition** (§7): telemetry-only, identity-tested; may land in
   the same post-freeze engineering window as (1) but is a separate concern.
3. **Population/objective correction** (if the audit confirms contamination):
   e.g. floor on `H_choice_avail` with an explicit λ_eff design
   (A: λ; B: λ·d; C: λ·min(1, n_choice/N_ref)) — currently the implementation
   implicitly mixes target and density via `d × H_choice`. This is an
   optimisation-objective change: **own PDR, pre-frozen hypotheses, OFF-only
   diagnostic or paired A/B.** Do not smuggle it into (1) or (2). No λ_eff
   selection before audit data.

## 9. Stage-2 implications

- **Validity: none.** The A/B arms differ in exactly one config line
  (`hra_value_decomposition`); horizon, floor definition, schedule, and
  coefficients are common-mode at assignment.
- **Mediator, not confound:** HRA may change decision density → floor pressure →
  further policy differences downstream. The total-effect estimate stands; only
  the narrow mechanistic claim "all observed changes are direct effects of
  cleaner value decomposition" is limited. Packet-read wording: *Stage-2
  estimates the effect of HRA under the current entropy-floor objective,
  including any downstream interaction with that objective.*
- Do not amend the frozen gate. At most, report exact op-head floor telemetry as
  descriptive context after the packet is scored.

## 10. Scope-boundary sentences (for the PDR record)

> The entropy-floor objective is normalised over legal actions but averaged over
> an availability population that includes single-legal-action steps as zero.
> Consequently, its effective target may depend on choice density. This is a
> post-freeze audit item, not a finding about the current Stage-2 A/B.

> Any salvaged rounds-251–451 long-horizon read is descriptive: it establishes
> behaviour during a window where the two known entropy schedules are constant,
> but does not causally identify the source of within-window drift.
