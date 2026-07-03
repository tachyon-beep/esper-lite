# Pre-A/B build — paid-event telemetry, fossilize-rate guard, ON-arm config, scoring pre-registration

```yaml
id: pre-ab-build
title: Pre-A/B build (committed-Shapley enablement gate, criteria 3+4 conditions)
type: ready
status: in-review            # drl-expert review of the guard design pending
tracker: esper-lite-3f3e9b9b4f
gate: esper-lite-f22a1d48a7 (comments #105-#106)
decisions: PDR-0018, PDR-0019
reviewed_by: [drl-expert 2026-07-03 — verdict ADJUST, all adjustments incorporated below]
```

**Review adjustments incorporated (drl-expert 2026-07-03):** MAJOR-1 extend the shipped
`CommittedShapleyTopUpPayload`, never a second event; keep `g`/`sum_raw`. MAJOR-2 the
rate guards cannot see k=3 CONCENTRATION (rate at/under 3× while paid events inflate
10–100×) — condition 3's frequency mandate is discharged by a **paid-event REVIEW
TRIGGER** (never a hard auto-reject: the honest success case also raises paid count)
that forces the G3 + Δcorr adjudication; division of labor stated explicitly: G1 = fast
runaway breaker, G2 = end-of-run gross-spike backstop, **G3 + Δcorr = the correctness
adjudicators**. MINOR-3 G3's 10% is PROVISIONAL pending a banked paired noise floor for
the efficiency read. MINOR-4 G1 abort flushes telemetry before raising; aborted-seed
pairing pre-registered in WI-5 (neither positive nor null). NOTE-6 trip threshold pinned
to the integer ≥13 fossils/batch (5×0.207×12 = 12.42). NOTE-7 adopted: carry the raw
2^k v(S) table on the event (masks+accs over the slot_ids ordering) — retires the
transient-proxy caveat on the first ON run.

Everything between the F2 freeze and the paired n=5 OFF/ON A/B launch. Scored per
PDR-0019. Enabling `shapley_synergy_scale > 0` on real runs remains owner-gated after
this completes (gate criteria 4–7 standing).

## WI-1 — Leyline payload + event type — **ALREADY BUILT (scope correction 2026-07-03)**

Recon against the shipped build found `COMMITTED_SHAPLEY_TOPUP`
(`leyline/telemetry.py:112`) with `CommittedShapleyTopUpPayload` (`:1648`) already
carrying per-slot `phi, c_paid, gap, raw, top_up, t_f, credit_buf` plus env-level
`g, sum_raw, clamp_binding, std_used` (= the review's `met_std`:
`max(running_std, std_floor)`) and `dropped_no_std`. No new payload or event type is
needed. WI-1 is CLOSED as pre-existing. Review bonus (MAJOR-1): the shipped emission is
NON-TRUNCATED — every env with k≥1 fossils at scale>0, including non-payers and k=1 —
which is exactly the null-player population the PDR-0017 ON-recalibration P99 needs.

## WI-2 — Delivery-seam emission — **MOSTLY BUILT; small residual**

`apply_committed_shapley_credits` (`ppo_coordinator.py:45-132`) already computes the
floored divisor and the coordinator already emits one event per credited env
(`:435-447`, hub-guarded; the buffer write never depends on telemetry). Residual for
drl condition 4, EXTENDING the shipped payload (never a second event; `g`/`sum_raw`
untouched):
- **`running_std_raw: float | None`** (the pre-floor `current_std()`; None on the
  dropped_no_std path) — env-level.
- **`std_floor_bound: bool`** (env-level, computed at write: `std_used > running_std`)
  and **`normalized_cap_bound: tuple[bool, ...]`** (per-slot, computed at write:
  `top_up/std_used > normalized_cap`) — authoritative in-code booleans, no offline
  float-equality derivation.
- **`v_table_masks: tuple[int, ...]` + `v_table_accs: tuple[float, ...]`** (NOTE-7):
  the raw 2^k coalition table, mask bits over the `slot_ids` ordering — one-shot
  retirement of the transient-factorial proxy caveat on the first ON run. Requires
  plumbing `coalition_accs` onto `CommittedShapleyEnvCredits` (trainer already holds it
  at the populator).
- Round-trip test updated (`_make_field_value` trap: esper-lite-obs-52b8d18afc). Bind
  RATES stay offline aggregations; the confound gate keys on `std_used` (present).

## WI-3 — Fossilize-rate guard (gate criterion 3; drl-review condition 3) — THE DESIGN

**Fact constraining the design:** fossilize count per env-episode is structurally ≤ 3
(one per slot; FOSSILIZED is terminal within an episode — probe data: max k observed = 3).
So a *within-episode* counter cannot express the over-commitment surface; the surface is
**rate inflation across episodes** (control baseline 0.207 ± 0.006 fossilize/ep).

Three components, escalating from in-run to scoring:

- **G1 — in-run HARD abort (wired, config-gated `scale > 0`):** per PPO batch
  (12 env-episodes), count successful fossilizations. Trip condition: **≥ 13 fossils in
  a batch** (pinned integer; 5× baseline = 5×0.207×12 = 12.42) for **2 consecutive
  batches** → flush accumulated telemetry (MINOR-4: the paid-event/null-player data must
  survive the abort), emit `FOSSILIZE_RATE_GUARD_TRIPPED` (batch idx, counts, threshold,
  consecutive count), then abort fail-loud (raise). Aborting (vs credit-freezing)
  preserves experiment integrity — mutating reward semantics mid-run would confound the
  arm. Division of labor (review MAJOR-2): G1 is the fast RUNAWAY breaker only — it
  cannot see k=3 concentration and is not the correctness adjudicator.
- **G2 — scoring gate (pre-registered, WI-5):** end-of-run ON-arm fossilize/ep
  > **3×** the paired OFF-arm rate ⇒ the A/B is NOT bankable regardless of Δcorr
  (over-commitment regime; also design criterion (iv) "no degenerate spike"). Review
  note: a GROSS-SPIKE backstop only — a loose proxy that concentration can sit under.
- **G3 — off-switch-J efficiency HARD gate (pre-registered, WI-5):** ON-arm efficiency
  (acc-per-param via the suppress/off-switch read) degraded > 10% vs OFF ⇒ not bankable
  (PDR-0018 §3 safety floor, made HARD per drl-review condition 3). The 10% is
  **PROVISIONAL** (MINOR-3) pending a banked paired noise floor for the efficiency read.
- **G4 — paid-event REVIEW TRIGGER (pre-registered, WI-5; discharges condition 3's
  frequency mandate — review MAJOR-2):** realized paid events per ON run materially
  above the ~12 expected (pre-registered line: > 3× = > 36) **forces** the G3 + Δcorr
  adjudication before any positive result is banked. NEVER a hard auto-reject — the
  honest success case (the term working) also raises paid count; a count gate would
  Goodhart the guard itself. Derivable offline (count `top_up > 0` events); no new code.

Explicit non-choices: no in-run credit cap mutation, no per-episode credit-eligibility
counter (structurally moot at k ≤ 3), no hard paid-count gate (G4 rationale above).
G3 + Δcorr(reward,J) are the actual correctness adjudicators; G1/G2 are breakers.

## WI-4 — ON-arm config + validation bound

`configs/config-3slot-3seed-baseline-shaped-shapley-on.json` = the OFF control config
plus: `shapley_synergy_scale=1.0, shapley_synergy_noise_floor=0.28,
shapley_synergy_cap=5.0, shapley_synergy_std_floor=0.25,
shapley_synergy_normalized_cap=3.0`. Config validation gains the missing upper sanity
bound: `shapley_synergy_normalized_cap ≤ reward-normalizer clip (10.0)` (fail-loud).
Paired same-commit with the OFF config (NUM_BLUEPRINTS constraint, PDR-0015).

## WI-5 — Pre-registered scoring doc

`docs/analysis/2026-07-0X-shapley-ab-scoring-preregistration.md`: operationalizes
PDR-0019 — primary = design criteria (i)–(v); effect-size floor = episode-level paired
Δcorr(reward,J) ≥ +0.10; asymmetric null (a null n=5 fires nothing); the std<1.0
confound gate (re-confirm at std_floor ≥ 1.0 if the effect is carried by early
low-std events); the conditional-payment-magnitude table as DESCRIPTIVE-ONLY (vs 2×tau
diagnostic); G2 backstop + G3 hard gate (10% provisional) + **G4 paid-event review
trigger and its adjudication path**; **G1-abort pairing rule: an aborted ON seed is
neither positive nor null — its pair is DROPPED from direction scoring and the abort is
investigated before any re-run** (pre-registered so post-hoc discretion cannot bias the
read); paired seeds 41–45; same-commit hash recorded at launch.

## Test plan
- WI-1: payload round-trip + enum/serialization tests.
- WI-2: unit test the emission with a stubbed normalizer (known std) and a crafted
  `CommittedShapleyEnvCredits` — assert met_std, flags, and one event per credited slot.
- WI-3: unit test G1 with synthetic batch fossilize counts (below/at/above threshold;
  1-batch spike does NOT trip; 2-batch does; abort raises + event emitted). Guard
  inactive at scale=0 (OFF arm and all current runs unaffected).
- WI-4: config validation tests (bound accepted/rejected); load the ON config end-to-end.
- Full suite green before the task closes.

## Acceptance
Criteria 3+4 conditions on the gate satisfied in code + docs; `uv run pytest` green;
no reward-value change at scale=0 (OFF-arm bitwise identity preserved); scoring doc
committed; then the A/B launch decision goes to the owner with GPU cost stated.
