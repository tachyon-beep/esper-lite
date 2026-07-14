# Current State — Esper        Checkpoint: 2026-07-14 (#71) · make-permanence-visible: **ALL FREEZE READS DONE; PLAN APPROVED (3 drl rounds); BUILD STARTED (increment 1 landed `3b36aa56`)**. Owner convergence-rule applied (PDR-0090): Phase-2 authorized, dissolution/bounds/price-don't-fix/Option-A/max-n ratified. Owner forks OPEN: **F1 abort-payment · F2 pending-bit · F3 threshold-gate redesign**. PDR-0085..0097; branch `feat/ev-stab-stage2-hra`

## The bet right now
**Unchanged: the commitment defect is a permanent-value MEASUREMENT gap (PDR-0074/0076)** — the counterfactual is
undefined at permanence, every consumer inherits the `None→0` lie, and the fix is L1 freeze-don't-zero + L2
non-selectable boundary settlement priced to neutralize the forfeited stream. **Phase-1 replay re-certified the
premise: the forfeited attribution stream (−274.7 PV from commit) still dominates the shipping commit ledger by two
orders of magnitude.** What changed is the PRICING LAYER (PDR-0085): the fine structure the verdict gates read was
wrong in three places and is now measured, not hand-derived. · metric: does a genuinely-contributing seed retain its
measured value across FOSSILIZE; guardrail = the fix does NOT farm fossils.

## In flight / freeze-blocker scoreboard
- **Replay (blocker #3): PHASE 1 DONE + reviewed + LANDED `b7b7c6d5`** (`tests/simic/rewards/replay/`, 84 tests; drl
  APPROVE-WITH-NITS, 5 nits closed; findings `docs/analysis/2026-07-14-gledger-replay-phase1-findings.md`).
  **HEADLINE (PDR-0085, r9-confirmed both seeds, 0/4,151 commits):** the trainer epoch ordering makes the PBRS
  `eis==0` branch unreachable live → ALL cross-stage PBRS transition deltas are SKIPPED. Audit **F5's forfeiture never
  fires**; **fossils re-accrue potential 6.0→8.0**; **PDR-0084 #4's park>fossil +0.31 residual REVERSES** (pro-fossil
  +0.02…+0.57 measured); holding_warning relief (+27.7) is the 2nd-largest pro-commit term; **F8 hatch measured +45.8
  PV**. Six pre-reg amendments PROPOSED for round 21 — **body deliberately NOT edited** (one touches the owner-pending
  Φ(PENDING) call). Phase 2 (B path, 7 gates) stays owner-gated.
- **Power calc (blocker #2): DRAFT accepted at review (PDR-0086)** — `2026-07-14-paired-run-power-calc-DRAFT.md`.
  n=10 ≥80% iff σ_d ≤ 1.17·Δ; Exp-1 arm-B P2 = BOUNDING not detection; tercile gates set required n (σ_d ×1.7–2.5);
  blinded reassessment = unsigned within-pair spreads, point-estimate primary; max-n=10 proposed + escalate branch.
  **Circularity fix: Δ_material must freeze ABSOLUTE via a NEW r9 q_decision-IQR read — task esper-lite-e12e2d1543
  (open, pre-freeze).** Six §8 owner ratifications outstanding.
- **r9 cadence read (was queued): DONE (PDR-0087)** — 100.000% per-epoch HOLDING measurement, zero gaps, both seeds →
  **PDR-0084 reversal trigger #3 does NOT fire**; zero-slack window feasible; bridge assumption named (pending seeds
  stay ablatable to boundary, §2.0). obs-v3 indirect verification ratified; missing positive obs-version stamp filed
  as TIP observation (esper-lite-obs-77569bfa0a).
- **Remaining before freeze:** (1) round-21 fold-in of the six PDR-0085 amendments (needs the owner's Φ(PENDING)
  call); (2) the q_decision-IQR read + owner ratification of ΔP_req; (3) final adversarial pass on the corrected,
  integrated doc; (4) owner gates below. **No GPU; nothing pushed.**
- **Harness enablers** (esper-lite-7fe21bd091, OPEN) and **branch-survivor** (esper-lite-1f1e55f58f, parked on merge
  window; stale-claim flag is bookkeeping) — unchanged.

## Facts the next session must not relitigate
- **PBRS live semantics (PDR-0085, executed read):** `record_accuracy` ticks before decisions; reward sees pre-action
  state; mutations execute after → first post-transition reward row has eis=1 → cross-stage PBRS deltas skipped.
  Code-cited, drl-review-re-verified (incl. FOSSILIZE synchronous inline), r9-population-confirmed. The kasmina
  fidelity test pins MECHANICS given the ordering, not the ordering itself (scope stated in the test).
- **F5 is refuted as a live penalty; A7's park residual is pro-fossil; do NOT quote the old −0.31/+0.31 figures.**
- **The finding does NOT weaken the epic** — forfeited attribution still dominates; the annuity remains the revenue line.
- **Matched-control holds across arms** (all share the trainer ordering) — the correction is to Δ_designed's pricing,
  not to arm design. **Price the skip, don't fix it mid-arc** (fixing changes arm A's shipping semantics — recommended
  owner ruling, PDR-0085).
- Carried from before: measurement gap fire-rate not magnitude (PDR-0074); premise UNMEASURED not refuted; floor +
  reward are a PACKAGE (FOSSILIZE 93–94% floor-bound); critic = symptom (round-11/H6); no reward transform fixes a
  missing INPUT (ESCROW contraindicated); Read B (K=1 KL guard inoperative — fixed by landed K=4); the r9 archival
  regime stamp must be asserted by every r9 read.
- **Power-calc discipline (PDR-0086):** never present an assumed-σ_d power as predictive; Exp-1 P2 is bounding; a
  relative-to-relative reassessment is vacuous — Δ_material is absolute or nothing.

## Open questions / blocked-on-owner  (Step-2 escalations — flagged, NOT enacted)
- **RATIFIED (PDR-0089): ΔP_req = 0.10.** Δ_material = 0.10/IQR_q becomes a number when the in-flight read lands; the
  0.10 family propagates as working constants to the gate-UCB and threshold-variance bounds (they inherit, not
  separately ruled). IQR-read legitimacy: needs a §1.12 citation, not a ruling (claude-prime r22) — dispatched.
- **Still yours — evidence-forced per claude-prime's sort (ratification ≈ bookkeeping):** Φ(PENDING) DISSOLUTION
  (both alternatives dead by executed facts; PDR-0088); gates-as-BOUNDS (detection n≈29–63 unreachable; wide bound =
  INCONCLUSIVE-on-gates); price-don't-fix the eis==0 skip (the matched-control ruling you've made twice).
- **Still yours — genuinely open:** **Phase-2 B-path authorization** (the first core-reward-path change of the arc;
  gpt-prime's proposed scope: default-off impl + tests + executable replay + drl review + docs; NOT training
  activation, NOT GPU); **Option-A cross-slot lock** (a real tradeoff — the lock enters the protocol-package estimand;
  lock-burden quantification queued before freeze); **pending-visibility fork** (gpt: obs must carry pending status +
  time-to-boundary or state-aliasing; vs the frozen §2.5.8 obs-dim-31-OFF pin + claude-prime's interpretation-note
  posture — PDR-0089 #4); **threshold-variance gate form** (data-conditional: raw EWMA fails ΔP<0.10 at σ≳0.15–0.2
  near the cliff, LCB holds to σ≈0.3 — the in-flight noise read decides; PDR-0089 #5); UCB posture (point-estimate
  primary recommended); max-n=10 confirmation; premium signature (lands with the freeze).
- Standing rules: git identity tachyon-beep; no push/tag/release/branch-deletion/telemetry-deletion/remote action
  without an explicit ask.

## Last did (this checkpoint, #62; #61 earlier today)
- **#61:** owner-requested first-principles review of #55–#60 (upheld); built + landed replay Phase 1 (84 tests,
  drl-reviewed, nits closed, `b7b7c6d5`) → PBRS live-semantics finding banked (PDR-0085); power-calc framework
  accepted (PDR-0086); r9 cadence read banked, trigger #3 not fired (PDR-0087).
- **#62 (this one):** both round-21 primes adjudicated against the harness before banking — **the harness corrected a
  prime hand-derivation a SIXTH time** (the "−0.09·d" flag-delta budget → measured |Δ|≤~0.14, dwell-sign-varying).
  **ROUND-21 FOLDED IN:** pre-reg §2.5.9 rewritten (Φ(PENDING) dissolution PROPOSED, both dead options recorded,
  "telescoping" clause withdrawn), G-PBRS gate rewritten to skip-semantics, §ROUND-21 block added (ledger pins:
  −274.7-vs−113 convention, F8→GATE-INFL, +27.7→suspension clause; gates-as-bounds), blockers updated; audit A8
  appended. PDR-0088. Prime ordering disagreement resolved: fold-in carries PROPOSED markers, owner ratifies on the
  integrated doc.

## Next session, start here
1. **Freeze reads: ALL DONE.** Cadence 100% (PDR-0087) · Δ_material = 0.0075/pp, IQR_q 13.32pp (PDR-0092) ·
   near-cliff σ ≈ 0.82pp → BOTH pre-registered gate forms fail → **F3** (PDR-0092) · observability: committed
   state genuinely ALIASED, masks logit-only, countdown derivable → **F2 narrowed to ONE pending bit,
   recommendation = add it** (PDR-0093) · lock burden 1.75% → Option-A condition DISCHARGED (PDR-0095).
2. **Build state:** plan APPROVED after 3 adversarial drl rounds (B1–B5/N1–N5/W1/W1-A/W1-B/N-r2a all closed;
   PDR-0091/0094/0096/0097 — key constructions: single-owner ledger owns the committed seed's ENTIRE per-seed
   row [R+1,B]; T_annuity ≡ T_provisional LITERALLY incl. both ratio branches at windowed-cf; ledger-driven
   window row; epoch-phase placement §0). **Increment 1 LANDED `3b36aa56`** (leyline FossilSettlementConfig,
   settlement.py pure core, ratio_penalty extraction; 257/257 green). **Next increments:** kasmina committed
   fields + controller freeze → action layer (request routing, §6 mask/validity table incl. slot_by_op[WAIT] +
   index-0 fallback, Option-A lock + burden counters) → reward-path adapter (§2 branch table, single-owner
   guard) → replay B path (§A ten areas, both F1 branches, both ratio-check failure directions). drl
   pre-commit review gates feature completion.
3. **Owner forks (the ONLY unmade decisions):** F1 abort payment (settle-partial vs void; config REQUIRES the
   choice); F2 pending bit (recommend: add — one per-slot dim, arm A zeros, schema version); F3 threshold-gate
   mechanism (options PDR-0092 #3: longer window / larger z / margin / price-it).
4. **Then:** blinded power certification (80%-UCB primary, PDR-0091 #6) → final fresh-context adversarial pass →
   no-peek freeze (owner signs: premium + F-rulings) → GPU decision (owner).
5. Unchanged logged items: `test_ev_liftoff_k4` threshold drift (unread cause); observations
   esper-lite-obs-73926b0291 (pgrep scar, expires 2026-07-26), obs-77569bfa0a (obs-version stamp),
   obs-3811e62ea1 (scratchpad filename collisions).
