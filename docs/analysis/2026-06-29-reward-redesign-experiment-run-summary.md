# Reward-Redesign Experimentation Run — Findings Summary (Phase −1 → R1 pilot)

**Date:** 2026-06-29 · **Scope:** the morphogenetic-PPO reward-redesign investigation from the GATE −1
cheap-fix falsifier through the in-flight R1 causal-contribution pilot. · **Status:** GATE −1 = SURVIVE;
causal-run harness built, reviewed, and validated on GPU at full scale; R1 pilot DONE — harness validated, but the causal read is blocked by a pervasive entropy collapse (wrong population).

**Evidence-strength ledger (read the rest through this):**

- **n=5-locked (paired, 5 seeds):** the GATE −1 SURVIVE verdict; the Phase-0 aggregates share_attribution
  0.925±0.004 and fossilize/ep 0.207±0.006.
- **n=1 / control:41-provisional (NOT yet replicated to ≥5 — Phase-0 marked "DO NOT harden"):** the efficiency
  figures below — host-alone 40.0% → with-seeds 46.1% (+6.1pp), the +252% params / ~7× compute estimate, the
  54%-of-fossils-at-r0c0 and −0.34 LOO motif.
- **Unresolved / pending:** the (a)/(b) fork (the lifecycle closer came back INCONCLUSIVE); the Stage-2 paired-Δ SD
  (running). Some GATE −1 per-arm numbers below live in the GATE1_VERDICT scratchpad + Karn, not yet backfilled
  into the GATE −1 evidence packet (traceability gap, flagged).

**The defect under investigation (n=1/control:41 read):** the controller commits **inefficient** structure — real
accuracy at high cost — dominated by early-conv stems at r0c0 with a negative per-seed LOO marginal that keep
getting committed. This is an **efficiency / credit-assignment (steering) problem** — real gain committed
inefficiently — **not** a do-nothing or reward-farming one (an earlier "degenerate" thesis was overturned).

---

## 1. GATE −1 — the cheap-fix falsifier: **NO STOP (redesign SURVIVES)**

**Question:** does a cheap reward rescale (clip / unit-normalize / escrow) make the pathology go away — cut the
inefficient churn **and** hold/raise committed accuracy — so the credit-assignment redesign is moot?

**Method:** 5 arms × 5 seeds (41–45), 25/25 runs complete. Scored with **seed-matched paired deltas**,
block-bootstrap resampling the **5 SEEDS as the unit, not the 12 vec-envs** (the sole pre-registered invalidator).
B=20000. *(Pre-registered at 90% seed-level CI; the numbers below are reported at 95% — wider, and the verdict is
robust at both levels since every acc-Δ CI spans 0.)*

**Result (Δ vs control, 95% seed CI):**

- clip2: churn **−0.148** [−0.23,−0.06] (sig) · acc −0.30 [−0.70,+0.07] n.s.
- clip5: churn **−0.125** [−0.23,−0.03] (sig) · acc +0.04 [−0.61,+0.54] n.s.
- unitnorm churn **+0.195** (UP); escrow churn **+0.297** (UP).
- **final val-acc is arm-invariant ~42–44% (every acc-Δ CI spans 0).**

**Verdict: SURVIVE.** No non-escrow lever achieves churn↓ AND acc↑. clip2/clip5 trim churn but **trivially**
(<1% of a ~24/ep base) with no accuracy benefit ⇒ the cheap rescale levers are inert on the pathology; it is **not
a scale artifact**. (Escrow was an in-scope cheap fix too and also failed — churn UP; it is the arm the mid-scoring
classification bug had mislabeled as clip2, now corrected.) **SURVIVE ≠ ADMIT:** this closes only the cheap-scale
escape hatch; it does not license building the reward. *Validation:* the rebuilt pipeline reproduced three prior
numbers (share≈0.92, churn≈24/ep, foss≈0.20/ep).

---

## 2. The (a)/(b) fork — reopened, and **unresolvable on existing telemetry**

SURVIVE reopened the central question: are the neg-LOO r0c0 stems
**(a) freeloaders** [→ penalize survival-without-contribution] or
**(b) LOO-undervalued enabling stems** the controller is right to keep [→ synergy/hindsight credit]? Opposite fixes.

**Finding (mechanistic): existing data cannot answer it.** Removal-cost ablation is structurally blind here —
(i) post-fossilization a committed stem is baked into the always-on baseline (un-togglable), and (ii) the puzzle
stems **commit before any downstream neighbour co-resides** (the cohort's running `interaction_sum` ≈ 0). So any
enabling, *if real*, is **developmental/sequential**, which no contemporaneous ablation can see. The
lifecycle-conditioned synergy closer — after correcting an initial buggy join (re-keyed on envelope seed_id,
reconciling 482/482 fossils, validated 3 ways) — came back **INCONCLUSIVE**: a weak per-seed median lean that
collapsed at n=5 (cohort medians +1.92/+0.48/+0.12/+0.24/**−0.84**, not distinguishable from 0). A *separate*
factorial showed real positive synergy (+0.79±0.22) but among the **wrong population** (transient co-resident
*helpers*, not the committed fossils that are the puzzle). **Net: the fork is unresolved; only an owner-gated
parallel-control causal run can resolve it (at the system level — see §3).**

*(Banked discipline, hit 3–4× this run: no directional read off <30 hand-picked samples; verify the population you
measured IS the population the question is about; median-led + paired + pre-registered always.)*

---

## 3. The causal-contribution run — design → adversarial review → v2

**v1** (whole-run, matched-seed parallel control; snapshot-branch proven infeasible) was **dual-reviewed**
(morphogenesis + governor) → **5 blocking findings** → NEEDS REVISION. A 4-phase agent workflow produced **v2**
(`docs/plans/concepts/2026-06-28-causal-contribution-run-design-v2.md`); **four specialists reviewed the v2
fix-list / intent** (not the artifact — the v2 design's *first* full review is the implementation pass, R2 below).

**The load-bearing v2 decision — estimand pivot (owner ratification pending):** rather than patch a placebo arm
(a frequency-matched placebo cannot bound the action-space-reduction artifact in either direction — too-inert
under-bounds, leverage-matched over-subtracts), v2 **deletes it and adopts the TOTAL-SYSTEM estimand.** This
dissolves the placebo-can't-bound and the §1-vs-§3 estimand-contradiction blockers but **reduces Goal-1's claim**:
the run can show *the system is worse without r0c0's ability to commit*, **not** that r0c0 *mechanistically enables*
downstream. It still bifurcates the reward decision (Δ_struct < 0 → r0c0 load-bearing, don't penalize; the (a)
freeloader cell — penalize — requires Δ_struct spanning 0 AND a null acc-Δ AND relocation=FALSE). *Caveat:* the
pivot was executed by a recovery agent on a reviewer's argument against the placebo *concept*, not an evaluated
construction — so it is the owner's scope call, not settled.

**Verdict: NEEDS-ANOTHER-PASS** — the design has **converged**, but three residuals are **irreducibly
empirical/code**, not closable by a doc edit:

- **R1 (dispositive):** a single global RNG was shared by controller-sampling, host-stochasticity, and
  blueprint-init; suppressing r0c0 changes the blueprint draw count and **offsets the controller stream**. That
  injects a *deterministic, seed-reproducible confound* into each per-seed Δ_struct — a **bias on the estimate**,
  not sampling noise — and degrades the matched-seed (CRN) variance cancellation. The fix is two-part: split the RNG
  and **prove the offset is gone** (the *offset-free gate*, not the SD, is what rules out the bias), then re-measure
  the realized paired-Δ SD.
- **R2:** the §5 mask/RNG fixes were spec, needing implementation + a fresh review of the real diff.
- **R3:** confirm the fresh control emits `GOVERNOR_ROLLBACK` rows (still owed).

---

## 4. Building the harness (estimand-invariant core) + R2

The owner chose **"build §5 + run R1."** Implemented the estimand-invariant subset (SUPPRESS-COMMIT deferred):
**SUPPRESS-SLOT(r0c0)** stationary op-mask with the op-validity repair; the **three-domain RNG split** — a
dedicated controller generator; a per-env host generator (**hygiene only**: it backs the discarded germination
shape-probe, never training values; it just keeps the global stream from advancing); and **content-addressed**
blueprint-init via CPU save/seed/restore (draws that advance no persistent stream yet give matched-coordinate
germinations identical weights across arms) — plus the compare-point and `INTERVENTION_CONFIGURED`/`INTERVENTION_STEP`
telemetry. ~724 insertions + 8 new files; 30 new tests; OFF-path a structural no-op (byte-identical *call sequence*,
not a GPU bit-reproducibility claim). *(Uncommitted on `feat/phase-minus1-scale-falsifier`; figures from the
implementation — verify exact counts with `git diff --stat`.)*

**R2 — three independent specialists on the actual diff: NO BLOCKERS** (all conditionally-fit). Verified: the
**controller generator advances a fixed, mask-independent number of draws/step** so no blueprint or host draw can
offset it (the bootstrap step uses the global default stream, but symmetrically across arms — no artifact);
content-addressing is coordinate-keyed; **governor independence preserved** (the mask sits downstream of the panic
verdict and the preflight veto, only narrows, force-prunes nothing); OFF-path no-op. **Hardening applied:** a
**device-portable `controller_draw_count`** + a within-run constancy assert (hard raise) + analysis-side
`OffsetParityError` gates — the dispositive offset proof, *given* that CUDA `multinomial` consumes a fixed,
value-independent amount of RNG per call (which §5 establishes — the two are complementary, not substitutes).

---

## 5. Stage-1 — CUDA verification of the compare-point: **ALL GATES PASS (smoke scale)**

A short on-GPU smoke (3 arms, seed 41, 80 steps), the gate the reviewers required before any expensive run:

- **Offset-free (`control == on`):** draw-count parity holds ⇒ suppressing r0c0 does **not** offset the controller
  stream — the dispositive proof that the Δ_struct artifact channel is closed **at smoke scale, on CUDA**
  (full-run closure is gated in Stage-2, §6).
- **CRN no-op (`control == off`):** identical draw-counts **and** state-hashes across 80 steps, two processes.
- **`control vs on` state-hash agreement = 80/80** ⇒ `torch.multinomial` is value-independent on **this CUDA
  build/arch/shape** — which validates the state-hash as a cross-arm detector and **resolves the reviewers'
  CUDA-value-independence sub-condition** (previously CPU-verified only; re-check on a PyTorch/GPU upgrade). The hash
  digests the controller *generator position*, not the trajectory — the arms sample different actions (control
  germinated 35 r0c0; ON, **zero**).
- **Suppression works:** ON germinated **zero r0c0**.

`gpu_preload` (cifar data-pipeline) is orthogonal to the RNG *generators* and to the offset-proof gates — re-running
the gate analyzer on a `gpu_preload` smoke passed all of them — but it is **not** trajectory-neutral (same seed, ON
germinations 91→92), as expected under the *CRN/statistical, not bit-exact* class. Both arms run identical
`gpu_preload` so the data order is shared; note the current gate set checks controller-stream parity, not data-order
pairing directly. Wall-time: ~17h/run → **~6.6h/run** with `gpu_preload`.

---

## 6. Stage-2 — the R1 measurement: **DONE (n=3) — harness VALIDATED; NO causal read**

`control` + `suppress_slot_on`, seeds 41–43, full config, `gpu_preload`. The first attempt crashed (~ep 161) on a
transient non-finite logit reaching `multinomial` — a latent fragility (the floor *propagates* non-finite; training
`validate=False` removes the guard), exposed by the split's altered trajectory, **not** the offset proof. A NaN-safe
sampling guard (contained, tested; leaves the stored log_prob un-guarded so the PPO finiteness gate still catches the
batch — robustness, not bug-hiding) fixed it; the re-run completed all 6 runs full-length, **0 finiteness-gate trips**.

The result splits in two (full detail: `docs/analysis/2026-06-30-r1-pilot-result.md`):

- **Part 1 — harness VALIDATED (banked, policy-independent):** offset-free (control==on draw-count parity), zero-r0c0,
  stride-constant — all PASS on the full 30k-step runs ⇒ the systematic RNG-bias channel into Δ_struct is closed.
- **Part 2 — NO read on (a)/(b):** both arms suffered slot-head entropy collapse **from ~update 3**; final accuracy is
  **pinned ~52% across every decile, both arms, regardless of structure committed**. So all fossils commit
  post-collapse, and Δacc≈0 is accuracy-**insensitivity**, not preservation — the wrong population for the causal
  question. The raw paired numbers (Δ_struct median +707k, Δacc≈0) are **recorded but NOT interpreted**: Δ_struct>0 is
  near-tautological germination-redirection (block 1 of 3 slots → pressure lands on the other 2), not "relocation." (An
  earlier "r0c0 is replaceable" read was caught and pulled — population mismatch.) **No causal direction banked.**

Caveat carried forward: the *pairing-strength* SD was measured on the collapsed regime; it must be **re-measured on a
healthy policy** before "CIs trustworthy" applies to the causal regime.

---

## 7. Open decisions & next steps

1. **Entropy-collapse fix — the gate to ANY causal read** (n=5 on this config buys only uninterpretable seeds): fix
   `entropy_anneal_episodes=3000` vs 200-episode mismatch; activate the dormant std-floor / strengthen the
   entropy-floor penalty on the raw distribution; keep policy logits fp32 (also closes the crash's overflow surface).
2. Then **re-measure the paired SD on a healthy policy** → n=5 → n=10 (the morphogenesis seed floor, zero margin).
3. **Check whether GATE-1's runs collapsed similarly** (Karn) — collapse may be a general config feature (GATE-1
   SURVIVE still stands — all arms collapse alike — but the collapse itself wants explaining).
4. **Owner estimand decision (pending, deferred):** ratify the total-system pivot vs commission a concrete
   placebo/DUMMY-R0C0 design. Unchanged by this pilot.
5. **R3:** Karn-confirm control `GOVERNOR_ROLLBACK` rows. Owed: per-stage placebo noise floor; GATE −1 numbers
   backfill into the packet; J's missing fixed-schedule baseline (ADMIT unmeasured).

**Net:** the cheap escape hatch is closed (GATE −1 SURVIVE); the (a)/(b) question is real and needs a causal run;
that run's harness is **built, independently reviewed, and validated on GPU at full scale** — but the pilot surfaced a
pervasive entropy collapse that **blocks the causal read until the controller is healthy.** Tools ready, one clear
prerequisite to clear, science still open. Trackers: `esper-lite-a221da47ea` (GATE −1), `esper-lite-3d67b09687`
(Stage-0 instrumentation); PLAN_TRACKER row `reward-redesign-causal-run`.
