# PDR-0072 — Round-8 evaluation: the floor-dead-zone reframe is calibrated (reprioritization sound, supersession overstated); workspace drift corrected

Date: 2026-07-14   Status: accepted (independent evaluation + owner-directed wording correction, within grant)
Supersedes: —   Amends: PDR-0069 (calibration banner added, body preserved). Related: `docs/analysis/2026-07-13-round8-findings-evaluation.md` (the deliverable), PDR-0071, PDR-0073. Memory: `floor-gradient-dead-zone`, `instrument-validity-before-interpretation`.

## Context
The owner asked, as the first job of this ownership session, to **evaluate the round-7 findings** ("they're quite numerous"). Those findings (PDR-0069/0070/0071) reframed the entire EV-stabilization epic and now gate the next GPU spend, and this project's own memory flags repeated diagnostic overclaim — so they warranted an independent adversarial pass, not a rubber-stamp. Evaluation used: my own primary-source code + math verification; an independent **drl-expert** (RL theory) and **pytorch-expert** (autograd/numerics), both of which *numerically executed the shipping symbols*; a **telemetry recompute** of the empirical cluster (direct DuckDB, ~1.08M decisions/seed); and two relayed **prime** reviews (gpt-prime, claude-prime) treated as first-class.

## What does this buy, and how is that measured?  (REQUIRED — PDR-0068)
It prevents building on an overclaimed premise. The buy: the epic proceeds on a **calibrated** claim (mechanism established, causal sufficiency pending an experiment) instead of an overstated one ("root cause found"), and the next expensive step is correctly re-ordered to a zero-GPU read (PDR-0073) that can confirm-or-collapse the premise before any GPU spend. Measured: the drift is removed from the living artifacts (verified by grep — no residual "can't learn"/"gate passes" causal-sufficiency claim), and the reframe's one unproven bridge (advantage-vs-LOO) is named as the gate rather than assumed.

## Options considered
- **Rubber-stamp the round-7 findings and proceed to the λ-sweep.** Cheapest; but repeats the overclaim pattern and spends GPU on an unproven premise. Rejected.
- **Independent adversarial evaluation (dual-expert + telemetry + primes), then calibrate.** Chosen. Higher cost, but the findings gate an epic reframe + GPU spend.
- **Reject the reframe outright.** Not supported — the mechanism is forced by the autograd.

## The call
**The reframe is CALIBRATED, not overturned.** Verdict:
- **Mechanism (dead-zone) SOUND** — floor-bound sampled actions make the PPO log-prob locally independent of the underweight raw logits; at |H|=1 the whole post-floor op distribution is locally constant (bit-exact zero gradient). Forced by the code, reproduced numerically by both experts, in the differentiable update path across all 8 heads.
- **One real overclaim (claim 5):** "immediate reward is LOO-graded ⇒ advantage rises ⇒ signal exists" is a **non-sequitur** (V(s) absorbs the LOO-driven part of the return; advantage is unknown in *both* directions). PDR-0071 §4 already self-corrected; PDR-0069's headline verbs did not — hence the calibration banner on PDR-0069 and the wording fix below.
- **Owner-directed wording correction (applied this session):** across `current-state.md`, `metrics.md`, `roadmap.md`, replace unqualified "the op head can't learn" with "the op-policy loss cannot directly update the op logits on these decisions" (shared-trunk + other-state gradients still move them; NOT "permanently frozen"), under an explicit Established / Not-established split. PDR-0069 preserved as historical record with a prominent correction banner (owner-endorsed approach).
- **New findings (verified) dispositions:** N1 wide-head expressiveness clamp — mechanism real but **severity REVISED DOWN by telemetry** (realized num_valid=7 not ≈13; ceiling ~2× uniform; realized 1.3-1.4× peaked → NOT a near-uniform sampler, NOT a new epic item). N2 aggregate-KL dilution — bank the mechanism; magnitude pending P8 (PDR-0073). N3 entropy/collapse monitoring structurally blind to the dead-zone — bank; a Telemetry Improvement Program scar. N4 floor-bound samples enter global advantage standardization — bank mechanism; direction pending. N5 fixed-λ confounds the sweep — banked → per-num_valid / op-isolated design (PDR-0073).
- **`hindsight_credit` comes OFF the shipping gate** — it is keyed on the non-causal `total_improvement` (over-read #12) and its 0.005% is a rare-trigger artifact (functional, fires 43-58×/seed), not a broken channel. Restate the co-requisite in advantage terms.
- **Single-pass floor confirmed (verified on the shipping fn):** the transform does NOT strictly enforce its advertised floor/cap at the margin (counterexample: overweight 0.151 → 0.1242 < 0.15; max 0.5758 > 0.55; entropy 0.8267 < H_min 0.853). Consequence: an "entropy ≥ H_min or STOP" validation gate would false-fire — do not use it. Mechanism untouched.

Reframe status: **justified as a reprioritization** (a critic/reward arm cannot move a zero-gradient logit), **overstated as a supersession** ("root cause found"). Hold the epic to PDR-0071's calibration.

## Reversal trigger
- If the offline advantage-vs-LOO read (PDR-0073, Read A) returns **GRADED at a parity-checked VALID tier**, the "causal sufficiency unproven" caveat upgrades toward confirmed and the fix is licensed to build. If **FLAT**, the reframe collapses to "immediate-reward gradient cancelled by baseline/continuation value" and the critic/reward line reopens (N2 prime suspect). If **INVERTED**, the epic's founding premise ("under-fossilization is a defect") is itself the over-read → escalate a strategy re-examination (owner-gated vision question).
- If the telemetry recompute's confirmed numbers (LOO overlap; |H|=1 63/72%; hindsight 0.005%) are contradicted by a larger-n or higher-precision read, revisit.
