# PDR-0069 — Root cause found: the anti-WAIT floor gradient dead-zone; the EV-stabilization commitment-premise is superseded

Date: 2026-07-13   Status: accepted (diagnostic finding, within grant) — the epic-direction reframe it implies is **owner-gated (proposed), flagged**
Supersedes: the EV-stabilization epic's founding premise ("advantage noise / value-target variance is the live suspect for suppressed commitment", roadmap Now bet / PDR-0027). Extends PDR-0066 ("critic is the wrong lever"). Related analysis (committed, multi-model reviewed): `docs/analysis/2026-07-13-decision-point-diagnosis.md`, `...-reviewer-catchup-brief.md`. Memory: `floor-gradient-dead-zone`.

## Context
The EV epic exists to fix *suppressed commitment* (Tamiyo rarely FOSSILIZEs). It litigated the critic (A/A′/B/C, HRA) and, after PDR-0066, the cf reward-shaping term — always on the premise that value-target/advantage noise is what suppresses commitment. This session ran a zero-GPU reward/commitment diagnostic on `stage2_on_longdiag/seed{41,42}` (advisor + claude-prime + gpt-prime; ~11 over-reads, all caught) and traced the defect to a **code-level policy-optimization bug**, not the critic or the reward.

## What does this buy, and how is that measured?  (REQUIRED — PDR-0068)
It replaces a mis-targeted optimization program (critic/reward-shaping arms that keep returning null) with the actual lever. Measured: **the reward signal for committing is already correct** — FOSSILIZE reward rises monotonically with contribution (c<0 → −0.7 ; c≥15 → +8.2, both seeds), so the "gate" that any commitment fix needs is PASSED; and **~94% of FOSSILIZE / 93-95% of PRUNE samples are floor-bound**, i.e. produce zero op-head gradient. The buy is: stop paying for critic/reward experiments that cannot move a frozen logit, and fix the primitive that froze it. Efficacy of the fix is the experiment in PDR-0070.

## The finding (both seeds, code-confirmed, zero-GPU)
- **Mechanism.** The anti-WAIT probability floor (`_apply_floor_to_logits`, `action_masks.py:522-581`) is in the DIFFERENTIABLE PPO update path (`factored_lstm.py:1533`) and applied to ALL 8 factored heads. A floor-bound action's post-floor prob is a constant, so its PPO log-prob has **zero gradient to the raw logits** (autograd-verified; partition cancels; PPO ratio = 1.0; zero op-head approx-KL). Forced exploration the policy cannot learn from.
- **Blast radius.** FOSSILIZE floor-bound in 93-94% of HOLDING decisions, PRUNE 93-95%; SET_ALPHA 0% (median 0.55 — the one op that stayed above floor and LEARNED).
- **Gate passes.** The immediate commit reward is strongly LOO-graded (−0.7→+8.2); with `V(s)` op-independent (P0-1), the advantage rises too. So the learning signal exists — the dead-zone is why the policy can't act on it. Explains flat-in-LOO fossilize, the FOSSILIZE/PRUNE-from-HOLDING current-LOO overlap (medians ~10 both), and the ~10% negative-current-LOO fossils (penalty → zero gradient to stop).

## The reframe (owner-gated DECIDE)
- The commitment defect is a **policy-optimization bug (the exploration primitive), not the critic (PDR-0066 confirmed) and not the cf reward-shaping term** (the reward signal is strong; the gate passes). The EV-stab value-target-variance line and the cf-earns-keep/SNR line (PDR-0067/0068) are both downstream of a bug neither addresses → demoted to secondary/parallel.
- **Two SEPARABLE problems, confirmed:** (A) commit-learning = dead-zone + a live reward → fixable in code (PDR-0070); (B) transfer/developmental value = `hindsight_credit` inert (0.005%) + no causal net-ensemble-value field → separate instrumentation track, unaffected by any floor fix.

## Reversal trigger
- If the no-floor necessity smoke (PDR-0070) shows the op head does NOT collapse to WAIT under the twice-changed reward, the floor may be obsolete rather than mis-designed (stronger conclusion, same direction).
- If a floors-off / mixture-floor arm restores commit-selectivity, the dead-zone was load-bearing and this reframe is confirmed as the epic's new spine. If commit-selectivity does NOT emerge despite a differentiable floor AND a live advantage-vs-LOO signal (logged in the re-run), the dead-zone was NOT the binding cause and the critic/reward line re-opens with a measured baseline.
