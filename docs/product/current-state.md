# Current State — Esper        Checkpoint: 2026-07-13 (#42) · dead-zone reshaped WHOLE-HEAD + co-requisite sequencing (PDR-0069/0071; on `feat/ev-stab-stage2-hra`)

## The bet right now
**Fix the commitment defect at its root — the floor gradient dead-zone — via the exploration
primitive (PDR-0069/0071).** The commit reward is already strongly LOO-graded (−0.7→+8.2); the op
head can't learn from it because the anti-WAIT floor censors its gradient. **Round-7 reshaping: the
dead-zone is WHOLE-HEAD** — in 67% (s41)/80% (s42) of HOLDING decisions only one op is above the
floor, so the output vector is CONSTANT → zero gradient to ALL op logits; SET_ALPHA's 0.55 is the
analytic cap, not a learned preference (round-6 "learned" RETRACTED). Fix + experiment are
**owner-gated**.

## In flight
- **Harness enablers** (esper-lite-7fe21bd091, ACCEPTED, no GPU): default-on checkpointing +
  per-decision **advantage + pre-floor logits + per-head approx-KL** logging (comment #180). Land
  FIRST — none of those were captured (PDR-0026 recurrence), so the true gate and the free KL
  confirmation are unreadable.
- **EV-stabilization epic** (esper-lite-f25b71c165): commitment-premise reframed (comments #178/#179);
  the ON-leg A′/B DECIDE (PDR-0064/0065) is moot (critic is downstream of the floor bug); cf-earns-
  keep/SNR (PDR-0067/0068) demoted to secondary.
- **Branch-survivor** (esper-lite-1f1e55f58f): main-as-trunk decided (PDR-0062), sign-offs in;
  execution gated on the merge window — untouched.

## Facts the next session must not relitigate
- **Whole-head dead-zone (PDR-0071):** at |H|=1 the floor output is a constant → zero gradient to all
  op logits; |H|=1 in 67-80% of HOLDING decisions; SET_ALPHA 0.55 = cap `1−(n−1)f`, not learned. The
  head learns only in the ~20-33% |H|≥2 minority. Policy-WIDE (all 8 heads, differentiable PPO path).
- **Floor fix SCOREABLE but NOT SHIPPABLE alone (PDR-0071):** `hindsight_credit` inert → unfreezing
  the gradient → LOO-greedy commit collapse. `hindsight_credit`/transfer is a **CO-REQUISITE shipping
  gate** (separate engineering, joint ship). The dead-zone may be an accidental safety property.
- **Experiment = ONE primitive + λ-sweep {0,0.3,0.6}** (`q=(1−λ)softmax+λ·uniform`; λ=0 = no-floor
  smoke), NOT straight-through, NOT two builds. Pre-register: λ=0 likely collapses to SET_ALPHA.
- **Gate caveat:** −0.7→+8.2 is IMMEDIATE reward; `V(s′)` is NOT op-independent — log per-decision
  advantage. **Scope:** the floor does NOT erase the prior HRA fit-failure or coverage-bound Shapley null.
- Doc: `docs/analysis/2026-07-13-decision-point-diagnosis.md` (through round-7); memory `floor-gradient-dead-zone`.

## Open questions / blocked-on-owner  (Step-2 escalations)
- **APPROVE building the one λ-sweep floor primitive?** Core-RL-primitive change (action dist / PPO
  ratio) — needs drl-expert review before landing. (proposed)
- **AUTHORIZE the λ-sweep GPU arm** (λ=0 necessity smoke + λ>0 fix, n=5 for any directional claim)?
  GPU + owner acceptance. Note it is SCOREABLE but the fix does not SHIP until `hindsight_credit`
  carries signal.
- **Sequencing:** pause cf-earns-keep/SNR (PDR-0067/0068) in favour of the floor + hindsight co-req?
- North-star target/date, rent ceiling, host-accuracy floor: owner-unset.
- Standing: no push/tag/release/branch-deletion/telemetry-deletion/remote action.

## Last did
- Round-7 review (claude-prime + gpt-prime) → ran a zero-GPU |H| read; reshaped the dead-zone to
  whole-head; corrected the sequencing (co-requisite hindsight) + experiment (one primitive) + gate
  caveat + scope; PDR-0071 (supersedes PDR-0070).

## Next session, start here
Two remaining **zero-GPU** audits before implementation: **entropy-gradient audit** (does the
entropy-floor loss move floor-bound logits indirectly, and does it carry outcome info — it doesn't)
and **advantage-standardisation audit** (do zero-gradient floor-bound samples distort the global
scale). Then land the harness enablers (esper-lite-7fe21bd091, now incl. per-head KL). THEN bring the
owner the λ-sweep primitive go/no-go + GPU authorization — framed as scoreable-not-shippable-alone.
Do NOT start a reward/critic arm; the primitive is the lever.
