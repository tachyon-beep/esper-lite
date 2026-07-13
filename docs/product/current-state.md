# Current State — Esper        Checkpoint: 2026-07-13 · reward/commitment diagnostic → ROOT CAUSE found (PDR-0069/0070; on `feat/ev-stab-stage2-hra`)

## The bet right now
**Commitment root cause FOUND → exploration-primitive fix (PDR-0069/0070).** The commitment
defect the EV epic (esper-lite-f25b71c165) exists to fix is root-caused to the **anti-WAIT floor
gradient dead-zone**: `_apply_floor_to_logits` sits in the differentiable PPO path across all 8
heads, and a floor-bound action gets ZERO gradient to its logit → ~94% of FOSSILIZE / 93-95% of
PRUNE samples can't learn (both seeds), despite a strong LOO-graded commit reward (−0.7→+8.2; the
gate PASSES). The value-target-variance premise is SUPERSEDED; the critic (PDR-0066) and
cf-earns-keep/SNR (PDR-0067/0068) lines are demoted to secondary. New lever = a **differentiable
mixture floor**, staged — but the fix + experiment are **owner-gated**.

## In flight
- **Harness enablers** (esper-lite-7fe21bd091, task under the epic; ACCEPTED, no GPU): default-on
  checkpointing + per-decision advantage/pre-floor logging. Land FIRST — the July diagnostic
  captured no weights, no advantage, no pre-floor logits (PDR-0026 recurrence), so the true
  advantage-vs-LOO gate was unreadable.
- **EV-stabilization epic** (esper-lite-f25b71c165): commitment-premise reframed (comment #178);
  the ON-leg A′/B DECIDE (PDR-0064/0065) is now moot/deprioritized — the critic is downstream of
  the floor bug.
- **Branch-survivor consolidation** (esper-lite-1f1e55f58f): main-as-trunk decided (PDR-0062), all
  sign-offs in; execution still gated on the merge window — untouched this session.

## Facts the next session must not relitigate
- **The dead-zone (PDR-0069, both seeds, code-confirmed, zero-GPU):** floor is floor-PRESERVING
  renormalization in the differentiable update path (all 8 heads); floor-bound action → zero
  gradient (autograd-verified; partition cancels; PPO ratio=1.0). Blast radius FOSSILIZE 93-94% /
  PRUNE 93-95% floor-bound; SET_ALPHA 0% (median 0.55, LEARNED). Gate PASSES (commit reward
  −0.7→+8.2). Doc: `docs/analysis/2026-07-13-decision-point-diagnosis.md` (+ `...-reviewer-catchup-brief.md`).
- **Straight-through is the WRONG fix** (biased PPO ratio) — use a differentiable mixture floor /
  smooth `P(non-WAIT)≥ε` (PDR-0070).
- **Two SEPARABLE problems:** (A) commit-learning = dead-zone + live reward → fixable; (B) transfer/
  developmental value = `hindsight_credit` inert (0.005%), no causal net-ensemble-value field →
  separate track, unaffected by the floor fix.
- ~11 over-reads this session, all caught (memory `instrument-validity-before-interpretation`,
  `floor-gradient-dead-zone`); "read the variable / print the number" discipline held.

## Open questions / blocked-on-owner  (Step-2 escalations)
- **APPROVE the mixture-floor fix?** A change to a core RL primitive (the action distribution / PPO
  ratio) — the exact class that caused this arc; needs drl-expert review before landing. (proposed)
- **AUTHORIZE the staged GPU experiment?** Stage-1 no-floor NECESSITY smoke (does WAIT-collapse
  still recur under the twice-changed reward?), then Stage-2 hard-vs-mixture A/B (n=5). GPU + owner
  acceptance gate.
- **Sequencing:** pause the cf-earns-keep/SNR line (PDR-0067/0068) in favour of the floor fix?
- North-star target/date, rent ceiling, host-accuracy floor: still owner-unset.
- Standing: no push/tag/release/branch-deletion/telemetry-deletion/remote action.

## Last did
- Ran a zero-GPU reward/commitment diagnostic (advisor + claude-prime + gpt-prime); traced
  "why Tamiyo rarely fossilises" to the floor gradient dead-zone; confirmed via autograd + both
  seeds; wrote PDR-0069/0070; reference doc `docs/reference/tamiyo-lifecycle-and-action-masking.md`.

## Next session, start here
Two remaining zero-GPU reads before any GPU: **entropy-coefficient audit** (rule out uniform-forcing
as a competing cause of the near-uniform live region) and **whether floor-bound zero-gradient samples
still enter global advantage standardisation**. Then land the harness enablers (esper-lite-7fe21bd091).
THEN bring the owner the mixture-floor go/no-go + staged-experiment authorization. Do NOT start a
reward/critic arm — the reward signal is fine; the primitive is the lever.
