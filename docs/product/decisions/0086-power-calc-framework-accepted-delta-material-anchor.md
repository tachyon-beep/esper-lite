# PDR-0086 — Paired-run power-calc framework ACCEPTED at review; Δ_material circularity fix adopted (new r9 q_decision-IQR read queued); max-n = 10 proposed

Date: 2026-07-14   Status: accepted (draft acceptance + read-queuing within standing authority; every §8 ratification — ΔP_req, power split, UCB posture, max-n stretch, IQR-read legitimacy — is OWNER-PENDING at freeze)
Follows PDR-0081 (B1) / PDR-0082 (§2). Artifact: `docs/analysis/2026-07-14-paired-run-power-calc-DRAFT.md` (stays DRAFT until owner ratifies; exact noncentral-t, self-checked vs Monte-Carlo to ≤0.001). Tracker: esper-lite-2852cf851c closed; NEW read task esper-lite-e12e2d1543.

## What does this buy?  (REQUIRED — PDR-0068)
Freeze blocker #2 moves from "no power number exists" to "a reviewed framework + a frozen-form reassessment rule exist, pending owner ratification." It also caught its own would-be failure mode before freeze: a blinded reassessment comparing a relative MDE to a relative target is vacuous — without the fix, the load-bearing certification step would have certified nothing.

## Decisions
1. **Framework ACCEPTED:** one paired one-sample t on per-run differences (df = n−1, one-sided α=0.05, = the E4 LCB convention) parameterizes P2 + GATE-DOM + GATE-INFL. Scale-free core: **n=10 is ≥80%-powered iff σ_d ≤ 1.17·Δ** (≥90% iff ≤0.995·Δ); n=5 reaches only σ_d ≤ 0.74·Δ.
2. **Expected-null discipline:** Exp-1 arm-B P2 is a BOUNDING read, never detection — a null bounds the effect below 0.953·σ_d (n=5) / 0.580·σ_d (n=10). The gates, not P2, carry the Exp-1 safety verdict; gates get the stricter 90% target (proposed). Tercile estimators inflate σ_d ~1.7–2.5× → **the gates set required n** (~3–6× runs if estimation-error-dominated).
3. **Circularity fix ADOPTED: Δ_material freezes in ABSOLUTE slope units** = ΔP_req / IQR_q with IQR_q from a NEW r9-only, no-peek-safe calibration read (q_decision interquartile range over eligible-HOLDING decisions) — task esper-lite-e12e2d1543 queued pre-freeze. ΔP_req = 0.10 proposed, owner-ratified. Known limitation stated: r9 scale is a K=1/obs-v3 analogy; σ_d itself is measured in-regime at the screen.
4. **Blinded reassessment rule (frozen form):** unsigned within-pair spreads only (σ̃_d² = Σδ_s²/(n−1), conservatively mean-inflated); prohibited: any signed difference, contrast, per-arm mean, or screen-tier point estimate/LCB. Certification: n=10 iff σ̃_d ≤ Δ_material/d_z*(π). Variance-of-variance posture: point-estimate PRIMARY, 80% UCB as sensitivity (a 95% UCB at df=4 inflates n 5.6× and double-counts conservatism).
5. **Frozen max-n = 10 PROPOSED** (= seed ceiling 41–50), interim blinded reassessment after the n=5 screen gates the second seed-wave; explicit escalate branch (stretch amendment / descriptive-tier with stated live-fatal-case / σ_d-reduction redesign — no silent truncation). Budget: n=10 claim = 120 GPU-h (Exp-1) / 180 GPU-h (Exp-2); optional owner-pre-authorized stretch n=12/16 laid out for the tercile-gate risk.
6. **Multiplicity stance recorded:** no α-correction on the safety gates (correction would blunt harm sensitivity; false alarms cost a redesign, not a bad ship) — stated in the frozen doc rather than silently imported.

## Reversal trigger
- If the r9 q_decision-IQR read shows a degenerate/unusable spread → the Δ_material anchor fails; materiality must be re-derived in-regime at screen time and the reassessment rule re-opens (do NOT fall back to a relative target).
- If the owner sets ΔP_req ≠ 0.10 or a different power split → recompute the ceilings; the framework itself stands.
- If the blinded σ̃_d makes the gates infeasible even at the stretch ceiling → §5.5 escalate branch; the claim tier does not proceed silently.
