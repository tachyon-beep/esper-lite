# Reward-credit term (Committed-Shapley top-up) — reward-function-reviewer verdict

**Date:** 2026-07-02 · **Reviewer:** yzmir-deep-rl reward-function-reviewer (SME protocol) · **Scope:** "safe to
BUILD default-OFF and present for owner sign-off" — enablement criteria are a separate later gate.
**Reviewed at:** design `docs/plans/concepts/2026-07-01-reward-credit-shapley-synergy-design.md` + GATE 2
pre-registration/result (`2026-07-02-gate2-learnability-{probe-preregistration,result}.md`) + n=5 J-read result +
live reward code, at commit `cee960ed`.

## VERDICT: APPROVE_WITH_CHANGES

Safe to build default-OFF: at `scale=0` the term is byte-identical to status quo (conditional on F5 below), the
evidence gates (GATE 1 proxy-vs-estimand; GATE 2 learnability with retro-write mandated; n=5 (b) direction) have
passed, the double-pay diagnosis premise is empirically intact, and units are consistent (accuracy is percentage
points end-to-end; `vectorized_trainer.py:1466`).

## Changes that MUST be in the build

1. **F5 — true no-op at `scale=0`:** gate the ENTIRE terminal 2^k alpha-mask apparatus behind `scale>0`; if a
   telemetry-only path runs at 0, prove it RNG- and state-neutral. The byte-identical claim and the causal
   harness's RNG-split determinism depend on this.
2. **F1/F8 — double-pay reconciliation with the two LIVE sibling channels:** per-step `synergy_bonus`
   (`contribution.py:741`) and retroactive `compute_scaffold_hindsight_credit` (`fossilize.py:215` →
   `action_execution.py:1084`) target the same concept and are near-dormant today (62/359,987 steps; ~12 events
   in n=5) — but the new term's incentive could WAKE them into feedback double-pay. Net them into `c_paid`,
   disable them when `scale>0`, or assert-dormant + monitor; document the choice. Also resolve the "synergy"
   naming collision (No-Legacy policy).
3. **F2 — scale-space safety:** `divide_by_std` is the ONLY unclipped reward channel (no clip, 1e-8 floor;
   `normalization.py:241-257`); at small early-run std (~0.3-0.4) the credit is amplified 2.5-5x and can exceed
   the ±10 that bounds every other reward. Build must expose the per-seed cap AND a normalized-space clamp or
   std-floor for the credit path. (GATE 2's "no clip" mandate was about the normalizer's clip for GAE linearity;
   a separate shipped-credit bound is fully compatible.)
4. **F4/F6 — kernel verification broadened + terminal same-time baseline:** assert alpha=0-override zeroes a
   slot's contribution for ARBITRARY coalitions (not just G's endpoints) and that the fused-val `v()` returns
   percentage points (a fraction-returning kernel silently corrupts `gap` by 100x). Recompute `c_paid`'s
   standalone baseline from the terminal 2^k family (`v({s})−v(∅)`, free in the factorial) instead of the
   fossilize-time snapshot — otherwise post-fossilize standalone growth is mis-credited as "synergy".
5. **GAE retro-write unit test** vs hand-computed advantages (already banked in the GATE 2 result).

## Banked as ENABLEMENT criteria (not build blockers)

- F2 calibration of scale/cap against the minimum running std the term will meet.
- F3 over-fossilization economics: rent (~0.3 reward over 150 steps) is NOT a real counterweight to a terminal
  credit lump of order 1-10; elevate "monitor fossilize/ep (0.207±0.006)" to a HARD enablement gate on
  off-switch-J efficiency + a per-episode fossilize-count guard.
- F7 entrenchment/opportunity-cost stays a live monitored gate (the proxy is validated for r0c0, untested for
  the entrenched-worthless failure seed).
- tau from the GATE-0 PIN-E placebo (still pending; design already blocks enablement on it).
- Re-run the F1 dormancy check on an ON run (dormancy is regime-dependent).

## Sign-off state after this review

GATE 1 ✅ · GATE 2 ✅ (retro-write mandated) · signal verification ✅ · reward-function-reviewer ✅
(APPROVE_WITH_CHANGES) · **owner sign-off: PENDING** — the flag stays `shapley_synergy_scale=0.0`; nothing is
enabled autonomously.
