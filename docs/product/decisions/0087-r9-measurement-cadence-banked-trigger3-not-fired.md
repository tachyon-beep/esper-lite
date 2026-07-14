# PDR-0087 — r9 HOLDING measurement-cadence read BANKED: 100% per-epoch, zero gaps; PDR-0084 reversal trigger #3 does NOT fire; obs-v3 indirect verification ratified

Date: 2026-07-14   Status: accepted
Follows PDR-0084 (#5, queued read). Artifacts: `docs/analysis/2026-07-14-r9-measurement-rate-read.md` + preserved scripts/outputs under `docs/analysis/scripts/2026-07-14-r9-measurement-rate/` (script SHA-256 `8f14063a…`). Tracker: esper-lite-5e9affe207 closed. Read executed by a dispatched agent; main-session review verified the mechanical grounding against independently-read trainer code.

## What does this buy?  (REQUIRED — PDR-0068)
It closes the last feasibility hole in the settlement's confirmation-window arithmetic with a census, not an assumption: the zero-slack `min_window=5 / ≥5-valid` rule was designed against a hypothesized missed-measurement path that turns out not to exist in the evidence base. The extension clause is the rare/never path; the annuity-horizon assumptions stand.

## Decisions
1. **Reading BANKED:** 100.000% per-epoch HOLDING measurement, ZERO gaps, both seeds (26,703 HOLDING seed-epochs; rule-of-three 95% miss bound ≤ ~0.011%/epoch, ≤ ~0.056%/5-window). Mechanically forced: a solo ablation config is built for EVERY active slot at EVERY validation pass — no rotation, no interval (`vectorized_trainer.py:984-992`); HOLDING ⇒ α=1 ⇒ always measured. Distinct from (and consistently one point above) the banked 98–99% decisions-acting-on-fresh figure — different quantity, direction predicted by the mechanism.
2. **PDR-0084 reversal trigger #3 ruled NOT FIRED.** Conditioned on the NAMED bridge assumption (recorded, not assumed silently): a committed/PENDING seed keeps landing in `active_slot_list` (α frozen > 0, ablated to the boundary per pre-reg §2.0/§2.1). The assumption survives the Option-A audit lock (it suspends morphology, not validation). The one implementation that breaks it — excluding pending seeds from ablation before the boundary — is already forbidden by §2.0.
3. **obs-v3 indirect verification RATIFIED** (main-session call): no positive obs-version stamp exists in this telemetry generation; verified via key-absence + code default + the "Obs V3" reset-site comment. Immaterial to the measured quantity (obs version changes encoding, never solo-eval timing); every other regime stamp verified positively. Gap filed as a TIP observation (esper-lite-obs-77569bfa0a: emit a positive obs-schema version in TRAINING_STARTED).
4. **Caveat carried, not buried:** r9 has no frozen-α window (spans P50=1 epoch; the 346/346 ≥5-in-5 figure is a sparse self-selected subset). The transferable quantity is the per-epoch 100% cadence; the settlement bridge is a mechanically-grounded proxy, not a direct observation.

## Reversal trigger
- If the Phase-2 settlement implementation excludes pending seeds from the ablation before the boundary (violating §2.0) → the bridge assumption breaks; the window arithmetic and this ruling re-open.
- If a K=4/obs-v4 run changes the eval cadence (rotation/interval introduced, or fused-val restructured) → this read is regime-bound; re-run in-regime before relying on it for the frozen doc.
