# PDR-0085 — G-LEDGER replay Phase 1 executed: PBRS live-semantics finding (F5 refuted, fossils re-accrue, park residual reversed); six pre-reg amendments proposed for round 21

Date: 2026-07-14   Status: accepted (execution + finding within standing authority; the round-21 fold-in content, the Φ(PENDING) re-adjudication, and the price-vs-fix question are OWNER-PENDING — flagged, not enacted)
Follows PDR-0084. Artifacts: `tests/simic/rewards/replay/` (84 tests, landed `b7b7c6d5`), `docs/analysis/2026-07-14-gledger-replay-phase1-findings.md` (R1–R5), drl-expert review APPROVE-WITH-NITS (all five nits closed same day, incl. proxy-path coverage + production-derived PRUNE cull). Preceded by an owner-requested first-principles review of checkpoints #55–#60 (all checkable claims verified against source; conclusion upheld) before clearance to proceed.

## What does this buy?  (REQUIRED — PDR-0068)
It converts freeze blocker #3's Phase-1 scope from "queued" to "executed" and does exactly what the blocker existed to do: it caught the pricing layer being wrong before anything froze. Twenty rounds of review, an executed code audit, and this session's own first-principles pass all priced a PBRS forfeiture that a trainer-ordering fact makes unreachable. Measured: the harness is landed and reviewed; the finding is confirmed at population scale on sealed r9 (both seeds); six specific pre-reg amendments are drafted; and an owner-pending decision (Φ(PENDING)) gained material new information before signature rather than after.

## The finding (R1–R4, code-verified + r9-confirmed + review-re-verified)
1. **The PBRS `eis==0` branch is unreachable live.** `record_accuracy` ticks `epochs_in_current_stage` in the metrics phase BEFORE decisions (`vectorized_trainer.py:1870-1875`); the reward is computed on the pre-action state (`action_execution.py:226-230/:973-977/:1089`); mutations + `step_epoch` run after (`:1304/:1643`); `transition()` zeroes the clock (`slot.py:520-531`). First post-transition reward row sees `eis==1` → **every cross-stage PBRS transition delta is skipped** — the branch's own warning message shows the design expected otherwise. r9: **0/4,151 commits** show the F5 forfeiture band; fossil rows show the exact skip signature (−0.012 cap-drag ×92.8k/96.6k rows; +0.078–0.081 climbs).
2. **Audit F5 corrected:** the dwell-graded commit-time forfeiture (−0.31…−0.46) never fires. A2's economic classification is settled: skipped, not telescoped, not refunded.
3. **Fossils re-accrue potential 6.0→8.0** (missed by the audit entirely); a settled fossil out-potentials a parked seed (7.5).
4. **PDR-0084 #4 reversed:** the shaping residual is PRO-fossil (+0.02…+0.57 A−S from commit, both conventions), not +0.31 pro-park. The GATE-DOM interpretation suspect flips direction (commit-side subsidy).
5. **Ledger precision:** forfeited attribution −274.7 PV still dominates (epic premise untouched); holding_warning relief +27.7 is the second-largest pro-commit term (16× the flat prior — G-MATCHED-CONTROL's warning-cessation parity leg is load-bearing); F8 measured at **+45.8 PV** (first-order, not a corner case). G-THRESHOLD-VARIANCE machinery delivered (windfall real and monotone below the cliff; LCB strictly reduces it; materiality bound = freeze-time owner decision).

## Decisions
1. **Finding BANKED** (executed read, both-seeds confirmation, independent review re-verification of the ordering chain and FOSSILIZE synchronicity). Severity-marker discipline satisfied.
2. **Six pre-reg amendments PROPOSED for a round-21 fold-in — body deliberately NOT edited** (one amendment touches the owner-pending Φ(PENDING) call): Φ(PENDING) premise re-adjudication; Δ_designed PBRS terms replaced with skip-semantics; G-PBRS gate rewritten to within-stage-identity + priced skips; G-MATCHED-CONTROL PBRS leg against real semantics; audit F5/A2/A7 corrected inline; the production-fix question separated from the experiment.
3. **RECOMMENDED (owner-gated): PRICE the skip, do not FIX it mid-arc** — fixing changes arm A's shipping semantics, the exact mistake the matched-control frame exists to prevent. The fix decision belongs after the experiment.
4. **Phase 2 (B path, seven gates) remains owner-gated** — unchanged.

## Reversal trigger
- If a real-trainer integration probe or the Phase-2 replay ever observes an `eis==0` reward row for a policy-initiated transition → the ordering finding re-opens; roll back the round-21 amendments and re-derive.
- If the r9 scan is shown to have mis-read the event schema (components nesting) → re-run the population confirmation before relying on it.
- If the owner rules FIX-not-PRICE → arm A's semantics change; the matched-control baseline, Δ_designed, and the R3/R4 tables must be re-derived on the fixed code before freeze.
