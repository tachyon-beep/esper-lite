# PDR-0099 — Obs audit BANKED (both specialists + main-session verification): heads are slot-blind (corrected: degraded, not impossible); obs dim +13 is a LIVE DEFECT (dead since inception, both versions); six top signals feed reward/heuristic but not the obs; prime brief prepared

Date: 2026-07-14   Status: accepted (audit + verification + brief within standing authority; every design consequence routes through the prime round + owner — nothing enacted on the frozen track)
Follows PDR-0098 (dispatch) / PDR-0093 (the F2 finding that seeded the owner's question). Artifacts: `docs/analysis/2026-07-14-obs-audit-{drl,pytorch}.md` (persisted from scratchpad), `docs/analysis/2026-07-14-obs-audit-prime-brief.md` (the paste-ready round-24 package), bug esper-lite-f0a82adccb, task esper-lite-f7b46a5f52.

## What does this buy?  (REQUIRED — PDR-0068)
The owner's question ("does Tamiyo see enough about the hosts?") now has a verified, three-layer answer —
information (six top signals unplumbed), architecture (heads cannot condition on the sampled slot), and
pipeline (a dim that has been silently dead in every run ever) — with the class-level lesson that F2 was
indeed an instance of a class. The main-session verification pass earned its keep twice: it corrected the drl
audit's severity overstatement AND caught a cross-report interaction neither specialist saw (the dead velocity
dim invalidates part of the drl audit's PRUNE grade).

## Decisions
1. **Audit findings BANKED at stated verification tiers** (the brief's [VERIFIED]/[AGENT]/[CORRECTED] tags are
   the record): head-independence factorization (factored_lstm.py:838-845, incl. a negative check for
   autoregressive sampling); the dead velocity dim four-link chain; committed_val_acc + plateau_epochs
   unplumbed; the V3 removal of the scaffolding trio; num_holding conflating HOLDING+FOSSILIZED; the F2
   layout-collision landmine (lockstep slot_feature_size bump).
2. **Main-session corrections applied before banking:** (a) "slot-averaged / regardless of obs" →
   "strongly degraded, not impossible" (trunk sees per-slot blocks position-wise; 0.68–0.75 slot confidence
   partially mitigates); (b) PRUNE's information grade downgraded (its velocity leg is the dead dim).
3. **Bug filed, fix DEFERRED as a design question (brief Q3):** the dim is constant in all historical data, so
   not-fixing is the maximally matched-control choice for the permanence experiment; fix-with-F2 vs
   post-experiment-Obs-V5 goes to the primes. Do NOT fix unilaterally mid-arc.
4. **F2 implementation landmines handed to the Phase-2 build** (plan-relevant now): lockstep size-constant
   bump, schema-version-forced re-warm, same-schema-both-arms, and the to_leyline copy-line half-tier — the
   omission class that caused the defect.
5. **Prime brief prepared** (the owner pastes it); consolidation into a concepts/ doc + a candidate roadmap
   bet happens AFTER the prime round lands — their round hits the brief, not a fait accompli. Sequencing
   stance carried in the brief: post-permanence-experiment bet, offline read ladder only before then.

## Reversal trigger
- If the `obs_normalizer.var` runtime dump contradicts the dead-dim chain (variance ≫ 0 on any +13 dim) → the
  four-link reading missed a write path; re-verify before the bug proceeds.
- If the primes overturn the sequencing stance (Q1) → the offline read ladder re-scopes accordingly.
- If the D2 ladder (when run) shows candidate site-features add NO predictive power → gap #1's obs half is
  refuted and the future bet re-centers on architecture/credit (the rival ordering flips).
