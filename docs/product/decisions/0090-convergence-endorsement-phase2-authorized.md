# PDR-0090 — Owner ruling: "anything the two primes converged on, I'm endorsing" — convergence audit executed; Phase-2 B-path AUTHORIZED; dissolution/bounds/price-don't-fix/Option-A/max-n RATIFIED; pending-visibility remains open (genuine divergence)

Date: 2026-07-14   Status: accepted (owner ruling quoted; the item-by-item convergence audit below is this session's interpretation of it — auditable, vetoable line by line)
Follows PDR-0089. The ruling (verbatim): "you can assume that anything the two primes converged on, I'm endorsing".

## What does this buy?  (REQUIRED — PDR-0068)
It converts a ten-item ratification queue into a two-item one without the owner rubber-stamping each line — while keeping the authority boundary honest: the delegation covers CONVERGENCE only, so the audit below must show its work, and the one genuinely divergent item stays escalated. Prime consensus ≠ owner ruling remains the standing rule; this PDR records the owner making convergence-endorsement an explicit, scoped exception.

## Convergence audit (the interpretation record)

| # | Item | claude-prime | gpt-prime | Verdict |
|---|---|---|---|---|
| 1 | Φ(PENDING) dissolution (flag-on-HOLDING) | proposed it (r21); "evidence-forced" (r22) | ratified as item 1 + Phase-2 invariants list | **CONVERGED → RATIFIED** |
| 2 | Gates-as-BOUNDS (95% UCB vs Δ_material; null ≠ pass; wide = INCONCLUSIVE-ON-GATES) | forced by arithmetic (r21 §3); evidence-forced (r22) | ratified as item 2 with explicit bound forms | **CONVERGED → RATIFIED** (exact frozen formulas land in the fold-in; adversarial pass checks them) |
| 3 | Price-don't-fix the eis==0 skip | evidence-forced ("the matched-control ruling made twice") | ratified as item 3 | **CONVERGED → RATIFIED** |
| 4 | **Phase-2 B-path authorization** | "authorizing is right precisely because the replay contract was built to certify exactly this change" (r22) | authorized as item 10: default-off impl + tests + executable replay + DRL review + docs; NOT training activation, NOT GPU | **CONVERGED → AUTHORIZED** (gpt's scope; fail-closed flag discipline per the escrow precedent) |
| 5 | Option-A cross-slot global audit lock | recommended Option A since r20 (PDR-0084 #6); r22 frames the estimand honesty as the owner's tradeoff | ratified as item 8 + REQUIRED burden quantification before freeze | **CONVERGED → RATIFIED CONDITIONALLY** — Option A for Exp-1; lock-burden quantification (locked-epoch %, suppressed ops by type, high-quote overlap, episode distribution) required before freeze; B−A labeled "complete LOCKED settlement protocol"; PDR-0084's escalate-to-Option-B trigger stands |
| 6 | **Pending-visibility (obs)** | invisibility + interpretation note (r22 pin) | MUST be visible to the value function (state-aliasing defect) | **DIVERGED → NOT COVERED — stays on the owner's desk.** Session recommendation unchanged: minimal-delta for Exp-1 + pre-registered aliasing note + INSTRUMENT the cost (value-error during windows as telemetry); obs amendment becomes an Exp-2 decision made from data |
| 7 | UCB posture for the blinded reassessment (point-estimate primary) | not addressed | not addressed (item 2 concerns the gate bounds, a different UCB) | **NOT ADJUDICATED by primes → recommended default carries** (point-estimate primary, 80% UCB sensitivity), owner may override until freeze |
| 8 | Max-n = 10, no automatic stretch | r21 arithmetic makes a detection-stretch pointless; no contest in r22 | ratified as item 6 (no automatic expansion; redesign-or-return on infeasibility) | **CONVERGED → RATIFIED** |
| 9 | Threshold-variance mechanism (ΔP<0.10 bound; gate form from the near-cliff σ data; LCB fallback, never a q_decision gate) | "cite the phase-1 number / defer to data" (r22) | ratified as item 9 with the same fallback | **CONVERGED → RATIFIED as mechanism**; the gate FORM resolves from the in-flight noise read (only returns to the owner if σ lands ambiguous, ~0.1–0.15) |
| 10 | Premium figure (`0.5+3·tanh(1/3)` × legitimacy, boundary-paid, PV-neutral, q_settle-qualified, never retuned) | endorsed since r16–17 | ratified as item 7 | **CONVERGED → figure ENDORSED**; the formal signature still lands with the no-peek freeze (by design — the freeze signature stays the experiment-authorization gate, not a decision record) |
| — | Execution order (IQR read → Phase-2 replay → lock burden → power/UCB finalization → adversarial pass → freeze → GPU) | IQR before adversarial pass (r22 pin d) | full order as above | **CONVERGED → ADOPTED** (already PDR-0089 #6) |

## Consequences enacted this session
1. **Phase-2 implementation begins** under the authorized scope: implementation plan drafted to `docs/plans/ready/`, specialist review dispatched (CLAUDE.md plan-review requirement), build behind the default-off flag after review. Acceptance contract = gpt's ten Phase-2 invariants + round-22 pins (exactly-one-STAGE-ENTRY-MINT wording; PBRS-only layer fence) + the no-unenumerated-`committed=True`-effects sweep.
2. Remaining owner items: **pending-visibility fork** (the one divergence) + UCB-posture default confirmation + the freeze-time signature package. Everything else is ratified or self-resolving.

## Reversal trigger
- If the owner vetoes any line of the audit above → that line reverts to PROPOSED and its dependent work pauses (Phase-2 work is severable by construction: flag off, no training activation).
- If either prime RESCINDS a converged position before freeze → convergence for that item is void; it returns to the owner queue (the delegation covers standing convergence, not historical snapshots).
- PDR-0084/0088/0089 triggers all remain live and unmodified by this ruling.
