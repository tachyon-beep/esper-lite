# PDR-0093 — Observability proof BANKED: committed-HOLDING is genuinely aliased (masks logit-only; no obs delta); countdown already derivable → fork F2 narrows to ONE per-slot bit vs instrument-only; recommendation flips to the bit

Date: 2026-07-14   Status: accepted (reading banked; the F2 ruling itself remains OWNER-OPEN — primes still diverge)
Follows PDR-0091 (#5). Artifact: `docs/analysis/2026-07-14-observability-source-proof.md`. Tracker esper-lite-8ca95a0ecf closed.

## What does this buy?  (REQUIRED — PDR-0068)
It converts the arc's last prime divergence from a clash of priors into a read fact plus a one-bit decision. gpt-prime's state-aliasing claim is structurally CONFIRMED from source (not assumed); claude-prime's "neither of us has read the number" objection is half-discharged — the STRUCTURE is now read (aliased), only the COST remains unread. And the remedy shrank: not two fields, one bit.

## Decisions
1. **Proof BANKED** (four questions, all file:line-cited): Q1 committed-state in obs = ABSENT (aliased —
   including the obs-v4 `cf_frozen` flag, which fires only at actual fossilization); Q2 masks as network input =
   ABSENT (logit-only via `masked_fill` post-forward); Q3 epoch/countdown = ESTABLISHED (base dim 0; fixed
   cadence ⇒ time-to-boundary is a deterministic function of an observed feature); Q4 previous action =
   partial (op-only, global, one-step, no reward input). Net: two identical-telemetry HOLDING seeds, one
   pending, produce byte-identical observations except a one-step op echo.
2. **The PDR-0091 reversal branch did NOT fire** ("observability established → F2 collapses to the
   interpretation-note resolution") — the proof went the other way. F2 stays owner-open.
3. **F2's options, as narrowed by the read:** (a) **one per-slot `pending_settlement` bit** (obs-v4 +35; arm A
   emits constant 0; schema identical across arms; explicit obs-schema version; re-warm/retrain accepted; the
   explicit countdown dim is skipped as information-redundant per gpt-prime's own rule) + the aliasing telemetry
   retained as verification; or (b) **no obs change** + pre-registered interpretation note + the aliasing
   telemetry & symmetry check as the instrument, with the Exp-2 decision made from the measured cost.
4. **Session recommendation UPDATED (flip): option (a), the bit.** Rationale: the structure is now read —
   aliased state during windows contaminates V(s) and GAE in arm B ONLY (an arm-asymmetric noise source that
   could masquerade as "the settlement hurt learning" — a B−A validity concern, not just a critic nicety);
   the remedy is one bit with a constant-zero control arm, which answers the minimal-delta objection; and the
   latent accidental channel (cf-staleness dims becoming an implicit discriminator if measurement ever stopped)
   argues for a DESIGNED signal over an accidental one. claude-prime's instrument (telemetry + symmetry check)
   is kept either way — under (a) it verifies the bit closed the gap.
5. **Cost stated, not hidden:** the bit is an obs-schema change → re-warm/retrain of the obs-v4 path and a
   pre-reg §2.5.8 amendment (the "obs dim 31 OFF / no new obs dim" pin) — which is exactly why this stays an
   OWNER ruling, not a convergence item.

## Reversal trigger
- If the owner rules (b) and the measured aliasing cost (value-error concentrated in open-window epochs, arm B
  only per the symmetry check) exceeds a pre-registered materiality at the screen → the bit enters at Exp-2 as
  the pre-planned response, not a redesign.
- If the owner rules (a) and the re-warm materially shifts baseline behavior (screen A-arm vs prior A-arm
  telemetry) → the schema change itself becomes a confound suspect; the A-arm comparison read is the check.
- If the boundary schedule ever becomes commit-relative rather than fixed-cadence → Q3's countdown derivation
  fails and the explicit countdown dim re-enters (PDR-0093 #3's redundancy argument is cadence-dependent).
