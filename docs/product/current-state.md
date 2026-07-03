# Current State — Esper        Checkpoint: 2026-07-03 (checkpoint #8 — tau DELIVERED, plateau trigger FIRED; SHAPED audit banked; prioritization review is next)

## The bet right now
**Reward credit-assignment redesign.** Term BUILT default-OFF (PDR-0013); enablement gate
OPEN (PDR-0014) with its **first leg DELIVERED**: tau = +0.28 pp CONSERVATIVE-PROVISIONAL
LOWER BOUND (PDR-0017; memo = esper-lite-f22a1d48a7 comment #98). `shapley_synergy_scale=0.0`
everywhere. Metric: committed-J / corr(reward,J) in the enablement A/B.

## In flight
- Nothing on GPU. Both subagents idle. esper-lite-94869250f1 CLOSED (@ 05569309);
  the gate esper-lite-f22a1d48a7 is now UNBLOCKED and waiting on the owner review below.
- **Telemetry hygiene** (esper-lite-425dcc4ca2): unchanged.

## Open questions / blocked-on-owner — THE PRIORITIZATION REVIEW (owner, with full deck)
The owner deferred all prioritization until this data landed; it has. The deck:
1. **tau disposition (PDR-0017, trigger FIRED):** epsilon plateau FAILED — the noise
   floor is magnitude-dependent. Accept +0.28 as lower bound and proceed to F2
   (recommended); supplement with methodology option (a); or re-scope. Owner's call.
2. **SHAPED audit findings (PDR-0016, esper-lite-a7ef375203, 9 findings PARKED):**
   F1 (prune pays unbounded positive, clip-exempt) + F5 (fossilize spuriously suppressed
   on coalitions) + F3 (PBRS not invariant) + F2 (dense credit keys on r0c0 via WAIT
   canonicalization). Owner-endorsed synthesis: these jointly rationalize the
   never-fossilize pathology (fossilize 0.207/ep vs germinate 12.4). Zero-GPU occurrence
   probes are defined and ready; audit note: run the F1 probe BEFORE reading any Phase −1
   clip-arm verdict.
3. **A/B false-negative risk (recorded on the gate):** the top-up pays only at FOSSILIZE;
   if F1/F5 keep OFF-arm fossilization rare, the A/B can't show an effect. Decide whether
   F1/F5 fixes (or probes) precede the A/B leg.
4. Standing: n=10 vs bank-at-n=5 (PDR-0009 — now carries the F2-audit interpretation
   caveat); metrics.md TARGET placeholders; owner's uncommitted working-tree files
   (seed_residency gap esper-lite-obs-7240279be3). NUM_BLUEPRINTS 13→14: paired A/Bs
   must be same-commit (PDR-0015).

## Last checkpoint did (checkpoint #8)
- **PDR-0016:** SHAPED adversarial audit accepted (5/5 load-bearing claims re-verified in
  code); 9 findings PARKED per owner directive; committed f3d8db95.
- **PDR-0017:** full-run tau delivered; PDR-0015 plateau trigger honestly FIRED and
  flagged (⚠ in metrics.md); disposition proposed, owner decides; esper-lite-94869250f1
  closed; plan → completed/ (d36c2395); analysis doc 05569309.
- Metrics: tau row updated to full-run readings with the trigger flag.

## Next session, start here
**The owner's prioritization review** — everything is staged on esper-lite-f22a1d48a7
(tau memo + options) and esper-lite-a7ef375203 (audit + probe menu). The cheapest first
move regardless of direction: the zero-GPU F1/F2 occurrence probes over the existing
n=5 telemetry (telemetry/causal_r1_n5/). Nothing proceeds to F2 calibration or the A/B
until the owner rules on tau disposition and audit sequencing.
