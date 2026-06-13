# Architecture Health Report Task Slotting

Date: 2026-06-13

Source report: `docs/arch-analysis-2026-06-13-0836/01-kasmina-tolaria-blueprint-health.md`

## Triage Position

The report is useful and should change the near-term order of work. It confirms the working theory is not blocked by missing core machinery: Kasmina slots, Tolaria rollback, blueprint catalogues, Simic rewards, and Tamiyo observations all exist. The confounders are authority, ordering, and evidence hygiene problems around that machinery.

The right product move is to stop treating the next proof run as an experiment-execution problem. It is now a proof-validity problem. The reward-efficiency exam stays deferred until the controller cannot mutate after rollback, missing evidence cannot look healthy, and morphogenesis events can be tied to stable proposal/verdict/mutation records.

## Slotting Summary

| Report finding | Disposition | Tracker item | Priority |
| --- | --- | --- | --- |
| Snapshot can overwrite last-known-good state before panic detection | Next bug-fix batch | `morphogenesis-governor-integrity` Task 1 | P1 |
| Rollback does not halt current action execution | Next bug-fix batch | `morphogenesis-governor-integrity` Task 1 | P1 |
| No independent Tolaria pre-flight veto | Foundational architecture in same package, after rollback ordering | `morphogenesis-governor-integrity` Task 3 | P1/P2 |
| Missing telemetry becomes healthy observation state | Next bug-fix batch | `morphogenesis-governor-integrity` Task 2, with overlap to `telemetry-domain-sep` | P1 |
| Missing counterfactual falls back to host drift | Next bug-fix batch | `morphogenesis-governor-integrity` Task 2, then `counterfactual-oracle` later | P1 |
| Blueprint smoke probe allows non-tensor output | Next bug-fix batch | `morphogenesis-governor-integrity` Task 2 | P1 |
| Blueprint registry identity and unknown IDs are not hardened | Contract hardening in next package | `morphogenesis-governor-integrity` Task 2 | P1/P2 |
| Growth events are not replay-identifiable or RNG-isolated | Foundation for future replay/proof claims | `morphogenesis-governor-integrity` Task 4 | P2 |
| Tolaria is loss-only and mean/std based | Defer full health model, add proposal/watch surface first | `morphogenesis-governor-integrity` Task 3 and Task 4 | P2 |
| Rollback signal is coarsely attributed | Telemetry enhancement tied to rollback halt semantics | `morphogenesis-governor-integrity` Task 1 and Task 4 | P2 |
| Generic telemetry IDs are not a causal morphology log | Foundation for Karn/proof evidence | `morphogenesis-governor-integrity` Task 4, then `karn-telemetry-quality-arc` | P2 |
| Lifecycle handlers are not production-wired | Cleanup after authority boundary is chosen | `morphogenesis-governor-integrity` Task 5 | P2 |
| `SIMPLIFIED` reward does not pay structural rent | Proof semantics fix before reward claims | `proof-baseline-controls` and `reward-efficiency` | P2 |
| Reward/action telemetry can mislead | Telemetry truthfulness repair | `morphogenesis-governor-integrity` Task 6 and `telemetry-domain-sep` | P2 |
| Baselines are not strong enough for final health claims | Experimental-design follow-up | `proof-baseline-controls` before final `reward-efficiency` verdict | P3 |

## Current Task List Changes

Add these work items to the tracker:

1. `morphogenesis-governor-integrity` - the next high-priority package. It drains the report's P1 correctness bugs and lays the first proposal/verdict/event-identity foundation.
2. `ppo-stability-oracle-sandbox` - still needed, but it should come after the P1 governor/truthfulness fixes. It isolates the proof rehearsal's value-collapse and gradient-anomaly blockers and proves lifecycle mechanics under an oracle or hardcoded policy.
3. `proof-baseline-controls` - a medium-priority experimental-control package for off-switch/static/fixed-schedule/lockstep baselines before final blueprint-health claims.

Update these existing items:

- `reward-efficiency`: keep infrastructure-ready, but mark execution deferred until `morphogenesis-governor-integrity`, `ppo-stability-oracle-sandbox`, and a clean rehearsal packet are complete.
- `telemetry-domain-sep`: keep as a broader contract cleanup, but move active-slot missing telemetry and action-success truthfulness into the nearer governor-integrity package because they are current proof confounders.
- `counterfactual-oracle`: keep blocked behind reward-efficiency; do not use it to paper over host-drift contamination in the current observation channel.
- `blueprint-compiler` and `kasmina2-phase0`: keep deferred until morphology safety/proof signals are clean.

## Product Rationale

This is the smallest major package that can make the next proof attempt meaningful. Running larger A/B experiments before these fixes would spend compute measuring a mixture of theory signal, rollback ordering bugs, policy observation contamination, and non-replayable growth events.

The expected milestone is not "the theory is proven." The milestone is: if a run underperforms after this package, the failure is much more likely to be an actual algorithmic problem rather than an instrumentation, authority-boundary, or recovery-ordering confounder.
