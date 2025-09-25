# RC1 Milestone Overview

## Objective
Bring Tolaria, Tamiyo, and Kasmina to the prototype-delta standard by removing masked failures, aligning on PyTorch 2.8 behaviour, tightening lifecycle/telemetry guarantees, and delivering clean complexity metrics across the execution stack.

## Success Criteria
- No subsystem masks missing dependencies (fallback kernels, synthetic pause commands, placeholder IDs).
- Shared async worker & telemetry routing adopted by Tolaria, Tamiyo, and Kasmina.
- All work packages (WP-T1..4, WP-A1..4, WP-K1..4) completed with passing tests and telemetry verification.
- Complexity hot-spots (F/E/D) reduced to ≤ C; `lint_static_analysis.md` updated.
- Performance harness executed; rollback SLA metrics instrumented.

## In-Scope
- Execution-layer code under `src/esper/tolaria`, `src/esper/tamiyo`, `src/esper/kasmina`.
- Shared RC1 foundations: async worker, telemetry routing, strict dependency policy.
- Test suites and telemetry for execution stack.

## Out-of-Scope
- Upstream subsystem changes outside execution plane (Urza, Weatherlight except telemetry routing hooks).
- New feature work beyond prototype-delta scope (e.g. KD training loops, distributed modes).

## Dependencies & Sequencing
1. Shared foundations (strict deps, async worker, telemetry routing, Leyline alignment).
2. Tolaria WP-T1/T2 → Tamiyo WP-A1 → Kasmina WP-K1.
3. Mid-phase: Tolaria WP-T3, Tamiyo WP-A2, Kasmina WP-K2.
4. Telemetry pass: Tolaria WP-T4, Tamiyo WP-A3, Kasmina WP-K3.
5. Persistence/cache: Tamiyo WP-A4, Kasmina WP-K4.
6. Cross-system QA (performance, rollback SLA, lint cleanup).

## Ownership Matrix
| Work Package | Lead | Reviewers |
|--------------|------|-----------|
| Shared Foundations | (tbd) | Leads from Tolaria, Tamiyo, Kasmina |
| Tolaria WP-T1/T2 | (tbd) | Tamiyo & Kasmina reps |
| Tolaria WP-T3/T4 | (tbd) | Shared foundations lead |
| Tamiyo WP-A1/A2 | (tbd) | Tolaria & Kasmina reps |
| Tamiyo WP-A3/A4 | (tbd) | Shared foundations lead |
| Kasmina WP-K1/K2 | (tbd) | Tolaria & Tamiyo reps |
| Kasmina WP-K3/K4 | (tbd) | Shared foundations lead |

## Checkpoints
- Kickoff: confirm owners, dependencies, test plan.
- Mid-iteration syncs: weekly status review using `08_status_tracker.md`.
- Telemetry validation checkpoint post WP-A3/WP-K3.
- Final sign-off: all success criteria met; change log updated.
