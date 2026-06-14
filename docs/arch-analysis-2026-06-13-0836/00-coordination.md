## Analysis Plan

- Scope: `src/esper/kasmina/`, `src/esper/tolaria/`, blueprint definitions/usages, and the cross-domain paths where Tamiyo/Simic/Nissa/Karn consume Kasmina/Tolaria signals.
- Deliverable: written health report covering models investigated, bugs, architectural issues, systemic risks, and key actions.
- Strategy: parallel exploration plus local synthesis. Use Loomweave for structural navigation, verify against live files because the index is stale.
- Complexity estimate: High. The target spans topology mutation, lifecycle FSM, training execution, governor safety, telemetry, and evaluation methodology.
- Constraints: do not revert or overwrite existing dirty worktree changes; keep report artifacts isolated to this directory.

## Review Lens

- Morphogenetic RL essentials: deterministic event replay, two-grain telemetry, factored/no-op actions, reward net of counterfactual utility and structural cost, independent governor, rollback signal, multi-seed arbitration, off-switch/static/fixed-schedule baselines.
- Esper constitution: sensors must match capabilities; complexity pays rent; Tolaria must not block Simic; Kasmina remains host-agnostic via `HostProtocol`; no unsupervised growth without a governor.
- Repo rules: no legacy compatibility shims, no bug-hiding defensive patterns, preserve user changes.

## Execution Log

- 2026-06-13 08:36: Created analysis workspace.
- 2026-06-13 08:36: Read `CLAUDE.md`, `README.md`, `ROADMAP.md`, Filigree session context, and relevant morphogenetic-RL sheets.
- 2026-06-13 08:36: Confirmed dirty worktree; report output will be isolated.
- 2026-06-13 08:36: Loomweave project status: available, SEIs populated, stale index due current worktree changes.
