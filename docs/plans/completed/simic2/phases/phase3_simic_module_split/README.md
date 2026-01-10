# Phase 3: Decompose Large Simic Modules (Rewards + PPO Agent)

**Intent:** Make the remaining large Simic modules easier to navigate and change.

## Rewards split (`src/esper/simic/rewards/rewards.py`)

Proposed split:
- `contribution.py`: contribution-primary reward computation
- `loss_primary.py`: loss-primary reward computation
- `shaping.py`: PBRS + intervention costs + shared shaping utilities
- `rewards.py`: thin dispatcher (or remove and update imports)

Constraints:
- Keep a single source of truth for PBRS potentials (no duplication).
- Do not leave deprecated functions around “for compatibility”; update call sites and delete old names.

## PPO agent split (`src/esper/simic/agent/ppo.py`)

Proposed split:
- `ppo_agent.py`: PPOAgent surface, checkpoint load/save, buffer coordination
- `ppo_update.py`: update internals (losses, ratio calculations, clipping, KL stop)
- `ppo_metrics.py`: metrics construction/types (prefer dataclasses)

Constraints:
- Preserve `TamiyoRolloutBuffer` tensor shapes and ordering invariants.
- Preserve head ordering contracts (`HEAD_NAMES`, slot ordering via `SlotConfig`).

## Done means

- Each file has a single primary responsibility.
- Imports are directional (training depends on agent/rewards/telemetry; not vice versa).
- The codebase has fewer “mega-modules” with unrelated concerns.

## Preflight

- `preflight_checklist.md`
