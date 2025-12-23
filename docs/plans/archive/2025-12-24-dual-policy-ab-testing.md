# Dual-Policy A/B Testing: One Policy Per GPU

**Status:** COMPLETED
**Date:** 2025-12-24
**Completed:** 2025-12-24
**Goal:** True A/B comparison of reward modes by training separate policies on separate GPUs

> **Implementation Notes:**
> - Phase 1 uses sequential training (not parallel lockstep) for simplicity
> - Task 3 (live comparative logging) deferred - requires parallel architecture
> - CLI: `esper ppo --dual-ab shaped-vs-simplified`
> - Files: `policy_group.py`, `dual_ab.py`, CLI in `train.py`

---

## Problem Statement

The current `--ab-test` flag trains a **single shared policy** with mixed reward signals. This conflates the effects of different reward modes and cannot answer: *"Which reward mode trains a better policy?"*

## Solution: One Policy Per GPU

Train independent policies in parallel, each on its own GPU with its own reward mode. Compare results at the end.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DUAL-POLICY TRAINING                          │
├─────────────────────────────────┬───────────────────────────────────┤
│           GPU 0                 │            GPU 1                  │
│  ┌─────────────────────────┐   │   ┌─────────────────────────┐     │
│  │     Policy A            │   │   │     Policy B            │     │
│  │  (SHAPED reward)        │   │   │  (SIMPLIFIED reward)    │     │
│  └───────────┬─────────────┘   │   └───────────┬─────────────┘     │
│              │                  │               │                   │
│  ┌───────────┴─────────────┐   │   ┌───────────┴─────────────┐     │
│  │  Envs 0-3 (all SHAPED)  │   │   │  Envs 0-3 (all SIMPLE)  │     │
│  └─────────────────────────┘   │   └─────────────────────────┘     │
└─────────────────────────────────┴───────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   A/B COMPARISON        │
                    │  Policy A acc vs B acc  │
                    │  Policy A reward vs B   │
                    │  Entropy trends         │
                    └─────────────────────────┘
```

---

## Architecture

### Core Concept: PolicyGroup

A `PolicyGroup` encapsulates one policy + its environments + its reward config:

```python
@dataclass
class PolicyGroup:
    """One policy with its dedicated environments."""
    group_id: str                    # "A" or "B"
    device: torch.device             # cuda:0 or cuda:1
    agent: PPOAgent                  # Independent policy
    envs: list[ParallelEnvState]     # Dedicated environments
    reward_config: ContributionRewardConfig
    episode_history: list[dict]      # Per-group tracking
```

### Training Loop Structure

```python
def train_dual_policy_ab(
    n_envs_per_group: int = 4,
    group_configs: list[tuple[str, RewardMode]] = [
        ("A", RewardMode.SHAPED),
        ("B", RewardMode.SIMPLIFIED),
    ],
    devices: list[str] = ["cuda:0", "cuda:1"],
    ...
) -> dict[str, tuple[PPOAgent, list[dict]]]:
    """Train multiple policies in parallel, one per GPU.

    Returns:
        Dict mapping group_id -> (agent, history)
    """

    # 1. Create one PolicyGroup per device
    groups = []
    for (group_id, reward_mode), device in zip(group_configs, devices):
        agent = PPOAgent(state_dim=..., device=device, ...)
        envs = [create_env(device) for _ in range(n_envs_per_group)]
        config = ContributionRewardConfig(reward_mode=reward_mode, ...)
        groups.append(PolicyGroup(group_id, device, agent, envs, config, []))

    # 2. Training loop: step all groups in lockstep
    for episode in range(n_episodes):
        for group in groups:
            # Each group does its own:
            # - Environment steps
            # - Reward computation (with group.reward_config)
            # - Buffer collection
            # - Policy update (only affects group.agent)
            step_group(group, epoch)

        # Sync point: log comparative metrics
        log_ab_comparison(groups)

    # 3. Final comparison
    print_ab_results(groups)

    return {g.group_id: (g.agent, g.episode_history) for g in groups}
```

---

## CLI Design

```bash
# New flag: --dual-ab
esper ppo --dual-ab shaped-vs-simplified --n-envs 4 --episodes 100

# Requires 2 GPUs (one per policy)
# Creates 4 envs per GPU (8 total)
# Trains Policy A on cuda:0 with SHAPED
# Trains Policy B on cuda:1 with SIMPLIFIED
# Prints comparative results at end
```

### Automatic GPU Assignment

```python
if args.dual_ab:
    available_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if len(available_gpus) < 2:
        raise ValueError("--dual-ab requires at least 2 GPUs")

    group_configs = parse_dual_ab_flag(args.dual_ab)  # e.g., shaped-vs-simplified
    devices = available_gpus[:len(group_configs)]
```

---

## Implementation Tasks

### Task 1: Add PolicyGroup Dataclass

**File:** `src/esper/simic/training/policy_group.py` (new)

```python
from dataclasses import dataclass, field
import torch
from esper.simic.agent import PPOAgent
from esper.simic.rewards import ContributionRewardConfig

@dataclass
class PolicyGroup:
    """Independent policy with dedicated environments for A/B testing."""
    group_id: str
    device: torch.device
    reward_mode: str
    agent: PPOAgent
    reward_config: ContributionRewardConfig
    episode_history: list[dict] = field(default_factory=list)

    # Per-group metrics
    total_episodes: int = 0
    total_steps: int = 0
    best_accuracy: float = 0.0
```

### Task 2: Create train_dual_policy_ab Function

**File:** `src/esper/simic/training/dual_ab.py` (new)

Core training loop that manages multiple PolicyGroups in lockstep.

Key differences from `train_ppo_vectorized`:
- Multiple agents (one per group)
- Multiple environment sets (partitioned by GPU)
- Independent buffers and updates
- Synchronized logging for comparison

### Task 3: Add Comparative Logging

**File:** `src/esper/simic/training/dual_ab.py`

```python
def log_ab_comparison(groups: list[PolicyGroup], episode: int) -> None:
    """Log comparative metrics between groups."""
    print(f"\n=== Episode {episode} Comparison ===")
    for group in groups:
        recent = group.episode_history[-10:] if group.episode_history else []
        avg_acc = sum(ep["final_accuracy"] for ep in recent) / len(recent) if recent else 0
        avg_reward = sum(ep["episode_reward"] for ep in recent) / len(recent) if recent else 0
        print(f"  {group.group_id} ({group.reward_mode}): "
              f"Acc={avg_acc:.1f}%, Reward={avg_reward:.2f}")
```

### Task 4: Wire CLI --dual-ab Flag

**File:** `src/esper/scripts/train.py`

```python
ppo_parser.add_argument(
    "--dual-ab",
    type=str,
    choices=["shaped-vs-simplified", "shaped-vs-sparse", "simplified-vs-sparse"],
    default=None,
    help="True A/B test: train separate policies on separate GPUs",
)
```

### Task 5: Final Comparison Report

**File:** `src/esper/simic/training/dual_ab.py`

```python
def print_dual_ab_results(groups: list[PolicyGroup]) -> None:
    """Print final A/B comparison."""
    print("\n" + "=" * 70)
    print("DUAL-POLICY A/B TEST RESULTS")
    print("=" * 70)

    for group in groups:
        eps = group.episode_history
        rewards = [ep["episode_reward"] for ep in eps]
        accs = [ep["final_accuracy"] for ep in eps]

        print(f"\n{group.group_id} - {group.reward_mode.upper()} (GPU: {group.device})")
        print(f"  Episodes: {len(eps)}")
        print(f"  Final Accuracy: {accs[-1]:.2f}%" if accs else "  No episodes")
        print(f"  Best Accuracy: {max(accs):.2f}%" if accs else "")
        print(f"  Avg Episode Reward: {sum(rewards)/len(rewards):.2f}" if rewards else "")

    # Winner determination
    if len(groups) == 2:
        a_final = groups[0].episode_history[-1]["final_accuracy"] if groups[0].episode_history else 0
        b_final = groups[1].episode_history[-1]["final_accuracy"] if groups[1].episode_history else 0
        winner = groups[0] if a_final > b_final else groups[1]
        margin = abs(a_final - b_final)
        print(f"\n>>> WINNER: {winner.group_id} ({winner.reward_mode}) by {margin:.2f}% <<<")

    print("=" * 70)
```

---

## Telemetry Integration

Each PolicyGroup emits events with `group_id` tag:

```python
hub.emit(TelemetryEvent(
    event_type=TelemetryEventType.EPISODE_COMPLETED,
    data={
        "group_id": group.group_id,
        "reward_mode": group.reward_mode,
        "device": str(group.device),
        "final_accuracy": acc,
        "episode_reward": reward,
    },
))
```

Karn/DuckDB can then query:
```sql
SELECT group_id, reward_mode, AVG(final_accuracy) as avg_acc
FROM episodes
GROUP BY group_id, reward_mode
```

---

## Resource Requirements

| Config | GPUs | Envs | Memory (est.) |
|--------|------|------|---------------|
| 2 groups × 4 envs | 2 | 8 | ~8GB per GPU |
| 2 groups × 8 envs | 2 | 16 | ~12GB per GPU |
| 3 groups × 4 envs | 3 | 12 | ~8GB per GPU |

Single-GPU fallback: Sequential training (less ideal but works):
```bash
esper ppo --reward-mode shaped --save-path a.pt && \
esper ppo --reward-mode simplified --save-path b.pt
```

---

## Success Criteria

1. `--dual-ab shaped-vs-simplified` runs on 2-GPU system
2. Each GPU trains an independent policy
3. Final report shows clear winner with accuracy delta
4. Telemetry tags events by group_id
5. Policies can be saved separately for later evaluation

---

## Open Questions

1. **Synchronized vs Independent Episodes:** Should groups complete episodes in lockstep, or run independently? Lockstep simplifies comparison; independent maximizes GPU utilization.

2. **Shared Observation Normalizer:** Should groups share RunningMeanStd stats, or each maintain their own? Sharing ensures fair input distribution; separate allows reward-specific adaptation.

3. **Cross-Evaluation:** After training, should we evaluate Policy A on SIMPLIFIED reward and vice versa? This would reveal reward-policy coupling.

---

## Estimated Effort

| Task | Complexity | Estimate |
|------|------------|----------|
| Task 1: PolicyGroup dataclass | Low | 30 min |
| Task 2: train_dual_policy_ab | High | 3-4 hours |
| Task 3: Comparative logging | Low | 30 min |
| Task 4: CLI wiring | Low | 30 min |
| Task 5: Final report | Medium | 1 hour |
| Testing | Medium | 2 hours |
| **Total** | | **~8 hours** |
