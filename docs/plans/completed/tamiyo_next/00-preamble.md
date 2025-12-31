# Tamiyo Next: Obs V3 + Policy V2 Implementation Guide

**Status:** Ready for Implementation
**Date:** 2025-12-30
**Prerequisites:** This document consolidates the approved designs from:

- `2025-12-30-obs-v3-design.md` — Observation space overhaul
- `2025-12-30-policy-v2-design.md` — Architecture and training enhancements

YOU MUST READ BOTH THIS DOCUMENT AND THE RELEVANT PREREQUISITE DESIGNS IN FULL BEFORE IMPLEMENTING. DO IT NOW. NO EXCEPTIONS.

---

## Risk Assessment

| Phase | Risk | Complexity | Key Hazards |
|-------|------|------------|-------------|
| 1: Leyline Constants | Low | Low | Most constants already exist; mainly dimension updates |
| 2: Obs V3 Extraction | **High** | **High** | Feature spec correctness, enum cardinalities, inactive-slot masking, CPU↔GPU copies |
| 3: Blueprint Embedding | Med | Med | dtype/int64, -1 handling, shape flattening |
| 4: Policy V2 + Q(s,op) | **High** | **High** | Critic becomes action-conditioned Q(s,op) baseline; bootstrap + GAE consistency |
| 5: PPO Entropy | Med | Med-High | Stability tuning, per-head causal masking/entropy |
| 6: Vectorized Integration | **Very High** | **Very High** | Rollout buffer schema changes, (obs, blueprint_indices) tuple through hot path |
| 7: Validation | Med | Risk-reducing | Verifies correctness; surface area for catching bugs |
| 8: Cleanup | Med | Med | Risk of leaving stale code; follow CLAUDE.md no-legacy policy |

**Top Technical Risks:**
1. **Op-conditioned value head:** Rollout value, truncation bootstrap_value, and PPO update must all condition on the same op
2. **Feature vectorization/perf:** Cached one-hot tables with `.to(device)` can accidentally allocate per-step
3. **Enum/cardinality drift:** Hard-coding `torch.eye(10)` for stages assumes fixed cardinality—use leyline constants
4. **API propagation:** `(obs, blueprint_indices)` touches network forward, rollout collection, buffer storage, PPO update

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Observation dims | 218 | 133 |
| Model params | ~227K | ~2.1M |
| LSTM hidden | 128 | 512 |
| Feature dim | 128 | 512 |
| Episode length | 25 epochs | 150 epochs |
| Decision horizon | ~25 epochs | 150+ epochs (3-seed sequential) |
| Value aliasing | Present | Solved |

**Why 512 hidden dim (not 256):**

With 150-epoch sequential scaffolding, the LSTM must maintain "archival memories" of earlier seeds while processing current ones. At epoch 140:
- Seed A: "I planted a CONV_HEAVY at epoch 5. It's fossilized and providing base structure."
- Seed B: "I planted ATTENTION at epoch 55. It fossilized at epoch 105, interacting with Seed A."
- Seed C: "I'm currently tuning DEPTHWISE to synergize with A+B."

**256 risks "Catastrophic Overwrite"** — as the LSTM processes Seed C's noisy gradients, it might evict the memory of what Seed A actually is. 512 provides the scratchpad space to keep archival memories safe.

**PPO "Horizon Cut" Bridge:** The LSTM hidden state is the only thing connecting step 150 to step 1 across rollout boundaries. Larger hidden state = more robust long-term state representation for the value function.

## Before You Start

### 1. Understand the Breaking Changes

**Checkpoint Incompatibility:** V1 checkpoints will NOT load into V2 networks due to:

- Different observation dimensions (218 → 133)
- Different LSTM hidden size (128 → 512)
- Different head input dimensions
- New value head architecture (op-conditioned)

**Action:** Plan to train from scratch. No migration path exists.

### 2. Verify Current Test Suite Passes

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/ -v
PYTHONPATH=src uv run pytest tests/simic/ -v
```

All tests should pass before starting. Any failures are pre-existing issues.

### 3. Create a Development Branch

```bash
git checkout -b tamiyo-v2 quality-sprint
```

### 4. Clean Replacement Strategy (No Dual Paths)

Per `CLAUDE.md` no-legacy policy, we do **clean replacement**, NOT dual-path version toggles:

- **Delete** old Tamiyo network/feature code as you implement V2
- **No** `_OBS_VERSION` or `_POLICY_VERSION` toggles
- **No** backwards compatibility shims
- Rollback via git branch, not code toggles

**The old Tamiyo doesn't work.** That's why we're here. There's no value in keeping it.

### 5. Key Files to Read First

| File | Purpose |
|------|---------|
| `src/esper/tamiyo/policy/features.py` | Current feature extraction (V2) |
| `src/esper/tamiyo/networks/factored_lstm.py` | Current network architecture |
| `src/esper/leyline/__init__.py` | Default dimensions and constants |
| `src/esper/simic/agent/ppo.py` | PPO training loop |
| `src/esper/simic/training/vectorized.py` | Rollout collection |

---

## Implementation Order

