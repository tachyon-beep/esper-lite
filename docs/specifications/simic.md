---
module: simic
version: 2.0
status: active
last_updated: 2026-01-01
maintainers: [John]
specialist_reviews:
  - agent: drl-expert
    date: 2025-12-14
    focus: PPO correctness, GAE, PBRS, reward shaping, Goodhart risks
  - agent: pytorch-expert
    date: 2025-12-14
    focus: CUDA streams, LSTM patterns, memory management, gradient flow
---

# Simic Module Bible

## 1. Prime Directive

### Role

Simic is the **reinforcement learning training infrastructure** for Tamiyo, Esper's seed lifecycle controller. It implements PPO (Proximal Policy Optimization) with:

- **Factored action heads**: 8 independent policy heads (slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve, op) with causal masking
- **LSTM temporal memory**: Tracks seed lifecycle state across 150-epoch episodes
- **Op-conditioned critic**: Q(s, op) baseline reduces value aliasing
- **Counterfactual reward attribution**: Measures seed value via ablation (alpha=0 validation)
- **PBRS reward shaping**: Potential-based shaping that preserves optimal policy
- **Vectorized parallel training**: N environments with CUDA stream optimization

The module trains an LSTM policy that observes training signals and makes lifecycle decisions (WAIT, GERMINATE, ADVANCE, SET_ALPHA_TARGET, FOSSILIZE, PRUNE) to maximize host model accuracy while minimizing parameter bloat.

### Anti-Scope

Simic is **NOT** responsible for:

- Model architecture (Kasmina's domain)
- Data loading or preprocessing (Runtime's domain)
- Seed state machine transitions (Leyline defines contracts, Kasmina executes)
- Telemetry aggregation (Nissa/Karn handle event routing)
- Heuristic decision logic (Tamiyo heuristics live in tamiyo module)

**Boundary violations to avoid:**
- Simic should never modify seed module weights directly (only optimizer.step())
- Simic should never advance seed stages (call slot.advance_stage(), not internal methods)
- Simic should never access DataLoader internals (only iterate over batches)

---

## 2. Interface Contract

### Primary Entry Points

```python
# Vectorized training (main training API)
from esper.simic.training import train_ppo_vectorized

history = train_ppo_vectorized(
    task="cifar10",
    n_episodes=100,
    n_envs=4,
    max_epochs=150,
    device="cuda:0",
    slots=["r0c0", "r0c1", "r0c2"],
)

# PPO Agent (for custom training loops)
from esper.leyline.slot_config import SlotConfig
from esper.simic.agent import PPOAgent
from esper.tamiyo.policy.factory import create_policy
from esper.tamiyo.policy.features import get_feature_size

slot_config = SlotConfig.default()
policy = create_policy(
    policy_type="lstm",
    state_dim=get_feature_size(slot_config),
    slot_config=slot_config,
    device="cuda:0",
    compile_mode="default",
)
agent = PPOAgent(
    policy=policy,
    slot_config=slot_config,
    device="cuda:0",
)

# Reward computation
from esper.simic.rewards import compute_contribution_reward, SeedInfo

reward = compute_contribution_reward(
    action=LifecycleOp.GERMINATE,
    seed_contribution=0.05,  # Counterfactual delta
    val_acc=65.0,
    seed_info=SeedInfo(...),
    epoch=10,
    max_epochs=150,
)
```

### Key Data Structures

```python
# SeedInfo - Lightweight seed state for reward computation
class SeedInfo(NamedTuple):
    stage: int                    # SeedStage.value
    improvement_since_stage_start: float
    total_improvement: float      # Since germination
    epochs_in_stage: int
    seed_params: int = 0
    previous_stage: int = 0       # For PBRS telescoping
    previous_epochs_in_stage: int = 0
    seed_age_epochs: int = 0

# TamiyoRolloutStep - Single transition for buffer storage
class TamiyoRolloutStep(NamedTuple):
    state: torch.Tensor           # [state_dim]
    slot_action: int
    blueprint_action: int
    style_action: int
    tempo_action: int
    alpha_target_action: int
    alpha_speed_action: int
    alpha_curve_action: int
    op_action: int
    slot_log_prob: float
    blueprint_log_prob: float
    style_log_prob: float
    tempo_log_prob: float
    alpha_target_log_prob: float
    alpha_speed_log_prob: float
    alpha_curve_log_prob: float
    op_log_prob: float
    value: float
    reward: float
    done: bool
    truncated: bool
    bootstrap_value: float
    slot_mask: torch.Tensor
    blueprint_mask: torch.Tensor
    style_mask: torch.Tensor
    tempo_mask: torch.Tensor
    alpha_target_mask: torch.Tensor
    alpha_speed_mask: torch.Tensor
    alpha_curve_mask: torch.Tensor
    op_mask: torch.Tensor
    hidden_h: torch.Tensor        # [num_layers, hidden_dim]
    hidden_c: torch.Tensor

# TrainingConfig - Unified hyperparameter configuration
@dataclass
class TrainingConfig:
    lr: float = 3e-4
    gamma: float = 0.995          # MUST match DEFAULT_GAMMA in rewards.py
    gae_lambda: float = 0.97
    clip_ratio: float = 0.2
    entropy_coef: float = 0.05
    # ... (see config.py for full list)
```

### Reward Mode Selection

```python
from esper.simic.rewards import RewardMode, ContributionRewardConfig

# Dense shaped rewards (default) - PBRS + attribution + warnings
config = ContributionRewardConfig(reward_mode=RewardMode.SHAPED)

# Sparse terminal-only - Tests credit assignment capability
config = ContributionRewardConfig(reward_mode=RewardMode.SPARSE)

# Minimal - Sparse + early-prune penalty only
config = ContributionRewardConfig(reward_mode=RewardMode.MINIMAL)
```

---

## 3. Tensor Contracts

### Network Input/Output Shapes

```python
# FactoredRecurrentActorCritic
# Input
features: [batch, seq_len, 113]   # Obs V3 non-blueprint (23 base + 30 per slot * 3)
blueprint_indices: [batch, seq_len, 3]  # Per-slot blueprint indices
# Network concatenates blueprint embeddings (num_slots * 4) before feature_net

# Hidden state (LSTM)
hidden_h: [1, batch, 512]         # [num_layers, batch, hidden_dim]
hidden_c: [1, batch, 512]

# Action masks (True = valid action)
slot_mask: [batch, seq_len, num_slots]
blueprint_mask: [batch, seq_len, NUM_BLUEPRINTS]
style_mask: [batch, seq_len, NUM_STYLES]
tempo_mask: [batch, seq_len, NUM_TEMPO]
alpha_target_mask: [batch, seq_len, NUM_ALPHA_TARGETS]
alpha_speed_mask: [batch, seq_len, NUM_ALPHA_SPEEDS]
alpha_curve_mask: [batch, seq_len, NUM_ALPHA_CURVES]
op_mask: [batch, seq_len, NUM_OPS]

# Outputs
slot_logits: [batch, seq_len, num_slots]
blueprint_logits: [batch, seq_len, NUM_BLUEPRINTS]
style_logits: [batch, seq_len, NUM_STYLES]
tempo_logits: [batch, seq_len, NUM_TEMPO]
alpha_target_logits: [batch, seq_len, NUM_ALPHA_TARGETS]
alpha_speed_logits: [batch, seq_len, NUM_ALPHA_SPEEDS]
alpha_curve_logits: [batch, seq_len, NUM_ALPHA_CURVES]
op_logits: [batch, seq_len, NUM_OPS]
value: [batch, seq_len]           # Q(s, op) baseline
```

### Buffer Shapes

```python
# TamiyoRolloutBuffer pre-allocated tensors
# Shape: [num_envs, max_steps_per_env, ...]
states: [4, 150, 113]
slot_actions: [4, 150]             # dtype=torch.long
values: [4, 150]
rewards: [4, 150]
advantages: [4, 150]               # Computed after rollout
returns: [4, 150]                  # advantages + values
hidden_h: [4, 150, 1, 512]         # [envs, steps, layers, hidden]
```

### Feature Vector Composition

```python
# Total non-blueprint dims: 23 base + 30 per-slot * num_slots (113 for 3 slots)
# Blueprint identity moved to embeddings inside the network (4 dims per slot).

# Base features (23 dims): tamiyo/policy/features.py
- epoch_norm, val_loss_norm, val_accuracy_norm (3)
- loss_history_5 (5)
- accuracy_history_5 (5)
- stage distribution (num_training/blending/holding) (3)
- action feedback: last_action_success + last_action_op one-hot (7)

# Per-slot features (30 dims each)
- is_active (1)
- stage one-hot (10)
- current_alpha (1)
- contribution_norm (1)
- contribution_velocity (1)
- blend_tempo_norm (1)
- alpha scaffolding (8): target, mode, steps_total, steps_done, time_to_target, velocity, algorithm, interaction_sum
- telemetry merged (4): gradient_norm, gradient_health, has_vanishing, has_exploding
- gradient_health_prev (1)
- epochs_in_stage_norm (1)
- counterfactual_fresh (1)
```

---

## 4. Operational Physics

### State Machine: Episode Lifecycle

```
EPISODE_START
    │
    ├─► HOST_STABILIZING ──[stable]──► TRAINING_ACTIVE
    │
    ▼
TRAINING_ACTIVE
    │
    ├─► EPOCH_LOOP (1..max_epochs)
    │       │
    │       ├─► TRAIN_BATCH ──► VALIDATION ──► COUNTERFACTUAL
    │       │                                     │
    │       │       ┌─────────────────────────────┘
    │       │       ▼
    │       ├─► COMPUTE_REWARD ──► SELECT_ACTION ──► EXECUTE_ACTION
    │       │                                              │
    │       │       ┌──────────────────────────────────────┘
    │       │       ▼
    │       └─► BUFFER_STORE ──► next epoch
    │
    ├─► TRUNCATED (epoch == max_epochs) ──► BOOTSTRAP_VALUE ──► EPISODE_END
    │
    └─► GOVERNOR_ROLLBACK (catastrophic loss) ──► DISCARD_BUFFER ──► EPISODE_END
```

### State Machine: Reward Attribution

```
OBSERVATION_AVAILABLE
    │
    ├─► [seed.stage >= BLENDING] ──► COUNTERFACTUAL_PATH
    │       │
    │       ├─► seed_contribution < 0 ──► TOXIC_PENALTY (negative attribution)
    │       │
    │       ├─► seed_contribution >= progress ──► GEOMETRIC_MEAN
    │       │       │
    │       │       └─► sqrt(progress * contribution)
    │       │
    │       └─► seed_contribution < progress ──► CAPPED_ATTRIBUTION
    │               │
    │               └─► min(contribution, progress)
    │
    └─► [seed.stage < BLENDING] ──► PROXY_PATH
            │
            └─► acc_delta > 0 ──► proxy_weight * acc_delta
```

### Anti-Ransomware Defenses

The reward system implements multi-layer defenses against seeds that create structural dependencies without adding value:

1. **Attribution Discount** (`rewards.py:434-435`): Seeds with `total_improvement < 0` receive sigmoid-scaled attribution
2. **Ratio Penalty** (`rewards.py:437-451`): High contribution / low improvement triggers penalty
3. **Ransomware Signature Detection** (`rewards.py:922-928`): Seeds with `contribution > 0.1` AND `total_delta < -0.2` get extra penalty
4. **Terminal Bonus Gate** (`rewards.py:629-633`): Only seeds with `total_improvement >= MIN_FOSSILIZE_CONTRIBUTION` receive fossilize terminal bonus

### Performance Cliffs

| Scenario | Symptom | Root Cause | Mitigation |
|----------|---------|------------|------------|
| Gamma mismatch | Policy diverges from optimal | `ppo.py:gamma=0.99` vs `rewards.py:DEFAULT_GAMMA=0.995` | Use `TrainingConfig.to_ppo_kwargs()` which ensures gamma=0.995 |
| Hidden state staleness | Erratic policy after epoch 1 | `recurrent_n_epochs > 1` with LSTM | Keep `recurrent_n_epochs=1` (default) |
| Value explosion | `value_loss > 1000` | Value clip too tight (0.2) | Use `value_clip=10.0` (different from `clip_ratio`) |
| Entropy collapse | Policy becomes deterministic | `entropy_coef_min` too low | Set `entropy_coef_min=0.01`, enable `adaptive_entropy_floor` |
| GAE interleaving | Incorrect advantages | Cross-environment GAE contamination | P0 fix: GAE computed per-environment (`rollout_buffer.py:266`) |

---

## 5. Dependencies

### Internal Dependencies

```
simic
├── leyline (contracts)
│   ├── SeedStage         # Stage enumeration
│   ├── FactoredAction    # Action space definition
│   ├── TelemetryEvent    # Event types
│   └── MIN_PRUNE_AGE, MIN_HOLDING_EPOCHS  # Constants
├── kasmina (model)
│   ├── MorphogeneticModel  # Model with seed slots
│   ├── SeedSlot           # Per-slot management
│   └── MIN_FOSSILIZE_CONTRIBUTION  # Gate threshold
├── nissa (telemetry)
│   └── get_hub()         # Telemetry event hub
├── runtime (task specs)
│   └── get_task_spec()   # DataLoader creation
└── utils/loss
    └── compute_task_loss_with_metrics()
```

### External Dependencies

```python
torch >= 2.0              # torch.compile support required
torch.distributions       # Categorical for action sampling
torch.optim              # AdamW with fused=True (CUDA)
numpy                    # Limited use (prefer torch)
```

### Import Graph (Critical Path)

```
train_ppo_vectorized
├── PPOAgent (ppo.py)
│   ├── FactoredRecurrentActorCritic (network.py)
│   └── TamiyoRolloutBuffer (rollout_buffer.py)
├── compute_contribution_reward (rewards.py)
│   └── SeedInfo, STAGE_POTENTIALS, DEFAULT_GAMMA
├── compute_action_masks (action_masks.py)
│   └── MaskedCategorical
└── ParallelEnvState (vectorized.py)
    └── MorphogeneticModel (kasmina)
```

---

## 6. Esper Integration

### Lifecycle Hooks

```python
# In vectorized training loop (vectorized.py:1100-1200)

# 1. Action Selection (with frozen normalizer)
with torch.inference_mode():
    action_result = agent.policy.get_action(
        features,
        blueprint_indices=blueprint_indices,
        masks=masks,
        hidden=hidden,
        deterministic=False,
    )

# 2. Action Execution (via MorphogeneticModel)
if factored_action.is_germinate:
    model.germinate_seed(blueprint_id, seed_id, slot=target_slot)
elif factored_action.op == LifecycleOp.SET_ALPHA_TARGET:
    slot.set_alpha_target(alpha_target, steps, curve, steepness, alpha_algorithm)
elif factored_action.is_fossilize:
    slot.advance_stage(SeedStage.FOSSILIZED)
elif factored_action.is_prune:
    model.prune_seed(slot=target_slot)

# 3. Telemetry Emission
hub.emit(TelemetryEvent(
    event_type=TelemetryEventType.STEP_COMPLETE,
    data={
        "env_id": env_id,
        "epoch": epoch,
        "action": factored_action.op.name,
        "reward": reward,
        ...
    }
))
```

### Governor Integration

```python
# TolariaGovernor monitors for catastrophic failures (vectorized.py:820-850)

governor = TolariaGovernor(
    sensitivity=6.0,
    absolute_threshold=12.0,
    death_penalty=10.0,
    random_guess_loss=math.log(10),  # CIFAR-10
)

# Check after each validation
catastrophic = governor.check_catastrophic(val_loss)
if catastrophic:
    # Rollback model to pre-germination checkpoint
    env_state.rollback_to_checkpoint()
    # Clear buffer to prevent training on bad transitions
    agent.buffer.reset()
```

### Checkpoint Protocol

```python
# Save (ppo.py:508-548)
agent.save("checkpoint.pt", metadata={
    "episode": ep,
    "best_accuracy": best_acc,
    "obs_normalizer_state": normalizer.state_dict(),
})

# Load (ppo.py:550-572)
agent = PPOAgent.load("checkpoint.pt", device="cuda:0")
# Normalizer stats must be restored separately!
```

---

## 7. Cross-References

### Related Bibles

- **[kasmina.md](kasmina.md)**: Model architecture, seed grafting, slot management
- **[leyline.md](leyline.md)**: Contracts, action space, stage definitions
- **[nissa.md](nissa.md)**: Telemetry hub, event routing

### Key Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `vectorized.py` | 1687 | Vectorized training loop with CUDA streams |
| `rewards.py` | 1190 | Reward computation, PBRS, anti-ransomware |
| `ppo.py` | 578 | PPO agent with factored recurrent architecture |
| `training.py` | 518 | Heuristic training, compiled train step |
| `gradient_collector.py` | 429 | Async gradient statistics collection |
| `rollout_buffer.py` | 395 | Pre-allocated rollout buffer |
| `network.py` | 334 | LSTM policy/value network |
| `action_masks.py` | 332 | MaskedCategorical distribution |
| `config.py` | 238 | TrainingConfig dataclass |
| `features.py` | 230 | Feature extraction |
| `anomaly_detector.py` | 226 | Training anomaly detection |
| `normalization.py` | 185 | Running mean/std normalizer |

### Test Coverage

- `tests/tamiyo/test_ppo_unified.py` - PPO agent tests
- `tests/tamiyo/test_rewards.py` - Reward function tests
- `tests/tamiyo/properties/test_pbrs_telescoping.py` - PBRS property tests
- `tests/tamiyo/test_action_masks.py` - Action masking tests

---

## 8. Tribal Knowledge

### Critical Configuration Gotchas

| Pitfall | Why It Happens | Correct Approach | Reference |
|---------|----------------|------------------|-----------|
| ~~Gamma mismatch breaks PBRS~~ | **FIXED 2025-12-14**: All gamma defaults now use `DEFAULT_GAMMA` from leyline (0.995). Previously, `ppo.py` defaulted to 0.99 while PBRS used 0.995. | `DEFAULT_GAMMA` is the single source of truth in `leyline/__init__.py`. All simic modules import from there. | `leyline/__init__.py:41` |
| Value clipping too tight | Using `clip_ratio=0.2` for value clip. Values range -10 to +50, policy ratios are ~1.0. | Use separate `value_clip=10.0` (default), not policy `clip_ratio` | `ppo.py:158` |
| Multiple PPO epochs with LSTM | Reusing old hidden states for multiple update epochs causes policy drift | Keep `recurrent_n_epochs=1` (default). Only increase with extreme caution | `ppo.py:179-180` |
| chunk_length != max_epochs | LSTM sequences must match episode length. Mismatch causes truncated BPTT or shape errors | Auto-matched in `TrainingConfig.__post_init__()` | `config.py:114-116` |
| Normalizer stats not saved | Resuming training with fresh normalizer causes distribution shift | Save/restore normalizer stats in checkpoint manually | `vectorized.py:1663-1669` |

### Debugging Protocol

**DRL Expert Recommended Diagnostics:**

```python
# If training appears stuck:
if explained_variance < 0:
    # Critic is broken - check value_loss
    print("ALERT: Negative explained variance indicates critic failure")

if clip_fraction < 0.1:
    # Policy not changing - may be stuck in local minimum
    print("WARNING: Low clip fraction - policy may be stuck")

if clip_fraction > 0.3:
    # Policy changing too fast - may be unstable
    print("WARNING: High clip fraction - consider reducing lr")

if ratio_max > 2.0 or ratio_min < 0.5:
    # Old log probs are stale
    print("WARNING: Extreme ratios - check buffer/policy sync")

# If rewards are suspicious:
reward, components = compute_contribution_reward(..., return_components=True)
if components.attribution_discount < 0.5:
    print("Ransomware detection: High contribution but negative total_improvement")
if components.ratio_penalty != 0:
    print(f"Ratio penalty applied: {components.ratio_penalty}")
```

### PyTorch Expert Insights

| Pattern | Why It Matters | Location |
|---------|----------------|----------|
| Pre-step hidden state capture | Buffer must store INPUT hidden state, not OUTPUT. Many LSTM-PPO implementations get this wrong | `vectorized.py:1117-1139` |
| Frozen normalizer during rollout | All states in a batch must use identical normalization. Updating mid-rollout causes inconsistency | `vectorized.py:1100-1110` |
| EMA momentum=0.99 for normalizer | Prevents distribution shift during long training. Welford's algorithm adapts too fast | `vectorized.py:395-396` |
| @torch.compiler.disable on GAE | Python loops in GAE cause graph breaks. Acceptable since GAE runs once per rollout, not per batch | `rollout_buffer.py:255` |
| Action mask validation compiler-disabled | `.any()` check forces CPU sync. Safety check worth the cost but shouldn't block compilation | `action_masks.py:263-275` |
| Tensor accumulation in train loop | Single `.item()` sync at epoch end instead of per-batch. 10x faster | `training.py:134-138` |

### DRL Expert Insights

| Insight | Why It Matters | Location |
|---------|----------------|----------|
| Truncation bootstrapping is NOT terminal | Time-limit truncation (default 150 epochs) is artificial. Without bootstrap, value function underestimates returns near episode end | `rollout_buffer.py:283-289` |
| Counterfactual fused with validation | Eliminates second DataLoader pass. Sets alpha=0 temporarily during validation batch | `vectorized.py:893-903` |
| Causal advantage masking | Per-head masks follow op selection; non-causal heads get zeroed advantages | `advantages.py:33-70` |
| Ransomware = high removal cost, negative improvement | Counterfactual measures "removal cost" not "value added". Seed can create dependencies without improving accuracy | `rewards.py:426-451` |
| Governor clears buffer on rollback | Prevents training on transitions that led to catastrophic failure | `vectorized.py:1442-1450` |

### What Would Break if...

| Change | Consequence |
|--------|-------------|
| Remove LayerNorm on LSTM output | Hidden state magnitudes drift over long horizons, causing gradient explosion |
| Use joint log prob instead of per-head | Causal masking breaks - irrelevant heads contribute noise to policy gradient |
| Remove attribution_discount | Ransomware seeds accumulate positive rewards despite net harm |
| Use gamma=0.99 in PPO with gamma=0.995 PBRS | PBRS no longer preserves optimal policy - shaping can change learned behavior |
| Store post-step hidden states | Policy learned with wrong temporal context - sees "future" information |

---

## 9. Changelog

### 2025-12-14 (This Document)
- Created comprehensive Module Bible with specialist reviews
- **FIXED gamma mismatch**: Moved `DEFAULT_GAMMA=0.995` to leyline as single source of truth
  - Updated: `ppo.py`, `rollout_buffer.py`, `vectorized.py`, `config.py`, `train.py`
  - `rewards.py` now imports from leyline instead of defining its own
- Added anti-ransomware defense documentation
- Captured PyTorch expert CUDA stream patterns

### Recent Code Changes (December 2025)
- `feat(vectorized)`: Wire prints to telemetry system
- `feat(rewards)`: Add RewardMode enum (SHAPED/SPARSE/MINIMAL)
- `feat(rewards)`: Implement compute_sparse_reward and compute_minimal_reward
- `feat(simic)`: Track host_max_acc for sparse reward
- `feat(simic)`: Wire anomaly detection into vectorized training
- `fix(vectorized)`: Use compute_reward dispatcher

### Architecture Decisions
- **PBRS over pure sparse**: Sparse reward tested but LSTM credit assignment insufficient for 150-step horizons without shaping
- **Factored action space**: Enables causal masking, reduces gradient noise vs joint action
- **Single PPO epoch for recurrent**: Multiple epochs cause hidden state staleness
- **Pre-allocated buffer**: Fixed 150-epoch episodes enable compile-friendly direct indexing

---

## Validation Checklist

- [x] YAML frontmatter valid and complete
- [x] All 9 sections present
- [x] Prime Directive has both Role AND Anti-Scope
- [x] Specialists ACTUALLY DISPATCHED via Task tool (drl-expert, pytorch-expert)
- [x] Specialist insights cited in Tribal Knowledge with attribution
- [x] All files >500 lines had dedicated deep-dives
- [x] Tensor shapes verified against code (checked network.py, rollout_buffer.py)
- [x] State machines cover episode lifecycle and reward attribution
- [x] Tribal Knowledge has ≥5 entries with file:line references
- [x] Performance Cliffs explain WHY, not just WHAT
- [x] Critical observations present (gamma mismatch, ransomware defenses)
