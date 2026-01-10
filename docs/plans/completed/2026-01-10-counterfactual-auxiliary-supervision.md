# Counterfactual Auxiliary Supervision Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract more learning signal from counterfactual ablation by training an auxiliary head to predict seed contributions, improving sample efficiency and value function quality.

**Architecture:** Add a `ContributionPredictor` head to the value network that predicts per-slot counterfactual contributions. Train with auxiliary MSE loss using ground-truth contributions from validation ablation. The auxiliary loss provides self-supervised signal that improves representation learning without changing the action space.

**Tech Stack:** PyTorch nn.Module, existing CounterfactualEngine, PPO update loop

**Review Status:** v3 - All expert reviews complete (DRL + PyTorch + Python specialists)

**Expert Sign-offs:**
- ✅ DRL Expert: Core concept sound; critical fix for target propagation; recommends stop-gradient initially
- ✅ PyTorch Expert: Architecture correct; initialization and torch.compile fixes identified
- ✅ Python Expert: Type annotations and code organization fixes; no conflicts with above

---

## Expert Review Summary (2026-01-10)

### Critical Changes Required

| Issue | Severity | Resolution |
|-------|----------|------------|
| **Target propagation (last-known)** | HIGH | Use measurement-only supervision — only compute aux loss at timesteps with fresh counterfactual |
| **Gradient flow to LSTM** | MEDIUM | Start with `stop_gradient=True` from auxiliary head; don't affect representations initially |
| **Loss coefficient** | MEDIUM | Reduce from 0.1 → **0.05**; add warmup ramp over first 1000 steps |
| **Output initialization** | MEDIUM | Change from gain=0.01 → **gain=0.1** for regression targets |
| **Shortcut learning risk** | MEDIUM | Add **Dropout(0.1)** to auxiliary head |

### DRL Expert Key Insights

1. **Leaky credit assignment**: Contributions measure instantaneous causal effect; returns incorporate future consequences. Predicting contributions might create "myopic" representations. Mitigate by using stop-gradient initially.

2. **Stale targets are worse than noise**: Propagating 10-epoch-old contributions creates systematic bias (not random noise). Use measurement-only supervision.

3. **Representation collapse risk**: If aux task is "easier" than value prediction, LSTM may preferentially encode contribution-relevant features. Stop-gradient prevents this.

### PyTorch Expert Key Insights

1. **gain=0.01 too small**: For regression targets in [-10, +10], need gain=0.1 to allow faster output range expansion.

2. **torch.compile**: Always compute predictions (avoid `if return_contributions` conditional). Caller ignores unused tensor — no overhead.

3. **Memory overhead**: Negligible (~0.6% increase in rollout buffer).

### Python Expert Key Insights

1. **Return type mismatch**: `evaluate_actions()` currently returns a **tuple**, not a dict. Plan must match existing pattern or refactor consistently.

2. **Config pattern**: `PPOConfig` dataclass doesn't exist — agent uses individual `__init__` parameters. Add params to `PPOAgent.__init__()` to match existing pattern.

3. **Rollout buffer location**: Plan references `RolloutTimestep` but actual implementation uses `TamiyoRolloutBuffer`. Locate correct file before implementing Phase 2.

4. **Backward compatibility**: Add defensive check for `has_fresh_contribution` key in batch dict (old rollouts won't have it).

5. **Type annotations**: Add tensor shape annotations to docstrings: `pred_contributions: torch.Tensor  # [batch, num_slots]`

---

## Problem Statement

### Current Data Flow

```
Training epoch:
  CNN trains on batch → val accuracy measured

Validation boundary (every 10 epochs):
  CounterfactualEngine.compute_matrix() → ablation over 2^N configs
  ↓
  seed_contribution[slot_k] = acc_with_all - acc_without_k
  ↓
  compute_contribution_reward() → scalar reward shaping
  ↓
  PPO update (only sees final reward, not contribution structure)
```

### The Waste

1. **Compute spent, signal discarded**: Counterfactual ablation is expensive (2^N forward passes). The rich structure (which seeds help, which hurt, interactions) is collapsed to scalar reward.

2. **Value function is blind**: The critic predicts V(s) but has no mechanism to understand *why* different states lead to different returns. It can't distinguish "state where seed A is contributing" from "state where seed B is contributing."

3. **Credit assignment is indirect**: The policy learns "FOSSILIZE good seed → high reward" but the value function doesn't learn "this seed is good" directly.

### The Opportunity

Counterfactual contributions are **ground truth** for "how much does each seed matter right now." This is exactly what the value function needs to understand to make good predictions. Use it as auxiliary supervision.

---

## Proposed Solution

### New Architecture

```
                    ┌─────────────────────────────────────┐
                    │         FactoredLSTMPolicy          │
                    ├─────────────────────────────────────┤
                    │  feature_net → LSTM → shared_repr   │
                    │              ↓                      │
                    │  ┌─────────────────────────────┐   │
                    │  │     Policy Heads (8)        │   │
                    │  │ op, slot, blueprint, ...    │   │
                    │  └─────────────────────────────┘   │
                    │              ↓                      │
                    │  ┌─────────────────────────────┐   │
                    │  │       Value Head            │   │
                    │  │  Q(s, op) → scalar          │   │
                    │  └─────────────────────────────┘   │
                    │              ↓                      │
                    │  ┌─────────────────────────────┐   │  ← NEW
                    │  │  Contribution Predictor     │   │
                    │  │  shared_repr → [num_slots]  │   │
                    │  │  Predicts: "how much does   │   │
                    │  │   each seed contribute?"    │   │
                    │  └─────────────────────────────┘   │
                    └─────────────────────────────────────┘
```

### Data Flow with Auxiliary Supervision

```
Training epoch:
  CNN trains on batch → val accuracy measured

Validation boundary:
  CounterfactualEngine.compute_matrix() → ablation
  ↓
  seed_contribution[slot_k] = ground truth contribution
  ↓
  Store in RolloutBuffer: contributions_target[t] = [c_0, c_1, ..., c_N]

PPO Update:
  Forward pass:
    value, pred_contributions = policy.evaluate_with_contributions(...)

  Losses:
    policy_loss = PPO clipped surrogate
    value_loss = MSE(value, returns)
    aux_loss = MSE(pred_contributions, contributions_target)  ← NEW

  total_loss = policy_loss + value_coef * value_loss + aux_coef * aux_loss
```

### Why This Works

1. **Representation pressure**: The shared LSTM features must now encode "what are the active seeds doing?" to minimize aux_loss. This improves features for both policy and value.

2. **Direct credit signal**: Instead of learning "this state → high return" and inferring seed credit, the network directly learns "seed A contributes +2.3%, seed B contributes -0.5%."

3. **Temporal consistency**: Contributions are computed at validation boundaries but propagated to all timesteps since last validation. This provides dense supervision even though ground truth is sparse.

4. **No action space change**: This is purely additive — doesn't change masking, doesn't change reward structure, doesn't change policy gradient.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Auxiliary head vs multi-head value | Auxiliary head | Simpler, doesn't change value semantics |
| Loss type | MSE (clipped) | Contributions are continuous; Huber as backup if outliers dominate |
| Target propagation | **Measurement-only** | ~~Last-known~~ creates systematic bias; only supervise at fresh timesteps |
| Slot masking | Active slots only | Don't penalize predictions for empty slots |
| Gradient flow | **Stop-grad to LSTM** | Prevent representation collapse; aux head learns to decode, not shape features |
| Initialization | **gain=0.1** | ~~gain=0.01~~ too small for regression targets in [-10, +10] |
| Regularization | **Dropout(0.1)** in aux head | Prevent shortcut learning / memorization |

### Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `aux_contribution_coef` | **0.05** | 0.01–0.5 | ~~0.1~~ too high given existing loss complexity |
| `aux_warmup_steps` | 1000 | 500–2000 | Ramp coefficient from 0 → full over N steps |
| `contribution_loss_clip` | 10.0 | 5.0–20.0 | Prevent outlier contributions from dominating |
| `aux_stop_gradient` | **True** | — | Stop gradients from aux head to LSTM (initially) |

---

## Implementation Phases

### Phase 1: Add ContributionPredictor Head

Add the auxiliary head to FactoredLSTMPolicy without changing training.

### Phase 2: Store Contribution Targets in Rollout

Capture ground-truth contributions from counterfactual and propagate to rollout buffer.

### Phase 3: Compute Auxiliary Loss in PPO Update

Add the auxiliary loss term to the PPO update loop.

### Phase 4: Add Telemetry and Diagnostics

Track auxiliary loss, prediction accuracy, and correlation with actual contributions.

---

## Phase 1: Add ContributionPredictor Head

### Task 1.1: Add contribution_predictor to FactoredLSTMPolicy

**Files:**
- Modify: `src/esper/tamiyo/networks/factored_lstm.py:287-315`
- Test: `tests/tamiyo/networks/test_factored_lstm.py`

**Step 1: Write the failing test**

Add to `tests/tamiyo/networks/test_factored_lstm.py`:

```python
def test_policy_has_contribution_predictor():
    """FactoredLSTMPolicy has auxiliary contribution predictor head."""
    from esper.tamiyo.networks.factored_lstm import FactoredLSTMPolicy
    from esper.leyline.slot_config import SlotConfig

    slot_config = SlotConfig.default()
    policy = FactoredLSTMPolicy(
        state_dim=128,
        num_slots=len(slot_config.slot_ids),
        num_ops=6,
        num_blueprints=8,
        num_styles=3,
        num_tempos=5,
        num_alpha_targets=11,
        num_alpha_speeds=5,
        num_alpha_curves=3,
    )

    assert hasattr(policy, 'contribution_predictor')
    assert policy.contribution_predictor is not None


def test_contribution_predictor_output_shape():
    """Contribution predictor outputs [batch, num_slots]."""
    import torch
    from esper.tamiyo.networks.factored_lstm import FactoredLSTMPolicy
    from esper.leyline.slot_config import SlotConfig

    slot_config = SlotConfig.default()
    num_slots = len(slot_config.slot_ids)

    policy = FactoredLSTMPolicy(
        state_dim=128,
        num_slots=num_slots,
        num_ops=6,
        num_blueprints=8,
        num_styles=3,
        num_tempos=5,
        num_alpha_targets=11,
        num_alpha_speeds=5,
        num_alpha_curves=3,
    )

    batch_size = 4
    features = torch.randn(batch_size, 128)
    hidden = policy.get_initial_hidden(batch_size, torch.device('cpu'))

    # Forward to get LSTM output
    lstm_out, _ = policy.lstm(policy.feature_net(features).unsqueeze(0), hidden)
    lstm_out = lstm_out.squeeze(0)

    # Predict contributions
    pred_contributions = policy.contribution_predictor(lstm_out)

    assert pred_contributions.shape == (batch_size, num_slots)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/tamiyo/networks/test_factored_lstm.py::test_policy_has_contribution_predictor -v
```

Expected: FAIL with `AttributeError: 'FactoredLSTMPolicy' object has no attribute 'contribution_predictor'`

**Step 3: Write minimal implementation**

In `src/esper/tamiyo/networks/factored_lstm.py`, add after the value_head definition (around line 314):

```python
        # Auxiliary contribution predictor head
        # Predicts per-slot counterfactual contributions for auxiliary supervision.
        # Input: lstm_out (lstm_hidden_dim)
        # Output: [num_slots] predicted contributions (unbounded, can be negative)
        #
        # Architecture: 3-layer MLP with dropout to prevent shortcut learning.
        # We don't condition on op because contributions are state properties, not action-dependent.
        #
        # DRL Expert review: Add Dropout(0.1) to prevent memorization/shortcut learning.
        # PyTorch Expert review: Keep separate from value head to avoid gradient interference.
        self.contribution_predictor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),  # DRL Expert: prevent shortcut learning
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(),
            nn.Linear(head_hidden // 2, num_slots),
        )
```

Also add `num_slots` to `__init__` parameters and store it:

```python
def __init__(
    self,
    state_dim: int,
    num_slots: int,  # NEW
    num_ops: int,
    ...
):
    ...
    self.num_slots = num_slots
```

Update `_init_weights` to initialize contribution predictor output with **gain=0.1** (not 0.01):

```python
        # Contribution predictor output: gain=0.1 for regression targets
        # PyTorch Expert review: gain=0.01 is too small for targets in [-10, +10];
        # gain=0.1 allows faster initial range expansion while staying centered.
        contrib_last = self.contribution_predictor[-1]
        if isinstance(contrib_last, nn.Linear):
            nn.init.orthogonal_(contrib_last.weight.data, gain=0.1)
            nn.init.zeros_(contrib_last.bias.data)  # Center predictions at 0
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/tamiyo/networks/test_factored_lstm.py -k "contribution" -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/tamiyo/networks/factored_lstm.py tests/tamiyo/networks/test_factored_lstm.py
git commit -m "feat(tamiyo): add contribution_predictor auxiliary head

Predicts per-slot counterfactual contributions from LSTM features.
Will be used for auxiliary supervision in PPO update.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.2: Add predict_contributions method

**Files:**
- Modify: `src/esper/tamiyo/networks/factored_lstm.py`
- Test: `tests/tamiyo/networks/test_factored_lstm.py`

**Step 1: Write the failing test**

```python
def test_predict_contributions_method():
    """predict_contributions() returns predictions from stored LSTM output."""
    import torch
    from esper.tamiyo.networks.factored_lstm import FactoredLSTMPolicy

    policy = FactoredLSTMPolicy(
        state_dim=128,
        num_slots=6,
        num_ops=6,
        num_blueprints=8,
        num_styles=3,
        num_tempos=5,
        num_alpha_targets=11,
        num_alpha_speeds=5,
        num_alpha_curves=3,
    )

    batch_size = 4
    features = torch.randn(batch_size, 128)
    hidden = policy.get_initial_hidden(batch_size, torch.device('cpu'))

    # Full forward pass
    action_dict, log_probs, value, new_hidden = policy.get_action(
        features, hidden, action_masks={}, deterministic=False
    )

    # Predict contributions (should use cached LSTM output)
    pred = policy.predict_contributions(features, hidden)

    assert pred.shape == (batch_size, 6)
    assert pred.requires_grad  # Should be differentiable
```

**Step 2: Run test**

```bash
uv run pytest tests/tamiyo/networks/test_factored_lstm.py::test_predict_contributions_method -v
```

**Step 3: Implement**

Add method to FactoredLSTMPolicy:

```python
def predict_contributions(
    self,
    features: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor],
    stop_gradient: bool = True,  # DRL Expert: default True to prevent representation collapse
) -> torch.Tensor:
    """Predict per-slot counterfactual contributions.

    Args:
        features: Input features [batch, state_dim]
        hidden: LSTM hidden state tuple (h, c)
        stop_gradient: If True, detach LSTM output before prediction (prevents
            auxiliary gradients from affecting shared representations).
            DRL Expert recommendation: Start with True to avoid representation
            collapse; can experiment with False later.

    Returns:
        Predicted contributions [batch, num_slots]
    """
    # Process through feature net and LSTM (same as forward path)
    x = self.feature_net(features)
    x = x.unsqueeze(0)  # Add sequence dim
    lstm_out, _ = self.lstm(x, hidden)
    lstm_out = lstm_out.squeeze(0)  # Remove sequence dim

    # DRL Expert: Stop gradient to prevent aux task from shaping LSTM features
    if stop_gradient:
        lstm_out = lstm_out.detach()

    return self.contribution_predictor(lstm_out)
```

**Step 4: Run test**

```bash
uv run pytest tests/tamiyo/networks/test_factored_lstm.py::test_predict_contributions_method -v
```

**Step 5: Commit**

```bash
git add src/esper/tamiyo/networks/factored_lstm.py tests/tamiyo/networks/test_factored_lstm.py
git commit -m "feat(tamiyo): add predict_contributions() method

Exposes contribution predictions for auxiliary loss computation.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.3: Add contributions to evaluate_actions return

**Files:**
- Modify: `src/esper/tamiyo/networks/factored_lstm.py`
- Test: `tests/tamiyo/networks/test_factored_lstm.py`

**Step 1: Write the failing test**

```python
def test_evaluate_actions_returns_contributions():
    """evaluate_actions() can optionally return contribution predictions."""
    import torch
    from esper.tamiyo.networks.factored_lstm import FactoredLSTMPolicy

    policy = FactoredLSTMPolicy(
        state_dim=128,
        num_slots=6,
        num_ops=6,
        num_blueprints=8,
        num_styles=3,
        num_tempos=5,
        num_alpha_targets=11,
        num_alpha_speeds=5,
        num_alpha_curves=3,
    )

    batch_size = 4
    seq_len = 10

    # Create dummy rollout data
    features = torch.randn(batch_size, seq_len, 128)
    actions = {
        "op": torch.randint(0, 6, (batch_size, seq_len)),
        "slot": torch.randint(0, 6, (batch_size, seq_len)),
        "blueprint": torch.randint(0, 8, (batch_size, seq_len)),
        "style": torch.randint(0, 3, (batch_size, seq_len)),
        "tempo": torch.randint(0, 5, (batch_size, seq_len)),
        "alpha_target": torch.randint(0, 11, (batch_size, seq_len)),
        "alpha_speed": torch.randint(0, 5, (batch_size, seq_len)),
        "alpha_curve": torch.randint(0, 3, (batch_size, seq_len)),
    }
    masks = {k: torch.ones(batch_size, seq_len, v, dtype=torch.bool)
             for k, v in [("op", 6), ("slot", 6), ("blueprint", 8),
                          ("style", 3), ("tempo", 5), ("alpha_target", 11),
                          ("alpha_speed", 5), ("alpha_curve", 3)]}

    result = policy.evaluate_actions(
        features, actions, masks,
        return_contributions=True,  # NEW parameter
    )

    assert "pred_contributions" in result
    assert result["pred_contributions"].shape == (batch_size, seq_len, 6)
```

**Step 2-5: Implement, test, commit**

**PyTorch Expert review**: Always compute contribution predictions (avoid conditional code paths for torch.compile compatibility). Caller ignores unused tensor — no overhead.

**Python Expert review**: `evaluate_actions()` currently returns a **tuple**, not a dict. Match existing pattern.

Modify `evaluate_actions()` to include predictions in return tuple:

```python
def evaluate_actions(
    self,
    features: torch.Tensor,  # [batch, seq, state_dim]
    actions: dict[str, torch.Tensor],  # [batch, seq] per head
    masks: dict[str, torch.Tensor],  # [batch, seq, num_actions] per head
    # ... existing params ...
    aux_stop_gradient: bool = True,  # NEW: control gradient flow
) -> tuple[
    dict[str, torch.Tensor],  # log_probs
    torch.Tensor,             # values [batch, seq]
    dict[str, torch.Tensor],  # entropy
    torch.Tensor,             # pred_contributions [batch, seq, num_slots]  # NEW
]:
    # ... existing evaluation code ...

    # Always compute contribution predictions (torch.compile friendly)
    # PyTorch Expert: Avoid conditional `if return_contributions` — creates graph breaks
    lstm_out_for_aux = lstm_out.detach() if aux_stop_gradient else lstm_out
    pred_contributions = self.contribution_predictor(lstm_out_for_aux)

    return log_probs, values, entropy, pred_contributions  # Tuple, not dict
```

**Caller update** (in PPO update loop):
```python
# OLD: log_probs, values, entropy = policy.evaluate_actions(...)
# NEW: Unpack additional return value
log_probs, values, entropy, pred_contributions = policy.evaluate_actions(
    features, actions, masks,
    aux_stop_gradient=config.aux_stop_gradient,
)
```

---

## Phase 2: Store Contribution Targets in Rollout

### Task 2.1: Add contribution_targets field to rollout storage

**Files:**
- Modify: `src/esper/simic/agent/ppo_agent.py` (or wherever RolloutBuffer is defined)
- Test: `tests/simic/agent/test_ppo_agent.py`

The rollout buffer needs to store ground-truth contributions alongside other timestep data.

**Key considerations (DRL Expert critical fix):**
- ~~Between measurements, use last-known values~~ **NO** — only supervise at measurement timesteps
- Track `has_fresh_contribution` boolean per timestep — only compute aux loss when True
- This avoids systematic bias from stale targets (worse than random noise)

**Data structure:**

```python
@dataclass
class RolloutTimestep:
    # ... existing fields ...

    # Auxiliary supervision targets (ground truth from counterfactual)
    # DRL Expert: Only valid at measurement timesteps (has_fresh_contribution=True)
    contribution_targets: torch.Tensor | None = None  # [num_slots]
    contribution_mask: torch.Tensor | None = None     # [num_slots] bool - which slots are active
    has_fresh_contribution: bool = False              # True only at counterfactual measurement timesteps
```

---

### Task 2.2: Propagate contributions from counterfactual to rollout

**Files:**
- Modify: `src/esper/simic/training/vectorized_trainer.py`
- Modify: `src/esper/simic/agent/ppo_agent.py`

**DRL Expert critical fix**: Only set `has_fresh_contribution=True` at measurement timesteps:

```python
# In vectorized_trainer.py, at counterfactual measurement boundary:
contribution_targets = torch.tensor([
    seed_contributions.get(slot_id, 0.0)
    for slot_id in slot_config.slot_ids
])
contribution_mask = torch.tensor([
    slot_id in active_slots
    for slot_id in slot_config.slot_ids
])
has_fresh_contribution = True  # Only True at measurement time!

# When storing rollout timestep (non-measurement epochs):
# Leave contribution_targets as None or zeros
# has_fresh_contribution = False
```

---

## Phase 3: Compute Auxiliary Loss in PPO Update

### Task 3.1: Add auxiliary loss computation

**Files:**
- Modify: `src/esper/simic/agent/ppo_update.py`
- Test: `tests/simic/agent/test_ppo_update.py`

**Step 1: Write the failing test**

```python
def test_compute_contribution_auxiliary_loss():
    """Auxiliary loss computes MSE between predictions and targets."""
    import torch
    from esper.simic.agent.ppo_update import compute_contribution_aux_loss

    batch_size = 32
    num_slots = 6

    pred = torch.randn(batch_size, num_slots)
    targets = torch.randn(batch_size, num_slots)
    mask = torch.ones(batch_size, num_slots, dtype=torch.bool)
    mask[:, 4:] = False  # Last 2 slots inactive

    loss = compute_contribution_aux_loss(pred, targets, mask)

    assert loss.shape == ()  # Scalar
    assert loss >= 0  # MSE is non-negative

    # Loss should only consider active slots
    pred_active = pred[:, :4]
    targets_active = targets[:, :4]
    expected = ((pred_active - targets_active) ** 2).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


def test_contribution_aux_loss_with_clip():
    """Large target values are clipped to prevent outlier domination."""
    import torch
    from esper.simic.agent.ppo_update import compute_contribution_aux_loss

    pred = torch.zeros(4, 6)
    targets = torch.zeros(4, 6)
    targets[0, 0] = 100.0  # Outlier
    mask = torch.ones(4, 6, dtype=torch.bool)

    loss_unclipped = compute_contribution_aux_loss(pred, targets, mask, clip=None)
    loss_clipped = compute_contribution_aux_loss(pred, targets, mask, clip=10.0)

    assert loss_clipped < loss_unclipped
```

**Step 2: Implement**

```python
def compute_contribution_aux_loss(
    pred_contributions: torch.Tensor,
    target_contributions: torch.Tensor,
    active_mask: torch.Tensor,
    has_fresh_target: torch.Tensor,  # DRL Expert: measurement-only supervision
    clip: float | None = 10.0,
) -> torch.Tensor:
    """Compute auxiliary MSE loss for contribution predictions.

    DRL Expert critical fix: Only compute loss on timesteps with fresh counterfactual
    measurements. Stale targets create systematic bias, not random noise.

    Args:
        pred_contributions: Predicted contributions [batch, num_slots]
        target_contributions: Ground truth from counterfactual [batch, num_slots]
        active_mask: Boolean mask for active slots [batch, num_slots]
        has_fresh_target: Boolean mask for timesteps with fresh measurements [batch]
        clip: Optional clip value for targets (prevents outlier domination)

    Returns:
        Scalar MSE loss over active slots at measurement timesteps only.
        Returns 0.0 if no fresh measurements in batch.
    """
    # DRL Expert: Only supervise at measurement timesteps
    if not has_fresh_target.any():
        return torch.tensor(0.0, device=pred_contributions.device)

    if clip is not None:
        target_contributions = target_contributions.clamp(-clip, clip)

    # Combine active slot mask with fresh measurement mask
    # has_fresh_target: [batch] -> [batch, 1] for broadcasting
    combined_mask = active_mask & has_fresh_target.unsqueeze(-1)

    # MSE only over active slots at measurement timesteps
    sq_error = (pred_contributions - target_contributions) ** 2
    masked_error = sq_error * combined_mask.float()

    n_valid = combined_mask.sum().clamp(min=1)
    return masked_error.sum() / n_valid
```

---

### Task 3.2: Integrate auxiliary loss into PPO update loop

**Files:**
- Modify: `src/esper/simic/agent/ppo_agent.py`

In the PPO update epoch loop:

```python
# Forward pass with contribution predictions (always computed for torch.compile)
# Python Expert: evaluate_actions returns tuple, not dict
log_probs, values, entropy, pred_contributions = policy.evaluate_actions(
    features, actions, masks,
    aux_stop_gradient=self.aux_stop_gradient,  # DRL Expert: True by default
)

# Python Expert: Backward compatibility check for old rollouts
if "has_fresh_contribution" not in batch:
    logger.debug("Rollout missing has_fresh_contribution - skipping aux loss")
    aux_loss = torch.tensor(0.0, device=values.device)
else:
    # Compute auxiliary loss (measurement-only supervision)
    aux_loss = compute_contribution_aux_loss(
        pred_contributions,
        batch["contribution_targets"],
        batch["contribution_mask"],
        batch["has_fresh_contribution"],  # DRL Expert: only supervise at fresh measurements
        clip=self.contribution_loss_clip,
    )

# Warmup coefficient (DRL Expert + PyTorch Expert recommendation)
warmup_progress = min(1.0, self._training_step / self.aux_warmup_steps)
effective_aux_coef = self.aux_contribution_coef * warmup_progress

# Add to total loss
total_loss = (
    policy_loss
    + self.value_coef * value_loss
    + self.entropy_coef * entropy_loss
    + effective_aux_coef * aux_loss  # NEW: with warmup
)
```

---

## Phase 4: Telemetry and Diagnostics

### Task 4.1: Add auxiliary loss to telemetry

Track in PPO update telemetry (DRL Expert + PyTorch Expert recommendations):

```python
# Core metrics
"aux_contribution_loss": aux_loss.item(),
"aux_contribution_coef_effective": effective_aux_coef,  # Shows warmup progress

# Prediction quality (DRL Expert: monitor for collapse)
with torch.no_grad():
    valid_mask = batch["has_fresh_contribution"]
    if valid_mask.any():
        pred_valid = pred_contributions[valid_mask]
        target_valid = batch["contribution_targets"][valid_mask]

        # Variance (should NOT be ~0 — indicates collapse)
        pred_variance = pred_valid.var().item()

        # Explained variance (should increase over training)
        target_var = target_valid.var().clamp(min=0.01)
        residual_var = (pred_valid - target_valid).var()
        explained_var = 1.0 - (residual_var / target_var).item()

        # Correlation (should be > 0.5 eventually)
        if pred_valid.numel() > 2:
            corr = torch.corrcoef(
                torch.stack([pred_valid.flatten(), target_valid.flatten()])
            )[0, 1].item()
        else:
            corr = 0.0

# Emit telemetry
"aux_pred_variance": pred_variance,  # DRL Expert: warn if < 0.01
"aux_explained_variance": explained_var,
"aux_pred_target_correlation": corr,  # DRL Expert: warn if < 0.2 after 1000 steps
```

### Task 4.2: Add diagnostic visualization

In Karn TUI or Overwatch:
- Per-slot contribution prediction vs actual (scatter plot)
- Prediction error over time (should decrease as network learns)
- `aux_pred_variance` over time (should NOT collapse to 0)
- `aux_pred_target_correlation` over time (should increase toward 0.6-0.8)

### Task 4.3: Add collapse detection warnings

```python
# DRL Expert: Detect prediction collapse
if training_step > 1000:
    if pred_variance < 0.01:
        logger.warning(
            "Contribution predictor may have collapsed (variance=%.4f). "
            "Consider increasing aux_contribution_coef or disabling stop_gradient.",
            pred_variance,
        )
    if corr < 0.2:
        logger.warning(
            "Contribution predictor correlation low (%.3f). "
            "Aux task may not be learning.",
            corr,
        )
```

---

## Configuration

**Python Expert note**: `PPOConfig` dataclass doesn't exist — agent uses individual `__init__` parameters. Add to `PPOAgent.__init__()`:

Add to `src/esper/simic/agent/ppo_agent.py` in `PPOAgent.__init__()`:

```python
class PPOAgent:
    def __init__(
        self,
        # ... existing parameters ...

        # Auxiliary contribution supervision (Expert-reviewed defaults)
        aux_contribution_coef: float = 0.05,  # DRL Expert: reduced from 0.1
        aux_warmup_steps: int = 1000,  # DRL + PyTorch Expert: ramp from 0 → full
        aux_stop_gradient: bool = True,  # DRL Expert: prevent representation collapse
        contribution_loss_clip: float = 10.0,  # Clip targets to prevent outliers
        enable_contribution_aux: bool = True,  # Can disable for ablation
    ):
        # ... existing init ...

        # Store aux supervision config
        self.aux_contribution_coef = aux_contribution_coef
        self.aux_warmup_steps = aux_warmup_steps
        self.aux_stop_gradient = aux_stop_gradient
        self.contribution_loss_clip = contribution_loss_clip
        self.enable_contribution_aux = enable_contribution_aux
```

---

## Verification Plan

### Unit Tests

```bash
uv run pytest tests/tamiyo/networks/test_factored_lstm.py -k "contribution" -v
uv run pytest tests/simic/agent/test_ppo_update.py -k "contribution" -v
```

### Integration Test

```bash
# Short training run to verify no crashes
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
    --preset cifar10 \
    --task cifar10 \
    --rounds 10 \
    --envs 2 \
    --episode-length 20 \
    --no-tui
```

### Metrics to Monitor

After implementation, verify:

1. **aux_contribution_loss decreases** over training (network is learning)
2. **Value explained_variance improves** (auxiliary task helps representation)
3. **Policy entropy stable** (auxiliary task doesn't destabilize policy)

### Telemetry Query

```sql
SELECT
    batch,
    aux_contribution_loss,
    value_loss,
    explained_variance
FROM ppo_updates
WHERE run_dir = '<new_run>'
ORDER BY batch
```

---

## Risks and Mitigations (Expert-Reviewed)

| Risk | Mitigation | Source |
|------|------------|--------|
| **Representation collapse** | Stop-gradient from aux head (default True) | DRL Expert |
| **Stale targets create bias** | Measurement-only supervision (has_fresh_contribution mask) | DRL Expert |
| Auxiliary loss destabilizes training | Start with coef=0.05 + warmup over 1000 steps | DRL + PyTorch |
| Shortcut learning | Dropout(0.1) in aux head | DRL Expert |
| Output range mismatch | gain=0.1 init (not 0.01) | PyTorch Expert |
| Outlier contributions dominate | Clip targets to ±10.0 | Both |
| Prediction collapse | Telemetry warnings for variance < 0.01 | DRL Expert |
| torch.compile graph breaks | Always compute predictions (no conditionals) | PyTorch Expert |
| Increased memory | Negligible (~0.6% overhead) | PyTorch Expert |

---

## Future Extensions

1. **Temporal prediction**: Predict contribution N epochs in future (harder but more useful)
2. **Interaction prediction**: Predict pairwise seed interactions (synergy/interference)
3. **Multi-step targets**: Use N-step contribution targets like N-step returns
4. **Curriculum**: Start with high aux_coef early, decay as policy improves

---

## Summary

This plan adds auxiliary supervision from counterfactual ablation:

1. **Phase 1**: Add ContributionPredictor head to policy network (with Dropout, gain=0.1 init)
2. **Phase 2**: Store ground-truth contributions in rollout buffer (measurement-only, not propagated)
3. **Phase 3**: Compute auxiliary MSE loss in PPO update (stop-gradient, warmup, measurement-only)
4. **Phase 4**: Add telemetry for monitoring (collapse detection, correlation tracking)

**Key benefit**: Extract more learning signal from compute you're already paying for (counterfactual ablation), improving sample efficiency and representation quality without changing the action space or reward structure.

**Implementation complexity**: Medium - touches network architecture, rollout storage, and PPO update, but changes are additive and well-isolated.

**Expert-reviewed safeguards:**
- Stop-gradient prevents representation collapse
- Measurement-only supervision avoids stale target bias
- Warmup prevents early training disruption
- Telemetry detects prediction collapse early
