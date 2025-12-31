### Phase 4: Policy Network V2

**Goal:** Implement the new network architecture.

**Files:**

- `src/esper/tamiyo/networks/factored_lstm.py`

**Sub-phases:**

#### 4a. Replace FactoredRecurrentActorCritic with V2

Per clean replacement policy, delete the old class and create the new one:

- `feature_dim = 512`
- `lstm_hidden_dim = 512`
- `head_hidden = lstm_hidden_dim // 2 = 256`
- `BlueprintEmbedding` module
- 3-layer blueprint head
- Op-conditioned value head

**Note:** If you need to reference old code during implementation, use `git show HEAD~1:path/to/file`.

**⚠️ CRITICAL: Update `get_action()` for Op-Conditioned Value**

The current `get_action()` method returns value from the unconditioned value head. When implementing the op-conditioned critic:

1. Update `GetActionResult` dataclass to include `sampled_op: torch.Tensor`
2. Update `get_action()` to compute value via `_compute_value(lstm_out, sampled_op)`
3. Verify `vectorized.py` bootstrap code (around line 3212) uses the correctly conditioned value

This ensures bootstrap values at truncation are computed with the same Q(s,op) conditioning as rollout values.

#### 4b. Implement 3-Layer Blueprint Head

```python
self.blueprint_head = nn.Sequential(
    nn.Linear(lstm_hidden_dim, lstm_hidden_dim),  # 512 → 512
    nn.ReLU(),
    nn.Linear(lstm_hidden_dim, head_hidden),      # 512 → 256
    nn.ReLU(),
    nn.Linear(head_hidden, num_blueprints),       # 256 → 13
)

# Initialization: gain=0.01 for final layer, sqrt(2) for intermediate
def _init_blueprint_head(self):
    """Apply orthogonal init with appropriate gains to blueprint head."""
    layers = [m for m in self.blueprint_head if isinstance(m, nn.Linear)]
    for i, layer in enumerate(layers):
        if i == len(layers) - 1:  # Final layer
            nn.init.orthogonal_(layer.weight, gain=0.01)
        else:  # Intermediate layers
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
        nn.init.zeros_(layer.bias)
```

#### 4c. Implement Op-Conditioned Value Head

```python
self.value_head = nn.Sequential(
    nn.Linear(lstm_hidden_dim + NUM_OPS, head_hidden),  # 512+6 → 256
    nn.ReLU(),
    nn.Linear(head_hidden, 1),                          # 256 → 1
)

def _compute_value(self, lstm_out: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
    """Shared helper: compute Q(s, op) value conditioned on operation.

    Used by both forward() (with sampled op) and evaluate_actions() (with stored op).
    """
    op_one_hot = F.one_hot(op, num_classes=NUM_OPS).float()
    value_input = torch.cat([lstm_out, op_one_hot], dim=-1)
    return self.value_head(value_input).squeeze(-1)
```

#### 4d. Define Output Types

Use `NamedTuple` for strict typing and torch.jit compatibility:

```python
from typing import NamedTuple

class ForwardOutput(NamedTuple):
    """Output from forward() during rollout."""
    op_logits: torch.Tensor           # [batch, seq, NUM_OPS]
    slot_logits: torch.Tensor         # [batch, seq, num_slots]
    blueprint_logits: torch.Tensor    # [batch, seq, NUM_BLUEPRINTS]
    style_logits: torch.Tensor        # [batch, seq, num_styles]
    tempo_logits: torch.Tensor        # [batch, seq, num_tempos]
    alpha_target_logits: torch.Tensor # [batch, seq, num_targets]
    alpha_speed_logits: torch.Tensor  # [batch, seq, num_speeds]
    alpha_curve_logits: torch.Tensor  # [batch, seq, num_curves]
    value: torch.Tensor               # [batch, seq] - Q(s, sampled_op)
    sampled_op: torch.Tensor          # [batch, seq] - the op used for value conditioning
    hidden: tuple[torch.Tensor, torch.Tensor]  # LSTM state

class EvaluateOutput(NamedTuple):
    """Output from evaluate_actions() during PPO update."""
    log_probs: dict[str, torch.Tensor]  # Per-head log probs of stored actions
    entropy: dict[str, torch.Tensor]     # Per-head entropy
    value: torch.Tensor                  # [batch, seq] - Q(s, stored_op)
```

#### 4e. Implement forward() for Rollout

```python
def forward(
    self,
    state: torch.Tensor,              # [batch, seq, 121]
    blueprint_indices: torch.Tensor,  # [batch, seq, num_slots]
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    action_mask: torch.Tensor | None = None,
) -> ForwardOutput:
    """Rollout path: sample actions, compute Q(s, sampled_op).

    This is used during data collection. The value returned is conditioned
    on the sampled op and should be stored in the rollout buffer.
    """
    # ... feature processing and LSTM ...

    # Sample op from policy (with action masking)
    op_logits = self.op_head(lstm_out)
    # CRITICAL: Use MaskedCategorical, NOT raw Categorical
    # - Prevents sampling invalid/masked actions
    # - Computes entropy only over valid actions
    # - Uses -1e4 (not -inf) for numerical stability
    op_dist = MaskedCategorical(logits=op_logits, mask=op_mask)
    sampled_op = op_dist.sample()

    # Value conditioned on sampled op (what we store in buffer)
    value = self._compute_value(lstm_out, sampled_op)

    return ForwardOutput(
        op_logits=op_logits,
        # ... other logits ...
        value=value,
        sampled_op=sampled_op,
        hidden=new_hidden,
    )
```

#### 4f. Implement evaluate_actions() for PPO Update

```python
def evaluate_actions(
    self,
    state: torch.Tensor,              # [batch, seq, 121]
    blueprint_indices: torch.Tensor,  # [batch, seq, num_slots]
    actions: dict[str, torch.Tensor], # Stored actions from buffer
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    action_mask: torch.Tensor | None = None,
) -> EvaluateOutput:
    """PPO update path: compute log_probs, entropy, Q(s, stored_op).

    The value is conditioned on the STORED op (actions["op"]), ensuring
    consistency with what was stored during rollout.
    """
    # ... feature processing and LSTM (same as forward) ...

    # Compute distributions for all heads (with action masking)
    op_logits = self.op_head(lstm_out)
    # CRITICAL: Use MaskedCategorical, NOT raw Categorical
    # This ensures entropy is computed only over valid actions and
    # log_prob returns correct values for the stored action
    op_dist = MaskedCategorical(logits=op_logits, mask=action_mask["op"])

    # Log prob of the STORED action (not a fresh sample)
    stored_op = actions["op"]
    op_log_prob = op_dist.log_prob(stored_op)

    # Value conditioned on STORED op (must match what was stored)
    value = self._compute_value(lstm_out, stored_op)

    # Compute log_probs and entropy for all heads
    # MaskedCategorical.entropy() already excludes masked actions
    log_probs = {"op": op_log_prob, ...}
    entropy = {"op": op_dist.entropy(), ...}

    return EvaluateOutput(log_probs=log_probs, entropy=entropy, value=value)
```

#### 4g. Implement get_value() for Debugging (Optional)

**⚠️ WARNING: This method is NOT used in the main training loop.**

The primary use cases are handled by:
- `forward()` → samples op and returns `Q(s, sampled_op)` (rollout + bootstrap)
- `evaluate_actions()` → uses stored op and returns `Q(s, stored_op)` (PPO update)

This method exists only for debugging/monitoring scenarios where you want to evaluate a specific state-op pair's value without sampling actions.

```python
@torch.no_grad()
def get_value(
    self,
    state: torch.Tensor,              # [batch, 1, 121] - single step
    blueprint_indices: torch.Tensor,  # [batch, 1, num_slots]
    hidden: tuple[torch.Tensor, torch.Tensor],
    sampled_op: torch.Tensor,         # [batch] - op to condition on
) -> torch.Tensor:
    """Compute value for a given state-op pair without gradient tracking.

    ⚠️ DEBUG ONLY - Do NOT use for bootstrap at truncation!
    Use forward() instead (see Gotcha #2).

    This method is for debugging scenarios like:
    - Visualizing Q(s, op) landscape for different ops
    - Comparing value estimates between ops
    - Unit testing value head behavior
    """
    # ... feature processing and LSTM ...
    return self._compute_value(lstm_out, sampled_op)
```

**@torch.no_grad() rationale:** Prevents gradient accumulation during debugging calls.

**Design rationale:** We use hard one-hot conditioning in **both** rollout and PPO update:
- `forward()`: `Q(s, sampled_op)` — value stored matches the op sampled
- `evaluate_actions()`: `Q(s, stored_op)` — stored_op == sampled_op from rollout
- `get_value()`: `Q(s, sampled_op)` — for bootstrap at truncation

This ensures the value function trained matches what was stored, avoiding GAE mismatch.

**Validation:**

```bash
PYTHONPATH=src python -c "
import torch
from esper.tamiyo.networks.factored_lstm import (
    FactoredRecurrentActorCritic, ForwardOutput, EvaluateOutput
)
net = FactoredRecurrentActorCritic(state_dim=121, num_slots=3)
state = torch.randn(2, 5, 121)  # batch=2, seq=5
bp_idx = torch.randint(0, 13, (2, 5, 3))

# Test forward
out = net(state, bp_idx)
assert isinstance(out, ForwardOutput)
print(f'Op logits: {out.op_logits.shape}')  # [2, 5, 6]
print(f'Value: {out.value.shape}')          # [2, 5]
print(f'Sampled op: {out.sampled_op.shape}')  # [2, 5]

# Test evaluate_actions
actions = {'op': torch.randint(0, 6, (2, 5)), 'slot': torch.randint(0, 3, (2, 5)), ...}
eval_out = net.evaluate_actions(state, bp_idx, actions)
assert isinstance(eval_out, EvaluateOutput)
print(f'Value (eval): {eval_out.value.shape}')  # [2, 5]

print(f'Params: {sum(p.numel() for p in net.parameters())}')  # ~2.1M
"
```

---

