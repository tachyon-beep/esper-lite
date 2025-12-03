# Design Spec: LSTM Policy Recurrence for Tamiyo

## Status: Proposed

## Motivation: Why We Need This

### The Problem

The current `ActorCritic` network is a feedforward MLP that processes each observation independently. This creates a fundamental limitation for the meta-learning task:

**Tamiyo cannot learn temporal patterns in training dynamics.**

Consider what information is lost:
1. **Trend direction ambiguity**: A 65% accuracy with `acc_delta=+0.5%` could be:
   - Early training (trend: rapid improvement ahead)
   - Late plateau (trend: diminishing returns)
   - Recovery after a dip (trend: unstable)

   The MLP sees identical features for all three cases.

2. **Seed history blindness**: When deciding whether to CULL, the agent cannot see:
   - How quickly the seed ramped up
   - Whether improvement is accelerating or decelerating
   - If there were temporary dips that recovered

3. **Credit assignment over lifecycle**: A seed germinated at epoch 5 may not fossilize until epoch 20. The agent must learn patterns like "seeds that improve quickly in TRAINING tend to succeed in BLENDING" - this requires memory.

### Evidence from DRL Expert Review

> "For a meta-learning task where the agent should learn from training dynamics patterns, a recurrent architecture (LSTM/GRU) or transformer might be beneficial to capture longer-term patterns in training dynamics."

> "The fundamental credit assignment problem over 15+ timesteps remains challenging."

### What LSTM Enables

1. **Implicit trend detection**: LSTM can learn to track whether metrics are accelerating, decelerating, or oscillating without explicit feature engineering.

2. **Seed lifecycle patterns**: The hidden state can encode "this seed started slow but is accelerating" vs "this seed peaked early and is declining."

3. **Adaptive patience**: Rather than fixed MIN_CULL_AGE, the agent can learn context-dependent patience based on observed trajectory.

4. **Better exploration**: With memory of past actions and outcomes within an episode, the agent can avoid repeating failed strategies.

---

## Architecture Design

### New Class: `RecurrentActorCritic`

```python
class RecurrentActorCritic(nn.Module):
    """Actor-Critic with LSTM for temporal pattern learning.

    Architecture:
        state -> Linear(state_dim, hidden_dim) -> ReLU
              -> LSTM(hidden_dim, lstm_hidden_dim)
              -> Actor head -> action logits
              -> Critic head -> value

    Hidden state (h, c) persists across timesteps within an episode,
    enabling the policy to learn temporal patterns in training dynamics.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 1,
    ):
        super().__init__()

        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers

        # Feature encoder (same as MLP version)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim // 2, action_dim),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim // 2, 1),
        )

    def get_initial_hidden(self, batch_size: int, device: torch.device):
        """Return zero-initialized hidden state for new episodes."""
        h = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
        c = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
        return (h, c)

    def forward(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[MaskedCategorical, torch.Tensor, tuple]:
        """Forward pass with hidden state.

        Args:
            state: [batch, state_dim] or [batch, seq_len, state_dim]
            action_mask: [batch, action_dim] or [batch, seq_len, action_dim]
            hidden: (h, c) tuple or None for fresh episode

        Returns:
            dist: MaskedCategorical over actions
            value: State value estimate
            hidden: Updated (h, c) for next timestep
        """
        batch_size = state.size(0)

        # Initialize hidden if not provided
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, state.device)

        # Handle both single-step and sequence inputs
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]
            action_mask = action_mask.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        # Encode features
        features = self.encoder(state)  # [batch, seq_len, hidden_dim]

        # LSTM processing
        lstm_out, hidden = self.lstm(features, hidden)  # [batch, seq_len, lstm_hidden]

        # Get logits and values
        logits = self.actor(lstm_out)
        values = self.critic(lstm_out).squeeze(-1)

        if squeeze_output:
            logits = logits.squeeze(1)
            values = values.squeeze(1)
            action_mask = action_mask.squeeze(1)

        dist = MaskedCategorical(logits=logits, mask=action_mask)
        return dist, values, hidden
```

### Hidden State in RolloutBuffer

The buffer must store hidden states to enable proper BPTT during PPO updates:

```python
@dataclass
class RecurrentRolloutStep:
    """Single step with hidden state for recurrent policies."""
    state: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    action_mask: torch.Tensor
    hidden_h: torch.Tensor  # LSTM hidden state
    hidden_c: torch.Tensor  # LSTM cell state
```

### Truncated BPTT for PPO Updates

Full backprop through entire episodes is expensive. Use truncated BPTT:

```python
def compute_loss_recurrent(
    self,
    states: torch.Tensor,      # [batch, seq_len, state_dim]
    actions: torch.Tensor,      # [batch, seq_len]
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    action_masks: torch.Tensor,
    initial_hidden: tuple,
    truncation_length: int = 8,  # BPTT truncation
):
    """PPO loss with truncated BPTT for recurrent policy."""
    # Process in chunks to limit gradient flow
    total_loss = 0
    hidden = initial_hidden

    for t in range(0, seq_len, truncation_length):
        chunk_end = min(t + truncation_length, seq_len)

        # Detach hidden at chunk boundaries (truncated BPTT)
        if t > 0:
            hidden = (hidden[0].detach(), hidden[1].detach())

        # Forward through chunk
        dist, values, hidden = self.network(
            states[:, t:chunk_end],
            action_masks[:, t:chunk_end],
            hidden,
        )

        # Compute PPO losses for this chunk
        # ... (standard PPO loss computation)

        total_loss += chunk_loss

    return total_loss / (seq_len // truncation_length)
```

---

## Integration Points

### 1. vectorized.py Changes

```python
# Per-environment hidden state tracking
@dataclass
class ParallelEnvState:
    # ... existing fields ...
    lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None = None

# During rollout collection
for epoch in range(1, max_epochs + 1):
    # Get batched actions WITH hidden state
    actions, log_probs, values, new_hiddens = agent.network.get_action_batch(
        states_batch_normalized,
        masks_batch,
        hiddens_batch,  # Pass current hidden states
    )

    # Update per-env hidden states
    for i, env_state in enumerate(env_states):
        env_state.lstm_hidden = (
            new_hiddens[0][:, i:i+1, :],
            new_hiddens[1][:, i:i+1, :],
        )

    # Store hidden in transition
    agent.store_transition(
        state, action, log_prob, value, reward, done, mask,
        hidden_h=env_state.lstm_hidden[0],
        hidden_c=env_state.lstm_hidden[1],
    )

# Reset hidden at episode boundaries
if done:
    env_state.lstm_hidden = None  # Will reinitialize on next step
```

### 2. ppo.py Changes

```python
class PPOAgent:
    def __init__(self, ..., recurrent: bool = False, lstm_hidden_dim: int = 128):
        if recurrent:
            self.network = RecurrentActorCritic(
                state_dim, action_dim, lstm_hidden_dim=lstm_hidden_dim
            )
        else:
            self.network = ActorCritic(state_dim, action_dim)
        self.recurrent = recurrent

    def update(self, ...):
        if self.recurrent:
            return self._update_recurrent(...)
        else:
            return self._update_feedforward(...)
```

### 3. buffers.py Changes

```python
class RecurrentRolloutBuffer:
    """Buffer that preserves temporal ordering for LSTM training."""

    def get_sequences(self, seq_length: int = 8):
        """Yield sequences for truncated BPTT training."""
        # Group transitions by episode
        # Yield (states, actions, ..., initial_hidden) tuples
        # Each sequence is seq_length timesteps
```

---

## Hyperparameters

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| `lstm_hidden_dim` | 128 | Smaller than MLP hidden (256) since LSTM is more expressive |
| `num_lstm_layers` | 1 | Start simple; 2 layers rarely helps for this task scale |
| `truncation_length` | 8 | Balance between gradient flow and compute cost |
| `hidden_init` | zeros | Standard; learned init is an option but adds complexity |

---

## Risks and Mitigations

### Risk 1: Training Instability
LSTM gradients can explode/vanish over long sequences.

**Mitigation**:
- Truncated BPTT (seq_len=8)
- Gradient clipping (already have max_grad_norm=0.5)
- Layer normalization in LSTM (optional)

### Risk 2: Increased Complexity
More moving parts = more bugs.

**Mitigation**:
- Keep feedforward `ActorCritic` as default
- Add `recurrent=True` flag to enable LSTM
- Extensive testing before switching

### Risk 3: Slower Training
LSTM is sequential; can't parallelize across timesteps.

**Mitigation**:
- Small LSTM (128 hidden)
- Batch across environments (already done)
- Profile before/after to quantify impact

### Risk 4: Hidden State Staleness
In vectorized training, environments may be at different episode stages.

**Mitigation**:
- Per-environment hidden state tracking
- Reset hidden on episode boundaries
- Store hidden in buffer for proper BPTT

---

## Success Metrics

1. **Accuracy improvement**: Recurrent policy should achieve higher final accuracy than MLP baseline
2. **Faster learning**: Should require fewer episodes to reach same performance
3. **Better CULL decisions**: Measured by "cull precision" - % of culled seeds that were actually harmful
4. **Temporal awareness**: Ablation showing performance drops when hidden state is reset mid-episode

---

## Implementation Plan

### Phase 1: Core Architecture (networks.py)
- [ ] Implement `RecurrentActorCritic` class
- [ ] Add `get_initial_hidden()` method
- [ ] Update forward pass signature
- [ ] Add unit tests

### Phase 2: Buffer Support (buffers.py)
- [ ] Create `RecurrentRolloutStep` dataclass
- [ ] Implement `RecurrentRolloutBuffer` with sequence extraction
- [ ] Add hidden state storage/retrieval

### Phase 3: PPO Integration (ppo.py)
- [ ] Add `recurrent` flag to `PPOAgent`
- [ ] Implement `_update_recurrent()` with truncated BPTT
- [ ] Update `store_transition()` for hidden states

### Phase 4: Vectorized Training (vectorized.py)
- [ ] Add `lstm_hidden` to `ParallelEnvState`
- [ ] Update rollout loop for hidden state management
- [ ] Handle episode boundary resets

### Phase 5: Testing & Validation
- [ ] A/B test MLP vs LSTM on CIFAR-10 task
- [ ] Profile training speed impact
- [ ] Verify hidden state reset logic

---

## Alternatives Considered

### Transformer Policy
**Pros**: Better parallelization, attention over arbitrary history
**Cons**: More parameters, harder to implement, overkill for 25-epoch episodes

### Frame Stacking
**Pros**: Simple, no architectural changes
**Cons**: Fixed history window, scales poorly, doesn't capture variable-length patterns

### External Memory (NTM/DNC)
**Pros**: Explicit memory addressing
**Cons**: Massive complexity increase, research-grade stability

**Decision**: LSTM is the right balance of expressiveness and implementation complexity for this task.
