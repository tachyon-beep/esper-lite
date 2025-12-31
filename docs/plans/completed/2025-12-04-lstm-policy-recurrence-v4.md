# LSTM Policy Recurrence Implementation Plan (v4)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add LSTM recurrence to Tamiyo's ActorCritic network so the policy can learn temporal patterns in training dynamics (trends, acceleration, lifecycle history).

**Architecture:** Episode-aligned BPTT with chunk_length=25 (matches max_epochs default exactly). LSTM sits between feature encoder and actor/critic heads. Per-environment step storage to avoid interleaving corruption. Single PPO epoch to avoid hidden state drift (standard for recurrent PPO). Episodes are hard boundaries (never chunk across episodes). Batched chunk processing for GPU efficiency.

**Tech Stack:** PyTorch, existing PPO infrastructure (simic module)

**Key Reviewer Fixes (v3 -> v4):**
1. GAE wired through to chunks properly
2. Single PPO epoch default (but n_epochs>1 viable with PPO clipping - see note)
3. Per-env step lists (no interleaving corruption)
4. Consistent hidden shapes [num_layers, batch, hidden]
5. Batched recurrent forward (no per-env loops)
6. **chunk_length=25 matches max_epochs default** (no mid-episode chunking needed)
7. Value clipping in recurrent update (parity with feedforward)
8. No unauthorized hasattr usage
9. Device-aware fallback hidden initialization
10. **Explicit `.detach().clone()` for hidden state storage** (.clone() prevents view retention)
11. **Graceful handling of mid-episode chunking** (warning + zeros, not crash)
12. **TrainingConfig integration** with `__post_init__` validation
13. **Gradient flow test** verifies GAE bootstrap AND BPTT weight changes
14. **Relaxed ratio test tolerance** (atol=1e-3 for float precision)
15. **`.detach().clone()` in single-env get_action()** (was missing in v3)
16. **train_steps increment** in update_recurrent() for entropy annealing
17. **n_epochs info log** when using n_epochs>1 (not a warning - CleanRL uses n_epochs=4)
18. **Multi-layer LSTM bias init** fixed to handle all layers
19. **Entropy normalization test** added
20. **Learning rate sensitivity note** and **KL monitoring** added
21. **Edge case tests** (empty episode, single step, exact chunk match)
22. **Burn-in BPTT upgrade path** documented
23. **Phase 0: API migration** - find and update all get_action() call sites
24. **Forget gate bias unit tests** - verify bias_ih/bias_hh[h:2h]=1.0 for all layers
25. **approx_kl warning threshold** - warn if KL > 0.03 during recurrent update
26. **LR info log in TrainingConfig** - note when lr > 2.5e-4 with recurrent=True

**Note on n_epochs:** While n_epochs=1 is safest, CleanRL's recurrent PPO uses n_epochs=4 by default. Multiple epochs work because PPO clipping limits policy divergence. The stored `old_log_probs` become slightly stale after each epoch, but this bias is small. Use n_epochs=1 for correctness guarantees, n_epochs=2-4 for sample efficiency.

**Note on learning rate:** Recurrent policies can be more sensitive to learning rate than feedforward. Start with `lr=2.5e-4` for recurrent, vs `lr=3e-4` for feedforward. If training is unstable, try reducing to `lr=1e-4`.

---

## Phase 0: API Migration Analysis (MUST DO FIRST)

### Task 0.1: Find All get_action() Call Sites

**Rationale:** The recurrent `get_action()` returns a 4-tuple `(action, log_prob, value, hidden)` instead of 3-tuple. All existing call sites must be updated to handle this.

**Step 1: Search for all get_action() usages**

```bash
rg "\.get_action\(" --type py src/ tests/
```

**Step 2: Document call sites**

Expected locations (verify with grep):
- `src/esper/simic/vectorized.py` - main training loop
- `src/esper/simic/training.py` - if exists
- `tests/test_simic_ppo.py` - PPO tests
- Any other files using PPOAgent

**Step 3: Update each call site**

For each call site, update from:
```python
# OLD: 3-tuple unpacking
action, log_prob, value = agent.get_action(state, mask)
```

To:
```python
# NEW: 4-tuple unpacking (hidden is None for non-recurrent)
action, log_prob, value, hidden = agent.get_action(state, mask)
# Or if hidden is needed:
action, log_prob, value, hidden = agent.get_action(state, mask, hidden=prev_hidden)
```

**Step 4: Verify no breakage**

```bash
uv run pytest tests/ -x -q
```

**Step 5: Commit**

```bash
git add -u
git commit -m "refactor: update get_action() call sites for 4-tuple return"
```

---

## Phase 1: Recurrent Network Architecture

### Task 1.1: RecurrentActorCritic - Core Class

**Files:**
- Modify: `src/esper/simic/networks.py` (after ActorCritic class)
- Test: `tests/test_simic_networks.py`

**Step 1: Write the failing test**

Add to `tests/test_simic_networks.py`:

```python
class TestRecurrentActorCritic:
    """Tests for LSTM-based actor-critic network."""

    def test_init_creates_lstm_layer(self):
        """RecurrentActorCritic should have LSTM between encoder and heads."""
        net = RecurrentActorCritic(state_dim=27, action_dim=7, lstm_hidden_dim=128)
        # Direct attribute access (no hasattr per CLAUDE.md)
        assert isinstance(net.lstm, nn.LSTM)
        assert net.lstm.hidden_size == 128

    def test_get_initial_hidden_returns_correct_shape(self):
        """Initial hidden state should be zeros with correct shape."""
        net = RecurrentActorCritic(state_dim=27, action_dim=7, lstm_hidden_dim=128)
        h, c = net.get_initial_hidden(batch_size=4, device=torch.device('cpu'))
        # Shape: [num_layers, batch, hidden]
        assert h.shape == (1, 4, 128)
        assert c.shape == (1, 4, 128)
        assert (h == 0).all()
        assert (c == 0).all()

    def test_forward_single_step_returns_dist_value_hidden(self):
        """Forward with single step should return distribution, value, and new hidden."""
        net = RecurrentActorCritic(state_dim=27, action_dim=7, lstm_hidden_dim=128)
        state = torch.randn(4, 27)  # [batch, state_dim]
        mask = torch.ones(4, 7, dtype=torch.bool)

        dist, value, hidden = net.forward(state, mask, hidden=None)

        assert value.shape == (4,)
        assert len(hidden) == 2  # (h, c)
        assert hidden[0].shape == (1, 4, 128)  # [layers, batch, hidden]

    def test_forward_sequence_returns_sequence_outputs(self):
        """Forward with sequence should return outputs for each timestep."""
        net = RecurrentActorCritic(state_dim=27, action_dim=7, lstm_hidden_dim=128)
        states = torch.randn(4, 16, 27)  # [batch, seq_len, state_dim]
        masks = torch.ones(4, 16, 7, dtype=torch.bool)

        dist, values, hidden = net.forward(states, masks, hidden=None)

        assert dist._dist.logits.shape == (4, 16, 7)
        assert values.shape == (4, 16)

    def test_hidden_state_persists_across_calls(self):
        """Hidden state from one forward should affect next forward."""
        torch.manual_seed(42)  # Deterministic for reliable test
        net = RecurrentActorCritic(state_dim=27, action_dim=7, lstm_hidden_dim=128)
        state = torch.randn(1, 27)
        mask = torch.ones(1, 7, dtype=torch.bool)

        # First call builds hidden state
        _, _, hidden1 = net.forward(state, mask, hidden=None)

        # Build up more context
        for _ in range(5):
            _, _, hidden1 = net.forward(torch.randn(1, 27), mask, hidden=hidden1)

        # Compare value with context vs fresh
        _, value_with_context, _ = net.forward(state, mask, hidden=hidden1)
        _, value_fresh, _ = net.forward(state, mask, hidden=None)

        # Values must differ (context changes output)
        assert not torch.allclose(value_with_context, value_fresh, atol=1e-5)

    def test_evaluate_actions_returns_log_probs_for_sequence(self):
        """evaluate_actions should return log probs for all sequence positions."""
        net = RecurrentActorCritic(state_dim=27, action_dim=7, lstm_hidden_dim=128)
        states = torch.randn(2, 8, 27)  # [batch, seq, state_dim]
        actions = torch.randint(0, 7, (2, 8))  # [batch, seq]
        masks = torch.ones(2, 8, 7, dtype=torch.bool)

        log_probs, values, entropy, hidden = net.evaluate_actions(states, actions, masks)

        assert log_probs.shape == (2, 8)
        assert values.shape == (2, 8)
        assert entropy.shape == (2, 8)
        assert hidden[0].shape == (1, 2, 128)

    def test_device_handling_gpu_to_cpu(self):
        """Network should handle device placement correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        net = RecurrentActorCritic(state_dim=27, action_dim=7).cuda()
        state = torch.randn(1, 27, device='cuda')
        mask = torch.ones(1, 7, dtype=torch.bool, device='cuda')

        _, _, hidden = net.forward(state, mask)
        assert hidden[0].device.type == 'cuda'

    def test_entropy_is_normalized(self):
        """Verify entropy is normalized to [0, 1] range (MaskedCategorical)."""
        net = RecurrentActorCritic(state_dim=27, action_dim=7, lstm_hidden_dim=128)
        states = torch.randn(2, 8, 27)
        actions = torch.randint(0, 7, (2, 8))
        masks = torch.ones(2, 8, 7, dtype=torch.bool)

        log_probs, values, entropy, hidden = net.evaluate_actions(states, actions, masks)

        # Entropy must be in [0, 1] (normalized)
        assert (entropy >= 0).all(), f"Entropy has negative values: {entropy.min()}"
        assert (entropy <= 1).all(), f"Entropy exceeds 1: {entropy.max()}"

    def test_forget_gate_bias_initialized_to_one(self):
        """Verify LSTM forget gate bias is initialized to 1.0 for better gradient flow."""
        net = RecurrentActorCritic(state_dim=27, action_dim=7, lstm_hidden_dim=128)
        h = net.lstm_hidden_dim
        # Forget gate is second quarter of bias vector (LSTM gate order: input, forget, cell, output)
        assert (net.lstm.bias_ih_l0.data[h:2*h] == 1.0).all(), "bias_ih forget gate should be 1.0"
        assert (net.lstm.bias_hh_l0.data[h:2*h] == 1.0).all(), "bias_hh forget gate should be 1.0"

    def test_forget_gate_bias_multilayer(self):
        """Verify forget gate bias is 1.0 for all LSTM layers."""
        net = RecurrentActorCritic(state_dim=27, action_dim=7, lstm_hidden_dim=64, num_lstm_layers=2)
        h = net.lstm_hidden_dim
        # Layer 0
        assert (net.lstm.bias_ih_l0.data[h:2*h] == 1.0).all(), "Layer 0 bias_ih forget gate should be 1.0"
        assert (net.lstm.bias_hh_l0.data[h:2*h] == 1.0).all(), "Layer 0 bias_hh forget gate should be 1.0"
        # Layer 1
        assert (net.lstm.bias_ih_l1.data[h:2*h] == 1.0).all(), "Layer 1 bias_ih forget gate should be 1.0"
        assert (net.lstm.bias_hh_l1.data[h:2*h] == 1.0).all(), "Layer 1 bias_hh forget gate should be 1.0"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_simic_networks.py::TestRecurrentActorCritic -v
```

Expected: FAIL with `ImportError` (RecurrentActorCritic doesn't exist)

**Step 3: Write minimal implementation**

Add to `src/esper/simic/networks.py` after the `ActorCritic` class:

```python
class RecurrentActorCritic(nn.Module):
    """Actor-Critic with LSTM for temporal pattern learning.

    Architecture:
        state -> Linear(state_dim, hidden_dim) -> ReLU
              -> LSTM(hidden_dim, lstm_hidden_dim)
              -> Actor head -> action logits
              -> Critic head -> value

    Hidden state shape convention: [num_layers, batch, lstm_hidden_dim]
    This is PyTorch's native LSTM format - never deviate from this.
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

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers

        # Feature encoder
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

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim // 2, action_dim),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

        # Smaller init for output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

        # LSTM-specific initialization (handles all layers)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                # Works for any layer: bias_ih_l0, bias_hh_l0, bias_ih_l1, etc.
                hidden_size = self.lstm_hidden_dim
                param.data[hidden_size:2*hidden_size].fill_(1.0)

    def get_initial_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized hidden state for new episodes.

        Returns:
            Tuple of (h, c) each with shape [num_layers, batch, lstm_hidden_dim]
        """
        h = torch.zeros(
            self.num_lstm_layers, batch_size, self.lstm_hidden_dim,
            device=device,
        )
        c = torch.zeros(
            self.num_lstm_layers, batch_size, self.lstm_hidden_dim,
            device=device,
        )
        return (h, c)

    def forward(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[MaskedCategorical, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with hidden state.

        Handles both single-step [batch, state_dim] and sequence
        [batch, seq_len, state_dim] inputs.

        Args:
            state: Observation tensor
            action_mask: Boolean mask (True=valid, False=invalid)
            hidden: (h, c) tuple with shape [num_layers, batch, hidden] or None

        Returns:
            dist: MaskedCategorical over actions
            value: State value estimate(s)
            hidden: Updated (h, c) for next timestep
        """
        # Handle both single-step and sequence inputs
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]
            action_mask = action_mask.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = state.size(0)

        # Initialize hidden if not provided
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, state.device)

        # Encode features: [batch, seq_len, hidden_dim]
        features = self.encoder(state)

        # LSTM processing: [batch, seq_len, lstm_hidden_dim]
        lstm_out, hidden = self.lstm(features, hidden)

        # Actor and critic heads
        logits = self.actor(lstm_out)
        values = self.critic(lstm_out).squeeze(-1)

        # Squeeze back if single-step input
        if squeeze_output:
            logits = logits.squeeze(1)
            values = values.squeeze(1)
            action_mask = action_mask.squeeze(1)

        dist = MaskedCategorical(logits=logits, mask=action_mask)
        return dist, values, hidden

    def get_action(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> tuple[int, float, float, tuple[torch.Tensor, torch.Tensor]]:
        """Sample action for single step during rollout collection.

        Args:
            state: Single observation [state_dim] or [1, state_dim]
            action_mask: Boolean mask [action_dim] or [1, action_dim]
            hidden: Current hidden state or None
            deterministic: If True, return argmax

        Returns:
            action: Selected action index
            log_prob: Log probability of action
            value: State value estimate
            hidden: Updated hidden state (detached and cloned for safe storage)
        """
        with torch.inference_mode():
            # Ensure batch dimension
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

            dist, value, new_hidden = self.forward(state, action_mask, hidden)

            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

            # CRITICAL: .clone() is essential, .detach() is defensive
            # - .clone() creates independent tensor (not a view that keeps parent alive)
            #   Without clone, sliced tensors retain reference to full batched tensor
            # - .detach() is technically redundant under inference_mode (no graph exists)
            #   but included for safety if code is ever called outside inference_mode
            detached_hidden = (
                new_hidden[0].detach().clone(),
                new_hidden[1].detach().clone(),
            )

            return action.item(), log_prob.item(), value.item(), detached_hidden

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Evaluate actions for PPO update (supports batched sequences).

        Args:
            states: [batch, seq_len, state_dim]
            actions: [batch, seq_len]
            action_masks: [batch, seq_len, action_dim]
            hidden: Initial hidden state [num_layers, batch, hidden]

        Returns:
            log_probs: [batch, seq_len]
            values: [batch, seq_len]
            entropy: [batch, seq_len] - NORMALIZED entropy from MaskedCategorical
            hidden: Final hidden state
        """
        dist, values, hidden = self.forward(states, action_masks, hidden)
        log_probs = dist.log_prob(actions)
        # NOTE: MaskedCategorical.entropy() returns NORMALIZED entropy [0, 1]
        # This matches the existing PPOAgent which uses entropy_coef scaled for
        # normalized values (e.g., 0.015 instead of 0.01 for raw entropy).
        entropy = dist.entropy()
        return log_probs, values, entropy, hidden
```

**Step 4: Update exports in networks.py**

Add to `__all__` list:

```python
__all__ = [
    "PolicyNetwork",
    "print_confusion_matrix",
    "ActorCritic",
    "RecurrentActorCritic",  # Add this
    "MaskedCategorical",
    "InvalidStateMachineError",
    "QNetwork",
    "VNetwork",
]
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_simic_networks.py::TestRecurrentActorCritic -v
```

Expected: All 10 tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/networks.py tests/test_simic_networks.py
git commit -m "feat(simic): add RecurrentActorCritic with LSTM for temporal patterns"
```

---

## Phase 2: Recurrent Rollout Buffer (Per-Environment Storage)

### Task 2.1: RecurrentRolloutBuffer with Per-Env Step Lists

**Files:**
- Modify: `src/esper/simic/buffers.py`
- Test: `tests/test_simic_buffers.py`

**Step 1: Write the failing test**

Add to `tests/test_simic_buffers.py`:

```python
class TestRecurrentRolloutBuffer:
    """Tests for LSTM-compatible rollout buffer with per-env storage."""

    def test_add_step_to_correct_env(self):
        """Steps should be stored per-environment, not interleaved."""
        buffer = RecurrentRolloutBuffer(chunk_length=4)

        # Add to env 0
        buffer.start_episode(env_id=0)
        buffer.add(
            state=torch.randn(27), action=0, log_prob=-0.5, value=1.0,
            reward=0.1, done=False, action_mask=torch.ones(7, dtype=torch.bool),
            env_id=0,
        )

        # Add to env 1 (different env)
        buffer.start_episode(env_id=1)
        buffer.add(
            state=torch.randn(27), action=1, log_prob=-0.3, value=1.2,
            reward=0.2, done=False, action_mask=torch.ones(7, dtype=torch.bool),
            env_id=1,
        )

        # Each env should have 1 step
        assert len(buffer.env_steps[0]) == 1
        assert len(buffer.env_steps[1]) == 1
        assert buffer.env_steps[0][0].action == 0
        assert buffer.env_steps[1][0].action == 1

    def test_episode_boundaries_tracked_per_env(self):
        """Episode boundaries should be independent per environment."""
        buffer = RecurrentRolloutBuffer(chunk_length=4)

        # Env 0: 2-step episode
        buffer.start_episode(env_id=0)
        for i in range(2):
            buffer.add(
                state=torch.randn(27), action=i, log_prob=-0.5, value=1.0,
                reward=0.1, done=(i == 1), action_mask=torch.ones(7, dtype=torch.bool),
                env_id=0,
            )
        buffer.end_episode(env_id=0)

        # Env 1: 3-step episode (started before env 0 ended)
        buffer.start_episode(env_id=1)
        for i in range(3):
            buffer.add(
                state=torch.randn(27), action=i, log_prob=-0.3, value=1.2,
                reward=0.2, done=(i == 2), action_mask=torch.ones(7, dtype=torch.bool),
                env_id=1,
            )
        buffer.end_episode(env_id=1)

        # Check boundaries are correct
        assert buffer.episode_boundaries[0] == [(0, 2)]
        assert buffer.episode_boundaries[1] == [(0, 3)]

    def test_get_chunks_respects_episode_boundaries(self):
        """Chunks should never cross episode boundaries."""
        buffer = RecurrentRolloutBuffer(chunk_length=4)

        # Episode 1: 3 steps (shorter than chunk)
        buffer.start_episode(env_id=0)
        for i in range(3):
            buffer.add(
                state=torch.randn(27), action=i % 7, log_prob=-0.5, value=1.0,
                reward=0.1, done=(i == 2), action_mask=torch.ones(7, dtype=torch.bool),
                env_id=0,
            )
        buffer.end_episode(env_id=0)

        # Episode 2: 6 steps (will be 2 chunks: 4 + 2 padded)
        # NOTE: This triggers mid-episode chunking warning for 2nd chunk
        buffer.start_episode(env_id=0)
        for i in range(6):
            buffer.add(
                state=torch.randn(27), action=i % 7, log_prob=-0.5, value=1.0,
                reward=0.1, done=(i == 5), action_mask=torch.ones(7, dtype=torch.bool),
                env_id=0,
            )
        buffer.end_episode(env_id=0)

        # Compute GAE first (required before get_chunks)
        buffer.compute_gae(gamma=0.99, gae_lambda=0.95)

        # Mid-episode chunking should warn but not crash
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chunks = buffer.get_chunks(device='cpu')
            # Should have warning for mid-episode chunk
            assert any("Mid-episode chunking" in str(warning.message) for warning in w)

        # Episode 1: 1 chunk (3 steps padded to 4)
        # Episode 2: 2 chunks (4 steps + 2 steps padded to 4)
        assert len(chunks) == 3

        # Each chunk should have length 4 (padded)
        for chunk in chunks:
            assert chunk['states'].shape[1] == 4  # [1, seq, state_dim]

    def test_mid_episode_chunking_warns_not_crashes(self):
        """Mid-episode chunking should emit warning but still work."""
        buffer = RecurrentRolloutBuffer(chunk_length=3)  # Smaller than episode

        # 5-step episode -> 2 chunks (3 + 2), second chunk is mid-episode
        buffer.start_episode(env_id=0)
        for i in range(5):
            buffer.add(
                state=torch.randn(27), action=0, log_prob=-0.5, value=1.0,
                reward=0.1, done=(i == 4), action_mask=torch.ones(7, dtype=torch.bool),
                env_id=0,
            )
        buffer.end_episode(env_id=0)
        buffer.compute_gae(gamma=0.99, gae_lambda=0.95)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chunks = buffer.get_chunks(device='cpu')

            # Should get warning about mid-episode chunking
            mid_episode_warnings = [
                warning for warning in w
                if "Mid-episode chunking" in str(warning.message)
            ]
            assert len(mid_episode_warnings) == 1, "Expected exactly 1 mid-episode warning"

        # Should still produce valid chunks
        assert len(chunks) == 2
        # Second chunk uses zeros for initial hidden (suboptimal but not wrong)

    def test_get_chunks_includes_gae_returns_advantages(self):
        """Chunks must include computed returns and advantages."""
        buffer = RecurrentRolloutBuffer(chunk_length=4)

        buffer.start_episode(env_id=0)
        for i in range(4):
            buffer.add(
                state=torch.randn(27), action=0, log_prob=-0.5, value=1.0,
                reward=0.1 * (i + 1), done=(i == 3),
                action_mask=torch.ones(7, dtype=torch.bool), env_id=0,
            )
        buffer.end_episode(env_id=0)

        buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = buffer.get_chunks(device='cpu')

        assert len(chunks) == 1
        chunk = chunks[0]

        # CRITICAL: returns and advantages must be in chunk
        assert 'returns' in chunk
        assert 'advantages' in chunk
        assert chunk['returns'].shape == (1, 4)
        assert chunk['advantages'].shape == (1, 4)
        # Advantages should be non-zero (we have rewards)
        assert chunk['advantages'].abs().sum() > 0

    def test_valid_mask_indicates_real_vs_padded(self):
        """Chunks should have mask indicating valid timesteps."""
        buffer = RecurrentRolloutBuffer(chunk_length=4)

        # 2 steps - will be padded to 4
        buffer.start_episode(env_id=0)
        buffer.add(
            state=torch.randn(27), action=0, log_prob=-0.5, value=1.0,
            reward=0.1, done=False, action_mask=torch.ones(7, dtype=torch.bool),
            env_id=0,
        )
        buffer.add(
            state=torch.randn(27), action=1, log_prob=-0.3, value=1.2,
            reward=0.2, done=True, action_mask=torch.ones(7, dtype=torch.bool),
            env_id=0,
        )
        buffer.end_episode(env_id=0)

        buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = buffer.get_chunks(device='cpu')
        chunk = chunks[0]

        # First 2 are valid, last 2 are padding
        expected = torch.tensor([[True, True, False, False]])
        assert torch.equal(chunk['valid_mask'], expected)

    def test_initial_hidden_stored_at_chunk_start(self):
        """Each chunk should have initial hidden from first step only."""
        buffer = RecurrentRolloutBuffer(chunk_length=4, lstm_hidden_dim=64)

        buffer.start_episode(env_id=0)
        for i in range(4):
            buffer.add(
                state=torch.randn(27), action=0, log_prob=-0.5, value=1.0,
                reward=0.1, done=(i == 3),
                action_mask=torch.ones(7, dtype=torch.bool), env_id=0,
            )
        buffer.end_episode(env_id=0)

        buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = buffer.get_chunks(device='cpu')

        # Initial hidden should be zeros (episode start)
        chunk = chunks[0]
        assert 'initial_hidden_h' in chunk
        assert 'initial_hidden_c' in chunk
        assert chunk['initial_hidden_h'].shape == (1, 1, 64)  # [1, batch=1, hidden]
        assert (chunk['initial_hidden_h'] == 0).all()  # Episode start = zeros

    def test_batched_chunks_stack_correctly(self):
        """Multiple chunks should batch together correctly."""
        buffer = RecurrentRolloutBuffer(chunk_length=4, lstm_hidden_dim=64)

        # Two short episodes (each becomes 1 chunk)
        for ep in range(2):
            buffer.start_episode(env_id=0)
            for i in range(3):
                buffer.add(
                    state=torch.randn(27), action=ep, log_prob=-0.5, value=1.0,
                    reward=0.1, done=(i == 2),
                    action_mask=torch.ones(7, dtype=torch.bool), env_id=0,
                )
            buffer.end_episode(env_id=0)

        buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        batches = buffer.get_batched_chunks(device='cpu', batch_size=2)
        batches = list(batches)

        assert len(batches) == 1  # 2 chunks batched together
        batch = batches[0]
        assert batch['states'].shape == (2, 4, 27)  # [batch, seq, state_dim]
        assert batch['initial_hidden_h'].shape == (1, 2, 64)  # [layers, batch, hidden]
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_simic_buffers.py::TestRecurrentRolloutBuffer -v
```

Expected: FAIL with `ImportError` (RecurrentRolloutBuffer doesn't exist)

**Step 3: Write minimal implementation**

Add to `src/esper/simic/buffers.py`:

```python
from typing import Iterator


class RecurrentRolloutStep(NamedTuple):
    """Single step in a recurrent PPO rollout.

    Note: Hidden states are NOT stored per-step (memory optimization).
    They're only stored at chunk boundaries during get_chunks().
    """
    state: torch.Tensor       # [state_dim]
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    action_mask: torch.Tensor  # [action_dim]


@dataclass
class RecurrentRolloutBuffer:
    """Buffer for recurrent PPO with per-environment storage.

    Key design decisions (from reviewer feedback):
    1. Per-env step lists - avoids interleaving corruption
    2. chunk_length=25 matches max_epochs default (no mid-episode chunking)
    3. GAE computed per-episode, then distributed to chunks
    4. Returns/advantages included in every chunk dict
    5. Single PPO epoch avoids hidden state drift between old/new log_probs

    Args:
        chunk_length: Length of chunks for BPTT (default 25 = max_epochs default)
        lstm_hidden_dim: Hidden dimension for zero-init at episode starts
    """
    chunk_length: int = 25
    lstm_hidden_dim: int = 128

    # Per-environment step storage (no interleaving)
    env_steps: dict[int, list[RecurrentRolloutStep]] = field(default_factory=dict)
    episode_boundaries: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    _current_episode_start: dict[int, int] = field(default_factory=dict)

    # GAE results (computed once, distributed to chunks)
    _episode_returns: dict[tuple[int, int, int], torch.Tensor] = field(default_factory=dict)
    _episode_advantages: dict[tuple[int, int, int], torch.Tensor] = field(default_factory=dict)
    _gae_computed: bool = field(default=False, init=False)

    def start_episode(self, env_id: int) -> None:
        """Mark start of a new episode for an environment."""
        if env_id not in self.env_steps:
            self.env_steps[env_id] = []
        self._current_episode_start[env_id] = len(self.env_steps[env_id])

    def end_episode(self, env_id: int) -> None:
        """Mark end of episode, recording boundary."""
        if env_id not in self._current_episode_start:
            return
        start = self._current_episode_start[env_id]
        end = len(self.env_steps[env_id])
        if env_id not in self.episode_boundaries:
            self.episode_boundaries[env_id] = []
        self.episode_boundaries[env_id].append((start, end))
        del self._current_episode_start[env_id]

    def add(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_mask: torch.Tensor,
        env_id: int,
    ) -> None:
        """Add a step to the correct environment's list."""
        if env_id not in self.env_steps:
            self.env_steps[env_id] = []

        self.env_steps[env_id].append(RecurrentRolloutStep(
            state=state.cpu(),  # Store on CPU to save GPU memory
            action=action,
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=done,
            action_mask=action_mask.cpu(),
        ))

    def clear(self) -> None:
        """Clear all data."""
        self.env_steps = {}
        self.episode_boundaries = {}
        self._current_episode_start = {}
        self._episode_returns = {}
        self._episode_advantages = {}
        self._gae_computed = False

    def __len__(self) -> int:
        return sum(len(steps) for steps in self.env_steps.values())

    def compute_gae(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE returns/advantages for all episodes.

        Must be called before get_chunks(). Stores results keyed by
        (env_id, episode_start, episode_end) for chunk retrieval.
        """
        for env_id, boundaries in self.episode_boundaries.items():
            steps = self.env_steps[env_id]

            for start_idx, end_idx in boundaries:
                episode_steps = steps[start_idx:end_idx]
                n_steps = len(episode_steps)

                if n_steps == 0:
                    continue

                returns = torch.zeros(n_steps)
                advantages = torch.zeros(n_steps)

                last_gae = 0.0
                next_value = 0.0  # Terminal

                for t in reversed(range(n_steps)):
                    step = episode_steps[t]
                    if step.done:
                        next_value = 0.0
                        last_gae = 0.0

                    delta = step.reward + gamma * next_value - step.value
                    advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae
                    returns[t] = advantages[t] + step.value
                    next_value = step.value

                # Store keyed by episode identity
                key = (env_id, start_idx, end_idx)
                self._episode_returns[key] = returns
                self._episode_advantages[key] = advantages

        self._gae_computed = True

    def get_chunks(self, device: str | torch.device) -> list[dict]:
        """Get all chunks with returns/advantages included.

        Each chunk is a dict with shape [1, chunk_len, ...] for batching later.
        """
        if not self._gae_computed:
            raise RuntimeError("Must call compute_gae() before get_chunks()")

        chunks = []

        for env_id, boundaries in self.episode_boundaries.items():
            steps = self.env_steps[env_id]

            for start_idx, end_idx in boundaries:
                episode_steps = steps[start_idx:end_idx]
                episode_len = len(episode_steps)

                # Get precomputed GAE for this episode
                key = (env_id, start_idx, end_idx)
                episode_returns = self._episode_returns[key]
                episode_advantages = self._episode_advantages[key]

                # Chunk this episode
                for chunk_start in range(0, episode_len, self.chunk_length):
                    chunk_end = min(chunk_start + self.chunk_length, episode_len)
                    chunk_steps = episode_steps[chunk_start:chunk_end]
                    actual_len = len(chunk_steps)
                    pad_len = self.chunk_length - actual_len

                    # Get GAE slice for this chunk
                    chunk_returns = episode_returns[chunk_start:chunk_end]
                    chunk_advantages = episode_advantages[chunk_start:chunk_end]

                    # Pad if needed
                    if pad_len > 0:
                        pad_step = chunk_steps[-1]
                        for _ in range(pad_len):
                            chunk_steps = list(chunk_steps) + [RecurrentRolloutStep(
                                state=torch.zeros_like(pad_step.state),
                                action=0,
                                log_prob=0.0,
                                value=0.0,
                                reward=0.0,
                                done=True,
                                action_mask=pad_step.action_mask,
                            )]
                        # Pad returns/advantages with zeros
                        chunk_returns = torch.cat([
                            chunk_returns,
                            torch.zeros(pad_len),
                        ])
                        chunk_advantages = torch.cat([
                            chunk_advantages,
                            torch.zeros(pad_len),
                        ])

                    # Initial hidden state determination:
                    # - Episode start (chunk_start == 0): zeros (correct)
                    # - Mid-episode chunk: would need stored hidden from previous chunk
                    #
                    # With chunk_length=25 matching max_epochs exactly,
                    # episodes fit in a single chunk, so chunk_start is ALWAYS 0.
                    # This avoids the hidden state drift problem entirely.
                    #
                    # GRACEFUL HANDLING: If chunk_length < episode_length (mid-episode
                    # chunking), we use zeros for initial hidden and log a warning.
                    # This is suboptimal (loses temporal context at chunk boundaries)
                    # but won't crash. For best results, set chunk_length >= episode_length.
                    is_episode_start = (chunk_start == 0)
                    if not is_episode_start:
                        import warnings
                        warnings.warn(
                            f"Mid-episode chunking detected (chunk_start={chunk_start}). "
                            f"Using zeros for initial hidden state, which loses temporal "
                            f"context. For correct BPTT, set chunk_length >= episode_length "
                            f"or implement burn-in BPTT.",
                            RuntimeWarning,
                        )

                    # Stack into tensors with batch dim [1, seq, ...]
                    chunk = {
                        'states': torch.stack([s.state for s in chunk_steps]).unsqueeze(0).to(device),
                        'actions': torch.tensor([s.action for s in chunk_steps], dtype=torch.long).unsqueeze(0).to(device),
                        'old_log_probs': torch.tensor([s.log_prob for s in chunk_steps], dtype=torch.float32).unsqueeze(0).to(device),
                        'old_values': torch.tensor([s.value for s in chunk_steps], dtype=torch.float32).unsqueeze(0).to(device),
                        'action_masks': torch.stack([s.action_mask for s in chunk_steps]).unsqueeze(0).to(device),
                        'returns': chunk_returns.unsqueeze(0).to(device),
                        'advantages': chunk_advantages.unsqueeze(0).to(device),
                        'valid_mask': torch.tensor(
                            [True] * actual_len + [False] * pad_len,
                            dtype=torch.bool,
                        ).unsqueeze(0).to(device),
                        # Initial hidden: zeros for episode start, zeros for chunk boundaries (TBPTT)
                        'initial_hidden_h': torch.zeros(1, 1, self.lstm_hidden_dim, device=device),
                        'initial_hidden_c': torch.zeros(1, 1, self.lstm_hidden_dim, device=device),
                        'is_episode_start': is_episode_start,
                    }
                    chunks.append(chunk)

        return chunks

    def get_batched_chunks(
        self,
        device: str | torch.device,
        batch_size: int = 8,
    ) -> Iterator[dict]:
        """Yield batched chunks for efficient GPU processing.

        Stacks multiple chunks into batches for parallel LSTM processing.
        """
        chunks = self.get_chunks(device)

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            n_chunks = len(batch_chunks)

            # Stack along batch dimension
            batch = {
                'states': torch.cat([c['states'] for c in batch_chunks], dim=0),  # [batch, seq, state]
                'actions': torch.cat([c['actions'] for c in batch_chunks], dim=0),
                'old_log_probs': torch.cat([c['old_log_probs'] for c in batch_chunks], dim=0),
                'old_values': torch.cat([c['old_values'] for c in batch_chunks], dim=0),
                'action_masks': torch.cat([c['action_masks'] for c in batch_chunks], dim=0),
                'returns': torch.cat([c['returns'] for c in batch_chunks], dim=0),
                'advantages': torch.cat([c['advantages'] for c in batch_chunks], dim=0),
                'valid_mask': torch.cat([c['valid_mask'] for c in batch_chunks], dim=0),
                # Hidden: [num_layers, batch, hidden]
                'initial_hidden_h': torch.cat(
                    [c['initial_hidden_h'] for c in batch_chunks], dim=1
                ),
                'initial_hidden_c': torch.cat(
                    [c['initial_hidden_c'] for c in batch_chunks], dim=1
                ),
            }
            yield batch
```

**Step 4: Update exports**

```python
__all__ = [
    "RolloutStep",
    "RolloutBuffer",
    "RecurrentRolloutStep",
    "RecurrentRolloutBuffer",
]
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_simic_buffers.py::TestRecurrentRolloutBuffer -v
```

Expected: All 7 tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/buffers.py tests/test_simic_buffers.py
git commit -m "feat(simic): add RecurrentRolloutBuffer with per-env storage and batched chunks"
```

---

## Phase 3: Recurrent PPO Agent

### Task 3.1: PPOAgent Recurrence Support

**Files:**
- Modify: `src/esper/simic/ppo.py`
- Test: `tests/test_simic_ppo.py`

**Step 1: Write the failing test**

Add to `tests/test_simic_ppo.py`:

```python
class TestRecurrentPPOAgent:
    """Tests for recurrent PPO agent."""

    def test_init_with_recurrent_creates_lstm_network(self):
        """PPOAgent(recurrent=True) should use RecurrentActorCritic."""
        agent = PPOAgent(state_dim=27, action_dim=7, recurrent=True, lstm_hidden_dim=128)
        assert isinstance(agent.network, RecurrentActorCritic)
        assert agent.recurrent is True

    def test_init_without_recurrent_uses_mlp(self):
        """PPOAgent(recurrent=False) should use standard ActorCritic."""
        agent = PPOAgent(state_dim=27, action_dim=7, recurrent=False)
        assert isinstance(agent.network, ActorCritic)
        assert agent.recurrent is False

    def test_get_action_returns_hidden_when_recurrent(self):
        """get_action should return hidden state for recurrent agent."""
        agent = PPOAgent(state_dim=27, action_dim=7, recurrent=True, device='cpu')
        state = torch.randn(27)
        mask = torch.ones(7, dtype=torch.bool)

        result = agent.get_action(state, mask, hidden=None)

        assert len(result) == 4  # (action, log_prob, value, hidden)
        action, log_prob, value, hidden = result
        assert isinstance(action, int)
        assert hidden is not None
        assert len(hidden) == 2  # (h, c)

    def test_update_recurrent_uses_batched_chunks(self):
        """Recurrent update should use batched chunk processing."""
        agent = PPOAgent(
            state_dim=27, action_dim=7, recurrent=True, device='cpu',
            chunk_length=4, lstm_hidden_dim=64,
        )

        # Add episode
        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(6):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=0.1, done=(i == 5), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        metrics = agent.update_recurrent()

        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert metrics['policy_loss'] != 0.0

    def test_advantages_are_nonzero_in_update(self):
        """Verify GAE advantages flow through to update (critical bug fix)."""
        agent = PPOAgent(
            state_dim=27, action_dim=7, recurrent=True, device='cpu',
            chunk_length=4, lstm_hidden_dim=64,
        )

        # Add episode with increasing rewards
        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(4):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=float(i + 1),  # Increasing rewards
                done=(i == 3), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        # Compute GAE
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')

        # Advantages must be non-zero
        assert chunks[0]['advantages'].abs().sum() > 0, "Advantages should be non-zero with rewards"

    def test_value_coef_used_correctly(self):
        """Value coefficient should be from agent, not hardcoded."""
        agent = PPOAgent(
            state_dim=27, action_dim=7, recurrent=True, device='cpu',
            value_coef=0.25,  # Non-default value
        )
        assert agent.value_coef == 0.25
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_simic_ppo.py::TestRecurrentPPOAgent -v
```

Expected: FAIL

**Step 3: Implement recurrent support in PPOAgent**

Modify `src/esper/simic/ppo.py`:

1. Update imports at top:
```python
from esper.simic.networks import ActorCritic, RecurrentActorCritic
from esper.simic.buffers import RolloutBuffer, RecurrentRolloutBuffer
```

2. Add recurrent params to `__init__`:
```python
def __init__(
    self,
    state_dim: int,
    action_dim: int,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.1,
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_coef_min: float = 0.01,
    entropy_anneal_steps: int = 0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    value_coef: float = 0.5,
    device: str = "cuda",
    clip_value: bool = True,
    max_grad_norm: float = 0.5,
    # Recurrence params
    recurrent: bool = False,
    lstm_hidden_dim: int = 128,
    chunk_length: int = 25,  # Must match max_epochs to avoid hidden state issues
):
    self.recurrent = recurrent
    self.chunk_length = chunk_length
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    self.value_coef = value_coef
    self.max_grad_norm = max_grad_norm
    self.lstm_hidden_dim = lstm_hidden_dim
    # ... existing code ...

    if recurrent:
        self.network = RecurrentActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            lstm_hidden_dim=lstm_hidden_dim,
        ).to(device)
        self.recurrent_buffer = RecurrentRolloutBuffer(
            chunk_length=chunk_length,
            lstm_hidden_dim=lstm_hidden_dim,
        )
    else:
        self.network = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)

    # ... rest of existing init ...
```

3. Add `get_action` method:
```python
def get_action(
    self,
    state: torch.Tensor,
    action_mask: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    deterministic: bool = False,
) -> tuple[int, float, float, tuple | None]:
    """Get action, optionally with hidden state for recurrent policy."""
    if self.recurrent:
        return self.network.get_action(state, action_mask, hidden, deterministic)
    else:
        action, log_prob, value = self.network.get_action(state, action_mask, deterministic)
        return action, log_prob, value, None
```

4. Add `store_recurrent_transition`:
```python
def store_recurrent_transition(
    self,
    state: torch.Tensor,
    action: int,
    log_prob: float,
    value: float,
    reward: float,
    done: bool,
    action_mask: torch.Tensor,
    env_id: int,
) -> None:
    """Store transition for recurrent policy (no hidden stored per-step)."""
    self.recurrent_buffer.add(
        state=state, action=action, log_prob=log_prob, value=value,
        reward=reward, done=done, action_mask=action_mask, env_id=env_id,
    )
```

5. Add `update_recurrent` method:
```python
def update_recurrent(self, n_epochs: int | None = None, chunk_batch_size: int = 8) -> dict:
    """PPO update for recurrent policy using batched sequences.

    Key design decisions (from reviewer feedback):
    1. GAE computed once, distributed to chunks
    2. Batched chunk processing for GPU efficiency
    3. SINGLE EPOCH (n_epochs=1) default to avoid hidden state drift between epochs
       - After gradient updates, policy changes, so recomputed log_probs differ
       - With multiple epochs, ratio = new_log_prob / old_log_prob becomes biased
       - Single epoch is standard practice for recurrent PPO (OpenAI Five, CleanRL)
    4. value_coef from agent, not hardcoded
    5. Value clipping for training stability (parity with feedforward)
    6. Gradient clipping at max_grad_norm
    7. train_steps incremented for entropy annealing

    Args:
        n_epochs: Number of PPO epochs. Default 1 for recurrent (safest).
                  Higher values (2-4) work due to PPO clipping but emit warning.
        chunk_batch_size: Number of chunks per batch for GPU efficiency.
    """
    # Default to 1 epoch for recurrent (safest)
    if n_epochs is None:
        n_epochs = 1

    # Log info if using multiple epochs (this is fine - CleanRL uses n_epochs=4)
    if n_epochs > 1:
        import logging
        logging.info(
            f"Using n_epochs={n_epochs} with recurrent policy. This is supported "
            f"(CleanRL uses n_epochs=4). PPO clipping handles the stale log_prob bias."
        )

    # Compute GAE for all episodes
    self.recurrent_buffer.compute_gae(
        gamma=self.gamma,
        gae_lambda=self.gae_lambda,
    )

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0  # For diagnostics
    n_updates = 0

    for epoch in range(n_epochs):
        # Process chunks in batches
        for batch in self.recurrent_buffer.get_batched_chunks(
            device=self.device,
            batch_size=chunk_batch_size,
        ):
            batch_size = batch['states'].size(0)
            seq_len = batch['states'].size(1)

            # Get initial hidden [num_layers, batch, hidden]
            initial_hidden = (
                batch['initial_hidden_h'],
                batch['initial_hidden_c'],
            )

            # Forward through sequences
            log_probs, values, entropy, _ = self.network.evaluate_actions(
                batch['states'],
                batch['actions'],
                batch['action_masks'],
                hidden=initial_hidden,
            )

            # Get valid mask and flatten
            valid = batch['valid_mask']  # [batch, seq]

            # Extract valid timesteps
            log_probs_valid = log_probs[valid]
            values_valid = values[valid]
            entropy_valid = entropy[valid]
            old_log_probs_valid = batch['old_log_probs'][valid]
            old_values_valid = batch['old_values'][valid]
            returns_valid = batch['returns'][valid]
            advantages_valid = batch['advantages'][valid]

            # Normalize advantages
            if len(advantages_valid) > 1:
                advantages_valid = (advantages_valid - advantages_valid.mean()) / (advantages_valid.std() + 1e-8)

            # PPO clipped objective
            ratio = torch.exp(log_probs_valid - old_log_probs_valid)
            surr1 = ratio * advantages_valid
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_valid
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss with clipping (parity with feedforward PPO)
            values_clipped = old_values_valid + torch.clamp(
                values_valid - old_values_valid,
                -self.clip_ratio,
                self.clip_ratio,
            )
            value_loss_unclipped = (values_valid - returns_valid) ** 2
            value_loss_clipped = (values_clipped - returns_valid) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            # Entropy bonus
            entropy_loss = -entropy_valid.mean()

            # Total loss (using agent's value_coef, not hardcoded)
            loss = policy_loss + self.value_coef * value_loss + self.get_entropy_coef() * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Increment train_steps for entropy annealing (CRITICAL for get_entropy_coef)
            self.train_steps += 1

            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_valid.mean().item()
            total_approx_kl += (old_log_probs_valid - log_probs_valid).mean().item()
            n_updates += 1

    self.recurrent_buffer.clear()

    # Compute final metrics
    avg_approx_kl = total_approx_kl / max(n_updates, 1)

    # Warn if KL divergence is high (indicates policy changing too fast)
    if avg_approx_kl > 0.03:
        logger.warning(
            f"High KL divergence ({avg_approx_kl:.4f}) during recurrent update. "
            f"Consider reducing lr or n_epochs to stabilize training."
        )

    return {
        'policy_loss': total_policy_loss / max(n_updates, 1),
        'value_loss': total_value_loss / max(n_updates, 1),
        'entropy': total_entropy / max(n_updates, 1),
        'approx_kl': avg_approx_kl,
    }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_simic_ppo.py::TestRecurrentPPOAgent -v
```

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py tests/test_simic_ppo.py
git commit -m "feat(simic): add recurrent=True mode to PPOAgent with batched BPTT"
```

### Task 3.2: TrainingConfig Integration

**Files:**
- Modify: `src/esper/simic/config.py`
- Test: `tests/test_simic_config.py`

**Step 1: Add recurrent params to TrainingConfig**

Add these fields to the `TrainingConfig` dataclass:

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...

    # === Recurrence (LSTM Policy) ===
    recurrent: bool = False
    lstm_hidden_dim: int = 128
    chunk_length: int | None = None  # None = auto-match max_epochs; set explicitly if different

    def __post_init__(self):
        """Validate and set defaults for recurrent config."""
        # Auto-match chunk_length to max_epochs if not set
        if self.chunk_length is None:
            self.chunk_length = self.max_epochs

        # Warn if chunk_length < max_epochs (will cause mid-episode chunking)
        if self.recurrent and self.chunk_length < self.max_epochs:
            import warnings
            warnings.warn(
                f"chunk_length={self.chunk_length} < max_epochs={self.max_epochs}. "
                f"Mid-episode chunking will occur, losing temporal context at chunk "
                f"boundaries. For optimal BPTT, set chunk_length >= max_epochs.",
                RuntimeWarning,
            )

        # Note about learning rate for recurrent policies
        if self.recurrent and self.lr > 2.5e-4:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Using lr={self.lr} with recurrent=True. Recurrent policies can be "
                f"more sensitive to learning rate. If training is unstable, consider "
                f"reducing to lr=2.5e-4 or lr=1e-4."
            )
```

**Step 2: Update `to_ppo_kwargs()`**

```python
def to_ppo_kwargs(self) -> dict[str, Any]:
    """Extract PPOAgent constructor kwargs."""
    return {
        # ... existing keys ...
        "recurrent": self.recurrent,
        "lstm_hidden_dim": self.lstm_hidden_dim,
        "chunk_length": self.chunk_length,
    }
```

**Step 3: Update `to_train_kwargs()`**

```python
def to_train_kwargs(self) -> dict[str, Any]:
    """Extract train_ppo_vectorized kwargs."""
    return {
        # ... existing keys ...
        "recurrent": self.recurrent,
        "lstm_hidden_dim": self.lstm_hidden_dim,
        "chunk_length": self.chunk_length,
    }
```

**Step 4: Update `summary()`**

```python
def summary(self) -> str:
    """Human-readable summary of configuration."""
    lines = [
        # ... existing lines ...
        f"  Recurrent: {'LSTM' if self.recurrent else 'MLP'}" +
            (f" (hidden={self.lstm_hidden_dim}, chunk={self.chunk_length})" if self.recurrent else ""),
    ]
    return "\n".join(lines)
```

**Step 5: Add test**

Add to `tests/test_simic_config.py`:

```python
def test_recurrent_config_to_ppo_kwargs():
    """Recurrent params should flow to PPOAgent kwargs."""
    config = TrainingConfig(recurrent=True, lstm_hidden_dim=256, chunk_length=16)
    kwargs = config.to_ppo_kwargs()
    assert kwargs["recurrent"] is True
    assert kwargs["lstm_hidden_dim"] == 256
    assert kwargs["chunk_length"] == 16

def test_recurrent_config_to_train_kwargs():
    """Recurrent params should flow to train_ppo_vectorized kwargs."""
    config = TrainingConfig(recurrent=True)
    kwargs = config.to_train_kwargs()
    assert kwargs["recurrent"] is True
    assert "lstm_hidden_dim" in kwargs
    assert "chunk_length" in kwargs
```

**Step 6: Commit**

```bash
git add src/esper/simic/config.py tests/test_simic_config.py
git commit -m "feat(simic): add recurrent params to TrainingConfig"
```

---

## Phase 4: Vectorized Training Integration

### Task 4.1: Hidden State Tracking in Vectorized Training

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Test: `tests/test_recurrent_vectorized.py`

**Step 1: Write the failing test**

Create `tests/test_recurrent_vectorized.py`:

```python
"""Tests for recurrent policy integration with vectorized training."""

import torch
from unittest.mock import MagicMock, patch

from esper.simic.vectorized import ParallelEnvState


class TestParallelEnvStateRecurrence:
    """Tests for LSTM hidden state tracking in ParallelEnvState."""

    def test_lstm_hidden_field_exists(self):
        """ParallelEnvState should have lstm_hidden field."""
        env_state = ParallelEnvState(
            model=MagicMock(),
            optimizer=MagicMock(),
            seed_slot=MagicMock(),
            env_id=0,
        )
        # Direct attribute access (no hasattr)
        assert env_state.lstm_hidden is None  # Default

    def test_lstm_hidden_can_store_tuple(self):
        """lstm_hidden should accept (h, c) tuple."""
        env_state = ParallelEnvState(
            model=MagicMock(),
            optimizer=MagicMock(),
            seed_slot=MagicMock(),
            env_id=0,
        )
        h = torch.zeros(1, 1, 128)
        c = torch.zeros(1, 1, 128)
        env_state.lstm_hidden = (h, c)

        assert env_state.lstm_hidden is not None
        assert torch.equal(env_state.lstm_hidden[0], h)
        assert torch.equal(env_state.lstm_hidden[1], c)

    def test_lstm_hidden_reset_to_none(self):
        """lstm_hidden should be resettable to None for episode boundaries."""
        env_state = ParallelEnvState(
            model=MagicMock(),
            optimizer=MagicMock(),
            seed_slot=MagicMock(),
            env_id=0,
        )
        env_state.lstm_hidden = (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
        env_state.lstm_hidden = None  # Reset on episode end

        assert env_state.lstm_hidden is None
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_recurrent_vectorized.py -v
```

Expected: FAIL (lstm_hidden field doesn't exist)

**Step 3: Add lstm_hidden to ParallelEnvState**

In `src/esper/simic/vectorized.py`, update the dataclass:

```python
@dataclass
class ParallelEnvState:
    """State for a single parallel environment."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    seed_slot: SeedSlot
    env_id: int
    current_epoch: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    # LSTM hidden state for recurrent policy (None = fresh episode)
    lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_recurrent_vectorized.py -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/esper/simic/vectorized.py tests/test_recurrent_vectorized.py
git commit -m "feat(simic): add lstm_hidden field to ParallelEnvState"
```

### Task 4.2: Batched Recurrent Action Selection

**Files:**
- Modify: `src/esper/simic/vectorized.py`

**Step 1: Update train_ppo_vectorized signature**

Add recurrent params:

```python
def train_ppo_vectorized(
    # ... existing params ...
    recurrent: bool = False,
    lstm_hidden_dim: int = 128,
    chunk_length: int = 25,  # Must match max_epochs default (25)
) -> tuple[PPOAgent, list[dict]]:
```

**Step 2: Update agent creation**

```python
agent = PPOAgent(
    state_dim=state_dim,
    action_dim=len(ActionEnum),
    # ... existing params ...
    recurrent=recurrent,
    lstm_hidden_dim=lstm_hidden_dim,
    chunk_length=chunk_length,
)
```

**Step 3: Add batched recurrent action selection**

In the epoch loop, update action selection to support both modes:

```python
# Get actions - batched for both recurrent and non-recurrent
if recurrent:
    # Collect and batch hidden states from all envs
    hiddens_h = []
    hiddens_c = []
    for env_state in env_states:
        if env_state.lstm_hidden is None:
            h, c = agent.network.get_initial_hidden(1, device)
        else:
            h, c = env_state.lstm_hidden
        hiddens_h.append(h)
        hiddens_c.append(c)

    # Batch hidden: [num_layers, n_envs, hidden]
    batch_h = torch.cat(hiddens_h, dim=1)
    batch_c = torch.cat(hiddens_c, dim=1)

    # Batched forward (no per-env loop!)
    with torch.inference_mode():
        dist, values, (new_h, new_c) = agent.network.forward(
            states_batch_normalized,
            masks_batch,
            hidden=(batch_h, batch_c),
        )
        if deterministic:
            actions = dist.probs.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)

    # Store new hidden per-env
    # CRITICAL: .clone() is essential to avoid memory retention
    # - Slicing new_h[:, i:i+1, :] creates a VIEW into the batched tensor
    # - Views keep the entire parent tensor alive, preventing garbage collection
    # - .clone() creates an independent copy that allows the batch to be freed
    # - .detach() is technically redundant under inference_mode but defensive
    for i, env_state in enumerate(env_states):
        env_state.lstm_hidden = (
            new_h[:, i:i+1, :].detach().clone(),
            new_c[:, i:i+1, :].detach().clone(),
        )

    # Convert to lists for existing loop
    actions = actions.tolist()
    log_probs = log_probs.tolist()
    values = values.tolist()
else:
    # Existing non-recurrent batched action selection
    actions, log_probs, values = agent.network.get_action_batch(
        states_batch_normalized, masks_batch, deterministic=deterministic
    )
```

**Step 4: Update episode lifecycle hooks**

```python
# At episode start (when creating new env or resetting)
if recurrent:
    agent.recurrent_buffer.start_episode(env_id=env_idx)
    env_state.lstm_hidden = None  # Fresh hidden for new episode

# At episode end (when done=True)
if done:
    if recurrent:
        agent.recurrent_buffer.end_episode(env_id=env_idx)
        env_state.lstm_hidden = None  # Reset for next episode
```

**Step 5: Update transition storage**

```python
# Store transitions
if recurrent:
    agent.store_recurrent_transition(
        state=states_batch_normalized[env_idx],
        action=action_idx,
        log_prob=log_prob,
        value=value,
        reward=normalized_reward,
        done=done,
        action_mask=masks_batch[env_idx],
        env_id=env_idx,
    )
else:
    agent.store_transition(...)
```

**Step 6: Update PPO update call**

```python
# After batch collection
if recurrent:
    # Single epoch for recurrent to avoid hidden state drift
    # (Multiple epochs would cause ratio bias as policy changes between epochs)
    metrics = agent.update_recurrent(n_epochs=1)
else:
    metrics = agent.update(last_value=0.0, clear_buffer=True)
```

**Step 7: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(simic): integrate batched recurrent action selection in vectorized training"
```

---

## Phase 5: Integration Testing

### Task 5.1: Full Integration Test

**Files:**
- Create: `tests/test_recurrent_integration.py`

```python
"""Integration tests for recurrent PPO training."""

import torch
from esper.simic.ppo import PPOAgent
from esper.simic.networks import RecurrentActorCritic


class TestRecurrentIntegration:
    """End-to-end tests for recurrent policy."""

    def test_full_episode_training_loop(self):
        """Test complete training loop with recurrent policy."""
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=8,
            device='cpu',
        )

        # Simulate 2 episodes of 12 steps each
        for episode in range(2):
            agent.recurrent_buffer.start_episode(env_id=0)
            hidden = None

            for step in range(12):
                state = torch.randn(27)
                mask = torch.ones(7, dtype=torch.bool)
                mask[3] = False  # Mask one action

                action, log_prob, value, hidden = agent.get_action(state, mask, hidden)

                agent.store_recurrent_transition(
                    state=state,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=0.1 * (step + 1),
                    done=(step == 11),
                    action_mask=mask,
                    env_id=0,
                )

            agent.recurrent_buffer.end_episode(env_id=0)

        # Should have 2 episodes worth of chunks
        # 12 steps / 8 chunk = 2 chunks per episode = 4 total
        metrics = agent.update_recurrent(n_epochs=2)

        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert metrics['policy_loss'] != 0.0

    def test_multi_env_parallel_episodes(self):
        """Test parallel episodes from multiple environments."""
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=8,
            device='cpu',
        )

        # 4 parallel envs, each running 10-step episodes
        hiddens = {i: None for i in range(4)}

        for env_id in range(4):
            agent.recurrent_buffer.start_episode(env_id=env_id)

        for step in range(10):
            for env_id in range(4):
                state = torch.randn(27)
                mask = torch.ones(7, dtype=torch.bool)

                action, log_prob, value, hiddens[env_id] = agent.get_action(
                    state, mask, hiddens[env_id]
                )

                agent.store_recurrent_transition(
                    state=state,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=0.1,
                    done=(step == 9),
                    action_mask=mask,
                    env_id=env_id,
                )

        for env_id in range(4):
            agent.recurrent_buffer.end_episode(env_id=env_id)

        # Each env has 10 steps -> 2 chunks of 8 (padded)
        # 4 envs * 2 chunks = 8 chunks total
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')
        assert len(chunks) == 8

        metrics = agent.update_recurrent(n_epochs=2)
        assert 'policy_loss' in metrics

    def test_advantages_nonzero_with_rewards(self):
        """Critical: Verify GAE advantages are computed and non-zero."""
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=4,
            device='cpu',
        )

        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None

        for step in range(4):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)

            agent.store_recurrent_transition(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=float(step + 1),  # 1, 2, 3, 4
                done=(step == 3),
                action_mask=mask,
                env_id=0,
            )

        agent.recurrent_buffer.end_episode(env_id=0)
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')

        # Advantages MUST be non-zero (this was a critical bug)
        assert chunks[0]['advantages'].abs().sum() > 0

    def test_hidden_state_continuity_within_episode(self):
        """Verify hidden state evolves within episode."""
        torch.manual_seed(42)
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            device='cpu',
        )

        state = torch.randn(27)
        mask = torch.ones(7, dtype=torch.bool)

        # Get values at different points in "history"
        _, _, value_t0, hidden = agent.get_action(state, mask, None)

        # Build up context
        for _ in range(5):
            _, _, _, hidden = agent.get_action(torch.randn(27), mask, hidden)

        _, _, value_t5, _ = agent.get_action(state, mask, hidden)

        # Same state, different context -> different values
        assert not torch.isclose(torch.tensor(value_t0), torch.tensor(value_t5), atol=1e-4)

    def test_episode_boundary_resets_hidden(self):
        """Hidden state should reset at episode boundaries."""
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            device='cpu',
        )

        # Get hidden after some steps
        hidden = None
        for _ in range(5):
            _, _, _, hidden = agent.get_action(
                torch.randn(27), torch.ones(7, dtype=torch.bool), hidden
            )

        assert hidden is not None
        assert hidden[0].abs().sum() > 0  # Hidden evolved

        # Episode boundary: reset to None
        hidden = None  # This is what vectorized.py does on done=True

        # Get fresh hidden
        _, _, _, fresh_hidden = agent.get_action(
            torch.randn(27), torch.ones(7, dtype=torch.bool), hidden
        )

        # Fresh should be from zeros (episode start)
        # The network will have non-zero hidden after forward, but it started from zeros

    def test_ratio_near_one_before_update(self):
        """PPO ratio should be ~1.0 before any gradient updates (same policy).

        This is a CRITICAL test - if this fails, the hidden state handling
        is broken and the PPO ratio is biased.
        """
        torch.manual_seed(42)
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=4,
            device='cpu',
        )

        # Collect episode
        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(4):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=0.1, done=(i == 3), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        # Compute GAE and get chunks
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')
        chunk = chunks[0]

        # Recompute log_probs with SAME initial hidden (zeros for episode start)
        with torch.no_grad():
            log_probs_recomputed, _, _, _ = agent.network.evaluate_actions(
                chunk['states'],
                chunk['actions'],
                chunk['action_masks'],
                hidden=(chunk['initial_hidden_h'], chunk['initial_hidden_c']),
            )

        # Extract valid (non-padded) positions
        valid = chunk['valid_mask'].squeeze(0)
        old_log_probs = chunk['old_log_probs'].squeeze(0)[valid]
        new_log_probs = log_probs_recomputed.squeeze(0)[valid]

        # Ratio should be ~1.0 (same policy, same hidden)
        # Note: Using atol=1e-3 due to floating point accumulation in LSTM
        # (tanh/sigmoid approximations, softmax numerical stability, etc.)
        ratio = torch.exp(new_log_probs - old_log_probs)
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3), (
            f"Ratio should be ~1.0 before updates, got {ratio}"
        )

    def test_gradient_flows_to_early_timesteps(self):
        """CRITICAL: Verify BPTT propagates gradients to early timesteps.

        If this test fails, the LSTM isn't learning temporal dependencies.
        Gradients should flow from later rewards back to early policy params.

        This test verifies TWO things:
        1. GAE correctly bootstraps reward from final step to early steps (advantages non-zero)
        2. LSTM recurrent weights change (gradients flowed through time via weight_hh)
        """
        torch.manual_seed(42)
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=8,
            device='cpu',
        )

        # Collect episode with reward ONLY at LAST step
        # If BPTT works, early steps should have non-zero advantages (GAE bootstrap)
        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(8):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            # Reward structure: ONLY last step gets reward
            reward = 10.0 if i == 7 else 0.0
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=reward, done=(i == 7), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        # VERIFICATION 1: Early timesteps have non-zero advantage from GAE bootstrap
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')
        advantages = chunks[0]['advantages'].squeeze(0)

        # Early timesteps should have non-zero advantage from gamma-discounted final reward
        assert advantages[0].abs() > 0.01, (
            f"Early timestep advantage is ~0 ({advantages[0]:.4f}). "
            f"GAE not bootstrapping reward backward through time!"
        )

        # Get LSTM weights before update
        lstm_weight_hh_before = agent.network.lstm.weight_hh_l0.clone()
        lstm_weight_ih_before = agent.network.lstm.weight_ih_l0.clone()

        # Run update (need to re-add data since compute_gae consumed it for chunks)
        # Actually, update_recurrent calls compute_gae internally, so recreate episode
        agent.recurrent_buffer.clear()
        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        torch.manual_seed(42)  # Same seed for reproducibility
        for i in range(8):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            reward = 10.0 if i == 7 else 0.0
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=reward, done=(i == 7), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        metrics = agent.update_recurrent(n_epochs=1)

        # VERIFICATION 2: LSTM recurrent weights changed (BPTT worked)
        lstm_weight_hh_after = agent.network.lstm.weight_hh_l0
        weight_hh_diff = (lstm_weight_hh_after - lstm_weight_hh_before).abs().sum()

        # weight_hh connects h(t) -> h(t+1), so if it changed, gradients flowed through time
        assert weight_hh_diff > 1e-6, (
            f"LSTM weight_hh unchanged ({weight_hh_diff:.2e}). "
            f"Gradients not flowing through recurrent connection - BPTT broken!"
        )

        # VERIFICATION 3: Input weights also changed (full gradient flow)
        lstm_weight_ih_after = agent.network.lstm.weight_ih_l0
        weight_ih_diff = (lstm_weight_ih_after - lstm_weight_ih_before).abs().sum()
        assert weight_ih_diff > 1e-6, (
            f"LSTM weight_ih unchanged ({weight_ih_diff:.2e}). "
            f"Gradients not reaching input projection!"
        )

    def test_value_targets_equal_advantages_plus_values(self):
        """Verify GAE relationship: returns = advantages + old_values."""
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=4,
            device='cpu',
        )

        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(4):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=float(i + 1), done=(i == 3), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')
        chunk = chunks[0]

        # GAE property: returns = advantages + values
        computed_returns = chunk['advantages'] + chunk['old_values']
        assert torch.allclose(computed_returns, chunk['returns'], atol=1e-5), (
            f"GAE violated: returns != advantages + values"
        )

    # === Edge Case Tests (v4) ===

    def test_single_step_episode(self):
        """Single-step episode should work correctly."""
        agent = PPOAgent(
            state_dim=27, action_dim=7, recurrent=True,
            lstm_hidden_dim=64, chunk_length=4, device='cpu',
        )

        agent.recurrent_buffer.start_episode(env_id=0)
        state = torch.randn(27)
        mask = torch.ones(7, dtype=torch.bool)
        action, log_prob, value, hidden = agent.get_action(state, mask, None)
        agent.store_recurrent_transition(
            state=state, action=action, log_prob=log_prob, value=value,
            reward=1.0, done=True, action_mask=mask, env_id=0,
        )
        agent.recurrent_buffer.end_episode(env_id=0)

        # Should produce 1 chunk (padded)
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')
        assert len(chunks) == 1
        assert chunks[0]['valid_mask'].sum() == 1  # Only 1 valid step

    def test_exact_chunk_length_episode(self):
        """Episode exactly matching chunk_length needs no padding."""
        chunk_length = 4
        agent = PPOAgent(
            state_dim=27, action_dim=7, recurrent=True,
            lstm_hidden_dim=64, chunk_length=chunk_length, device='cpu',
        )

        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(chunk_length):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=0.1, done=(i == chunk_length - 1), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')

        assert len(chunks) == 1
        assert chunks[0]['valid_mask'].all()  # All steps valid (no padding)

    def test_approx_kl_returned_in_metrics(self):
        """update_recurrent should return approx_kl for diagnostics."""
        agent = PPOAgent(
            state_dim=27, action_dim=7, recurrent=True,
            lstm_hidden_dim=64, chunk_length=4, device='cpu',
        )

        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(4):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=0.1, done=(i == 3), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        metrics = agent.update_recurrent(n_epochs=1)

        assert 'approx_kl' in metrics
        # approx_kl should be near 0 before any policy drift
        assert abs(metrics['approx_kl']) < 0.1
```

**Step 2: Run tests**

```bash
uv run pytest tests/test_recurrent_integration.py -v
```

**Step 3: Commit**

```bash
git add tests/test_recurrent_integration.py
git commit -m "test: add comprehensive integration tests for recurrent PPO"
```

---

## Summary

| Phase | Task | Files Modified | Key Fix |
|-------|------|----------------|---------|
| 0 | API Migration | src/, tests/ | Update all get_action() call sites |
| 1 | RecurrentActorCritic | networks.py | No hasattr, proper shapes |
| 2 | RecurrentRolloutBuffer | buffers.py | Per-env storage, GAE in chunks |
| 3.1 | PPOAgent recurrent | ppo.py | Batched chunks, value_coef |
| 3.2 | TrainingConfig | config.py | Recurrent params in config |
| 4 | Vectorized training | vectorized.py | Batched action selection |
| 5 | Integration tests | tests/ | Critical bug coverage |

**Total commits:** 8

**Key improvements (v3 -> v4):**
- **CRITICAL: chunk_length=25** matches max_epochs default (was incorrectly 20)
- **Phase 0: API migration** - find and update all get_action() call sites first
- Single PPO epoch default with info log for n_epochs>1 (not warning - industry standard)
- Per-environment step lists (no interleaving corruption)
- GAE wired through to chunks correctly
- Batched chunk processing (GPU efficiency)
- Value clipping in recurrent update (parity with feedforward)
- No unauthorized hasattr usage
- Device-aware hidden initialization
- value_coef from agent config
- Comprehensive test coverage including ratio=1.0 test for hidden state correctness
- Explicit `.detach().clone()` for hidden state storage (prevents memory leak)
- **v4: `.detach().clone()` in single-env get_action()** (was missing in v3)
- Graceful mid-episode chunking handling (warning instead of crash)
- **v4: TrainingConfig.__post_init__** validates chunk_length >= max_epochs
- Gradient flow test verifies BPTT reaches early timesteps
- Relaxed ratio test tolerance (atol=1e-3 for float precision)
- **v4: train_steps increment** in update_recurrent() for entropy annealing
- **v4: approx_kl tracking** for training diagnostics
- **v4: Entropy normalization test** verifies [0,1] range
- **v4: Multi-layer LSTM bias init** documented for all layers
- **v4: Learning rate sensitivity note** added
- **v4: Edge case tests** (single step, exact chunk match)

**Critical design decisions:**
1. **chunk_length >= episode_length**: Avoids mid-episode chunking entirely, eliminating
   the need to store/restore hidden states at chunk boundaries. With chunk_length=25
   and episodes of 25 steps (max_epochs default), each episode is exactly one chunk.

2. **Single PPO epoch (default)**: Safest for recurrent PPO. However, n_epochs=2-4
   is viable because PPO clipping limits policy divergence. CleanRL uses n_epochs=4.
   Use n_epochs=1 for correctness guarantees, higher for sample efficiency.

3. **Graceful mid-episode handling**: If chunk_length < episode_length, emit warning
   and use zeros for initial hidden (suboptimal but won't crash). For correct BPTT,
   either increase chunk_length or implement burn-in BPTT (see upgrade path below).

4. **Explicit hidden state management**: Use `.detach().clone()` when storing hidden
   states to prevent memory leaks from computation graph accumulation and ensure
   independent tensors (not views into larger batched tensor).

---

## Upgrade Path: Burn-In BPTT (Future Enhancement)

If mid-episode chunking becomes necessary (chunk_length < max_epochs), implement burn-in BPTT:

```python
# Concept: Run initial steps WITHOUT gradients to "warm up" hidden state
def get_chunks_with_burnin(self, device, burn_in_length: int = 5):
    """Get chunks with burn-in for mid-episode chunks.

    For chunks that don't start at episode boundary:
    1. Store hidden state at previous chunk boundary
    2. During training, run burn_in_length steps with torch.no_grad()
    3. Then run remaining steps WITH gradients

    This preserves temporal context without full-episode BPTT memory cost.
    """
    # Implementation would:
    # - Store hidden at each chunk boundary during rollout
    # - Include burn_in_states in chunk dict
    # - In evaluate_actions, run burn-in forward WITHOUT grad, then continue WITH grad
```

**Trade-offs:**
- Burn-in adds compute overhead (re-run burn_in_length steps per chunk)
- But enables longer episodes with bounded memory
- OpenAI Five used 1000+ step sequences with burn-in BPTT

**Current approach (chunk_length=25 = max_epochs) avoids this complexity entirely.**

---

## torch.compile Compatibility Note

The current implementation should be `torch.compile` compatible, but watch for:

1. **Graph breaks** from conditional logic in `forward()` (squeeze_output branching)
2. **Dynamic shapes** from variable batch sizes

For production, consider:
```python
# After training is stable, wrap for inference speedup
model = torch.compile(agent.network, mode="reduce-overhead")
```

Test compile compatibility with:
```bash
TORCH_COMPILE_DEBUG=1 uv run python -c "
import torch
from esper.simic.networks import RecurrentActorCritic
net = RecurrentActorCritic(27, 7)
compiled = torch.compile(net)
x = torch.randn(1, 10, 27)
mask = torch.ones(1, 10, 7, dtype=torch.bool)
compiled(x, mask)  # Check for graph breaks
"
```
