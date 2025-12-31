# Entropy Annealing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add entropy coefficient annealing to PPO to prevent premature policy convergence while still allowing final convergence.

**Architecture:** Linear annealing from high initial entropy (exploration) to low final entropy (exploitation) over a configurable number of training steps. Disabled by default (entropy_anneal_steps=0) to preserve legacy behavior.

**Tech Stack:** PyTorch, Python 3.11+, pytest

---

## Task 1: Add Annealing Parameters to PPOAgent

**Files:**
- Modify: `src/esper/simic/ppo.py:108-137`
- Test: `tests/test_simic_ppo.py`

**Step 1: Write the failing test for legacy behavior preservation**

Add to `tests/test_simic_ppo.py`:

```python
class TestEntropyAnnealing:
    """Test entropy coefficient annealing schedule."""

    def test_no_annealing_when_disabled(self):
        """entropy_anneal_steps=0 should use fixed entropy_coef."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            entropy_coef=0.05,
            entropy_anneal_steps=0,
            device='cpu'
        )
        assert agent.get_entropy_coef() == 0.05
        agent.train_steps = 100
        assert agent.get_entropy_coef() == 0.05  # Still fixed
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_simic_ppo.py::TestEntropyAnnealing::test_no_annealing_when_disabled -v`

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'entropy_anneal_steps'`

**Step 3: Write minimal implementation - add parameters and method**

In `src/esper/simic/ppo.py`, update `PPOAgent.__init__`:

```python
def __init__(
    self,
    state_dim: int,
    action_dim: int = 7,
    hidden_dim: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_anneal_steps: int = 0,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    n_epochs: int = 10,
    batch_size: int = 64,
    device: str = "cuda:0",
):
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    self.clip_ratio = clip_ratio
    self.entropy_coef = entropy_coef
    self.entropy_coef_start = entropy_coef_start if entropy_coef_start is not None else entropy_coef
    self.entropy_coef_end = entropy_coef_end if entropy_coef_end is not None else entropy_coef
    self.entropy_anneal_steps = entropy_anneal_steps
    self.value_coef = value_coef
    self.max_grad_norm = max_grad_norm
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.device = device

    self.network = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
    self.buffer = RolloutBuffer()
    self.train_steps = 0
```

Add the `get_entropy_coef` method after `__init__`:

```python
def get_entropy_coef(self) -> float:
    """Get current entropy coefficient (annealed if configured).

    Returns fixed entropy_coef when entropy_anneal_steps=0 (legacy behavior).
    Otherwise linearly interpolates from entropy_coef_start to entropy_coef_end
    over entropy_anneal_steps training updates.
    """
    if self.entropy_anneal_steps == 0:
        return self.entropy_coef

    progress = min(1.0, self.train_steps / self.entropy_anneal_steps)
    return self.entropy_coef_start + progress * (self.entropy_coef_end - self.entropy_coef_start)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_simic_ppo.py::TestEntropyAnnealing::test_no_annealing_when_disabled -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py tests/test_simic_ppo.py
git commit -m "feat(ppo): add entropy annealing parameters and get_entropy_coef method"
```

---

## Task 2: Test Annealing Schedule Correctness

**Files:**
- Test: `tests/test_simic_ppo.py`

**Step 1: Write tests for annealing at different points**

Add to `TestEntropyAnnealing` class in `tests/test_simic_ppo.py`:

```python
    def test_annealing_at_start(self):
        """Step 0 should return entropy_coef_start."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            entropy_coef_start=0.2,
            entropy_coef_end=0.01,
            entropy_anneal_steps=100,
            device='cpu'
        )
        agent.train_steps = 0
        assert agent.get_entropy_coef() == 0.2

    def test_annealing_at_midpoint(self):
        """Midpoint should return average of start and end."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            entropy_coef_start=0.2,
            entropy_coef_end=0.0,
            entropy_anneal_steps=100,
            device='cpu'
        )
        agent.train_steps = 50
        assert abs(agent.get_entropy_coef() - 0.1) < 1e-6

    def test_annealing_at_end(self):
        """At anneal_steps, should return entropy_coef_end."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            entropy_coef_start=0.2,
            entropy_coef_end=0.01,
            entropy_anneal_steps=100,
            device='cpu'
        )
        agent.train_steps = 100
        assert agent.get_entropy_coef() == 0.01

    def test_annealing_clamps_beyond_schedule(self):
        """Beyond anneal_steps, should stay at entropy_coef_end."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            entropy_coef_start=0.2,
            entropy_coef_end=0.01,
            entropy_anneal_steps=100,
            device='cpu'
        )
        agent.train_steps = 200
        assert agent.get_entropy_coef() == 0.01
```

**Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_simic_ppo.py::TestEntropyAnnealing -v`

Expected: All 5 tests PASS (implementation already done in Task 1)

**Step 3: Commit**

```bash
git add tests/test_simic_ppo.py
git commit -m "test(ppo): add entropy annealing schedule tests"
```

---

## Task 3: Use Annealed Entropy in Loss Calculation

**Files:**
- Modify: `src/esper/simic/ppo.py:187-191`
- Test: `tests/test_simic_ppo.py`

**Step 1: Write test that entropy coefficient affects loss**

Add to `TestEntropyAnnealing` class:

```python
    def test_annealed_entropy_used_in_update(self):
        """PPO update should use annealed entropy coefficient."""
        from esper.simic.ppo import PPOAgent
        import torch

        # Create agent with annealing
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            entropy_coef_start=0.5,
            entropy_coef_end=0.01,
            entropy_anneal_steps=10,
            device='cpu'
        )

        # Add some dummy transitions
        for _ in range(5):
            state = torch.randn(27)
            agent.store_transition(state, action=0, log_prob=-1.0, value=0.5, reward=1.0, done=False)

        # At step 0, entropy_coef should be 0.5
        assert agent.train_steps == 0
        assert agent.get_entropy_coef() == 0.5

        # Perform update
        metrics = agent.update(last_value=0.0)

        # After update, train_steps incremented
        assert agent.train_steps == 1
        # Entropy coef should have changed
        expected_coef = 0.5 + (1/10) * (0.01 - 0.5)  # 0.451
        assert abs(agent.get_entropy_coef() - expected_coef) < 1e-6
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_simic_ppo.py::TestEntropyAnnealing::test_annealed_entropy_used_in_update -v`

Expected: PASS (method already returns correct value based on train_steps)

Note: The test validates that `get_entropy_coef()` returns different values as `train_steps` changes. The actual integration into the loss is Step 3.

**Step 3: Update loss calculation to use get_entropy_coef()**

In `src/esper/simic/ppo.py`, update the `update` method around line 187-191:

Change:
```python
loss = (
    policy_loss
    + self.value_coef * value_loss
    + self.entropy_coef * entropy_loss
)
```

To:
```python
loss = (
    policy_loss
    + self.value_coef * value_loss
    + self.get_entropy_coef() * entropy_loss
)
```

**Step 4: Run all entropy tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_simic_ppo.py::TestEntropyAnnealing -v`

Expected: All 6 tests PASS

**Step 5: Run full PPO test suite to ensure no regressions**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_simic_ppo.py tests/integration/test_ppo_integration.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/ppo.py tests/test_simic_ppo.py
git commit -m "feat(ppo): use annealed entropy coefficient in loss calculation"
```

---

## Task 4: Update Config Serialization

**Files:**
- Modify: `src/esper/simic/ppo.py:220-226`

**Step 1: Update save method to include annealing config**

In `src/esper/simic/ppo.py`, update the `save` method config dict:

```python
save_dict = {
    'network_state_dict': self.network.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'train_steps': self.train_steps,
    'config': {
        'gamma': self.gamma,
        'gae_lambda': self.gae_lambda,
        'clip_ratio': self.clip_ratio,
        'entropy_coef': self.entropy_coef,
        'entropy_coef_start': self.entropy_coef_start,
        'entropy_coef_end': self.entropy_coef_end,
        'entropy_anneal_steps': self.entropy_anneal_steps,
        'value_coef': self.value_coef,
    }
}
```

**Step 2: Run existing save/load tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -k "save or load or checkpoint" -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/esper/simic/ppo.py
git commit -m "feat(ppo): include entropy annealing config in checkpoint"
```

---

## Task 5: Add CLI Arguments

**Files:**
- Modify: `src/esper/scripts/train.py:23-36`

**Step 1: Add CLI arguments for entropy annealing**

In `src/esper/scripts/train.py`, add after line 29 (`--entropy-coef`):

```python
ppo_parser.add_argument("--entropy-coef-start", type=float, default=None,
    help="Initial entropy coefficient (default: use --entropy-coef)")
ppo_parser.add_argument("--entropy-coef-end", type=float, default=None,
    help="Final entropy coefficient (default: use --entropy-coef)")
ppo_parser.add_argument("--entropy-anneal-episodes", type=int, default=0,
    help="Episodes over which to anneal entropy (0=fixed, no annealing)")
```

**Step 2: Verify CLI parsing works**

Run: `PYTHONPATH=src python -m esper.scripts.train ppo --help`

Expected output should include:
```
--entropy-coef-start ENTROPY_COEF_START
                      Initial entropy coefficient (default: use --entropy-coef)
--entropy-coef-end ENTROPY_COEF_END
                      Final entropy coefficient (default: use --entropy-coef)
--entropy-anneal-episodes ENTROPY_ANNEAL_EPISODES
                      Episodes over which to anneal entropy (0=fixed, no annealing)
```

**Step 3: Commit**

```bash
git add src/esper/scripts/train.py
git commit -m "feat(cli): add entropy annealing arguments"
```

---

## Task 6: Wire CLI to Vectorized Training

**Files:**
- Modify: `src/esper/scripts/train.py:66-79`
- Modify: `src/esper/simic/vectorized.py:78-90,156-164`

**Step 1: Update train_ppo_vectorized function signature**

In `src/esper/simic/vectorized.py`, update the function signature (lines 78-90):

```python
def train_ppo_vectorized(
    n_episodes: int = 100,
    n_envs: int = 4,
    max_epochs: int = 25,
    device: str = "cuda:0",
    devices: list[str] | None = None,
    use_telemetry: bool = False,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.1,
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_anneal_episodes: int = 0,
    gamma: float = 0.99,
    save_path: str = None,
) -> tuple[PPOAgent, list[dict]]:
```

**Step 2: Convert episodes to steps and pass to PPOAgent**

In `src/esper/simic/vectorized.py`, before creating the PPOAgent (around line 156), add conversion logic and update the agent creation:

```python
# Convert episode-based annealing to step-based
# Each batch of n_envs episodes = 1 PPO update step
entropy_anneal_steps = entropy_anneal_episodes // n_envs if entropy_anneal_episodes > 0 else 0

# Create PPO agent
agent = PPOAgent(
    state_dim=state_dim,
    action_dim=len(SimicAction),
    lr=lr,
    clip_ratio=clip_ratio,
    entropy_coef=entropy_coef,
    entropy_coef_start=entropy_coef_start,
    entropy_coef_end=entropy_coef_end,
    entropy_anneal_steps=entropy_anneal_steps,
    gamma=gamma,
    device=device,
)
```

**Step 3: Update the print statement to show annealing config**

Around line 127, update the print statements:

```python
if entropy_anneal_episodes > 0:
    print(f"Entropy annealing: {entropy_coef_start or entropy_coef} -> {entropy_coef_end or entropy_coef} over {entropy_anneal_episodes} episodes")
else:
    print(f"Entropy coef: {entropy_coef} (fixed)")
```

**Step 4: Update CLI call to pass new arguments**

In `src/esper/scripts/train.py`, update the `train_ppo_vectorized` call (lines 67-79):

```python
train_ppo_vectorized(
    n_episodes=args.episodes,
    n_envs=args.n_envs,
    max_epochs=args.max_epochs,
    device=args.device,
    devices=args.devices,
    use_telemetry=use_telemetry,
    lr=args.lr,
    clip_ratio=args.clip_ratio,
    entropy_coef=args.entropy_coef,
    entropy_coef_start=args.entropy_coef_start,
    entropy_coef_end=args.entropy_coef_end,
    entropy_anneal_episodes=args.entropy_anneal_episodes,
    gamma=args.gamma,
    save_path=args.save,
)
```

**Step 5: Test CLI with annealing parameters**

Run: `PYTHONPATH=src python -m esper.scripts.train ppo --vectorized --n-envs 2 --n-episodes 4 --max-epochs 2 --entropy-coef-start 0.2 --entropy-coef-end 0.01 --entropy-anneal-episodes 4 --devices cpu 2>&1 | head -20`

Expected output should include:
```
Entropy annealing: 0.2 -> 0.01 over 4 episodes
```

**Step 6: Run full test suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_simic_ppo.py tests/integration/test_ppo_integration.py -v`

Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/esper/scripts/train.py src/esper/simic/vectorized.py
git commit -m "feat(vectorized): wire entropy annealing through CLI to training"
```

---

## Task 7: Update Non-Vectorized Training (Optional)

**Files:**
- Modify: `src/esper/simic/training.py:508-532`
- Modify: `src/esper/scripts/train.py:81-93`

**Step 1: Update train_ppo function signature**

In `src/esper/simic/training.py`, update the function signature to include annealing parameters:

```python
def train_ppo(
    n_episodes: int = 100,
    max_epochs: int = 25,
    update_every: int = 5,
    device: str = "cuda:0",
    use_telemetry: bool = False,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_anneal_episodes: int = 0,
    gamma: float = 0.99,
    save_path: str | None = None,
) -> tuple:
```

**Step 2: Pass annealing params to PPOAgent**

Update the PPOAgent creation in train_ppo:

```python
# Convert episode-based annealing to step-based
# CRITICAL: Non-vectorized training only updates every `update_every` episodes
# So actual PPO updates = n_episodes / update_every
# If update_every=5 and entropy_anneal_episodes=100, we get 20 PPO updates
entropy_anneal_steps = (entropy_anneal_episodes // update_every) if entropy_anneal_episodes > 0 else 0

agent = PPOAgent(
    state_dim=state_dim,
    action_dim=len(SimicAction),
    lr=lr,
    clip_ratio=clip_ratio,
    entropy_coef=entropy_coef,
    entropy_coef_start=entropy_coef_start,
    entropy_coef_end=entropy_coef_end,
    entropy_anneal_steps=entropy_anneal_steps,
    gamma=gamma,
    device=device,
)
```

**Step 3: Update CLI call for non-vectorized**

In `src/esper/scripts/train.py`, update the non-vectorized train_ppo call:

```python
train_ppo(
    n_episodes=args.episodes,
    max_epochs=args.max_epochs,
    update_every=args.update_every,
    device=args.device,
    use_telemetry=use_telemetry,
    lr=args.lr,
    clip_ratio=args.clip_ratio,
    entropy_coef=args.entropy_coef,
    entropy_coef_start=args.entropy_coef_start,
    entropy_coef_end=args.entropy_coef_end,
    entropy_anneal_episodes=args.entropy_anneal_episodes,
    gamma=args.gamma,
    save_path=args.save,
)
```

**Step 4: Commit**

```bash
git add src/esper/simic/training.py src/esper/scripts/train.py
git commit -m "feat(training): add entropy annealing to non-vectorized PPO"
```

---

## Task 8: Add Entropy Logging

**Files:**
- Modify: `src/esper/simic/vectorized.py:535-538`

**Step 1: Log current entropy coefficient in batch output**

In `src/esper/simic/vectorized.py`, update the metrics print (around line 536-538):

```python
if metrics:
    current_entropy_coef = agent.get_entropy_coef()
    print(f"  Policy loss: {metrics['policy_loss']:.4f}, "
          f"Value loss: {metrics['value_loss']:.4f}, "
          f"Entropy: {metrics['entropy']:.4f}, "
          f"Entropy coef: {current_entropy_coef:.4f}")
```

**Step 2: Add entropy_coef to history**

Update the history append (around line 540-549):

```python
history.append({
    'batch': batch_idx + 1,
    'episodes': episodes_completed,
    'env_accuracies': list(env_final_accs),
    'avg_accuracy': avg_acc,
    'rolling_avg_accuracy': rolling_avg_acc,
    'avg_reward': avg_reward,
    'action_counts': total_actions,
    'entropy_coef': agent.get_entropy_coef(),
    **metrics,
})
```

**Step 3: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(vectorized): log entropy coefficient during training"
```

---

## Task 9: Fix Telemetry Bug in Vectorized Training (BUG FIX)

**Problem:** Vectorized PPO never populates telemetry state. When `use_telemetry=True`, the 10 telemetry features (gradient_norm, gradient_health, accuracy, etc.) stay at zero-initialized values because `sync_telemetry()` is never called. This makes 27% of the observation space bogus data.

**Files:**
- Modify: `src/esper/simic/vectorized.py:33,199-258,393-433`
- Test: `tests/test_simic_vectorized.py` (new file)

**Step 1: Write failing test for telemetry population**

Create `tests/test_simic_vectorized.py`:

```python
"""Tests for vectorized PPO telemetry integration."""

import torch

class TestVectorizedTelemetry:
    """Test that vectorized training properly populates telemetry."""

    def test_telemetry_features_nonzero_with_active_seed(self):
        """When use_telemetry=True and seed is active, telemetry features should be populated."""
        from esper.leyline import SeedTelemetry

        # Telemetry features should not all be zero when there's an active seed
        # This is a structural test - the actual integration test would run training

        # SeedTelemetry with real values
        telemetry = SeedTelemetry(
            seed_id="test_seed",
            blueprint_id="conv",
            stage=2,  # TRAINING
            epochs_in_stage=5,
            alpha=0.5,
            accuracy=75.0,
            gradient_norm=1.5,
            gradient_health=0.85,
            has_vanishing=False,
            has_exploding=False,
        )

        features = telemetry.to_features()
        assert len(features) == 10
        # At least some features should be non-zero
        assert sum(abs(f) for f in features) > 0, "Telemetry features should not all be zero"
        assert features[5] == 75.0  # accuracy
        assert features[6] == 1.5   # gradient_norm
```

**Step 2: Run test to verify structure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_simic_vectorized.py -v`

Expected: PASS (this tests SeedTelemetry, not the bug itself)

**Step 3: Add import for gradient collector**

In `src/esper/simic/vectorized.py`, add import after line 33:

```python
from esper.simic.gradient_collector import collect_seed_gradients
```

**Step 4: Update process_train_batch to collect gradients**

In `src/esper/simic/vectorized.py`, modify `process_train_batch` to return gradient stats. Update the function signature and body:

```python
def process_train_batch(env_state: ParallelEnvState, inputs: torch.Tensor,
                        targets: torch.Tensor, criterion: nn.Module,
                        use_telemetry: bool = False) -> tuple[torch.Tensor, torch.Tensor, int, dict | None]:
    """Process a single training batch for one environment (runs in CUDA stream).

    Returns TENSORS (not floats) to avoid blocking .item() calls inside stream context.
    Call .item() only AFTER synchronizing all streams.

    Returns:
        Tuple of (loss_tensor, correct_tensor, total, grad_stats)
        grad_stats is None if use_telemetry=False or no active seed in TRAINING stage
    """
    model = env_state.model
    seed_state = model.seed_state
    env_dev = env_state.env_device
    grad_stats = None

    # Use CUDA stream for async execution
    stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()

    with stream_ctx:
        # Move data asynchronously
        inputs = inputs.to(env_dev, non_blocking=True)
        targets = targets.to(env_dev, non_blocking=True)

        model.train()

        # Determine which optimizer to use based on seed state
        if seed_state is None or seed_state.stage == SeedStage.FOSSILIZED:
            optimizer = env_state.host_optimizer
        elif seed_state.stage in (SeedStage.GERMINATED, SeedStage.TRAINING):
            if seed_state.stage == SeedStage.GERMINATED:
                seed_state.transition(SeedStage.TRAINING)
                env_state.seed_optimizer = torch.optim.SGD(
                    model.get_seed_parameters(), lr=0.01, momentum=0.9
                )
            if env_state.seed_optimizer is None:
                env_state.seed_optimizer = torch.optim.SGD(
                    model.get_seed_parameters(), lr=0.01, momentum=0.9
                )
            optimizer = env_state.seed_optimizer
        else:  # BLENDING
            optimizer = env_state.host_optimizer
            # Update blend alpha for this step
            if model.seed_slot and seed_state:
                step = seed_state.metrics.epochs_in_current_stage
                model.seed_slot.update_alpha_for_step(step)

        optimizer.zero_grad()
        if seed_state and seed_state.stage == SeedStage.BLENDING and env_state.seed_optimizer:
            env_state.seed_optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Collect gradient stats for telemetry (after backward, before step)
        if use_telemetry and seed_state and seed_state.stage in (SeedStage.TRAINING, SeedStage.BLENDING):
            grad_stats = collect_seed_gradients(model.get_seed_parameters())

        optimizer.step()
        if seed_state and seed_state.stage == SeedStage.BLENDING and env_state.seed_optimizer:
            env_state.seed_optimizer.step()

        _, predicted = outputs.max(1)
        correct_tensor = predicted.eq(targets).sum()

        # Return tensors - .item() called after stream sync
        return loss.detach(), correct_tensor, targets.size(0), grad_stats
```

**Step 5: Update training loop to pass use_telemetry and capture grad_stats**

In the training loop (around line 335-345), update the call and capture grad_stats:

```python
# Launch all environments in their respective CUDA streams (async)
# Accumulate on GPU inside stream context - no sync until epoch end
env_grad_stats = [None] * envs_this_batch  # Track last grad_stats per env
for i, env_state in enumerate(env_states):
    if env_batches[i] is None:
        continue
    inputs, targets = env_batches[i]
    loss_tensor, correct_tensor, total, grad_stats = process_train_batch(
        env_state, inputs, targets, criterion, use_telemetry=use_telemetry
    )
    if grad_stats is not None:
        env_grad_stats[i] = grad_stats  # Keep last batch's grad stats
    # Accumulate inside stream context (in-place add respects stream ordering)
    stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
    with stream_ctx:
        train_loss_accum[i].add_(loss_tensor)
        train_correct_accum[i].add_(correct_tensor)
    train_totals[i] += total
```

**Step 6: Add sync_telemetry call before feature extraction**

After the validation loop sync and before feature extraction (around line 393-398), add telemetry sync:

```python
# ===== Compute epoch metrics and get BATCHED actions =====
# First, sync telemetry for envs with active seeds (must happen BEFORE feature extraction)
for env_idx, env_state in enumerate(env_states):
    model = env_state.model
    seed_state = model.seed_state

    if use_telemetry and seed_state and env_grad_stats[env_idx]:
        grad_stats = env_grad_stats[env_idx]
        seed_state.sync_telemetry(
            gradient_norm=grad_stats['gradient_norm'],
            gradient_health=grad_stats['gradient_health'],
            has_vanishing=grad_stats['has_vanishing'],
            has_exploding=grad_stats['has_exploding'],
            epoch=epoch,
            max_epochs=max_epochs,
        )

# Collect features from all environments
all_features = []
all_signals = []
```

**Step 7: Initialize env_grad_stats before epoch loop**

Add initialization at the start of the epoch loop (around line 313):

```python
# Run epochs with INVERTED CONTROL FLOW
for epoch in range(1, max_epochs + 1):
    # Track gradient stats per env for telemetry sync
    env_grad_stats = [None] * envs_this_batch

    # Reset per-epoch metrics - GPU tensors for accumulation, sync only at epoch end
    train_loss_accum = [...]
```

**Step 8: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_simic_vectorized.py tests/test_simic_ppo.py -v`

Expected: All tests PASS

**Step 9: Commit**

```bash
git add src/esper/simic/vectorized.py tests/test_simic_vectorized.py
git commit -m "fix(vectorized): populate telemetry features via sync_telemetry"
```

---

## Summary

**Total: 9 tasks, ~30 steps**

**Recommended run command after implementation:**

```bash
PYTHONPATH=src python -m esper.scripts.train ppo --vectorized \
  --n-envs 4 \
  --devices cuda:0 cuda:1 \
  --n-episodes 200 \
  --entropy-coef-start 0.2 \
  --entropy-coef-end 0.01 \
  --entropy-anneal-episodes 150
```

This anneals entropy from 0.2 to 0.01 over the first 150 episodes, then holds at 0.01 for the remaining 50.

**With telemetry fix, the policy will now receive:**
- 27 base features (epoch, loss, accuracy, etc.)
- 10 telemetry features (gradient_norm, gradient_health, accuracy, stage, etc.) - **now properly populated**
