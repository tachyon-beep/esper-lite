# Simic Deep Dive Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical, high, and medium-priority issues identified in the Simic subsystem deep dive by code review, PyTorch, and DRL expert agents.

**Architecture:** Surgical fixes to existing files with no new abstractions. Focus on numerical stability, training correctness, and dead code removal per CLAUDE.md policy.

**Tech Stack:** Python 3.13, PyTorch 2.9, pytest, hypothesis

**Expert Review Status:** ✅ Reviewed by PyTorch Expert and DRL Expert (2025-12-13)

---

## Task 1: Implement KL Early Stopping (CRITICAL)

The PPO update loop has `early_stopped = False` that never becomes `True`. `approx_kl` is always 0.0. Without KL stopping, the policy can diverge during multi-epoch updates.

**Files:**
- Modify: `src/esper/simic/ppo.py:366-475`
- Test: `tests/simic/test_ppo.py`

**Step 1: Write the failing test**

Add to `tests/simic/test_ppo.py`:

```python
def test_kl_early_stopping_triggers():
    """Verify approx_kl is computed and can trigger early stopping."""
    agent = PPOAgent(
        state_dim=35,
        num_envs=2,
        max_steps_per_env=5,
        target_kl=0.001,  # Very low to ensure triggering
        recurrent_n_epochs=5,  # Multiple epochs to allow early stop
        compile_network=False,
    )

    # Fill buffer with synthetic data
    hidden = agent.network.get_initial_hidden(1, torch.device(agent.device))
    for env_id in range(2):
        agent.buffer.start_episode(env_id)
        for step in range(5):
            state = torch.randn(1, 35, device=agent.device)
            masks = {
                "slot": torch.ones(1, 3, dtype=torch.bool, device=agent.device),
                "blueprint": torch.ones(1, 5, dtype=torch.bool, device=agent.device),
                "blend": torch.ones(1, 3, dtype=torch.bool, device=agent.device),
                "op": torch.ones(1, 7, dtype=torch.bool, device=agent.device),
            }
            actions, log_probs, value, hidden = agent.network.get_action(
                state, hidden,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                blend_mask=masks["blend"],
                op_mask=masks["op"],
            )
            agent.buffer.add_step(
                env_id=env_id,
                state=state.squeeze(0),
                slot_action=actions["slot"].item(),
                blueprint_action=actions["blueprint"].item(),
                blend_action=actions["blend"].item(),
                op_action=actions["op"].item(),
                slot_log_prob=log_probs["slot"].item(),
                blueprint_log_prob=log_probs["blueprint"].item(),
                blend_log_prob=log_probs["blend"].item(),
                op_log_prob=log_probs["op"].item(),
                value=value.item(),
                reward=1.0,
                done=step == 4,
                truncated=False,
                slot_mask=masks["slot"].squeeze(0),
                blueprint_mask=masks["blueprint"].squeeze(0),
                blend_mask=masks["blend"].squeeze(0),
                op_mask=masks["op"].squeeze(0),
                hidden_h=hidden[0].squeeze(1),
                hidden_c=hidden[1].squeeze(1),
                bootstrap_value=0.0 if step == 4 else None,
            )
        agent.buffer.end_episode(env_id)

    metrics = agent.update(clear_buffer=True)

    # approx_kl must be computed (not always 0.0)
    assert "approx_kl" in metrics, "approx_kl should be in metrics"
    # With very low target_kl, early stopping should trigger
    assert "early_stop_epoch" in metrics or metrics.get("approx_kl", 0) > 0, \
        "Either early stopping triggered or KL was computed"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/simic/test_ppo.py::test_kl_early_stopping_triggers -v`
Expected: FAIL with KeyError or assertion error (approx_kl not in metrics)

**Step 3: Implement KL computation and early stopping**

In `src/esper/simic/ppo.py`, modify the `update` method. Find line ~458-464 (after computing policy_loss) and add KL computation:

```python
            # Track metrics
            joint_ratio = per_head_ratios["op"]  # Use op ratio as representative
            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(-entropy_loss.item())
            metrics["ratio_mean"].append(joint_ratio.mean().item())
            metrics["ratio_max"].append(joint_ratio.max().item())

            # Compute approximate KL divergence using the log-ratio trick:
            # KL(old||new) ≈ E[(ratio - 1) - log(ratio)]
            # This is the "KL3" estimator from Schulman's TRPO/PPO papers
            # Average across all heads for factored policy (PyTorch expert review)
            with torch.no_grad():
                head_kls = []
                for key in ["slot", "blueprint", "blend", "op"]:
                    log_ratio = log_probs[key] - old_log_probs[key]
                    head_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    head_kls.append(head_kl)
                approx_kl = torch.stack(head_kls).mean().item()
                metrics["approx_kl"].append(approx_kl)

                # Clip fraction: how often clipping was active
                clip_fraction = ((joint_ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                metrics["clip_fraction"].append(clip_fraction)

                # Early stopping if KL divergence exceeds threshold
                # 1.5x multiplier is standard (OpenAI baselines, Stable-Baselines3)
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    early_stopped = True
                    metrics["early_stop_epoch"] = [epoch_i]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/simic/test_ppo.py::test_kl_early_stopping_triggers -v`
Expected: PASS

**Step 5: Run full PPO test suite**

Run: `pytest tests/simic/test_ppo.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo.py
git commit -m "feat(simic): implement KL early stopping and clip_fraction tracking

- Add approx_kl computation using KL3 estimator (averaged across all heads)
- Add clip_fraction metric for PPO health monitoring
- Implement early stopping when approx_kl > 1.5 * target_kl
- Fixes CRITICAL issue C1 from deep dive audit

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Fix Float("-inf") Masking for torch.compile (CRITICAL)

Using `float("-inf")` causes issues with FP16/BF16 and torch.compile.

**IMPORTANT (PyTorch Expert Review):** Do NOT use `torch.finfo(dtype).min` - it can cause softmax overflow! When max valid logit is +5.0 and masked logits are -65504, after softmax's max-subtraction the valid logits become 0 and masked become -65509, which overflows FP16's range and causes NaN.

**Use `-1e4` instead** - a large negative that won't overflow after softmax normalization.

**Files:**
- Modify: `src/esper/simic/tamiyo_network.py:197-205`
- Test: `tests/simic/test_tamiyo_network.py` (create if not exists)

**Step 1: Write the failing test**

Create `tests/simic/test_tamiyo_network.py`:

```python
"""Tests for FactoredRecurrentActorCritic network."""

import torch
import pytest
from esper.simic.tamiyo_network import FactoredRecurrentActorCritic


def test_masking_produces_valid_softmax():
    """Verify masking produces valid probabilities after softmax.

    Critical test from PyTorch expert review: must verify softmax doesn't
    produce NaN/Inf, not just that mask value is finite.
    """
    net = FactoredRecurrentActorCritic(state_dim=35)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        net_typed = net.to(dtype)
        state = torch.randn(2, 3, 35, dtype=dtype)

        # Mask that disables some actions
        slot_mask = torch.ones(2, 3, 3, dtype=torch.bool)
        slot_mask[:, :, 1] = False  # Mask out middle action

        output = net_typed.forward(state, slot_mask=slot_mask)
        probs = torch.softmax(output["slot_logits"], dim=-1)

        # Masked positions should have ~0 probability
        assert (probs[:, :, 1] < 1e-6).all(), \
            f"Masked action should have near-zero probability with {dtype}"
        # Valid positions should have valid probabilities
        assert not torch.isnan(probs).any(), \
            f"Softmax should not produce NaN with {dtype}"
        assert not torch.isinf(probs).any(), \
            f"Softmax should not produce Inf with {dtype}"
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 3, dtype=dtype), atol=1e-3), \
            f"Probabilities should sum to 1 with {dtype}"


def test_forward_with_all_dtypes():
    """Verify forward pass works with float16, bfloat16, float32."""
    net = FactoredRecurrentActorCritic(state_dim=35)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        net_typed = net.to(dtype)
        state = torch.randn(2, 3, 35, dtype=dtype)
        slot_mask = torch.ones(2, 3, 3, dtype=torch.bool)
        slot_mask[:, :, 1] = False  # Mask out middle action

        output = net_typed.forward(state, slot_mask=slot_mask)

        # Should not contain inf or nan
        for key in ["slot_logits", "blueprint_logits", "blend_logits", "op_logits", "value"]:
            tensor = output[key]
            assert not torch.isinf(tensor).any(), f"{key} contains inf with {dtype}"
            assert not torch.isnan(tensor).any(), f"{key} contains nan with {dtype}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/simic/test_tamiyo_network.py -v`
Expected: FAIL (masked values cause softmax issues with FP16)

**Step 3: Fix the masking to use safe large negative**

In `src/esper/simic/tamiyo_network.py`, add constant at module level (around line 33):

```python
# Mask value for invalid actions. Use -1e4 (not -inf or dtype.min) because:
# 1. float("-inf") causes FP16 saturation issues
# 2. torch.finfo(dtype).min can cause softmax overflow after max-subtraction
# 3. -1e4 is large enough to zero out softmax but small enough to avoid overflow
# This is the standard practice in HuggingFace Transformers and PyTorch attention.
_MASK_VALUE = -1e4
```

Then replace lines 197-205:

```python
        # Apply masks (set invalid actions to large negative for softmax zeroing)
        # Using -1e4 instead of -inf or dtype.min to avoid FP16 overflow issues
        # (PyTorch expert review: dtype.min can cause NaN after softmax normalization)
        if slot_mask is not None:
            slot_logits = slot_logits.masked_fill(~slot_mask, _MASK_VALUE)
        if blueprint_mask is not None:
            blueprint_logits = blueprint_logits.masked_fill(~blueprint_mask, _MASK_VALUE)
        if blend_mask is not None:
            blend_logits = blend_logits.masked_fill(~blend_mask, _MASK_VALUE)
        if op_mask is not None:
            op_logits = op_logits.masked_fill(~op_mask, _MASK_VALUE)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/simic/test_tamiyo_network.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/tamiyo_network.py tests/simic/test_tamiyo_network.py
git commit -m "fix(simic): use -1e4 for action masking instead of -inf

Replace float('-inf') with -1e4 constant for:
- FP16/BF16 mixed precision compatibility
- torch.compile tracing compatibility
- Numerical stability (avoids softmax overflow after max-subtraction)

Note: torch.finfo(dtype).min also causes overflow - use moderate large negative.
Standard practice from HuggingFace Transformers.

Fixes CRITICAL issue C2 from deep dive audit

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Add Epsilon Guard for Entropy Normalization (CRITICAL)

If `num_slots=1`, then `log(1)=0`, causing division by near-zero (1e-8) which scales entropy by 1e8.

**Files:**
- Modify: `src/esper/simic/tamiyo_network.py:81-87, 314-315`
- Test: `tests/simic/test_tamiyo_network.py`

**Step 1: Write the failing test**

Add to `tests/simic/test_tamiyo_network.py`:

```python
def test_entropy_normalization_with_single_action():
    """Verify entropy normalization handles single-action case gracefully."""
    # Edge case: num_slots=1 means log(1)=0, could cause division issues
    net = FactoredRecurrentActorCritic(
        state_dim=35,
        num_slots=1,  # log(1) = 0!
        num_blueprints=5,
        num_blends=3,
        num_ops=7,
    )

    state = torch.randn(2, 3, 35)
    actions = {
        "slot": torch.zeros(2, 3, dtype=torch.long),  # Only one option
        "blueprint": torch.randint(0, 5, (2, 3)),
        "blend": torch.randint(0, 3, (2, 3)),
        "op": torch.randint(0, 7, (2, 3)),
    }

    log_probs, values, entropy, hidden = net.evaluate_actions(state, actions)

    # Entropy for single-action head should be 0 (no uncertainty), not inf/nan
    assert not torch.isnan(entropy["slot"]).any(), "Entropy should not be NaN"
    assert not torch.isinf(entropy["slot"]).any(), "Entropy should not be Inf"
    # With single action, normalized entropy should be 0 or 1 (not 1e8)
    assert entropy["slot"].abs().max() <= 1.0, f"Entropy out of range: {entropy['slot'].max()}"


def test_entropy_normalization_in_loss():
    """Verify entropy normalization doesn't blow up loss values."""
    net = FactoredRecurrentActorCritic(state_dim=35, num_slots=1)

    state = torch.randn(2, 3, 35)
    actions = {
        "slot": torch.zeros(2, 3, dtype=torch.long),
        "blueprint": torch.randint(0, 5, (2, 3)),
        "blend": torch.randint(0, 3, (2, 3)),
        "op": torch.randint(0, 7, (2, 3)),
    }

    log_probs, values, entropy, _ = net.evaluate_actions(state, actions)

    # Entropy loss should be bounded
    entropy_loss = sum(-ent.mean() for ent in entropy.values())
    assert entropy_loss.abs() < 100, f"Entropy loss too large: {entropy_loss}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/simic/test_tamiyo_network.py::test_entropy_normalization_with_single_action -v`
Expected: FAIL (entropy is inf or very large)

**Step 3: Fix entropy normalization**

In `src/esper/simic/tamiyo_network.py`, modify `__init__` (lines 81-87) to use a safer minimum:

```python
        # Max entropy for per-head normalization (different cardinalities)
        # Use max(log(n), 1.0) to prevent division-by-near-zero when n=1
        # When n=1, there's no uncertainty, so we set max_entropy=1.0 (entropy will be 0)
        # For n>=3: max_entropy=log(n), normalized entropy in [0, 1]
        # For n=2: max_entropy clamped to 1.0, normalized entropy in [0, 0.693]
        self.max_entropies = {
            "slot": max(math.log(num_slots), 1.0),
            "blueprint": max(math.log(num_blueprints), 1.0),
            "blend": max(math.log(num_blends), 1.0),
            "op": max(math.log(num_ops), 1.0),
        }
```

And update line 315 to remove the redundant epsilon guard:

```python
            # Normalize entropy by max possible (different head cardinalities)
            # The max() in __init__ ensures max_entropies[key] >= 1.0
            raw_entropy = dist.entropy().reshape(batch, seq)
            entropy[key] = raw_entropy / self.max_entropies[key]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/simic/test_tamiyo_network.py::test_entropy_normalization_with_single_action -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/simic/test_tamiyo_network.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/esper/simic/tamiyo_network.py tests/simic/test_tamiyo_network.py
git commit -m "fix(simic): prevent entropy normalization division by zero

When action dimension is 1, log(1)=0 causes division issues.
Use max(log(n), 1.0) to ensure safe normalization.

Fixes CRITICAL issue C3 from deep dive audit

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Fix Value Function Clipping Range (HIGH)

Value clipping uses `clip_ratio=0.2` which is designed for policy ratios (near 1.0), but value outputs can range from -10 to +50.

**DRL Expert Note:** Consider setting `clip_value=False` as default - research (Engstrom et al., 2020) found value clipping often hurts performance. However, for this fix we add the separate parameter and document the trade-off.

**Files:**
- Modify: `src/esper/simic/ppo.py:135-169, 433-441`
- Test: `tests/simic/test_ppo.py`

**Step 1: Write the failing test**

Add to `tests/simic/test_ppo.py`:

```python
def test_value_clipping_uses_appropriate_range():
    """Verify value clipping doesn't use the policy clip ratio."""
    agent = PPOAgent(
        state_dim=35,
        clip_ratio=0.2,  # Policy clip
        clip_value=True,
        value_clip=10.0,  # Should exist and be separate
        compile_network=False,
    )

    # Value clip should be much larger than policy clip
    # Note: Direct attribute access instead of hasattr per CLAUDE.md policy
    assert agent.value_clip == 10.0, "Agent should have value_clip=10.0"
    assert agent.value_clip > agent.clip_ratio, "Value clip should be larger than policy clip"


def test_value_clipping_disabled_option():
    """Verify clip_value=False disables value clipping entirely."""
    agent = PPOAgent(
        state_dim=35,
        clip_value=False,
        compile_network=False,
    )
    assert agent.clip_value is False, "clip_value should be configurable to False"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/simic/test_ppo.py::test_value_clipping_uses_appropriate_range -v`
Expected: FAIL (no value_clip attribute)

**Step 3: Add value_clip parameter**

In `src/esper/simic/ppo.py`, add the parameter to `__init__` (around line 154):

```python
        clip_value: bool = True,
        # Separate clip range for value function (larger than policy clip_ratio)
        # Note: Some research (Engstrom et al., 2020) suggests value clipping often
        # hurts performance. Consider clip_value=False if value learning is slow.
        value_clip: float = 10.0,
        max_grad_norm: float = 0.5,
```

And store it (around line 186):

```python
        self.clip_value = clip_value
        self.value_clip = value_clip
        self.max_grad_norm = max_grad_norm
```

Then update the value clipping logic (lines 433-439):

```python
            # Value loss
            valid_old_values = data["values"][valid_mask]
            if self.clip_value:
                # Use separate value_clip (not policy clip_ratio) since value scale differs
                # Value predictions can range from -10 to +50, so clip_ratio=0.2 is too tight
                values_clipped = valid_old_values + torch.clamp(
                    values - valid_old_values, -self.value_clip, self.value_clip
                )
                value_loss_unclipped = (values - valid_returns) ** 2
                value_loss_clipped = (values_clipped - valid_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = F.mse_loss(values, valid_returns)
```

Also add to the save config (line ~500):

```python
                'value_clip': self.value_clip,
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/simic/test_ppo.py::test_value_clipping_uses_appropriate_range -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/simic/test_ppo.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo.py
git commit -m "fix(simic): use separate value_clip parameter for value function

Policy clip_ratio (0.2) is designed for ratios near 1.0, but value
outputs can range from -10 to +50. Add value_clip=10.0 default.

Note: Consider clip_value=False if value learning is slow - some
research suggests value clipping can hurt performance.

Fixes HIGH issue H1 from deep dive audit

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Add non_blocking=True and Vectorize valid_mask (HIGH)

Two issues from PyTorch expert review:
1. All `.to(device)` calls are synchronous, serializing GPU transfers
2. The `valid_mask` construction uses a CPU-bound Python loop writing to GPU tensor

**Files:**
- Modify: `src/esper/simic/tamiyo_buffer.py:340-378`

**Step 1: Vectorize valid_mask construction**

In `src/esper/simic/tamiyo_buffer.py`, modify `get_batched_sequences` (around lines 345-351):

Replace:
```python
        # Build valid mask
        valid_mask = torch.zeros(
            self.num_envs, self.max_steps_per_env, dtype=torch.bool, device=device
        )
        for env_id in range(self.num_envs):
            valid_mask[env_id, : self.step_counts[env_id]] = True
```

With:
```python
        # Build valid mask (vectorized - avoids CPU loop with GPU tensor writes)
        # PyTorch expert review: Python loop writing to GPU forces sync each iteration
        step_counts_tensor = torch.tensor(self.step_counts, device=device)
        indices = torch.arange(self.max_steps_per_env, device=device).unsqueeze(0)
        valid_mask = indices < step_counts_tensor.unsqueeze(1)
```

**Step 2: Update buffer to use non_blocking transfers**

In `src/esper/simic/tamiyo_buffer.py`, modify `get_batched_sequences` return statement (lines 352-378):

```python
        # non_blocking=True enables overlapped CPU->GPU transfers.
        # Safe because first consumer is network.evaluate_actions() which
        # implicitly synchronizes via CUDA stream ordering.
        return {
            "states": self.states.to(device, non_blocking=True),
            "slot_actions": self.slot_actions.to(device, non_blocking=True),
            "blueprint_actions": self.blueprint_actions.to(device, non_blocking=True),
            "blend_actions": self.blend_actions.to(device, non_blocking=True),
            "op_actions": self.op_actions.to(device, non_blocking=True),
            "slot_log_probs": self.slot_log_probs.to(device, non_blocking=True),
            "blueprint_log_probs": self.blueprint_log_probs.to(device, non_blocking=True),
            "blend_log_probs": self.blend_log_probs.to(device, non_blocking=True),
            "op_log_probs": self.op_log_probs.to(device, non_blocking=True),
            "values": self.values.to(device, non_blocking=True),
            "rewards": self.rewards.to(device, non_blocking=True),
            "advantages": self.advantages.to(device, non_blocking=True),
            "returns": self.returns.to(device, non_blocking=True),
            "slot_masks": self.slot_masks.to(device, non_blocking=True),
            "blueprint_masks": self.blueprint_masks.to(device, non_blocking=True),
            "blend_masks": self.blend_masks.to(device, non_blocking=True),
            "op_masks": self.op_masks.to(device, non_blocking=True),
            "hidden_h": self.hidden_h.to(device, non_blocking=True),
            "hidden_c": self.hidden_c.to(device, non_blocking=True),
            "valid_mask": valid_mask,
            # Initial hidden states for each env (first timestep)
            "initial_hidden_h": self.hidden_h[:, 0, :, :].permute(1, 0, 2).contiguous().to(device, non_blocking=True),
            "initial_hidden_c": self.hidden_c[:, 0, :, :].permute(1, 0, 2).contiguous().to(device, non_blocking=True),
        }
```

**Step 3: Run existing buffer tests**

Run: `pytest tests/simic/test_tamiyo_buffer.py -v`
Expected: All PASS (non_blocking is backward compatible)

**Step 4: Commit**

```bash
git add src/esper/simic/tamiyo_buffer.py
git commit -m "perf(simic): vectorize valid_mask and use non_blocking transfers

Two optimizations from PyTorch expert review:
1. Vectorize valid_mask construction to avoid CPU loop with GPU tensor writes
2. Use non_blocking=True for overlapped CPU-GPU transfers

Fixes HIGH issue H3 from deep dive audit

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Ensure PBRS Gamma Consistency (MEDIUM)

`DEFAULT_GAMMA=0.99` in rewards.py differs from `gamma=0.995` in config.py. For PBRS policy invariance, they must match.

**Files:**
- Modify: `src/esper/simic/rewards.py:113-116`
- Modify: `src/esper/simic/config.py:53`

**Step 1: Align gamma values**

In `src/esper/simic/rewards.py`, line 115:

```python
# Default discount factor for PBRS. Must match TrainingConfig.gamma for
# proper telescoping (Ng et al., 1999). See config.py.
# DRL expert review: gamma_RL MUST equal gamma_PBRS for policy invariance.
DEFAULT_GAMMA = 0.995
```

**Step 2: Add cross-reference comment in config.py**

In `src/esper/simic/config.py`, line 53:

```python
    gamma: float = 0.995  # Must match rewards.DEFAULT_GAMMA for PBRS policy invariance
```

**Step 3: Run reward tests**

Run: `pytest tests/simic/test_rewards.py tests/properties/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/esper/simic/rewards.py src/esper/simic/config.py
git commit -m "fix(simic): align PBRS gamma with training config

PBRS requires gamma_rl == gamma_pbrs for policy invariance (Ng et al., 1999).
Changed DEFAULT_GAMMA from 0.99 to 0.995 to match TrainingConfig.

Fixes MEDIUM issue M1 from deep dive audit

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Delete Dead Code - factored_network.py (CLEANUP)

The non-recurrent `FactoredActorCritic` in `factored_network.py` is dead code - superseded by `FactoredRecurrentActorCritic` in `tamiyo_network.py`.

**Files:**
- Delete: `src/esper/simic/factored_network.py` (248 lines)
- Modify: `src/esper/simic/__init__.py` (remove export if present)

**Step 1: Verify no imports**

Run: `grep -r "from esper.simic.factored_network" src/ tests/`
Expected: No results (confirms dead code)

**Step 2: Delete the file**

```bash
rm src/esper/simic/factored_network.py
```

**Step 3: Remove from __init__.py if present**

Check `src/esper/simic/__init__.py` and remove any reference to `factored_network` or `FactoredActorCritic`.

**Step 4: Run test suite**

Run: `pytest tests/simic/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor(simic): delete dead factored_network.py

FactoredActorCritic superseded by FactoredRecurrentActorCritic in
tamiyo_network.py. Per CLAUDE.md No Legacy Code Policy.

Removes 248 lines of dead code from deep dive audit

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Delete Dead Code - curriculum.py (CLEANUP)

The `BlueprintCurriculum` (UCB1-based) is fully implemented but never wired to training.

**Files:**
- Delete: `src/esper/simic/curriculum.py` (181 lines)
- Modify: `src/esper/simic/__init__.py` (remove export if present)

**Step 1: Verify no imports**

Run: `grep -r "from esper.simic.curriculum\|BlueprintCurriculum" src/`
Expected: Only the file itself or __init__.py exports

**Step 2: Delete the file**

```bash
rm src/esper/simic/curriculum.py
```

**Step 3: Remove from __init__.py if present**

Check `src/esper/simic/__init__.py` and remove any reference to `curriculum` or `BlueprintCurriculum`.

**Step 4: Run test suite**

Run: `pytest tests/simic/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor(simic): delete unwired curriculum.py

BlueprintCurriculum UCB1 implementation never connected to training.
Per CLAUDE.md No Legacy Code Policy.

Removes 181 lines of unwired code from deep dive audit

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Delete Dead Code - simple_rewards.py (CLEANUP)

The `compute_simple_reward` function is only used in tests, not in actual training.

**Files:**
- Delete: `src/esper/simic/simple_rewards.py` (77 lines)
- Modify: `src/esper/simic/__init__.py` (remove export)
- Modify: Any tests that import it (update to use primary reward or delete test)

**Step 1: Check usage**

Run: `grep -r "simple_rewards\|compute_simple_reward" src/ tests/`
Expected: Only test files and __init__.py

**Step 2: Delete the file**

```bash
rm src/esper/simic/simple_rewards.py
```

**Step 3: Remove from __init__.py**

In `src/esper/simic/__init__.py`, remove any reference to `simple_rewards` or `compute_simple_reward`.

**Step 4: Update or delete tests that used it**

If tests import `compute_simple_reward`, either:
- Delete those tests (if they only tested the simple reward)
- Update them to use `compute_contribution_reward` instead

**Step 5: Run test suite**

Run: `pytest tests/ -v`
Expected: All PASS (or update tests as needed)

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor(simic): delete unwired simple_rewards.py

compute_simple_reward only used in tests, not in training pipeline.
Per CLAUDE.md No Legacy Code Policy.

Removes 77 lines of unwired code from deep dive audit

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Run Full Integration Tests

**Step 1: Run all simic tests**

Run: `pytest tests/simic/ -v`
Expected: All PASS

**Step 2: Run integration tests**

Run: `pytest tests/integration/ -v`
Expected: All PASS

**Step 3: Run property-based tests**

Run: `pytest tests/properties/ -v`
Expected: All PASS

**Step 4: Final commit summary**

Verify git log shows all 9 commits for the fixes.

---

## Summary

| Task | Issue | Severity | Lines Changed |
|------|-------|----------|---------------|
| 1 | KL Early Stopping | CRITICAL | +35 |
| 2 | Float("-inf") → -1e4 Masking | CRITICAL | ~15 |
| 3 | Entropy Normalization | CRITICAL | ~15 |
| 4 | Value Clip Range | HIGH | +20 |
| 5 | Non-blocking + Vectorize valid_mask | HIGH | ~25 |
| 6 | PBRS Gamma Consistency | MEDIUM | ~4 |
| 7 | Delete factored_network.py | CLEANUP | -248 |
| 8 | Delete curriculum.py | CLEANUP | -181 |
| 9 | Delete simple_rewards.py | CLEANUP | -77 |

**Net effect:** ~115 lines added, ~506 lines deleted = **~390 lines removed**

---

## Expert Review Changes Applied

| Change | Source | Applied |
|--------|--------|---------|
| Use `-1e4` instead of `finfo.min` for masking | PyTorch Expert | ✅ Task 2 |
| Test softmax validity, not just mask finiteness | PyTorch Expert | ✅ Task 2 |
| Average KL across all heads (not just op) | PyTorch Expert | ✅ Task 1 |
| Vectorize valid_mask construction | PyTorch Expert | ✅ Task 5 |
| Add sync safety comment for non_blocking | PyTorch Expert | ✅ Task 5 |
| Document clip_value=False option | DRL Expert | ✅ Task 4 |
| Remove unauthorized hasattr from test | PyTorch Expert | ✅ Task 4 |
| Add entropy loss magnitude test | PyTorch Expert | ✅ Task 3 |
