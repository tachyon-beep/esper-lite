# Tamiyo Policy Migration Design

**Status:** Design Complete - Expert Reviewed - Ready for Implementation
**Date:** 2025-12-17
**Authors:** John + Claude (Brainstorming Session)
**Reviewers:** DRL Expert Agent, PyTorch Expert Agent

---

## Executive Summary

This document captures the design for migrating the policy network from Simic to Tamiyo, aligning the architecture with the biological metaphor where Tamiyo is the "Brain/Cortex" and Simic is "Evolution/RL Training".

**Key Outcome:** Tamiyo becomes hotswappable - different policy implementations (LSTM, MLP, Heuristic) can be selected at config-time, and different training algorithms (PPO, SAC, TD3) can be used without modifying Tamiyo.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Design Decisions (Brainstorming Summary)](#2-design-decisions-brainstorming-summary)
3. [Architecture Overview](#3-architecture-overview)
4. [PolicyBundle Protocol](#4-policybundle-protocol)
5. [Module Structure](#5-module-structure)
6. [Simic ↔ Tamiyo Interface](#6-simic--tamiyo-interface)
7. [Policy Registry](#7-policy-registry)
8. [Checkpointing](#8-checkpointing)
9. [Migration Plan](#9-migration-plan)
10. [Expert Review Feedback](#10-expert-review-feedback)
11. [Appendix: Full Brainstorming Session](#appendix-full-brainstorming-session)

---

## 1. Problem Statement

### Current State

Despite Tamiyo being documented as the "Brain/Cortex" for strategic decision-making, the actual policy implementation is scattered across Simic:

| Module | LOC | Contents |
|--------|-----|----------|
| `simic/agent/` | 1,766 | PPO algorithm, neural network, trajectory buffer |
| `simic/control/` | 895 | Feature extraction, action masking, normalization |
| `simic/training/` | 3,507 | Training loop, vectorized environments |
| **Simic total** | ~6,168 | The actual "brain" |
| `tamiyo/` | ~658 | Heuristic baseline + signal tracker |

The naming is also confusing:
- `simic/agent/tamiyo_network.py` - The neural network named after Tamiyo, but living in Simic
- `simic/agent/tamiyo_buffer.py` - Same issue

### Design Intent

- Tamiyo should own the policy and be "semi-hotswappable"
- The heuristic system is temporary and will be removed
- Without the policy, Tamiyo would be "nothing" after heuristic removal

---

## 2. Design Decisions (Brainstorming Summary)

The following decisions were made through collaborative brainstorming:

| Question | Decision | Rationale |
|----------|----------|-----------|
| Hotswap level? | **Config-time swapping** | Choose policy at startup via config. Checkpoint compatibility as nice-to-have. |
| Policy architectures? | **LSTM, MLP, Transformer, Heuristic** | Support multiple architectures for experimentation |
| Observation processing? | **PolicyBundle (C)** | Each policy bundles its own observation processor since input representations are tightly coupled to architecture |
| Heuristic fate? | **Keep as PolicyBundle** | Useful for ablations and debugging |
| Normalization? | **Stays in Simic** | Training infrastructure - Simic owns running stats |
| Input to Tamiyo? | **TrainingSignals (expanded)** | Clean contract, expanded to include everything PolicyBundle needs |
| Buffer location? | **Stays in Simic** | RL training infrastructure, not part of the "brain" |
| PPO location? | **Stays in Simic** | Training algorithm - Simic evolves Tamiyo |
| Algorithm support? | **On-policy AND off-policy** | Future flexibility for SAC/TD3 |
| Config mechanism? | **JSON config file** | Better for reproducibility than CLI flags |

### Scope Clarification: Tactical vs Strategic Tamiyo

During brainstorming, we clarified the hierarchical control architecture:

- **Tactical Tamiyo (this refactoring):** LSTM/MLP policy for seed lifecycle control (germinate, fossilize, cull, wait)
- **Strategic Tamiyo (future):** GNN-based controller that coordinates multiple tactical Tamiyos

This refactoring concerns **tactical Tamiyo only**. Strategic Tamiyo is a separate future system.

---

## 3. Architecture Overview

### Before (Current)

```
┌─────────────────────────────────────────────────┐
│                    SIMIC                         │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ PPO Agent   │  │ Network     │  │ Features │ │
│  │ Buffer      │  │ (tamiyo_*)  │  │ Masks    │ │
│  │ Advantages  │  │             │  │ Norm     │ │
│  └─────────────┘  └─────────────┘  └──────────┘ │
└─────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────┐
│                   TAMIYO                         │
│  ┌─────────────┐  ┌─────────────┐               │
│  │ Heuristic   │  │ SignalTracker│              │
│  │ (baseline)  │  │             │               │
│  └─────────────┘  └─────────────┘               │
└─────────────────────────────────────────────────┘
```

### After (Proposed)

```
┌─────────────────────────────────────────────────┐
│              TAMIYO (The Brain)                  │
│  ┌─────────────────────────────────────────────┐│
│  │            PolicyBundle Registry             ││
│  │  ┌─────────┐ ┌─────────┐ ┌───────────────┐  ││
│  │  │  LSTM   │ │   MLP   │ │   Heuristic   │  ││
│  │  │ Bundle  │ │ Bundle  │ │    Bundle     │  ││
│  │  └─────────┘ └─────────┘ └───────────────┘  ││
│  └─────────────────────────────────────────────┘│
│  ┌─────────────┐  ┌─────────────┐               │
│  │ SignalTracker│ │ ActionMasks │              │
│  │             │  │ Features    │               │
│  └─────────────┘  └─────────────┘               │
└─────────────────────────────────────────────────┘
              ▲ get_action()    │ TrainingSignals
              │                 ▼
┌─────────────────────────────────────────────────┐
│             SIMIC (Evolution)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ PPO Agent   │  │  Rollout    │  │ Normalizer│ │
│  │ (trains     │  │  Buffer     │  │          │ │
│  │  Tamiyo)    │  │             │  │          │ │
│  └─────────────┘  └─────────────┘  └──────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 4. PolicyBundle Protocol

> **Note:** This protocol was enhanced based on DRL and PyTorch expert review.
> See [Section 10](#10-expert-review-feedback) for detailed rationale.

```python
# tamiyo/policy/protocol.py

from typing import Protocol, runtime_checkable, Any, Iterator
import torch
from torch import nn

@runtime_checkable  # [PyTorch expert] Enables isinstance() checks at registration
class PolicyBundle(Protocol):
    """Interface for swappable Tamiyo policy implementations.

    Tamiyo is the "brain" of Esper - she makes strategic decisions about
    seed lifecycle (germinate, fossilize, cull, wait). Different PolicyBundle
    implementations provide different decision-making strategies:

    - LSTMPolicyBundle: Recurrent neural policy with temporal memory
    - MLPPolicyBundle: Stateless feedforward policy (simpler baseline)
    - HeuristicPolicyBundle: Rule-based expert system (for ablations)

    ## Design Rationale (from expert review)

    - Protocol over ABC: Avoids MRO conflicts with nn.Module inheritance
    - runtime_checkable: Enables validation at policy registration time
    - Explicit state_dict: Required for checkpoint compatibility
    - Device management: Essential for multi-GPU and distributed training

    ## Adding a New Policy

    1. Create `tamiyo/policy/my_bundle.py`
    2. Implement the PolicyBundle protocol
    3. Decorate with @register_policy("my_policy")
    4. Add to config: {"tamiyo": {"policy": "my_policy", ...}}

    ## On-Policy vs Off-Policy

    Policies declare their capabilities via `supports_off_policy`.
    On-policy algorithms (PPO) use `evaluate_actions()`.
    Off-policy algorithms (SAC) use `get_q_values()` and `forward()`.

    ## Recurrent Policies

    Recurrent policies (LSTM) maintain hidden state across steps.
    They must implement `initial_hidden()` and set `is_recurrent = True`.
    Simic handles hidden state threading during rollout collection.

    ## torch.compile Guidance

    Compile the inner nn.Module, NOT the PolicyBundle wrapper.
    Keep torch.compile() calls in Simic (training infrastructure).
    """

    # === Observation Processing ===
    def process_signals(self, signals: "TrainingSignals") -> torch.Tensor:
        """Convert TrainingSignals to policy-specific features."""
        ...

    # === Action Selection (both paradigms) ===
    def get_action(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,  # For off-policy eval mode
    ) -> "ActionResult":
        """Select action given observations.

        Uses inference_mode internally - returned tensors are non-differentiable.
        """
        ...

    # === Forward (for off-policy) === [DRL expert recommendation]
    def forward(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> "ForwardResult":
        """Compute action distribution parameters without sampling.

        Required for:
        - SAC: Computing log_prob of sampled actions for entropy bonus
        - TD3: Getting deterministic action for target policy
        - Offline RL: Computing action distribution for OOD detection

        Returns:
            ForwardResult with logits/mean+std per head, value, new_hidden
        """
        ...

    # === On-Policy (PPO/A2C) ===
    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: "FactoredAction",
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> "EvalResult":  # log_prob, value, entropy
        """Evaluate actions for PPO update.

        Must be called with gradient tracking enabled (not in inference_mode).
        """
        ...

    # === Off-Policy (SAC/TD3) ===
    def get_q_values(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Twin Q-values for off-policy critic.

        Returns (Q1, Q2) for clipped double-Q learning.
        Raises NotImplementedError if supports_off_policy is False.
        """
        ...

    def sync_from(self, source: "PolicyBundle", tau: float = 0.005) -> None:
        """Polyak averaging update from source policy (for target networks).

        target = tau * source + (1 - tau) * target

        [DRL expert recommendation] Required for SAC/TD3 target network updates.
        """
        ...

    # === Value Estimation ===
    def get_value(
        self,
        features: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """State value estimate for baseline."""
        ...

    # === Recurrent State ===
    def initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Initial hidden state for recurrent policies (None if stateless).

        Should be called with inference_mode for efficiency.
        """
        ...

    # === Serialization === [PyTorch expert recommendation]
    def state_dict(self) -> dict[str, Any]:
        """Return policy state for checkpointing."""
        ...

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Load policy state from checkpoint.

        Args:
            state_dict: State dictionary from checkpoint
            strict: If True, keys must match exactly. If False, allows partial loading.
        """
        ...

    # === Device Management === [PyTorch expert recommendation]
    @property
    def device(self) -> torch.device:
        """Device where policy parameters reside."""
        ...

    def to(self, device: torch.device | str) -> "PolicyBundle":
        """Move policy to specified device. Returns self for chaining."""
        ...

    # === Introspection ===
    @property
    def is_recurrent(self) -> bool:
        """True if policy maintains hidden state across steps."""
        ...

    @property
    def supports_off_policy(self) -> bool:
        """True if policy supports off-policy algorithms (SAC/TD3).

        If False, get_q_values() and sync_from() raise NotImplementedError.
        """
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Data type of policy parameters (for AMP compatibility)."""
        ...

    # === Optional: Gradient Checkpointing === [PyTorch expert recommendation]
    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Enable/disable gradient checkpointing for memory efficiency.

        Optional - policies that don't support this should no-op.
        Primarily useful for Transformer-based policies.
        """
        ...
```

---

## 5. Module Structure

### Tamiyo (New Structure)

```
tamiyo/
├── __init__.py              # Exports PolicyBundle, registry, SignalTracker
├── signals.py               # TrainingSignals dataclass (expanded)
├── tracker.py               # SignalTracker (existing, aggregates signals)
├── decisions.py             # TamiyoDecision, ActionResult, EvalResult
│
└── policy/                  # NEW: PolicyBundle implementations
    ├── __init__.py          # Exports all bundles + registry
    ├── protocol.py          # PolicyBundle protocol
    ├── registry.py          # Policy registration + factory
    │
    ├── lstm_bundle.py       # LSTM policy (from simic/agent/tamiyo_network.py)
    ├── mlp_bundle.py        # Stateless MLP policy (new, simpler baseline)
    ├── heuristic_bundle.py  # HeuristicTamiyo wrapped as PolicyBundle
    │
    ├── features.py          # Feature extraction (from simic/control/features.py)
    └── action_masks.py      # Action masking (from simic/control/action_masks.py)
```

### Simic (What Stays)

```
simic/
├── agent/
│   ├── ppo.py               # PPO algorithm (trains any PolicyBundle)
│   ├── rollout_buffer.py    # Renamed from tamiyo_buffer.py
│   └── advantages.py        # GAE computation
│
├── control/
│   └── normalization.py     # Running mean/std (training infrastructure)
│
├── training/
│   └── vectorized.py        # Training loop (uses Tamiyo.PolicyBundle)
│
├── rewards/                 # Reward computation
└── telemetry/               # Training telemetry
```

### Renames

| Old Name | New Name | Rationale |
|----------|----------|-----------|
| `simic/agent/tamiyo_network.py` | `tamiyo/policy/lstm_bundle.py` | It's a bundle, not just a network |
| `simic/agent/tamiyo_buffer.py` | `simic/agent/rollout_buffer.py` | It's Simic's buffer, not Tamiyo's |
| `tamiyo/heuristic.py` | `tamiyo/policy/heuristic_bundle.py` | Conforms to PolicyBundle interface |

---

## 6. Simic ↔ Tamiyo Interface

```python
# simic/training/vectorized.py (simplified)

from tamiyo import PolicyBundle, TrainingSignals, SignalTracker
from tamiyo.policy import get_policy  # Registry factory

class VectorizedTrainer:
    def __init__(self, config: TrainingConfig):
        # Get policy from Tamiyo via registry
        self.policy: PolicyBundle = get_policy(
            name=config["tamiyo"]["policy"],
            config=config["tamiyo"].get("policy_config", {}),
        )

        # Simic owns training infrastructure
        self.ppo = PPOAgent(self.policy, config["ppo"])
        self.buffer = RolloutBuffer(config["buffer"])
        self.normalizer = ObservationNormalizer(config["normalization"])

        # Tamiyo's signal tracker (aggregates observations)
        self.signal_tracker = SignalTracker()

    def collect_rollout(self, env):
        hidden = self.policy.initial_hidden(batch_size=env.num_envs)

        for step in range(self.rollout_length):
            # 1. Get raw observation from environment
            obs = env.get_observation()

            # 2. Tamiyo's tracker aggregates into TrainingSignals
            signals = self.signal_tracker.update(obs)

            # 3. Policy processes signals → features (inside bundle)
            features = self.policy.process_signals(signals)

            # 4. Simic normalizes (training infrastructure)
            features_norm = self.normalizer(features)

            # 5. Compute masks (lives in Tamiyo, called by Simic)
            masks = compute_action_masks(signals, self.slot_config)

            # 6. Policy selects action
            result = self.policy.get_action(features_norm, masks, hidden)

            # 7. Execute action, store in Simic's buffer
            next_obs, reward, done = env.step(result.action)
            self.buffer.add(features_norm, result, reward, done)

            hidden = result.hidden

    def train_step(self):
        # Simic trains Tamiyo's policy using PPO
        self.ppo.update(self.buffer)
        self.buffer.clear()
```

---

## 7. Policy Registry

### Registry Implementation

```python
# tamiyo/policy/registry.py

from typing import Type
from tamiyo.policy.protocol import PolicyBundle

_REGISTRY: dict[str, Type[PolicyBundle]] = {}

def register_policy(name: str):
    """Decorator to register a PolicyBundle implementation."""
    def decorator(cls: Type[PolicyBundle]) -> Type[PolicyBundle]:
        _REGISTRY[name] = cls
        return cls
    return decorator

def get_policy(name: str, config: dict) -> PolicyBundle:
    """Factory function to instantiate a policy by name."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown policy: {name}. Available: {available}")
    return _REGISTRY[name](**config)

def list_policies() -> list[str]:
    """List all registered policy names."""
    return list(_REGISTRY.keys())
```

### Config-Based Selection

```json
{
  "training": {
    "algorithm": "ppo",
    "episodes": 100
  },
  "tamiyo": {
    "policy": "lstm",
    "policy_config": {
      "hidden_dim": 256,
      "num_layers": 1,
      "num_attention_heads": 4
    }
  }
}
```

```bash
# CLI points to config
esper train ppo --config experiments/lstm_baseline.json
esper train ppo --config experiments/heuristic_ablation.json
```

---

## 8. Checkpointing

### Checkpoint Structure

```python
# Full checkpoint
checkpoint = {
    "version": "1.0",
    "tamiyo": {
        "policy_name": "lstm",
        "policy_config": {"hidden_dim": 256, "num_layers": 1},
        "policy_state_dict": policy.state_dict(),
    },
    "simic": {
        "normalizer_state": normalizer.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "training_step": 50000,
    },
}

# Weights-only checkpoint (for transfer / fine-tuning)
checkpoint_weights_only = {
    "version": "1.0",
    "weights_only": True,
    "tamiyo": {
        "policy_name": "lstm",
        "policy_config": {"hidden_dim": 256, "num_layers": 1},
        "policy_state_dict": policy.state_dict(),
    },
    # No simic training state
}
```

### Loading with Compatibility Check

```python
def load_checkpoint(path: str, config: dict, weights_only: bool = False) -> tuple[PolicyBundle, dict | None]:
    ckpt = torch.load(path)

    # Policy compatibility check
    ckpt_policy = ckpt["tamiyo"]["policy_name"]
    requested_policy = config["tamiyo"]["policy"]

    if ckpt_policy != requested_policy:
        raise IncompatibleCheckpointError(
            f"Checkpoint uses '{ckpt_policy}' but config requests '{requested_policy}'. "
            f"Cannot load weights across different policy architectures."
        )

    # Instantiate and load
    policy = get_policy(
        name=ckpt_policy,
        config=ckpt["tamiyo"]["policy_config"],
    )
    policy.load_state_dict(ckpt["tamiyo"]["policy_state_dict"])

    # Weights-only: return None for simic state
    if weights_only or ckpt.get("weights_only", False):
        return policy, None

    return policy, ckpt.get("simic")
```

---

## 9. Migration Plan

### Phase 1: Create Tamiyo Policy Structure
1. Create `tamiyo/policy/` directory
2. Create `protocol.py` with PolicyBundle interface
3. Create `registry.py` with registration mechanism

### Phase 2: Move Files
1. Move `simic/control/features.py` → `tamiyo/policy/features.py`
2. Move `simic/control/action_masks.py` → `tamiyo/policy/action_masks.py`
3. Move `simic/agent/tamiyo_network.py` → `tamiyo/policy/lstm_bundle.py`
4. Rename `simic/agent/tamiyo_buffer.py` → `simic/agent/rollout_buffer.py`

### Phase 3: Wrap Heuristic
1. Create `tamiyo/policy/heuristic_bundle.py` wrapping `HeuristicTamiyo`
2. Keep original `heuristic.py` for backwards compatibility during transition
3. Delete original after migration complete

### Phase 4: Update Simic
1. Update imports in `simic/training/vectorized.py`
2. Update `simic/agent/ppo.py` to work with PolicyBundle interface
3. Update checkpoint save/load logic

### Phase 5: Config Integration
1. Add `tamiyo.policy` to config schema
2. Update CLI to use config-based policy selection
3. Update documentation

### Phase 6: Cleanup
1. Remove dead imports
2. Update all tests
3. Delete any remaining legacy code

---

## 10. Expert Review Feedback

The design was reviewed by two specialist agents before implementation approval.

### 10.1 DRL Expert Review

**Overall Assessment:** "Production-quality RL engineering" - design is sound.

#### Key Confirmations

| Aspect | Assessment |
|--------|------------|
| Action masking in Tamiyo | ✅ Correct - semantic constraints belong with the brain |
| LSTM + off-policy = False | ✅ Correct - needs R2D2 machinery, defer to separate bundle |
| Hidden state handling | ✅ Sequence-based forward is correct |
| Per-head advantage masking | ✅ "Excellent RL engineering" - causal masking is correct |

#### Recommendations Incorporated

1. **Added `forward()` method** - SAC needs `sample_and_log_prob` functionality. The new method returns distribution parameters without sampling.

2. **Added `sync_from()` method** - Required for Polyak updates in SAC/TD3 target networks.

3. **Per-head entropy weighting** (implementation note) - Blueprint/blend heads fire ~10% of time. Consider upweighting their entropy coefficient (2x) to encourage exploration when germinating.

4. **Per-head ratio monitoring** (implementation note) - Track ratio statistics per-head separately for debugging head-specific issues.

#### Algorithm Recommendation

For seed lifecycle control (semi-sparse rewards, 25-epoch episodes, factored discrete actions, recurrence need):

1. **PPO (current)** - Optimal choice, keep it
2. **PPO with entropy annealing** - Experiment with aggressive annealing
3. **SAC-Discrete (stateless MLP)** - As experiment, not replacement
4. **TD3** - Probably not a good fit (deterministic policy for discrete actions)

### 10.2 PyTorch Expert Review

**Overall Assessment:** "Fundamentally sound" - design is compile-safe with minor enhancements.

#### Key Confirmations

| Aspect | Assessment |
|--------|------------|
| Protocol over ABC | ✅ Correct - avoids MRO conflicts with nn.Module |
| Import structure | ✅ No concerns - lazy imports already in place |
| Checkpoint structure | ✅ Correct for single-node; use DCP for future distributed |

#### Recommendations Incorporated

1. **Added `@runtime_checkable`** - Enables `isinstance()` checks at policy registration time.

2. **Added explicit `state_dict()` / `load_state_dict()`** - Duck typing insufficient for checkpointing; compile-time guarantees needed.

3. **Added `device` property and `to()` method** - Essential for multi-GPU and future FSDP/tensor parallelism.

4. **Added `dtype` property** - Helps callers know whether to use `torch.float16` for inference.

5. **Added `enable_gradient_checkpointing()`** - Optional no-op for LSTM, useful for future Transformer bundles.

#### torch.compile Guidance

```python
# CORRECT: Compile inner nn.Module in Simic
class PPOAgent:
    def __init__(self, policy_bundle: PolicyBundle, compile_network: bool = True):
        if compile_network and hasattr(policy_bundle, '_network'):
            policy_bundle._network = torch.compile(
                policy_bundle._network,
                mode="default"
            )

# WRONG: Don't compile the PolicyBundle wrapper
# policy_bundle = torch.compile(policy_bundle)  # NO!
```

#### Code Cleanup Items (for migration)

1. **Unify mask values** - `_MASK_VALUE = -1e4` defined in two places. Define once in Leyline:
   ```python
   # leyline/constants.py
   MASKED_LOGIT_VALUE = -1e4  # Safe for FP16/BF16
   ```

2. **Fix `hasattr` pattern** - Replace:
   ```python
   # Before
   if hasattr(self.network, '_orig_mod'):
       return self.network._orig_mod

   # After
   return getattr(self.network, '_orig_mod', self.network)
   ```

3. **Add `@torch.inference_mode()`** to `get_initial_hidden()` - Prevents accidental gradient tracking.

### 10.3 Summary of Changes from Expert Review

| Category | Original | After Review |
|----------|----------|--------------|
| Protocol methods | 8 methods | 13 methods (+forward, +sync_from, +state_dict, +load_state_dict, +to) |
| Properties | 2 properties | 4 properties (+device, +dtype) |
| Decorators | None | @runtime_checkable |
| Optional methods | None | enable_gradient_checkpointing() |

The enhanced protocol supports:
- On-policy (PPO, A2C) ✅
- Off-policy (SAC, TD3) ✅
- Recurrent policies (LSTM) ✅
- Stateless policies (MLP) ✅
- Heuristic policies ✅
- Mixed precision training ✅
- Multi-GPU deployment ✅
- Gradient checkpointing ✅

---

## Appendix: Full Brainstorming Session

### Context Discovery

The investigation revealed that despite Tamiyo being documented as the "Brain/Cortex", the actual policy lived in Simic:

- `simic/agent/tamiyo_network.py` (363 LOC) - The neural network
- `simic/agent/ppo.py` (784 LOC) - Training algorithm
- `simic/control/features.py` (254 LOC) - Feature extraction
- `simic/control/action_masks.py` (363 LOC) - Action masking

Tamiyo only contained a heuristic baseline (~658 LOC total) that would eventually be removed, leaving Tamiyo "empty".

### Question 1: Hotswap Level

**Options:**
- A) Config-time swapping
- B) Checkpoint-compatible swapping
- C) Runtime swapping

**Decision:** A (with B as nice-to-have)

### Question 2: Policy Architectures

**Options:**
- A) Transformer-based
- B) Feedforward/MLP
- C) Heuristic as policy
- D) External plugins
- E) Multiple

**Decision:** A + B (plus mention of future GNN for strategic Tamiyo)

### Scope Clarification: Tactical vs Strategic

User clarified there are TWO levels of Tamiyo:
- **Tactical Tamiyo (this refactoring):** Seed lifecycle control
- **Strategic Tamiyo (future):** GNN-based, coordinates tactical Tamiyos

Decision: Focus on tactical only (YAGNI for strategic)

### Question 3: Observation Processing Boundary

**Options:**
- A) Inside policy
- B) Outside policy (separate swap point)
- C) PolicyBundle (bundled together)

**Decision:** C - Each policy bundles its observation processor

### Question 4: Heuristic Fate

**Options:**
- A) Keep as PolicyBundle
- B) Delete now
- C) Keep but don't migrate

**Decision:** A - Useful for ablations and debugging

### Question 5: Normalization Location

**Options:**
- A) Part of PolicyBundle
- B) Stays in Simic
- C) Shared in Leyline

**Decision:** B - Training infrastructure, Simic owns running stats

### Question 6: Input to Tamiyo

**Options:**
- A) Raw observation dict
- B) TrainingSignals dataclass
- C) Both
- D) New StatePacket

**Decision:** B (expanded to include everything needed)

### Question 7: Buffer Location

**Options:**
- A) Stays in Simic
- B) Moves to Tamiyo
- C) Split

**Decision:** A - RL training infrastructure

### Question 8: PPO Location

**Options:**
- A) Stays in Simic
- B) Moves to Tamiyo

**Decision:** A - Enables algorithm swapping too!

### Question 9: Algorithm Support

**Options:**
- A) On-policy only (PPO, A2C)
- B) Off-policy too (SAC, TD3)
- C) Not sure

**Decision:** B - Future flexibility

### Question 10: Config Mechanism

Original design used CLI flags, user corrected to JSON config for reproducibility.

---

## Summary

This design enables:

1. **Hotswappable policies** - LSTM, MLP, Heuristic selectable via config
2. **Algorithm flexibility** - PPO, SAC, TD3 can train any PolicyBundle
3. **Clean separation** - Tamiyo = brain, Simic = evolution
4. **Checkpoint compatibility** - Weights-only loading preserved
5. **Future-ready** - Interface supports strategic Tamiyo when needed

The migration minimizes disruption while aligning architecture with the biological metaphor.
