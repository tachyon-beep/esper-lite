# Work Package B: Dynamic Action Space Spike

**Status:** Ready for implementation
**Priority:** High (blocks M1.5)
**Effort:** ~3-4 hours
**Dependencies:** WP-C (Slot ID Module) for canonical ID format

---

## Goal

Prove that PPO training works with N≠3 slots before committing to the full action space migration.

## Why This De-risks M1.5

- `TamiyoNetwork` accepts `num_slots` parameter, but we've never tested N≠3 end-to-end
- Action masking, reward computation, and vectorized env may have hardcoded assumptions
- A spike proves feasibility and surfaces blockers early
- Fail fast: if fundamental issues exist, we find them in hours not days

## Background

Current state:
- `TamiyoNetwork.__init__` accepts `num_slots` with default `NUM_SLOTS=3`
- `tests/simic/test_tamiyo_network.py` has a `num_slots=1` edge case test
- Integration tests use `NUM_SLOTS` constant from `leyline.factored_actions`
- Action masks are shaped `[batch, NUM_SLOTS]`

What we don't know:
- Does vectorized environment work with different slot counts?
- Does reward computation assume 3 slots?
- Do action masks propagate correctly through the full training loop?

---

## Tasks

### B.1 Create isolated spike test

**File:** `tests/spike_dynamic_slots.py` (temporary, not for CI)

```python
"""Spike: Prove PPO works with dynamic slot counts.

This is a temporary spike test - delete after M1.5 implementation.
"""

import pytest
import torch

from esper.simic.tamiyo_network import FactoredRecurrentActorCritic
from esper.leyline.factored_actions import NUM_BLUEPRINTS, NUM_BLENDS, NUM_OPS


class TestDynamicSlotsSpike:
    """Prove network architecture works with N≠3 slots."""

    @pytest.mark.parametrize("num_slots", [1, 2, 3, 4, 5, 8])
    def test_network_construction(self, num_slots: int):
        """Network constructs with arbitrary slot counts."""
        state_dim = 50  # Approximate real state dim
        net = FactoredRecurrentActorCritic(
            state_dim=state_dim,
            num_slots=num_slots,
        )
        assert net.num_slots == num_slots

        # Verify slot head output dimension
        dummy_state = torch.randn(2, 1, state_dim)
        output = net(dummy_state)
        assert output["slot_logits"].shape == (2, 1, num_slots)

    @pytest.mark.parametrize("num_slots", [2, 5])
    def test_forward_backward(self, num_slots: int):
        """Forward/backward pass works with different slot counts."""
        state_dim = 50
        net = FactoredRecurrentActorCritic(
            state_dim=state_dim,
            num_slots=num_slots,
        )

        # Forward
        states = torch.randn(4, 3, state_dim, requires_grad=True)
        output = net(states)

        # Compute dummy loss and backward
        loss = output["value"].mean() + output["slot_logits"].mean()
        loss.backward()

        # Verify gradients exist
        assert states.grad is not None
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None

    @pytest.mark.parametrize("num_slots", [2, 5])
    def test_action_masks(self, num_slots: int):
        """Action masking works with different slot counts."""
        state_dim = 50
        net = FactoredRecurrentActorCritic(
            state_dim=state_dim,
            num_slots=num_slots,
        )

        batch_size = 4
        states = torch.randn(batch_size, 1, state_dim)

        # Create masks with some slots disabled
        slot_mask = torch.ones(batch_size, 1, num_slots, dtype=torch.bool)
        slot_mask[:, :, 0] = False  # Disable first slot

        output = net(states, slot_mask=slot_mask)

        # Verify masked slot has very low probability
        probs = torch.softmax(output["slot_logits"], dim=-1)
        assert probs[:, :, 0].max() < 0.01  # First slot should be ~0

    @pytest.mark.parametrize("num_slots", [2, 5])
    def test_get_action(self, num_slots: int):
        """Action sampling works with different slot counts."""
        state_dim = 50
        net = FactoredRecurrentActorCritic(
            state_dim=state_dim,
            num_slots=num_slots,
        )

        states = torch.randn(4, state_dim)
        slot_mask = torch.ones(4, num_slots, dtype=torch.bool)

        actions, log_probs, values, hidden = net.get_action(
            states, slot_mask=slot_mask
        )

        # Verify action is in valid range
        assert actions["slot"].max() < num_slots
        assert actions["slot"].min() >= 0

    @pytest.mark.parametrize("num_slots", [2, 5])
    def test_evaluate_actions(self, num_slots: int):
        """Action evaluation works with different slot counts."""
        state_dim = 50
        net = FactoredRecurrentActorCritic(
            state_dim=state_dim,
            num_slots=num_slots,
        )

        batch, seq = 4, 3
        states = torch.randn(batch, seq, state_dim)

        # Generate random actions in valid range
        actions = {
            "slot": torch.randint(0, num_slots, (batch, seq)),
            "blueprint": torch.randint(0, NUM_BLUEPRINTS, (batch, seq)),
            "blend": torch.randint(0, NUM_BLENDS, (batch, seq)),
            "op": torch.randint(0, NUM_OPS, (batch, seq)),
        }

        log_probs, values, entropy, hidden = net.evaluate_actions(states, actions)

        assert log_probs["slot"].shape == (batch, seq)
        assert entropy["slot"].shape == (batch, seq)


class TestVectorizedEnvSpike:
    """Prove vectorized environment works with dynamic slots.

    NOTE: This may require modifications to VectorizedMorphogeneticEnv
    to accept slot configuration. Document blockers.
    """

    @pytest.mark.skip(reason="Requires VectorizedEnv modifications - document blockers")
    def test_vectorized_env_dynamic_slots(self):
        """Vectorized environment accepts dynamic slot config."""
        # TODO: Implement after discovering required changes
        pass


# Blockers discovered during spike
BLOCKERS = []


def record_blocker(description: str, file: str, line: int | None = None):
    """Record a blocker discovered during spike testing."""
    BLOCKERS.append({
        "description": description,
        "file": file,
        "line": line,
    })


@pytest.fixture(scope="session", autouse=True)
def report_blockers():
    """Print blockers at end of test session."""
    yield
    if BLOCKERS:
        print("\n" + "=" * 60)
        print("BLOCKERS DISCOVERED:")
        for b in BLOCKERS:
            loc = f"{b['file']}:{b['line']}" if b['line'] else b['file']
            print(f"  - {b['description']}")
            print(f"    Location: {loc}")
        print("=" * 60)
```

### B.2 Run spike tests and document findings

- [ ] Run `pytest tests/spike_dynamic_slots.py -v`
- [ ] Note which tests pass/fail
- [ ] For failures, identify the root cause:
  - Hardcoded `3` or `NUM_SLOTS`?
  - Shape mismatch in masks?
  - Missing slot_config propagation?

### B.3 Attempt 10 PPO steps with N≠3 slots

- [ ] Modify spike to run minimal PPO training loop
- [ ] Test with `num_slots=2` and `num_slots=5`
- [ ] Document any crashes or incorrect behavior

### B.4 Document blockers

**File:** `docs/plans/dynamic-slots-spike-results.md`

Document:
- Which tests passed/failed
- Root cause of each failure
- List of files that need modification for dynamic slots
- Estimated complexity of each fix

---

## Acceptance Criteria

- [ ] All `TestDynamicSlotsSpike` tests pass
- [ ] Blockers documented with file locations
- [ ] Clear go/no-go assessment for M1.5

## Outputs

1. `tests/spike_dynamic_slots.py` — temporary spike tests (delete after M1.5)
2. `docs/plans/dynamic-slots-spike-results.md` — findings and blockers

## Success Criteria

**Green light for M1.5:** All network-level tests pass, blockers are in integration code only.

**Yellow light:** Some network tests fail, requires architecture changes.

**Red light:** Fundamental design issues prevent dynamic slots.
