# Q-Values Telemetry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire up op-conditioned Q-values (Q(s,op)) and Q-variance telemetry from Policy V2 to Sanctum UI

**Architecture:** Policy V2 already computes Q(s,op) via `_compute_value(lstm_out, op)`. We'll collect Q-values for all 6 ops during PPO updates, compute variance metrics, emit via telemetry pipeline, and wire to TamiyoState for display. UI rendering already exists but shows 0.0 defaults.

**Tech Stack:** PyTorch (value computation), Leyline telemetry (contracts), Simic PPO (collection), Karn aggregator (wiring)

---

## Task 1: Add Q-Value Fields to PPOUpdatePayload

**Files:**
- Modify: `src/esper/leyline/telemetry.py:605-760` (PPOUpdatePayload class and from_dict)
- Test: `tests/leyline/test_telemetry.py`

**Step 1: Write failing test for q_* fields**

Add to `tests/leyline/test_telemetry.py`:

```python
def test_ppo_update_payload_with_q_values():
    """PPOUpdatePayload accepts and serializes Q-values per op."""
    payload = PPOUpdatePayload(
        policy_loss=0.5,
        value_loss=0.3,
        entropy=1.2,
        grad_norm=2.0,
        kl_divergence=0.01,
        clip_fraction=0.15,
        nan_grad_count=0,
        # Q-values per operation
        q_germinate=5.2,
        q_advance=3.1,
        q_fossilize=2.8,
        q_prune=-1.5,
        q_wait=0.5,
        q_set_alpha=4.0,
        q_variance=2.3,
        q_spread=6.7,
    )

    assert payload.q_germinate == 5.2
    assert payload.q_advance == 3.1
    assert payload.q_fossilize == 2.8
    assert payload.q_prune == -1.5
    assert payload.q_wait == 0.5
    assert payload.q_set_alpha == 4.0
    assert payload.q_variance == 2.3
    assert payload.q_spread == 6.7


def test_ppo_update_payload_from_dict_with_q_values():
    """PPOUpdatePayload.from_dict parses Q-values."""
    data = {
        "policy_loss": 0.5,
        "value_loss": 0.3,
        "entropy": 1.2,
        "grad_norm": 2.0,
        "kl_divergence": 0.01,
        "clip_fraction": 0.15,
        "nan_grad_count": 0,
        "q_germinate": 5.2,
        "q_advance": 3.1,
        "q_fossilize": 2.8,
        "q_prune": -1.5,
        "q_wait": 0.5,
        "q_set_alpha": 4.0,
        "q_variance": 2.3,
        "q_spread": 6.7,
    }

    payload = PPOUpdatePayload.from_dict(data)

    assert payload.q_germinate == 5.2
    assert payload.q_variance == 2.3
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_ppo_update_payload_with_q_values -xvs
PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_ppo_update_payload_from_dict_with_q_values -xvs
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'q_germinate'`

**Step 3: Add Q-value fields to PPOUpdatePayload**

In `src/esper/leyline/telemetry.py`, after line 684 (`value_max: float = 0.0`), add:

```python
    # === Op-Conditioned Q-Values (Policy V2) ===
    # Q(s, op) for each operation - value estimates conditioned on the operation
    # These show how the critic values different operations in the current state
    q_germinate: float = 0.0  # Q(s, GERMINATE)
    q_advance: float = 0.0    # Q(s, ADVANCE)
    q_fossilize: float = 0.0  # Q(s, FOSSILIZE)
    q_prune: float = 0.0      # Q(s, PRUNE)
    q_wait: float = 0.0       # Q(s, WAIT)
    q_set_alpha: float = 0.0  # Q(s, SET_ALPHA_TARGET)

    # Q-value analysis metrics
    q_variance: float = 0.0  # Variance across ops (low = critic ignoring op conditioning)
    q_spread: float = 0.0    # max(Q) - min(Q) across ops
```

**Step 4: Update from_dict to parse Q-value fields**

In `src/esper/leyline/telemetry.py`, in the `from_dict` method after line 788 (`value_max=data.get("value_max", 0.0),`), add:

```python
            # Q-values
            q_germinate=data.get("q_germinate", 0.0),
            q_advance=data.get("q_advance", 0.0),
            q_fossilize=data.get("q_fossilize", 0.0),
            q_prune=data.get("q_prune", 0.0),
            q_wait=data.get("q_wait", 0.0),
            q_set_alpha=data.get("q_set_alpha", 0.0),
            q_variance=data.get("q_variance", 0.0),
            q_spread=data.get("q_spread", 0.0),
```

**Step 5: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_ppo_update_payload_with_q_values -xvs
PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_ppo_update_payload_from_dict_with_q_values -xvs
```

Expected: PASS (2 tests)

**Step 6: Commit**

```bash
git add src/esper/leyline/telemetry.py tests/leyline/test_telemetry.py
git commit -m "feat(telemetry): add Q-value fields to PPOUpdatePayload

Add op-conditioned Q-values (q_germinate, q_advance, q_fossilize, q_prune,
q_wait, q_set_alpha) and analysis metrics (q_variance, q_spread) to
PPOUpdatePayload for Policy V2 telemetry.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Collect Q-Values During PPO Update

**Files:**
- Modify: `src/esper/simic/agent/ppo.py:370-700` (update method)
- Test: `tests/simic/test_ppo.py`

**Step 1: Write failing test for Q-value collection**

Add to `tests/simic/test_ppo.py`:

```python
def test_ppo_update_collects_q_values(simple_env, lstm_policy):
    """PPO update collects Q(s,op) for all ops and computes variance."""
    import torch
    from esper.simic.agent import PPOAgent

    agent = PPOAgent(
        policy=lstm_policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        n_ppo_epochs=1,
    )

    # Collect some data
    state = simple_env.reset()
    blueprint_indices = torch.zeros((1, 6), dtype=torch.long)
    for _ in range(5):
        result = agent.get_action(state, blueprint_indices, training=True)
        next_state, reward, done, _ = simple_env.step(result.actions)
        agent.buffer.add(
            state=state,
            blueprint_indices=blueprint_indices,
            actions=result.actions,
            log_probs=result.log_probs,
            values=result.values,
            rewards=reward,
            dones=done,
            sampled_op=result.sampled_op,
            hidden=result.hidden,
        )
        state = next_state

    # Trigger update
    metrics = agent.update()

    # Verify Q-values were collected
    assert "q_germinate" in metrics
    assert "q_advance" in metrics
    assert "q_fossilize" in metrics
    assert "q_prune" in metrics
    assert "q_wait" in metrics
    assert "q_set_alpha" in metrics
    assert "q_variance" in metrics
    assert "q_spread" in metrics

    # Q-values should be floats
    assert isinstance(metrics["q_germinate"], float)

    # Q-variance should be >= 0
    assert metrics["q_variance"] >= 0.0

    # Q-spread should be >= 0
    assert metrics["q_spread"] >= 0.0
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/simic/test_ppo.py::test_ppo_update_collects_q_values -xvs
```

Expected: `KeyError: 'q_germinate'`

**Step 3: Add Q-value collection in PPO update**

In `src/esper/simic/agent/ppo.py`, after computing advantage stats (around line 467), add:

```python
        # === Collect Op-Conditioned Q-Values (Policy V2) ===
        # Compute Q(s, op) for all ops using a representative state
        # Use first valid state from batch to avoid bias from terminal states
        if valid_mask.any():
            # Get first valid state
            first_valid_idx = valid_mask.nonzero(as_tuple=True)
            if len(first_valid_idx[0]) > 0:
                sample_state_idx = (first_valid_idx[0][0].item(), first_valid_idx[1][0].item())
                sample_obs = data["states"][sample_state_idx[0], sample_state_idx[1]].unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
                sample_blueprints = data["blueprint_indices"][sample_state_idx[0], sample_state_idx[1]].unsqueeze(0).unsqueeze(0)  # [1, 1, num_slots]

                # Forward pass to get LSTM output
                with torch.no_grad():
                    forward_result = self.policy.forward(
                        state=sample_obs,
                        blueprint_indices=sample_blueprints,
                        hidden=None,  # Use initial hidden state for consistency
                    )
                    lstm_out = forward_result["lstm_out"]  # [1, 1, hidden_dim]

                # Compute Q(s, op) for each op
                from esper.leyline import NUM_OPS
                q_values = []
                for op_idx in range(NUM_OPS):
                    op_tensor = torch.tensor([[op_idx]], dtype=torch.long, device=self.device)
                    q_val = self.policy._compute_value(lstm_out, op_tensor)
                    q_values.append(q_val.item())

                # Unpack to named ops (order matches LifecycleOp enum)
                # LifecycleOp: GERMINATE=0, ADVANCE=1, FOSSILIZE=2, PRUNE=3, WAIT=4, SET_ALPHA_TARGET=5
                metrics["q_germinate"] = [q_values[0]]
                metrics["q_advance"] = [q_values[1]]
                metrics["q_fossilize"] = [q_values[2]]
                metrics["q_prune"] = [q_values[3]]
                metrics["q_wait"] = [q_values[4]]
                metrics["q_set_alpha"] = [q_values[5]]

                # Compute Q-variance and Q-spread
                q_variance = float(torch.tensor(q_values).var().item())
                q_spread = max(q_values) - min(q_values)
                metrics["q_variance"] = [q_variance]
                metrics["q_spread"] = [q_spread]
            else:
                # No valid states - use NaN
                metrics["q_germinate"] = [float("nan")]
                metrics["q_advance"] = [float("nan")]
                metrics["q_fossilize"] = [float("nan")]
                metrics["q_prune"] = [float("nan")]
                metrics["q_wait"] = [float("nan")]
                metrics["q_set_alpha"] = [float("nan")]
                metrics["q_variance"] = [float("nan")]
                metrics["q_spread"] = [float("nan")]
        else:
            # No valid data - use NaN
            metrics["q_germinate"] = [float("nan")]
            metrics["q_advance"] = [float("nan")]
            metrics["q_fossilize"] = [float("nan")]
            metrics["q_prune"] = [float("nan")]
            metrics["q_wait"] = [float("nan")]
            metrics["q_set_alpha"] = [float("nan")]
            metrics["q_variance"] = [float("nan")]
            metrics["q_spread"] = [float("nan")]
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/simic/test_ppo.py::test_ppo_update_collects_q_values -xvs
```

Expected: PASS

**Step 5: Run broader PPO test suite**

```bash
PYTHONPATH=src uv run pytest tests/simic/test_ppo.py -xvs
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/agent/ppo.py tests/simic/test_ppo.py
git commit -m "feat(ppo): collect Q(s,op) values during PPO update

Compute Q-values for all 6 operations using Policy V2's op-conditioned
value head. Sample first valid state from batch, forward through network,
and evaluate Q(s,op) for each op. Compute variance and spread metrics.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Emit Q-Values in Telemetry Event

**Files:**
- Modify: `src/esper/simic/telemetry/emitters.py:723-830` (emit_ppo_update_event)
- Test: Integration test in Task 4

**Step 1: Add Q-value emission to emit_ppo_update_event**

In `src/esper/simic/telemetry/emitters.py`, in the `emit_ppo_update_event` function, after line 788 (`value_max=metrics.get("value_max", 0.0),`), add:

```python
            # Q-values (Policy V2 op-conditioned critic)
            q_germinate=metrics.get("q_germinate", 0.0),
            q_advance=metrics.get("q_advance", 0.0),
            q_fossilize=metrics.get("q_fossilize", 0.0),
            q_prune=metrics.get("q_prune", 0.0),
            q_wait=metrics.get("q_wait", 0.0),
            q_set_alpha=metrics.get("q_set_alpha", 0.0),
            q_variance=metrics.get("q_variance", 0.0),
            q_spread=metrics.get("q_spread", 0.0),
```

**Step 2: Verify syntax**

```bash
PYTHONPATH=src uv run python -m py_compile src/esper/simic/telemetry/emitters.py
```

Expected: No output (successful compilation)

**Step 3: Commit**

```bash
git add src/esper/simic/telemetry/emitters.py
git commit -m "feat(telemetry): emit Q-values in PPO_UPDATE_COMPLETED event

Pass Q-values from PPO metrics to PPOUpdatePayload for telemetry emission.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Wire Q-Values in Sanctum Aggregator

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:653-830` (_handle_ppo_update)
- Test: `tests/karn/sanctum/test_aggregator.py`

**Step 1: Write failing test for Q-value aggregation**

Add to `tests/karn/sanctum/test_aggregator.py`:

```python
def test_aggregator_wires_q_values():
    """Aggregator wires Q-values from PPO_UPDATE_COMPLETED to TamiyoState."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline.telemetry import TelemetryEvent, TelemetryEventType, PPOUpdatePayload

    aggregator = SanctumAggregator(slot_ids=["r0c0", "r0c1"])

    # Emit PPO_UPDATE_COMPLETED with Q-values
    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        epoch=1,
        data=PPOUpdatePayload(
            policy_loss=0.5,
            value_loss=0.3,
            entropy=1.2,
            grad_norm=2.0,
            kl_divergence=0.01,
            clip_fraction=0.15,
            nan_grad_count=0,
            q_germinate=5.2,
            q_advance=3.1,
            q_fossilize=2.8,
            q_prune=-1.5,
            q_wait=0.5,
            q_set_alpha=4.0,
            q_variance=2.3,
            q_spread=6.7,
        ),
    )

    aggregator.handle_event(event)
    snapshot = aggregator.get_snapshot()

    # Verify Q-values are wired to TamiyoState
    assert snapshot.tamiyo.q_germinate == 5.2
    assert snapshot.tamiyo.q_advance == 3.1
    assert snapshot.tamiyo.q_fossilize == 2.8
    assert snapshot.tamiyo.q_prune == -1.5
    assert snapshot.tamiyo.q_wait == 0.5
    assert snapshot.tamiyo.q_set_alpha == 4.0
    assert snapshot.tamiyo.q_variance == 2.3
    assert snapshot.tamiyo.q_spread == 6.7
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_aggregator_wires_q_values -xvs
```

Expected: `AssertionError: assert 0.0 == 5.2` (Q-values not wired)

**Step 3: Wire Q-values in aggregator**

In `src/esper/karn/sanctum/aggregator.py`, in the `_handle_ppo_update` method, after line 790 (`self._tamiyo.value_max = payload.value_max`), add:

```python
        # Op-conditioned Q-values (Policy V2)
        self._tamiyo.q_germinate = payload.q_germinate
        self._tamiyo.q_advance = payload.q_advance
        self._tamiyo.q_fossilize = payload.q_fossilize
        self._tamiyo.q_prune = payload.q_prune
        self._tamiyo.q_wait = payload.q_wait
        self._tamiyo.q_set_alpha = payload.q_set_alpha
        self._tamiyo.q_variance = payload.q_variance
        self._tamiyo.q_spread = payload.q_spread
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_aggregator_wires_q_values -xvs
```

Expected: PASS

**Step 5: Run broader aggregator test suite**

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py -xvs
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator.py
git commit -m "feat(sanctum): wire Q-values from PPO telemetry to TamiyoState

Connect Q-value fields from PPOUpdatePayload to TamiyoState for UI display.
HealthStatusPanel already renders these values.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Integration Test - End-to-End Q-Value Flow

**Files:**
- Create: `tests/integration/test_q_values_telemetry.py`

**Step 1: Write end-to-end integration test**

Create `tests/integration/test_q_values_telemetry.py`:

```python
"""Integration test: Q-values flow from Policy V2 → PPO → Telemetry → Sanctum UI."""

import torch
from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline.telemetry import TelemetryHub
from esper.simic.agent import PPOAgent
from esper.simic.telemetry.emitters import emit_ppo_update_event
from esper.tamiyo.networks.factored_lstm import FactoredLSTMActorCritic


def test_q_values_end_to_end_flow(simple_env, slot_config):
    """Q-values flow from policy → PPO → telemetry → aggregator → UI."""
    # Setup telemetry hub
    hub = TelemetryHub()
    aggregator = SanctumAggregator(slot_ids=["r0c0", "r0c1", "r0c2"])
    hub.subscribe(aggregator.handle_event)

    # Create policy and agent
    state_dim = 95  # Obs V3 feature dim
    policy = FactoredLSTMActorCritic(
        state_dim=state_dim,
        slot_config=slot_config,
    )

    agent = PPOAgent(
        policy=policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        n_ppo_epochs=1,
    )

    # Collect rollout data
    state = simple_env.reset()
    blueprint_indices = torch.zeros((1, 6), dtype=torch.long)
    for _ in range(10):
        result = agent.get_action(state, blueprint_indices, training=True)
        next_state, reward, done, _ = simple_env.step(result.actions)
        agent.buffer.add(
            state=state,
            blueprint_indices=blueprint_indices,
            actions=result.actions,
            log_probs=result.log_probs,
            values=result.values,
            rewards=reward,
            dones=done,
            sampled_op=result.sampled_op,
            hidden=result.hidden,
        )
        state = next_state

    # Trigger PPO update (collects Q-values)
    metrics = agent.update()

    # Verify metrics contain Q-values
    assert "q_germinate" in metrics
    assert "q_variance" in metrics
    assert metrics["q_variance"] >= 0.0

    # Emit telemetry event
    emit_ppo_update_event(
        hub=hub,
        metrics=metrics,
        episodes_completed=1,
        batch_idx=0,
        epoch=0,
        optimizer=agent.optimizer,
        grad_norm=metrics.get("grad_norm"),
        update_time_ms=10.0,
    )

    # Verify aggregator received and wired Q-values
    snapshot = aggregator.get_snapshot()
    assert snapshot.tamiyo.q_germinate != 0.0  # Should have real value
    assert snapshot.tamiyo.q_variance >= 0.0

    # If variance > 0, Q-values are differentiated (op-conditioning works)
    # If variance ≈ 0, all Q-values are same (critic ignoring ops - BAD)
    print(f"Q-variance: {snapshot.tamiyo.q_variance:.4f}")
    print(f"Q-spread: {snapshot.tamiyo.q_spread:.4f}")
    print(f"Q-values: G={snapshot.tamiyo.q_germinate:.2f} "
          f"A={snapshot.tamiyo.q_advance:.2f} "
          f"F={snapshot.tamiyo.q_fossilize:.2f} "
          f"P={snapshot.tamiyo.q_prune:.2f} "
          f"W={snapshot.tamiyo.q_wait:.2f} "
          f"S={snapshot.tamiyo.q_set_alpha:.2f}")
```

**Step 2: Run integration test**

```bash
PYTHONPATH=src uv run pytest tests/integration/test_q_values_telemetry.py::test_q_values_end_to_end_flow -xvs
```

Expected: PASS with Q-values printed

**Step 3: Commit**

```bash
git add tests/integration/test_q_values_telemetry.py
git commit -m "test(integration): add end-to-end Q-values telemetry test

Verify Q-values flow from Policy V2 through PPO, telemetry emission,
aggregator wiring, and final display in Sanctum UI schema.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Validation - Run Full Test Suite

**Step 1: Run all relevant test suites**

```bash
# Telemetry tests
PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py -xvs

# PPO tests
PYTHONPATH=src uv run pytest tests/simic/test_ppo.py -xvs

# Aggregator tests
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py -xvs

# Integration tests
PYTHONPATH=src uv run pytest tests/integration/test_q_values_telemetry.py -xvs
```

Expected: All PASS

**Step 2: Run linter**

```bash
uv run ruff check src/esper/leyline/telemetry.py src/esper/simic/agent/ppo.py src/esper/simic/telemetry/emitters.py src/esper/karn/sanctum/aggregator.py
```

Expected: No errors

**Step 3: Type check**

```bash
uv run mypy src/esper/leyline/telemetry.py src/esper/simic/agent/ppo.py src/esper/karn/sanctum/aggregator.py --strict
```

Expected: No errors (or resolve any type issues)

---

## Task 7: Documentation Update

**Files:**
- Create: `docs/bugs/fixed/B11-TELEM-01.md`

**Step 1: Document the bug fix**

Create `docs/bugs/fixed/B11-TELEM-01.md`:

```markdown
# B11-TELEM-01: Q-Values Not Wired to Telemetry

**Ticket ID:** B11-TELEM-01
**Status:** Fixed
**Severity:** Medium
**Domain:** Telemetry / UI
**Date Discovered:** 2025-12-31
**Date Fixed:** 2025-12-31

## One-Line Summary

Op-conditioned Q-values (Policy V2) were displayed in UI but not wired to telemetry pipeline.

## Symptom

HealthStatusPanel showed Q-values and Q-variance, but all values were 0.0 defaults. UI correctly flagged this as "NO OP COND!" critical status.

## Root Cause

Q-value telemetry was not implemented when Policy V2 was developed. The UI was built first (schema + rendering), but the data pipeline was deferred.

## Fix

Added Q-value collection, emission, and aggregation:

1. **PPOUpdatePayload** (leyline/telemetry.py): Added `q_germinate`, `q_advance`, `q_fossilize`, `q_prune`, `q_wait`, `q_set_alpha`, `q_variance`, `q_spread` fields
2. **PPO Update** (simic/agent/ppo.py): Collect Q(s,op) for all ops using `_compute_value()`, compute variance/spread
3. **Telemetry Emitter** (simic/telemetry/emitters.py): Pass Q-values to PPOUpdatePayload
4. **Aggregator** (karn/sanctum/aggregator.py): Wire Q-values to TamiyoState

## Impact

- Q-values now visible in Sanctum UI during training
- Q-variance diagnostic works (detects if critic ignores op conditioning)
- Better visibility into Policy V2 value head behavior

## Prevention

Use "schema-first + telemetry TODO" pattern: when adding UI fields, immediately add TODO comment in telemetry pipeline to track missing wiring.
```

**Step 2: Commit documentation**

```bash
git add docs/bugs/fixed/B11-TELEM-01.md
git commit -m "docs: document Q-values telemetry bug fix (B11-TELEM-01)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Completion Checklist

- [x] PPOUpdatePayload has Q-value fields
- [x] PPO update collects Q(s,op) for all ops
- [x] Telemetry emitter passes Q-values
- [x] Aggregator wires Q-values to TamiyoState
- [x] Integration test validates end-to-end flow
- [x] All tests pass
- [x] Code passes linter
- [x] Documentation updated

**Total commits:** 7
**Estimated time:** 45-60 minutes (TDD with tests)
