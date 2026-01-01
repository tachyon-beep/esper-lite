# Integration Plan: _train_one_epoch into run_ppo_episode

**Goal:** Replace the 6 duplicated training loops in `run_ppo_episode` with calls to the `_train_one_epoch` helper.

**Current State:** `_train_one_epoch` exists and is tested, but NOT used by `run_ppo_episode`.

---

## Analysis: Current Duplication

`run_ppo_episode` has 6 nearly-identical training loops (lines 250-364):

| Block | Condition | Lines | seed_optimizer | grad_stats | Stage transition |
|-------|-----------|-------|----------------|------------|------------------|
| 1 | `seed_state is None` | 250-260 | No | No | No |
| 2 | `stage == GERMINATED` | 262-286 | Yes (created) | Yes | Yes (→TRAINING) |
| 3 | `stage == TRAINING` | 288-309 | Yes | Yes | No |
| 4 | `stage == BLENDING` | 311-331 | Optional | Yes | No |
| 5 | `stage in (SHADOWING, PROBATIONARY)` | 333-352 | Yes | Yes | No |
| 6 | `stage == FOSSILIZED` | 354-364 | No | No | No |

### Common Pattern (repeated 6x)
```python
for inputs, targets in trainloader:
    inputs, targets = inputs.to(device), targets.to(device)
    host_optimizer.zero_grad()
    if seed_optimizer:
        seed_optimizer.zero_grad()
    outputs = model(inputs)
    loss, correct_batch, batch_total = _loss_and_correct(...)
    loss.backward()
    # Optional: grad_stats = collect_seed_gradients(...)
    host_optimizer.step()
    if seed_optimizer:
        seed_optimizer.step()
    running_loss += loss.item()
    total += batch_total
    correct += correct_batch
```

### Differences
1. **Gradient collection:** Blocks 2-5 collect `grad_stats` for telemetry; blocks 1,6 don't
2. **Stage transition:** Block 2 (GERMINATED) calls `advance_stage(TRAINING)` before training
3. **seed_optimizer:** Blocks 1,6 don't use it; blocks 2-5 do

---

## Integration Strategy

### Option A: Simple - Keep stage transitions separate, unify loops only
- Pros: Minimal helper changes, clear separation of concerns
- Cons: Still some duplication in optimizer handling

### Option B: Add collect_gradients param to helper (RECOMMENDED)
- Pros: Full consolidation, helper handles telemetry path
- Cons: Helper signature grows slightly

### Option C: Separate training helper from gradient helper
- Pros: Single responsibility
- Cons: Requires two calls per stage

**Decision: Option B** - Add optional `collect_gradients` parameter to return gradient stats.

---

## Implementation Plan

### Step 1: Extend _train_one_epoch signature

Add `collect_gradients` parameter and return gradient stats:

```python
def _train_one_epoch(
    model: nn.Module,
    trainloader: "torch.utils.data.DataLoader",
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer | None,
    device: str,
    task_type: str,
    collect_gradients: bool = False,
) -> tuple[float, float, int, dict | None]:
    """Unified training loop for all seed stages.

    Returns:
        Tuple of (running_loss, correct_count, total_count, grad_stats)
        - grad_stats: Gradient statistics dict if collect_gradients=True, else None
    """
```

### Step 2: Update helper implementation

```python
def _train_one_epoch(
    model: nn.Module,
    trainloader: "torch.utils.data.DataLoader",
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer | None,
    device: str,
    task_type: str,
    collect_gradients: bool = False,
) -> tuple[float, float, int, dict | None]:
    """Unified training loop for all seed stages."""
    model.train()

    running_loss = 0.0
    correct = 0.0
    total = 0
    grad_stats = None

    for inputs, targets in trainloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        host_optimizer.zero_grad(set_to_none=True)
        if seed_optimizer:
            seed_optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss, correct_batch, batch_total = _loss_and_correct(
            outputs, targets, criterion, task_type
        )
        loss.backward()

        # Collect gradient stats on last batch only (matches current behavior)
        if collect_gradients:
            grad_stats = collect_seed_gradients(model.get_seed_parameters())

        host_optimizer.step()
        if seed_optimizer:
            seed_optimizer.step()

        running_loss += loss.item()
        correct += correct_batch
        total += batch_total

    return running_loss, correct, total, grad_stats
```

### Step 3: Refactor run_ppo_episode training section

Replace lines 244-365 with:

```python
        # Training phase
        seed_state = model.seed_state
        grad_stats = None

        # Handle GERMINATED→TRAINING transition
        if seed_state and seed_state.stage == SeedStage.GERMINATED:
            gate_result = model.seed_slot.advance_stage(SeedStage.TRAINING)
            if not gate_result.passed:
                raise RuntimeError(f"G1 gate failed: {gate_result}")

        # Initialize seed optimizer if needed (active seed, not fossilized)
        needs_seed_optimizer = (
            seed_state is not None
            and seed_state.stage not in (SeedStage.DORMANT, SeedStage.FOSSILIZED)
        )
        if needs_seed_optimizer and seed_optimizer is None:
            seed_optimizer = torch.optim.SGD(
                model.get_seed_parameters(), lr=0.01, momentum=0.9
            )

        # Determine if we should collect gradients (active training seed)
        should_collect_gradients = (
            use_telemetry
            and seed_state is not None
            and seed_state.stage in (
                SeedStage.GERMINATED, SeedStage.TRAINING,
                SeedStage.BLENDING, SeedStage.SHADOWING, SeedStage.PROBATIONARY
            )
        )

        # Single unified training call
        running_loss, correct, total, grad_stats = _train_one_epoch(
            model=model,
            trainloader=trainloader,
            criterion=criterion,
            host_optimizer=host_optimizer,
            seed_optimizer=seed_optimizer if needs_seed_optimizer else None,
            device=device,
            task_type=task_type,
            collect_gradients=should_collect_gradients,
        )

        train_loss = running_loss / len(trainloader) if len(trainloader) > 0 else 0.0
        train_acc = 100.0 * correct / total if total > 0 else 0.0
```

### Step 4: Update existing tests

Modify `tests/simic/test_training_helper.py` to handle new return signature:

```python
def test_returns_correct_tuple_types(self, simple_model, simple_dataloader):
    """Should return (float, float, int, None) tuple without gradient collection."""
    from esper.simic.training import _train_one_epoch

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

    result = _train_one_epoch(
        model=simple_model,
        trainloader=simple_dataloader,
        criterion=criterion,
        host_optimizer=optimizer,
        seed_optimizer=None,
        device="cpu",
        task_type="classification",
    )

    assert isinstance(result, tuple)
    assert len(result) == 4  # Now returns 4 values
    running_loss, correct, total, grad_stats = result
    assert isinstance(running_loss, float)
    assert isinstance(correct, float)
    assert isinstance(total, int)
    assert grad_stats is None  # Not collected by default
```

### Step 5: Add gradient collection test

```python
def test_gradient_collection(self):
    """Should return gradient stats when collect_gradients=True."""
    from esper.simic.training import _train_one_epoch
    from esper.tolaria import create_model
    from esper.runtime import get_task_spec

    # Need a real model with seed for gradient collection
    task_spec = get_task_spec("cifar10")
    model = create_model(task=task_spec, device="cpu")
    model.germinate_seed("conv_light", "test_seed")
    model.seed_slot.advance_stage(SeedStage.TRAINING)

    X = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=4)

    criterion = nn.CrossEntropyLoss()
    host_optimizer = torch.optim.SGD(model.get_host_parameters(), lr=0.01)
    seed_optimizer = torch.optim.SGD(model.get_seed_parameters(), lr=0.01)

    running_loss, correct, total, grad_stats = _train_one_epoch(
        model=model,
        trainloader=dataloader,
        criterion=criterion,
        host_optimizer=host_optimizer,
        seed_optimizer=seed_optimizer,
        device="cpu",
        task_type="classification",
        collect_gradients=True,
    )

    assert grad_stats is not None
    assert 'gradient_norm' in grad_stats
    assert 'gradient_health' in grad_stats
```

---

## Behavioral Differences to Verify

### Critical: Ensure identical behavior

1. **Gradient stats timing:** Current code collects on every batch (overwriting). Helper should match - collect on last batch only. ✓ (Current helper does this)

2. **Device transfer:** Current uses `.to(device)`, helper uses `.to(device, non_blocking=True)`. This is an **improvement** but verify no regressions.

3. **zero_grad:** Current uses `.zero_grad()`, helper uses `.zero_grad(set_to_none=True)`. This is an **improvement** but verify no regressions.

4. **model.train():** Helper calls `model.train()` at start. Current code calls it once at line 245. Should be equivalent.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Behavioral regression | Low | High | Full test suite + comparison run |
| Gradient stats timing | Low | Medium | Match existing behavior exactly |
| Device transfer change | Very Low | Low | `non_blocking=True` is strictly better |
| Return signature change | Low | Medium | Update all call sites (only tests currently) |

---

## Verification Plan

1. Run existing helper tests after signature change
2. Run full simic test suite: `pytest tests/simic/ -v`
3. Run integration test with actual PPO episode
4. Compare training metrics (loss, accuracy) before/after refactor

---

## Summary

**Before:** 6 duplicated ~15-line training loops in run_ppo_episode (90+ lines of duplication)

**After:** Single call to `_train_one_epoch` with stage-conditional logic for optimizer and gradient collection (~15 lines)

**Net reduction:** ~75 lines of duplicated code
