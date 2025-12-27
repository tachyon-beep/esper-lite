# BUG-003: PPO target_kl early-stop ineffective with n_epochs=1

- **Title:** PPO `target_kl` early-stop is effectively disabled with default settings
- **Context:** Simic / `PPOAgent.update` (`src/esper/simic/ppo.py`)
- **Impact:** P1 – `target_kl` is advertised/defaulted (0.015) but did nothing for `recurrent_n_epochs=1`
- **Environment:** Main branch
- **Status:** FIXED (2025-12-17)

## Root Cause Analysis

The KL early stopping check was placed at the **END** of each epoch, but only consulted at the **START** of the next epoch:

```python
for epoch_i in range(self.recurrent_n_epochs):
    if early_stopped:  # Check at START of epoch
        break

    # ... forward pass, loss computation ...
    self.optimizer.step()  # UPDATE HAPPENS

    # KL check happens AFTER update
    if approx_kl > 1.5 * target_kl:
        early_stopped = True  # Set at END of epoch
```

With `recurrent_n_epochs=1` (the default for LSTM stability):
1. Epoch 0 runs completely
2. `optimizer.step()` applies the update
3. KL check sets `early_stopped = True`
4. Loop ends (no epoch 1)
5. `early_stopped` was never used to prevent anything

## Fix

Moved KL computation and early stopping check to **BEFORE** `optimizer.step()`:

```python
for epoch_i in range(self.recurrent_n_epochs):
    if early_stopped:
        break

    # Forward pass
    log_probs, values, entropy, _ = self.network.evaluate_actions(...)

    # Extract ratios
    per_head_ratios = {key: torch.exp(log_probs[key] - old_log_probs[key]) ...}

    # KL CHECK NOW HAPPENS HERE (BUG-003 FIX)
    with torch.no_grad():
        approx_kl = compute_kl(...)
        if approx_kl > 1.5 * target_kl:
            early_stopped = True
            break  # Skip loss computation and optimizer.step()

    # Only reach here if KL is acceptable
    loss = compute_loss(...)
    loss.backward()
    self.optimizer.step()
```

Now with `recurrent_n_epochs=1`, if KL is already too high (e.g., from policy drift since rollout), the update is skipped entirely.

## Validation

Regression test added: `test_kl_early_stopping_with_single_epoch`
- Uses `recurrent_n_epochs=1` and extreme target_kl=0.0001
- Fills buffer with fake log_probs different from network output
- Verifies early stopping triggers at epoch 0
- Verifies no policy_loss in metrics (update skipped)

All 14 PPO tests pass.

## Links

- Fix: `src/esper/simic/ppo.py` (PPOAgent.update, lines 513-546)
- Test: `tests/simic/test_ppo.py::test_kl_early_stopping_with_single_epoch`
