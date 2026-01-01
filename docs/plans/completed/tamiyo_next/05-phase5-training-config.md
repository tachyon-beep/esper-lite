### Phase 5: Training Configuration Updates

**Goal:** Update PPO for differential entropy and new network API.

**Files:**

- `src/esper/simic/agent/ppo.py`

#### 5a. Add Differential Entropy Coefficients

Add to `PPOConfig` dataclass (or as module-level constants in ppo.py):

```python
# Sparse heads need higher entropy coefficients to maintain exploration
# when they receive fewer training signals due to causal masking
ENTROPY_COEF_PER_HEAD: dict[str, float] = {
    "op": 1.0,           # Always active (100% of steps)
    "slot": 1.0,         # Usually active (~60%)
    "blueprint": 1.3,    # GERMINATE only (~18%) — needs boost
    "style": 1.2,        # GERMINATE + SET_ALPHA_TARGET (~22%)
    "tempo": 1.3,        # GERMINATE only (~18%) — needs boost
    "alpha_target": 1.2, # GERMINATE + SET_ALPHA_TARGET (~22%)
    "alpha_speed": 1.2,  # SET_ALPHA_TARGET + PRUNE (~19%)
    "alpha_curve": 1.2,  # SET_ALPHA_TARGET + PRUNE (~19%)
}
# Note: Start conservative (1.2-1.3x), tune empirically if heads collapse
```

#### 5b. Update Entropy Loss Computation

```python
# In _ppo_update(), after calling evaluate_actions:
result = self.policy.evaluate_actions(states, blueprint_indices, actions, hidden, action_mask)

# result.entropy is dict[str, Tensor] from EvaluateOutput
total_entropy_loss = torch.zeros(1, device=device)
for head, entropy in result.entropy.items():
    coef = ENTROPY_COEF_PER_HEAD.get(head, 1.0)
    total_entropy_loss -= self.entropy_coef * coef * entropy.mean()
```

#### 5c. Update evaluate_actions() Call Site

Update the PPO update loop to pass blueprint_indices from the buffer:

```python
def _ppo_update(self, buffer: TamiyoRolloutBuffer) -> dict[str, float]:
    # Get batched data from buffer (Phase 6a adds blueprint_indices)
    batch = buffer.get_batched_sequences(self.device)

    states = batch["states"]                    # [batch, seq, 121]
    blueprint_indices = batch["blueprint_indices"]  # [batch, seq, num_slots]
    actions = {
        "op": batch["op_actions"],
        "slot": batch["slot_actions"],
        # ... other action heads ...
    }

    # evaluate_actions extracts stored_op from actions["op"] internally
    # (see Phase 4f - no separate sampled_op parameter needed)
    result = self.policy.evaluate_actions(
        states,
        blueprint_indices,
        actions,
        hidden,
        action_mask,
    )

    # result.value is Q(s, stored_op) - matches what was stored during rollout
    # result.log_probs is dict[str, Tensor] of per-head log probs
    # result.entropy is dict[str, Tensor] of per-head entropy
```

**Note:** The op used for value conditioning comes from `actions["op"]` (the stored action), extracted inside `evaluate_actions()`. This ensures the Q(s,op) value matches what was stored during rollout.

---

