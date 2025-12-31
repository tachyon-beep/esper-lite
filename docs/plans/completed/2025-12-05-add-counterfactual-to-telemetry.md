# Add Counterfactual to Telemetry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Propagate the TRUE counterfactual (`real_acc - baseline_acc`) to telemetry so logs show both `blending Δ` (temporal) and `causal Δ` (counterfactual).

**Architecture:** The counterfactual is already computed in `vectorized.py` for reward shaping. We add a field to `SeedMetrics` to store it, then include it in fossilize/cull telemetry events.

**Tech Stack:** Python, dataclasses

---

## Expected Log Output

**Before:**
```
[env2] Fossilized 'env2_seed_0' (conv_heavy, total Δacc +1.95%, blending Δ -1.81%)
```

**After:**
```
[env2] Fossilized 'env2_seed_0' (conv_heavy, total Δacc +1.95%, blending Δ -1.81%, causal Δ +0.32%)
```

The `causal Δ` shows the TRUE seed contribution from counterfactual validation.

---

### Task 1: Add counterfactual field to SeedMetrics

**Files:**
- Modify: `src/esper/kasmina/slot.py:55-75`

**Step 1: Add field after alpha_ramp_step**

Add after line 75:
```python
    alpha_ramp_step: int = 0

    # Counterfactual contribution (set by vectorized training)
    counterfactual_contribution: float | None = None
```

**Step 2: Verify change compiles**

Run: `python3 -m py_compile src/esper/kasmina/slot.py`
Expected: No output (success)

---

### Task 2: Store counterfactual in vectorized.py

**Files:**
- Modify: `src/esper/simic/vectorized.py:904-912`

**Step 1: Store counterfactual in seed metrics after computing it**

Find this block (~line 903-912):
```python
                # Compute seed_contribution from counterfactual if available
                seed_contribution = None
                if baseline_accs[env_idx] is not None:
                    seed_contribution = env_state.val_acc - baseline_accs[env_idx]

                # Use contribution-primary reward when counterfactual is available
                if seed_contribution is not None:
```

Change to:
```python
                # Compute seed_contribution from counterfactual if available
                seed_contribution = None
                if baseline_accs[env_idx] is not None:
                    seed_contribution = env_state.val_acc - baseline_accs[env_idx]
                    # Store in metrics for telemetry at fossilize/cull
                    if seed_state and seed_state.metrics:
                        seed_state.metrics.counterfactual_contribution = seed_contribution

                # Use contribution-primary reward when counterfactual is available
                if seed_contribution is not None:
```

**Step 2: Verify change compiles**

Run: `python3 -m py_compile src/esper/simic/vectorized.py`
Expected: No output (success)

---

### Task 3: Include counterfactual in telemetry events

**Files:**
- Modify: `src/esper/kasmina/slot.py:706-747` (fossilize) and `src/esper/kasmina/slot.py:776-802` (cull)

**Step 1: Update fossilize telemetry (~line 706-747)**

Find this block in advance_stage():
```python
            # Capture metrics before transition resets stage counters
            metrics = self.state.metrics
            improvement = metrics.total_improvement
            blending_delta = metrics.blending_delta
```

Change to:
```python
            # Capture metrics before transition resets stage counters
            metrics = self.state.metrics
            improvement = metrics.total_improvement
            blending_delta = metrics.blending_delta
            counterfactual = metrics.counterfactual_contribution
```

Then update the SEED_FOSSILIZED telemetry data (~line 736-747):
```python
                if target_stage == SeedStage.FOSSILIZED:
                    self._emit_telemetry(
                        TelemetryEventType.SEED_FOSSILIZED,
                        data={
                            "blueprint_id": blueprint_id,
                            "seed_id": seed_id,
                            "improvement": improvement,
                            "blending_delta": blending_delta,
                            "counterfactual": counterfactual,  # True causal attribution
                            "params_added": sum(
                                p.numel() for p in self.seed.parameters() if p.requires_grad
                            ),
                            "epochs_total": epochs_total,
                            "epochs_in_stage": epochs_in_stage,
                        }
                    )
```

**Step 2: Update cull telemetry (~line 776-802)**

Find this block in cull():
```python
        improvement = self.state.metrics.total_improvement
        blending_delta = self.state.metrics.blending_delta
```

Change to:
```python
        improvement = self.state.metrics.total_improvement
        blending_delta = self.state.metrics.blending_delta
        counterfactual = self.state.metrics.counterfactual_contribution
```

Then update the SEED_CULLED telemetry data:
```python
                "improvement": improvement,
                "blending_delta": blending_delta,
                "counterfactual": counterfactual,  # True causal attribution (may be None)
```

**Step 3: Verify change compiles**

Run: `python3 -m py_compile src/esper/kasmina/slot.py`
Expected: No output (success)

---

### Task 4: Update analytics to display counterfactual

**Files:**
- Modify: `src/esper/nissa/analytics.py:47,60-62,155-172,179-195,218,252`

**Step 1: Add counterfactual field to BlueprintStats (~line 47)**

After `blending_deltas`:
```python
    blending_deltas: list[float] = field(default_factory=list)  # Accuracy change during blending
    counterfactuals: list[float] = field(default_factory=list)  # True causal attribution
```

**Step 2: Add mean_counterfactual property (~line 63)**

After `mean_blending_delta`:
```python
    @property
    def mean_counterfactual(self) -> float:
        """Mean counterfactual contribution (true causal attribution)."""
        valid = [c for c in self.counterfactuals if c is not None]
        return sum(valid) / len(valid) if valid else 0.0
```

**Step 3: Update SEED_FOSSILIZED handler (~lines 155-172)**

After `blending_delta = event.data.get(...)`:
```python
            blending_delta = event.data.get("blending_delta", 0.0)
            counterfactual = event.data.get("counterfactual")  # May be None
```

After `self.stats[bp_id].blending_deltas.append(...)`:
```python
            self.stats[bp_id].blending_deltas.append(blending_delta)
            if counterfactual is not None:
                self.stats[bp_id].counterfactuals.append(counterfactual)
```

Update print statement:
```python
            # Show total improvement, blending delta, and causal contribution
            causal_str = f", causal Δ {counterfactual:+.2f}%" if counterfactual is not None else ""
            print(f"    [env{env_id}] Fossilized '{seed_id}' ({bp_id}, "
                  f"total Δacc {improvement:+.2f}%, blending Δ {blending_delta:+.2f}%{causal_str})")
```

**Step 4: Update SEED_CULLED handler (~lines 179-195)**

Same pattern as FOSSILIZED:
```python
            blending_delta = event.data.get("blending_delta", 0.0)
            counterfactual = event.data.get("counterfactual")  # May be None
            ...
            self.stats[bp_id].blending_deltas.append(blending_delta)
            if counterfactual is not None:
                self.stats[bp_id].counterfactuals.append(counterfactual)
            ...
            causal_str = f", causal Δ {counterfactual:+.2f}%" if counterfactual is not None else ""
            print(f"    [env{env_id}] Culled '{seed_id}' ({bp_id}, "
                  f"total Δacc {improvement:+.2f}%, blending Δ {blending_delta:+.2f}%{causal_str}){reason_str}")
```

**Step 5: Update summary_table (~line 218)**

Add counterfactual column:
```python
f"{s.mean_acc_delta:>+7.2f}% {s.mean_blending_delta:>+7.2f}% {s.mean_counterfactual:>+7.2f}% "
```

**Step 6: Update to_dict (~line 252)**

Add counterfactual:
```python
"mean_blending_delta": s.mean_blending_delta,
"mean_counterfactual": s.mean_counterfactual,
```

**Step 7: Verify change compiles**

Run: `python3 -m py_compile src/esper/nissa/analytics.py`
Expected: No output (success)

---

### Task 5: Run Tests

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests pass

---

### Task 6: Commit

**Step 1: Stage changes**

```bash
git add src/esper/kasmina/slot.py src/esper/simic/vectorized.py src/esper/nissa/analytics.py
```

**Step 2: Commit**

```bash
git commit -m "feat: add counterfactual contribution to telemetry

Propagate the TRUE causal attribution (real_acc - baseline_acc with alpha=0)
from vectorized training to fossilize/cull telemetry events.

Log output now shows both metrics:
- blending Δ: temporal delta (includes host drift)
- causal Δ: counterfactual contribution (true seed impact)

This helps distinguish between 'model accuracy dropped during blending'
(blending_delta negative) vs 'seed is actually hurting' (counterfactual
negative).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Notes

- `counterfactual_contribution` may be `None` if counterfactual validation wasn't run (e.g., during TRAINING stage when alpha=0)
- The log output gracefully handles None by omitting the causal Δ portion
- The summary table always shows the mean, which will be 0.0 if no counterfactuals were recorded
