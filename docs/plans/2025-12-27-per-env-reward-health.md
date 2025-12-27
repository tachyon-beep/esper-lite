# Per-Env Reward Health & Always-Visible Metrics

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-env PBRS fraction and gaming rate to EnvDetailScreen, and eliminate jarring layout shifts by making all routine data rows always visible with dim placeholders.

**Architecture:** Add gaming tracking fields to EnvState schema, update aggregator to increment counters on REWARD_COMPUTED events, modify UI rendering to always show rows with `--` placeholders when empty.

**Tech Stack:** Python dataclasses, Textual/Rich TUI, existing Sanctum widget patterns.

---

## Task 1: Add Gaming Tracking Fields to EnvState

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py`

**Step 1: Add fields to EnvState dataclass (after line ~211)**

```python
# Gaming rate tracking (for per-env reward health)
gaming_trigger_count: int = 0   # Steps where ratio_penalty or alpha_shock fired
total_reward_steps: int = 0     # Total steps with reward computed
```

**Step 2: Add computed property**

```python
@property
def gaming_rate(self) -> float:
    """Fraction of steps with anti-gaming penalties."""
    if self.total_reward_steps == 0:
        return 0.0
    return self.gaming_trigger_count / self.total_reward_steps
```

**Verification:** `python -c "from esper.karn.sanctum.schema import EnvState; e = EnvState(env_id=0); print(e.gaming_rate)"`

---

## Task 2: Update Aggregator to Track Gaming Rate

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`

**Step 1: In `_handle_reward_computed()`, after updating reward_components, add:**

```python
# Track gaming rate
env_state.total_reward_steps += 1
if rc.ratio_penalty != 0 or rc.alpha_shock != 0:
    env_state.gaming_trigger_count += 1
```

**Step 2: In episode reset logic (likely `_handle_episode_start` or similar), reset counters:**

```python
env_state.gaming_trigger_count = 0
env_state.total_reward_steps = 0
```

**Verification:** Run sanctum with a short training run and verify gaming_rate updates.

---

## Task 3: Make EnvDetailScreen Metrics Always Visible

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/env_detail_screen.py`

**Changes to `_render_metrics()` (lines 434-551):**

Replace conditional row additions with always-visible rows using dim placeholders:

| Row | Before | After |
|-----|--------|-------|
| Fossilized Params | `if foss_params > 0:` | Always show, `--` if 0 |
| Action Distribution | `if total_actions > 0:` | Always show, `--` if 0 |
| Reward Total | `if rc.total != 0:` | Always show, `0.000` if 0 |
| Signals | `if signals.plain:` | Always show, `--` if empty |
| Credits | `if credits.plain:` | Always show, `--` if empty |
| Warnings | `if warnings.plain:` | Always show, `--` if empty |
| Recent Actions | `if env.action_history:` | Always show, `--` if empty |

**Add PBRS fraction to Reward Total row:**

```python
# Compute PBRS fraction
pbrs_fraction = 0.0
if rc.total != 0:
    pbrs_fraction = abs(rc.stage_bonus) / abs(rc.total)

# Format: "+0.042  PBRS: 25% ✓" or "+0.042  PBRS: 65% ⚠"
pbrs_healthy = 0.1 <= pbrs_fraction <= 0.4
pbrs_icon = "✓" if pbrs_healthy else "⚠"
pbrs_style = "green" if pbrs_healthy else "yellow"
reward_text = Text()
reward_text.append(f"{rc.total:+.3f}", style=total_style)
reward_text.append(f"  PBRS: {pbrs_fraction:.0%} ", style="dim")
reward_text.append(pbrs_icon, style=pbrs_style)
```

**Add gaming rate to Signals row:**

```python
# Gaming rate with current state
gaming_rate = env.gaming_rate
gaming_active = rc.ratio_penalty != 0 or rc.alpha_shock != 0
gaming_healthy = gaming_rate < 0.05

if gaming_active:
    gaming_state = "SHOCK" if rc.alpha_shock != 0 else "RATIO"
    gaming_text = f"Gaming: {gaming_rate:.1%} ({gaming_state})"
    gaming_style = "red"
else:
    gaming_text = f"Gaming: {gaming_rate:.1%} (CLEAN)"
    gaming_style = "green" if gaming_healthy else "yellow"

signals.append(f"  {gaming_text}", style=gaming_style)
```

---

## Task 4: Make HistoricalEnvDetail Metrics Always Visible

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/historical_env_detail.py`

Apply same pattern as Task 3 to `_render_metrics()` method. Note: Historical view won't have gaming_rate since it's a frozen snapshot, so show `--` for that field.

---

## Task 5: Make EsperStatus Always Visible

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/esper_status.py`

**Changes to `render()` method:**

| Metric | Before | After |
|--------|--------|-------|
| Host Params | `if vitals.host_params > 0:` | Always show, `--` if 0 |
| Runtime | `if self._snapshot.runtime_seconds > 0:` | Always show, `0s` if 0 |
| GPU Memory | `if stats.memory_total_gb > 0:` | Always show, `--` if no GPU |
| GPU Util | `if stats.utilization > 0:` | Always show, `--` if no GPU |
| RAM | `if vitals.ram_total_gb > 0:` | Always show, `--` if unavailable |
| CPU | `if vitals.cpu_percent > 0:` | Always show, `--` if unavailable |

---

## Task 6: Make RunHeader Stats Always Visible

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/run_header.py`

**Changes:**

| Metric | Before | After |
|--------|--------|-------|
| Env status counts | `if healthy > 0:` etc. | Always show all three (H/S/D) |
| Seed stage counts | `if training > 0:` etc. | Always show all three (T/B/F) |
| Progress bar | `if s.max_epochs > 0:` | Always show, dim if no max |
| Throughput | `if eps > 0 or bpm > 0:` | Always show, `--` if 0 |

---

## Task 7: Write Tests

**Files:**
- Create: `tests/karn/sanctum/test_always_visible_metrics.py`

**Test cases:**
1. `test_env_detail_shows_all_rows_when_empty` - EnvState with all zeros shows all rows with `--`
2. `test_gaming_rate_computed_correctly` - gaming_trigger_count / total_reward_steps
3. `test_pbrs_fraction_displayed` - PBRS shown with healthy/warning indicator
4. `test_gaming_rate_resets_on_episode` - Counters reset at episode boundary

---

## Verification

After all tasks:
1. Run Sanctum with `--demo` mode
2. Open env detail modal - all rows visible immediately
3. Start training - values populate, no layout shifts
4. Verify PBRS and Gaming rate appear in correct locations
