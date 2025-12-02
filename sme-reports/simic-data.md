# SME Analysis: Simic Data Structures

**Files Analyzed:**
- `/home/john/esper-lite/src/esper/simic/episodes.py`
- `/home/john/esper-lite/src/esper/simic/sanity.py`

**Analyst:** DRL + PyTorch SME
**Date:** 2025-12-02

---

## 1. episodes.py

### Purpose
Defines the complete data pipeline for collecting, storing, and loading RL training trajectories. Implements observation space (`TrainingSnapshot`), action recording (`ActionTaken`), outcome tracking (`StepOutcome`), and episode management (`Episode`, `EpisodeCollector`, `DatasetManager`).

### Key Classes/Functions

| Class/Function | Role |
|----------------|------|
| `TrainingSnapshot` | Observation dataclass with `to_vector()` for 27-dim state representation |
| `TrainingSnapshot.batch_to_tensor()` | Optimized batch conversion to PyTorch tensors |
| `ActionTaken` | Action recording with one-hot encoding support |
| `StepOutcome` | Step reward/outcome with `compute_reward()` method |
| `DecisionPoint` | (obs, action, outcome) tuple with timestamp |
| `Episode` | Full trajectory container with serialization |
| `EpisodeCollector` | State machine for collecting decision points |
| `DatasetManager` | File-based episode storage and retrieval |
| `snapshot_from_signals()` | Bridge from Tamiyo signals to Simic observations |
| `action_from_decision()` | Bridge from Tamiyo decisions to Simic actions |

### DRL Assessment

**Observation Space Design (TrainingSnapshot):**
- **Good:** Fixed 27-dimensional vector suitable for MLP input
- **Good:** Handles inf/NaN values with `safe()` clamping (lines 70-74, 138-141)
- **Good:** Includes temporal features (history windows, plateau detection)
- **Concern:** `best_val_loss` initialized to `float('inf')` requires special handling during serialization (line 185, 210)
- **Concern:** No observation normalization built into the structure - running mean/std should be applied externally

**Action Space Design (ActionTaken):**
- **Good:** One-hot encoding for discrete actions
- **Concern:** `action: object` type annotation is too permissive (line 233) - should be typed to the specific action enum
- **Concern:** `to_vector()` assumes `action.value` is a valid index (line 243) - fragile if enum values are non-contiguous

**Reward Design (StepOutcome):**
- **Critical:** `compute_reward()` uses raw accuracy change scaled by 10x (line 301) - this is a placeholder reward that will cause issues:
  - No reward shaping for sparse signals
  - No discount handling
  - Accuracy delta is bounded [-1, 1] but reward is [-10, 10] - may cause gradient magnitude issues with PPO

**Episode Structure:**
- **Good:** Proper trajectory storage with timestamps
- **Good:** JSON serialization for offline RL data collection
- **Concern:** `field_reports` serialization skipped (line 423 comment) - data loss during save/load

### PyTorch Assessment

**Tensor Handling:**
- **Good:** `batch_to_tensor()` (lines 105-170) pre-allocates tensor on device - avoids CPU-to-GPU copy overhead
- **Good:** Direct indexing into pre-allocated tensor minimizes GC pressure
- **Minor:** Import of `torch` inside method (line 127) - should be module-level for clarity

**Serialization:**
- **Good:** Complete `to_dict()`/`from_dict()` round-trip for all dataclasses
- **Good:** ISO 8601 datetime handling
- **Concern:** `from_dict()` requires `action_enum` parameter everywhere - fragile API design

**Memory Efficiency:**
- **Concern:** `DatasetManager.load_all()` (line 611-613) loads entire dataset into memory - problematic for large offline RL datasets
- **Concern:** `get_training_data()` (lines 615-619) converts all episodes to Python lists before tensor conversion - defeats purpose of `batch_to_tensor()`

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| **CRITICAL** | Placeholder reward function (`accuracy_change * 10`) provides no meaningful learning signal for PPO | `StepOutcome.compute_reward()` L301 |
| **HIGH** | `action: object` type annotation bypasses type safety | `ActionTaken` L233 |
| **HIGH** | `load_all()` and `get_training_data()` don't scale for offline RL | `DatasetManager` L611-619 |
| **MEDIUM** | `field_reports` not serialized - data loss during episode save/load | `Episode.to_dict()` L423 |
| **MEDIUM** | No observation normalization infrastructure | `TrainingSnapshot` |
| **MEDIUM** | `from_dict()` API requires threading `action_enum` everywhere | All dataclasses |

### Recommendations

1. **Replace placeholder reward with proper reward shaping:**
   ```python
   # Use potential-based reward shaping (Ng et al., 1999)
   # Phi(s) = best_val_accuracy provides theoretical guarantees
   def compute_reward(self, prev_best: float) -> float:
       intrinsic = self.accuracy_change * 2.0  # Keep small
       potential_diff = self.accuracy_after - prev_best  # PBRS
       return intrinsic + potential_diff
   ```

2. **Add streaming support for large datasets:**
   ```python
   def iter_episodes(self) -> Iterator[Episode]:
       """Lazily iterate episodes without loading all into memory."""
       for path in self.data_dir.glob("*.json"):
           yield Episode.load(path, action_enum=self.action_enum)
   ```

3. **Add observation normalization wrapper:**
   ```python
   class NormalizedSnapshot:
       def __init__(self, running_stats: RunningMeanStd):
           self.stats = running_stats

       def normalize(self, snap: TrainingSnapshot) -> torch.Tensor:
           vec = snap.to_vector()
           return (vec - self.stats.mean) / (self.stats.std + 1e-8)
   ```

4. **Type the action field properly:**
   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from esper.leyline import Action

   @dataclass
   class ActionTaken:
       action: "Action"  # Proper typing
   ```

---

## 2. sanity.py

### Purpose
Provides lightweight runtime sanity checks for training debugging. Guards against reward explosion, logs parameter ratios for rent calibration, and validates tensor shapes for slot compatibility.

### Key Classes/Functions

| Function | Role |
|----------|------|
| `check_reward_magnitude()` | Warns on reward values exceeding threshold |
| `log_params_ratio()` | Debug logging for parameter ratio (seed vs host) |
| `assert_slot_shape()` | Validates tensor dimensionality for CNN/Transformer slots |

### DRL Assessment

**Reward Magnitude Check:**
- **Good:** Catches reward explosion which indicates reward scaling issues
- **Concern:** Threshold of 10.0 (line 16) may be too tight if using curiosity/intrinsic rewards
- **Concern:** Only warns, doesn't clip - in production should also return clipped value

**Parameter Ratio Logging:**
- **Good:** Useful for debugging seed capacity utilization
- **Concern:** Debug-level logging may not appear in production runs

**Slot Shape Validation:**
- **Good:** Catches architecture mismatches early
- **Concern:** Only covers CNN (4D) and Transformer (3D) - no RNN/other topologies

### PyTorch Assessment

**Tensor Operations:**
- **Good:** Uses `x.dim()` and `x.shape` - standard PyTorch idioms
- **Good:** Raises `AssertionError` for shape mismatches - fail-fast behavior

**Environment Variable Control:**
- **Concern:** `SANITY_CHECKS_ENABLED` (line 12) defined but never used in the module - dead code or incomplete feature

### Issues

| Severity | Issue | Location |
|----------|-------|----------|
| **HIGH** | `SANITY_CHECKS_ENABLED` flag defined but never used | L12 |
| **MEDIUM** | `check_reward_magnitude()` only warns, doesn't clip | L15-21 |
| **MEDIUM** | Hardcoded threshold 10.0 may conflict with intrinsic reward schemes | L15 |
| **LOW** | No RNN/LSTM slot shape validation | `assert_slot_shape()` L31-45 |

### Recommendations

1. **Use the sanity check flag:**
   ```python
   def check_reward_magnitude(reward: float, epoch: int, max_epochs: int,
                               threshold: float = 10.0, clip: bool = True) -> float:
       if not SANITY_CHECKS_ENABLED:
           return reward
       if abs(reward) > threshold:
           logger.warning(...)
           return threshold if reward > 0 else -threshold if clip else reward
       return reward
   ```

2. **Make threshold configurable via environment:**
   ```python
   REWARD_THRESHOLD = float(os.getenv("ESPER_REWARD_THRESHOLD", "10.0"))
   ```

3. **Add validation decorator for common checks:**
   ```python
   def sanity_checked(func):
       """Decorator that enables sanity checks based on env var."""
       if not SANITY_CHECKS_ENABLED:
           return func
       return func  # Full implementation wraps with checks
   ```

---

## Summary

| File | Overall Assessment | Priority Actions |
|------|-------------------|------------------|
| `episodes.py` | Solid foundation with critical reward design gap | Fix reward function, add streaming API |
| `sanity.py` | Minimal utility, incomplete | Wire up `SANITY_CHECKS_ENABLED`, add reward clipping |

### Architecture Notes

The episode data structures follow the standard RL trajectory format (Sutton & Barto notation: `(s_t, a_t, r_t, s_{t+1})`). The `batch_to_tensor()` optimization is appropriate for PPO's vectorized environment pattern. However, the reward signal is placeholder-quality and will not support effective policy learning.

For PPO specifically:
- Observation normalization is critical - consider integrating `RunningMeanStd` from stable-baselines3 or implementing in `normalization.py`
- The 27-dim observation is reasonable for the task complexity
- Action space appears discrete (one-hot) which simplifies policy architecture

For offline RL (if planned):
- The JSON-based `DatasetManager` won't scale past ~10k episodes
- Consider switching to memory-mapped formats (HDF5, Arrow/Parquet) for larger datasets
- Current `get_training_data()` pattern defeats streaming
