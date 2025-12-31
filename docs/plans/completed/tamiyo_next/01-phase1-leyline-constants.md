### Phase 1: Leyline Constants (Foundation) ✅ COMPLETE

**Status:** Already merged to `quality-sprint` branch.

**Goal:** Update shared constants that everything depends on.

**Files:**

- `src/esper/leyline/__init__.py`

**Already implemented (verify with validation command below):**

- `DEFAULT_LSTM_HIDDEN_DIM = 512` (line 103)
- `DEFAULT_FEATURE_DIM = 512` (line 418)
- `BLUEPRINT_NULL_INDEX = 13` (line 432)
- `DEFAULT_BLUEPRINT_EMBED_DIM = 4` (line 438)
- `NUM_BLUEPRINTS = 13` (line 426)
- `NUM_OPS = 6`, `NUM_STAGES = 10`, etc.

**Validation (run to confirm):**

```bash
PYTHONPATH=src python -c "
from esper.leyline import (
    DEFAULT_LSTM_HIDDEN_DIM, DEFAULT_FEATURE_DIM,
    DEFAULT_EPISODE_LENGTH, NUM_OPS, NUM_STAGES, NUM_BLUEPRINTS,
    BLUEPRINT_NULL_INDEX, DEFAULT_BLUEPRINT_EMBED_DIM
)
print(f'LSTM={DEFAULT_LSTM_HIDDEN_DIM}, FEATURE={DEFAULT_FEATURE_DIM}')
print(f'EPISODE_LENGTH={DEFAULT_EPISODE_LENGTH}')
print(f'OPS={NUM_OPS}, STAGES={NUM_STAGES}, BLUEPRINTS={NUM_BLUEPRINTS}')
print(f'NULL_INDEX={BLUEPRINT_NULL_INDEX}, EMBED_DIM={DEFAULT_BLUEPRINT_EMBED_DIM}')
"
# Should print:
# LSTM=512, FEATURE=512
# EPISODE_LENGTH=150
# OPS=6, STAGES=10, BLUEPRINTS=13
# NULL_INDEX=13, EMBED_DIM=4
```

#### Phase 1a: Additional Leyline Constants (Pre-Implementation)

Before beginning Phase 2, define these constants in `src/esper/leyline/__init__.py` to eliminate magic numbers in subsequent phases:

```python
# =============================================================================
# Obs V3 Dimension Constants
# =============================================================================

# Non-blueprint feature dimension for Obs V3
# Breakdown: 24 base + 7 temporal + 30×3 slots = 121
OBS_V3_NON_BLUEPRINT_DIM = 121

# Default number of slots in training configurations
DEFAULT_NUM_SLOTS = 3

# =============================================================================
# PPO Architecture Constants
# =============================================================================

# Number of action heads in the factored policy
# (slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve, op)
# See ACTION_HEAD_SPECS in factored_actions.py for the authoritative list
NUM_ACTION_HEADS = 8

# Note: DEFAULT_CLIP_RATIO = 0.2 already exists in leyline (line 131)
# Do NOT add a duplicate DEFAULT_PPO_CLIP_EPSILON - use DEFAULT_CLIP_RATIO instead

# Minimum log probability for numerical stability
# exp(-100) ≈ 3.7e-44 (tiny but non-zero in float64)
# In float32: exp(-88) ≈ 1e-38 (smallest normal), exp(-104) underflows to 0.0
# For ratio stability, we need exp(new_lp - old_lp) to be finite
LOG_PROB_MIN = -100.0
```

**Add to `__all__`:**

```python
"OBS_V3_NON_BLUEPRINT_DIM",
"DEFAULT_NUM_SLOTS",
"NUM_ACTION_HEADS",
# DEFAULT_CLIP_RATIO already exported
"LOG_PROB_MIN",
```

**Validation:**

```bash
PYTHONPATH=src python -c "
from esper.leyline import (
    OBS_V3_NON_BLUEPRINT_DIM, DEFAULT_NUM_SLOTS,
    NUM_ACTION_HEADS, DEFAULT_CLIP_RATIO, LOG_PROB_MIN
)
print(f'OBS_DIM={OBS_V3_NON_BLUEPRINT_DIM}, SLOTS={DEFAULT_NUM_SLOTS}')
print(f'HEADS={NUM_ACTION_HEADS}, CLIP={DEFAULT_CLIP_RATIO}, LOG_MIN={LOG_PROB_MIN}')
"
# Should print:
# OBS_DIM=121, SLOTS=3
# HEADS=8, CLIP=0.2, LOG_MIN=-100.0
```

---

