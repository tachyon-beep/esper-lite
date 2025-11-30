# Telemetry Enhancements: Blueprint Analytics

**Source**: External review feedback
**Priority**: Medium (high value, low friction)
**Location**: Likely `simic/vectorized.py` or new `nissa/analytics.py`

---

## Problem

Current logging is verbose but doesn't aggregate the key question: **which blueprints are working?**

Narrative like "Tamiyo rediscovered depthwise" needs to become data product.

---

## Enhancement 1: Blueprint Outcome Summary Per Batch

Track per blueprint type, per batch (or every N episodes):

| Metric | Description |
|--------|-------------|
| `germinated` | Count of germinations |
| `fossilized` | Count of successful integrations |
| `culled` | Count of failures |
| `mean_acc_delta` | Mean accuracy delta while seed was active |
| `mean_churn` | Mean accuracy drop on removal (if any) |

### Example Output

```
Blueprint stats (episodes 160-164):
  depthwise:     germ 12, foss 7, cull 1, mean Δacc +2.3%, churn -0.1%
  conv_enhance:  germ 19, foss 3, cull 9, mean Δacc +0.4%, churn -0.2%
  norm:          germ 2,  foss 0, cull 2, mean Δacc -0.1%, churn ~0
  attention:     germ 3,  foss 0, cull 3, mean Δacc +0.2%, churn ~0
```

### Implementation Sketch

```python
@dataclass
class BlueprintStats:
    germinated: int = 0
    fossilized: int = 0
    culled: int = 0
    acc_deltas: list[float] = field(default_factory=list)
    churn_on_removal: list[float] = field(default_factory=list)

    @property
    def mean_acc_delta(self) -> float:
        return sum(self.acc_deltas) / len(self.acc_deltas) if self.acc_deltas else 0.0

    @property
    def mean_churn(self) -> float:
        return sum(self.churn_on_removal) / len(self.churn_on_removal) if self.churn_on_removal else 0.0

# In vectorized.py or training loop:
blueprint_stats: dict[str, BlueprintStats] = defaultdict(BlueprintStats)

# On germinate:
blueprint_stats[blueprint_id].germinated += 1

# On fossilize:
blueprint_stats[blueprint_id].fossilized += 1
blueprint_stats[blueprint_id].acc_deltas.append(seed_state.metrics.total_improvement)

# On cull:
blueprint_stats[blueprint_id].culled += 1
blueprint_stats[blueprint_id].acc_deltas.append(seed_state.metrics.total_improvement)
# Track churn = accuracy drop after removal (measure on next epoch)
```

---

## Enhancement 2: Seed Population Scoreboard

Per environment, track:

| Metric | Description |
|--------|-------------|
| `live_seeds` | Current count of active seeds |
| `total_params_added` | Sum of parameters from all fossilized seeds |
| `compute_cost_estimate` | Relative forward pass cost (1.0 = baseline host) |
| `blueprint_distribution` | Count by blueprint type |

### Example Output

```
Seed Scoreboard (env 0):
  Live seeds: 2 (depthwise, depthwise)
  Fossilized: 5 (depthwise x3, conv_enhance x2)
  Total params added: +127K (+8.3% of host)
  Compute cost: 1.12x baseline

  Blueprint distribution:
    depthwise:     5 fossilized, 2 live
    conv_enhance:  2 fossilized, 0 live
    attention:     0 fossilized, 0 live
    norm:          0 fossilized, 0 live
```

### Implementation Notes

Currently Esper has a single SeedSlot (one seed at a time). This scoreboard becomes more valuable when:
1. Multiple slots are supported (Phase 2+ per ROADMAP)
2. Fossilized seeds accumulate over training

For single-slot case, simpler version:
```python
@dataclass
class SeedScoreboard:
    total_germinated: int = 0
    total_fossilized: int = 0
    total_culled: int = 0
    fossilized_by_blueprint: dict[str, int] = field(default_factory=dict)
    params_added: int = 0
```

---

## Integration Points

| Location | Change |
|----------|--------|
| `simic/vectorized.py` | Track stats in training loop, print per batch |
| `simic/training.py` | Same for non-vectorized |
| `nissa/analytics.py` | New module for aggregation (optional) |
| `history` return value | Include `blueprint_stats` in training history |

---

## Value

These two tables answer:
1. **Which blueprints should Tamiyo prefer?** → Fossilization rate
2. **Is Tamiyo learning or thrashing?** → Cull rate, churn metrics
3. **What's the cost of growth?** → Params added, compute cost
4. **Is there a plateau strategy?** → Blueprint distribution at convergence

Turns "Tamiyo rediscovered depthwise" into: "depthwise: 58% fossilization rate, +1.8% mean accuracy, 0.02% churn"
