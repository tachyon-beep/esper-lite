# Blueprint Analytics Design

**Date**: 2025-11-30
**Status**: Approved
**Priority**: High (roadmap priority)

---

## Overview

Implement telemetry enhancements to answer: **which blueprints are working?**

Two components:
1. **BlueprintAnalytics** - Aggregate stats across all blueprint types
2. **SeedScoreboard** - Per-environment tracking of seeds, params, compute cost

---

## Data Structures

### BlueprintStats

```python
@dataclass
class BlueprintStats:
    """Performance statistics for a single blueprint type."""
    germinated: int = 0
    fossilized: int = 0
    culled: int = 0
    acc_deltas: list[float] = field(default_factory=list)  # improvement at terminal
    churns: list[float] = field(default_factory=list)      # accuracy drop post-cull

    @property
    def mean_acc_delta(self) -> float:
        return sum(self.acc_deltas) / len(self.acc_deltas) if self.acc_deltas else 0.0

    @property
    def mean_churn(self) -> float:
        return sum(self.churns) / len(self.churns) if self.churns else 0.0

    @property
    def fossilization_rate(self) -> float:
        """Percentage of germinated seeds that fossilized."""
        total = self.fossilized + self.culled
        return (self.fossilized / total * 100) if total > 0 else 0.0
```

### SeedScoreboard

```python
@dataclass
class SeedScoreboard:
    """Tracks cumulative seed population and cost for an environment."""
    total_germinated: int = 0
    total_fossilized: int = 0
    total_culled: int = 0
    fossilized_by_blueprint: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    live_blueprint: str | None = None  # Currently active seed's blueprint
    params_added: int = 0              # Cumulative from fossilized seeds
    host_params: int = 0               # Baseline for percentage calc

    @property
    def compute_cost(self) -> float:
        """Estimated compute cost relative to baseline (1.0)."""
        cost = 1.0
        for bp_id, count in self.fossilized_by_blueprint.items():
            cost += (compute_cost_for_blueprint(bp_id) - 1.0) * count
        return cost
```

---

## Compute Cost Multipliers

Blueprint-specific constants (can tune with empirical data later):

```python
BLUEPRINT_COMPUTE_MULTIPLIERS: dict[str, float] = {
    "depthwise": 1.08,      # Cheap - depthwise separable
    "conv_enhance": 1.15,   # Moderate - adds conv layers
    "norm": 1.02,           # Minimal - just normalization
    "attention": 1.35,      # Expensive - O(n²) attention
}

def compute_cost_for_blueprint(blueprint_id: str) -> float:
    """Return compute multiplier for a blueprint type."""
    return BLUEPRINT_COMPUTE_MULTIPLIERS.get(blueprint_id, 1.1)
```

---

## BlueprintAnalytics Backend

```python
class BlueprintAnalytics(OutputBackend):
    """Aggregates blueprint performance from telemetry events."""

    def __init__(self):
        self.stats: dict[str, BlueprintStats] = defaultdict(BlueprintStats)
        self.scoreboards: dict[int, SeedScoreboard] = {}  # env_id -> scoreboard

    def emit(self, event: TelemetryEvent) -> None:
        """Process lifecycle events to update stats."""
        if event.event_type == TelemetryEventType.SEED_GERMINATED:
            bp_id = event.data.get("blueprint_id", "unknown")
            env_id = event.data.get("env_id", 0)
            self.stats[bp_id].germinated += 1
            self._get_scoreboard(env_id).total_germinated += 1
            self._get_scoreboard(env_id).live_blueprint = bp_id

        elif event.event_type == TelemetryEventType.SEED_FOSSILIZED:
            bp_id = event.data.get("blueprint_id", "unknown")
            env_id = event.data.get("env_id", 0)
            improvement = event.data.get("improvement", 0.0)
            params = event.data.get("params_added", 0)

            self.stats[bp_id].fossilized += 1
            self.stats[bp_id].acc_deltas.append(improvement)

            sb = self._get_scoreboard(env_id)
            sb.total_fossilized += 1
            sb.fossilized_by_blueprint[bp_id] += 1
            sb.params_added += params
            sb.live_blueprint = None

        elif event.event_type == TelemetryEventType.SEED_CULLED:
            bp_id = event.data.get("blueprint_id", "unknown")
            env_id = event.data.get("env_id", 0)
            improvement = event.data.get("improvement", 0.0)

            self.stats[bp_id].culled += 1
            self.stats[bp_id].churns.append(improvement)

            sb = self._get_scoreboard(env_id)
            sb.total_culled += 1
            sb.live_blueprint = None

    def _get_scoreboard(self, env_id: int) -> SeedScoreboard:
        if env_id not in self.scoreboards:
            self.scoreboards[env_id] = SeedScoreboard()
        return self.scoreboards[env_id]

    def summary_table(self) -> str:
        """Pretty-print blueprint performance stats."""
        lines = ["Blueprint Stats:"]
        lines.append("  " + "-" * 75)
        lines.append(f"  {'Blueprint':<14} {'Germ':>5} {'Foss':>5} {'Cull':>5} {'Rate':>6} {'ΔAcc':>8} {'Churn':>8}")
        lines.append("  " + "-" * 75)

        for bp_id in sorted(self.stats.keys()):
            s = self.stats[bp_id]
            lines.append(
                f"  {bp_id:<14} {s.germinated:>5} {s.fossilized:>5} "
                f"{s.culled:>5} {s.fossilization_rate:>5.1f}% "
                f"{s.mean_acc_delta:>+7.2f}% {s.mean_churn:>+7.2f}%"
            )
        return "\n".join(lines)

    def scoreboard_table(self, env_id: int = 0) -> str:
        """Pretty-print scoreboard for an environment."""
        sb = self._get_scoreboard(env_id)
        pct = (sb.params_added / sb.host_params * 100) if sb.host_params > 0 else 0

        dist = ", ".join(f"{bp} x{count}" for bp, count in sb.fossilized_by_blueprint.items())

        lines = [
            f"Seed Scoreboard (env {env_id}):",
            f"  Fossilized: {sb.total_fossilized} (+{sb.params_added/1000:.1f}K params, +{pct:.1f}% of host)",
            f"  Compute cost: {sb.compute_cost:.2f}x baseline",
            f"  Distribution: {dist or 'none'}",
        ]
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Return serializable snapshot for history."""
        return {
            "stats": {bp: asdict(s) for bp, s in self.stats.items()},
            "scoreboards": {env_id: asdict(sb) for env_id, sb in self.scoreboards.items()},
        }
```

---

## Kasmina Integration

Update `SeedSlot` event emissions to include required data:

### germinate()
```python
self._emit_telemetry(
    TelemetryEventType.SEED_GERMINATED,
    data={
        "blueprint_id": blueprint_id,
        "seed_id": seed_id,
        "params": sum(p.numel() for p in self.seed.parameters() if p.requires_grad),
    }
)
```

### advance_stage() - on FOSSILIZED
```python
self._emit_telemetry(
    TelemetryEventType.SEED_FOSSILIZED,
    data={
        "blueprint_id": self.state.blueprint_id,
        "seed_id": self.state.seed_id,
        "improvement": self.state.metrics.total_improvement,
        "params_added": sum(p.numel() for p in self.seed.parameters() if p.requires_grad),
    }
)
```

### cull()
```python
self._emit_telemetry(
    TelemetryEventType.SEED_CULLED,
    data={
        "reason": reason,
        "blueprint_id": self.state.blueprint_id,
        "seed_id": self.state.seed_id,
        "improvement": self.state.metrics.total_improvement,
    }
)
```

**Note**: Kasmina remains environment-agnostic. `env_id` is injected at the Simic layer.

---

## Simic Integration

### Setup: Wire analytics and callbacks

```python
def train_ppo_vectorized(...) -> tuple[PPOAgent, list[dict]]:
    from esper.nissa.output import NissaHub
    from esper.nissa.analytics import BlueprintAnalytics

    # Create analytics backend
    analytics = BlueprintAnalytics()
    hub = NissaHub()
    hub.add_backend(analytics)

    # Create environments with env_id-injecting callbacks
    for env_idx in range(n_envs):
        def make_callback(idx: int):
            def callback(event: TelemetryEvent):
                event.data["env_id"] = idx
                hub.emit(event)
            return callback

        model = create_model(device=env_device)
        model.slot._telemetry_callback = make_callback(env_idx)

        # Set host_params baseline for scoreboard
        analytics._get_scoreboard(env_idx).host_params = sum(
            p.numel() for p in model.host.parameters() if p.requires_grad
        )
```

### Periodic Summary

```python
# Print analytics every 5 episodes
if (episode + 1) % 5 == 0:
    print("\n" + analytics.summary_table())
    for env_idx in range(n_envs):
        print(analytics.scoreboard_table(env_idx))
    print()
```

### Return analytics in history

```python
return agent, {
    "episode_rewards": all_rewards,
    "blueprint_analytics": analytics.snapshot(),
}
```

---

## Example Output

```
Blueprint Stats:
  ---------------------------------------------------------------------------
  Blueprint       Germ  Foss  Cull   Rate     ΔAcc    Churn
  ---------------------------------------------------------------------------
  attention          8     1     7  12.5%   +0.42%   -0.15%
  conv_enhance      15     5    10  33.3%   +1.15%   -0.22%
  depthwise         22    14     8  63.6%   +2.31%   -0.08%
  norm               4     0     4   0.0%   -0.18%   -0.31%

Seed Scoreboard (env 0):
  Fossilized: 5 (+87.2K params, +5.7% of host)
  Compute cost: 1.42x baseline
  Distribution: depthwise x3, conv_enhance x2
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/esper/nissa/analytics.py` | **CREATE** - BlueprintStats, SeedScoreboard, BlueprintAnalytics |
| `src/esper/nissa/__init__.py` | Update exports |
| `src/esper/kasmina/slot.py` | Enrich telemetry event data |
| `src/esper/simic/vectorized.py` | Wire analytics, inject env_id, print summaries |

---

## Value Delivered

These analytics answer:
1. **Which blueprints should Tamiyo prefer?** → Fossilization rate
2. **Is Tamiyo learning or thrashing?** → Cull rate, churn metrics
3. **What's the cost of growth?** → Params added, compute cost
4. **Is there a plateau strategy?** → Blueprint distribution at convergence

Turns narrative ("Tamiyo rediscovered depthwise") into data product:
> "depthwise: 63.6% fossilization rate, +2.31% mean accuracy, -0.08% churn"
