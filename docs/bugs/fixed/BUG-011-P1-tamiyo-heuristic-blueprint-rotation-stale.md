# BUG-011: Heuristic blueprint rotation ignores topology

- **Title:** Heuristic blueprint rotation ignores enabled slots/topology and can select invalid actions
- **Context:** Tamiyo heuristic (`src/esper/tamiyo/heuristic.py`)
- **Impact:** P1 – heuristic germination could crash or select invalid actions
- **Environment:** Main branch
- **Status:** FIXED (P1-B fix already in codebase)

## Root Cause

The default `blueprint_rotation` in `HeuristicPolicyConfig` was CNN-specific:
```python
blueprint_rotation: list[str] = field(
    default_factory=lambda: ["conv_light", "conv_heavy", "attention", "norm", "depthwise"]
)
```

When used with transformer topology, `getattr(Action, GERMINATE_CONV_LIGHT)` would fail
because those blueprints don't exist for transformers.

## Fix (Already Implemented)

The P1-B fix validates `blueprint_rotation` against available actions at init:

```python
# P1-B fix: Validate blueprint_rotation against available actions at init
# Prevents AttributeError crash during training when getattr fails
available_blueprints = {
    name[len("GERMINATE_"):].lower()
    for name in dir(self._action_enum)
    if name.startswith("GERMINATE_")
}
invalid_blueprints = set(self.config.blueprint_rotation) - available_blueprints
if invalid_blueprints:
    raise ValueError(
        f"blueprint_rotation contains blueprints not available for "
        f"topology '{topology}': {sorted(invalid_blueprints)}. "
        f"Available: {sorted(available_blueprints)}"
    )
```

## Validation

Tested all three scenarios:

```
✓ CNN with default config works
✓ Transformer with default config correctly raises ValueError:
  blueprint_rotation contains blueprints not available for topology
  'transformer': ['conv_heavy', 'conv_light', 'depthwise'].
  Available: ['attention', 'flex_attention', 'lora', 'mlp', 'noop', 'norm']
✓ Transformer with custom config works
```

## Usage

For transformer topology, provide a custom config:
```python
config = HeuristicPolicyConfig(blueprint_rotation=['norm', 'lora', 'attention'])
policy = HeuristicTamiyo(config=config, topology='transformer')
```

## Links

- Fix: `src/esper/tamiyo/heuristic.py` (lines 97-110, P1-B fix)
