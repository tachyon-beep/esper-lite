# Multi-Location Seed Architecture Design

## Current State (Single Seed Validation)

The current implementation validates the core concept with a single seed slot:
- One `SeedSlot` at a fixed injection point (after block2, 64 channels)
- Seed lifecycle: DORMANT → GERMINATED → TRAINING → BLENDING → FOSSILIZED/CULLED
- Tamiyo (RL agent) learns when to germinate, what blueprint to use, when to advance/cull

This is working as intended – it validates that an RL controller can successfully manage a single seed lifecycle and improve the host model, before we scale out to multiple locations.

## Conceptual Evolution: From "Many Kasminas" to a Single Morphogenetic Plane

### The Initial Mental Model

Our initial mental model treated Kasmina as "the thing you wrap around a host", which naturally suggested "many Kasminas" for many regions or modules.

At small scale, that works: each Kasmina wraps a host-like subnetwork and manages its own seed slot.

### Why This Breaks at Scale

At large scale, this breaks down. A modern model is not "a bag of independent hosts", it is a single computation graph with many *potential growth sites*. Treating each region as a separate Kasmina introduces several problems:

- Ownership ambiguity: "which Kasmina owns this block?"
- Duplicated lifecycle logic in every wrapper
- RL actions that must first choose *which host* to touch
- Seed management that scales like a big unordered bag, not a structured surface

### The Plane Model

What we really want is:

- One task model (e.g. a CNN, a transformer, a hybrid)
- One **Kasmina plane** over that model, which:
  - Enumerates all injection points
  - Owns all `SeedSlot`s
  - Manages all seed lifecycles in a single, consistent state machine

In this view, Kasmina is not a collection of little wrappers; it is the **global morphogenetic coordinate system** for the model. Each slot is a point on that plane, and Tamiyo's job is to decide *where* on the plane to grow, *what* to grow there, and *when* to fossilise.

### Summary

| Mental Model | Description | Scales? |
|--------------|-------------|---------|
| Many Kasminas | Each region has its own wrapper instance | No - coordination nightmare |
| One Kasmina Plane | Single coordinate system over the whole model | Yes – unified view; Kasmina is the map over the territory |

The whole model is a continuous surface you can plant seeds on. Kasmina is the map, not the territory.

## Design Intent: Multiple Location Seeds

### Core Concept

Seeds are **location-based containers** in the computation graph:

```
# Legend: [Seed Slot N] = SeedSlot, noop until burst

Input
  ↓
block1(32ch) ─→ [Seed Slot 0] ─→ (noop until burst)
  ↓ pool
block2(64ch) ─→ [Seed Slot 1] ─→ (current single-slot location)
  ↓ pool
block3(128ch) ─→ [Seed Slot 2] ─→ (noop until burst)
  ↓ pool
classifier
```

Each slot:
1. **Starts as noop** - identity pass-through, no parameters
2. **Provides telemetry** - Tamiyo sees a local state vector from each slot (stage, alpha, Δacc, gradient stats, etc.), so decisions are always conditioned on local behaviour
3. **Can be burst** - Tamiyo decides if that location needs more complexity, and with which blueprint + blending algorithm, based on that local state
4. **Owns an independent lifecycle** - each slot's seed trains, blends, fossilises, and dies independently of the others

### Key Insight

The agent doesn't just learn *what* to add - she learns *where* to add it. Different locations may benefit from different architectural patterns:
- Early layers (low-level features) → might benefit from normalization or depthwise
- Mid layers (feature composition) → might benefit from attention or conv enhancement
- Late layers (semantic features) → might benefit from attention

## Architecture Changes Required

### 1. Host Network: Multiple Injection Points

Here we show a simple CNN host, but the same pattern applies to transformers or hybrids: the key is that the host exposes segment boundaries where `SeedSlot`s can attach.

```python
class HostCNN(nn.Module):
    def __init__(self):
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Linear(128, 10)

    def forward_segment(self, x, start_block: int, end_block: int):
        """Forward through a segment of the network."""
        # Allows seed slots to be inserted between any blocks
```

### 2. MorphogeneticModel: Slot Registry

```python
class MorphogeneticModel(nn.Module):
    def __init__(self, host: HostCNN, device):
        self.host = host
        self.seed_slots = {
            "early": SeedSlot("early", channels=32, device=device),   # After block1
            "mid": SeedSlot("mid", channels=64, device=device),       # After block2
            "late": SeedSlot("late", channels=128, device=device),    # After block3
        }

    def forward(self, x):
        x = self.host.block1(x)
        x = self.seed_slots["early"].forward(x)  # Noop or burst blueprint
        x = self.host.pool(x)

        x = self.host.block2(x)
        x = self.seed_slots["mid"].forward(x)
        x = self.host.pool(x)

        x = self.host.block3(x)
        x = self.seed_slots["late"].forward(x)
        x = self.host.pool(x)

        return self.host.classifier(x.flatten(1))
```

### 3. Noop Seed: Identity Pass-Through

```python
class NoopSeed(nn.Module):
    """Identity seed - placeholder before bursting."""
    blueprint_id = "noop"

    def __init__(self, channels: int):
        super().__init__()
        # No parameters - pure pass-through

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
```

All slots start with `NoopSeed`. When Tamiyo germinates, she replaces the noop with a real blueprint.

### 4. Action Space Design: Factored Actions for PPO

We **do not** represent the action space as a single giant "flat" enumeration. Instead, we factor the decision into multiple smaller heads (slot, blueprint, blend, op) and sample them jointly.

#### The Explosion Problem

The full decision tuple is:

```
Action = (slot, blueprint, blend_method, lifecycle_op)
```

Where:
- **slot**: which injection point (early, mid, late, ...)
- **blueprint**: what to germinate (conv_enhance, attention, norm, depthwise, ...)
- **blend_method**: how to blend (linear, sigmoid, gated, ...)
- **lifecycle_op**: what operation (GERMINATE, ADVANCE, CULL, WAIT)

Naive enumeration explodes:
- 3 slots × 4 blueprints × 3 blend methods × 4 ops = 144 actions
- 10 slots × 8 blueprints × 5 blend methods × 4 ops = 1600 actions

PPO doesn't love this, and it destroys the nice locality in the decision structure (slot vs blueprint vs blend vs lifecycle).

#### Recommended: Factored Multi-Head Policy

Instead of one giant action space, use **multiple policy heads** that output independently:

```python
class FactoredPolicy(nn.Module):
    def __init__(self, obs_dim, num_slots, num_blueprints, num_blend_methods):
        self.shared = nn.Sequential(...)  # Shared feature extraction

        # Separate heads for each decision dimension
        self.slot_head = nn.Linear(hidden, num_slots)           # Which slot?
        self.blueprint_head = nn.Linear(hidden, num_blueprints) # What blueprint?
        self.blend_head = nn.Linear(hidden, num_blend_methods)  # How to blend?
        self.op_head = nn.Linear(hidden, 4)                     # GERMINATE/ADVANCE/CULL/WAIT

    def forward(self, obs):
        features = self.shared(obs)
        return {
            "slot": Categorical(logits=self.slot_head(features)),
            "blueprint": Categorical(logits=self.blueprint_head(features)),
            "blend": Categorical(logits=self.blend_head(features)),
            "op": Categorical(logits=self.op_head(features)),
        }

# Sampling: separate heads, then compose
dists = policy(obs)
slot_idx = dists["slot"].sample()
op = dists["op"].sample()
blueprint_idx = dists["blueprint"].sample()
blend_idx = dists["blend"].sample()
```

**Benefits:**
- Action space is additive not multiplicative: 3 + 4 + 3 + 4 = 14 outputs
- Each head learns its own sub-policy
- Scales gracefully to many slots/blueprints
- Easy to mask invalid combinations (e.g., can't GERMINATE if slot is active)

#### Action Masking

Not all action combinations are valid. Apply masks per-head:

```python
def get_action_masks(model_state):
    masks = {}

    # Slot mask: which slots can accept an action?
    masks["slot"] = [slot.can_accept_action() for slot in model.seed_slots]

    # Op mask: depends on selected slot's state
    # - GERMINATE only if slot is noop
    # - ADVANCE/CULL only if slot has active seed
    # - WAIT always valid
    masks["op"] = compute_op_mask(selected_slot)

    # Blueprint/blend masks: only relevant if op == GERMINATE
    masks["blueprint"] = [1, 1, 1, 1] if op == GERMINATE else [0, 0, 0, 0]
    masks["blend"] = [1, 1, 1] if op == GERMINATE else [0, 0, 0]

    return masks
```

#### Alternative: Hierarchical Action Selection

For even cleaner semantics, use two-stage selection:

1. **Stage 1**: Select operation (WAIT, GERMINATE, ADVANCE, CULL)
2. **Stage 2**: If GERMINATE → select (slot, blueprint, blend_method)
               If ADVANCE/CULL → select slot only

This avoids sampling invalid combinations but requires two forward passes or autoregressive sampling.

### 5. Observation Space Extension

Add per-slot features to observation:

```python
def extract_features(model, training_signals):
    global_features = [...]  # 11 dims: loss, acc, deltas, etc.

    slot_features = []
    for slot_id, slot in model.seed_slots.items():
        slot_features.extend([
            1.0 if slot.is_active else 0.0,    # Has burst seed?
            slot.state.stage.value if slot.state else 0,
            slot.alpha,                          # Blending progress
            slot.state.improvement if slot.state else 0,
        ])  # 4 dims per slot

    return global_features + slot_features
    # With 3 slots: 11 + (4 × 3) = 23 dims
```

This keeps each slot's local state visible, while still giving Tamiyo a single flat observation vector that standard PPO implementations can consume without custom architecture changes.

### 6. Reward Shaping

Per-slot rewards aggregated:

```python
def compute_reward(action, slot_id, acc_delta, slot_states):
    base_reward = acc_delta * 0.5  # Global accuracy improvement

    # Slot-specific shaping
    slot = slot_states[slot_id]
    if action.is_germinate and slot.is_noop:
        reward += 0.1  # Bonus for bursting unused slot
    elif action == CULL and slot.is_failing:
        reward += 0.1  # Bonus for removing bad seed

    # Penalty for action on wrong slot
    if action.is_germinate and not slot.is_noop:
        reward -= 0.3  # Can't germinate already-active slot

    return reward
```

The reward remains primarily global (accuracy deltas are shared), but we add light slot-specific shaping so Tamiyo has a clearer signal about good/bad interventions per location.

## Design Decisions (Multi-Slot Behaviour)

To keep the mental model clean and scalable, we treat each slot as an independent morphogenetic container with its own lifecycle and blending policy. Tamiyo operates at the *control* level (where / what / when), not at the micro-scheduling level of individual gradient steps.

### 1. Independent slots, sequential data flow

- Data still flows early → mid → late in the forward pass (for gradient flow sanity)
- But each `SeedSlot` owns its **own** seed, lifecycle, and blending
- There is no shared "global blend" knob; every slot runs its own schedule

### 2. Per-seed blending algorithms (Tamiyo has a library)

Each seed is paired with a **blending algorithm** (e.g. linear alpha schedule, sigmoid warmup, gated residual, etc.):
- Tamiyo's decisions include:
  - *which blueprint* to germinate
  - *which blending algorithm* to use for that seed
- Implementation options:
  - Blueprint attribute (blueprint includes its preferred blending strategy), or
  - Separate "blend policy" choice in the action space
- Key point: **each seed blends independently**, using its own chosen scheme

### 3. Per-slot telemetry: full local state

Each slot reports a rich local state vector:
- `is_active`, current `SeedStage`
- `alpha` (blending progress)
- local Δacc / loss contribution estimates
- gradient health / variance summaries

Tamiyo sees **all slots at once** in the observation:
- She can decide to invest in early only
- Or replace everything with attention heads
- Or keep some regions simple and over-parameterise others

### 4. Independent lifecycles and fossilisation

Each slot's seed lifecycle is fully independent:
- Early slot can be FOSSILIZED while mid and late are still TRAINING or BLENDING

Fossilisation is **final** for that container:
- "Throw away the keys" for that specific structural chunk
- Any further changes in that region come from *new* seeds grafted on top/around, not by mutating the fossil
- Each fossilised seed is a stable structural asset in the network
- Simplifies reasoning: each fossil is a permanent line item in the parameter budget

### 5. Gradient isolation per slot

We keep gradient isolation **per-slot**:
- Each seed can be trained in isolation from the host and other seeds when needed (e.g. in TRAINING stage)
- Avoids cross-slot interference
- Keeps local credit assignment tractable

## Migration Path

1. **Phase 1 (current)**: Single-slot validation ✓
2. **Phase 2**: Add noop seed concept, slots start as noop
3. **Phase 3**: Add second slot (mid + late), prove two-slot learning
4. **Phase 4**: Full multi-slot (early + mid + late)
5. **Phase 5**: Dynamic slot count based on network depth

## Key Files to Modify (Future)

| File | Change |
|------|--------|
| `src/esper/kasmina/host.py` | `MorphogeneticModel` gets `seed_slots: dict[str, SeedSlot]` |
| `src/esper/kasmina/blueprints.py` | Add `NoopSeed` blueprint |
| `src/esper/leyline/actions.py` | Extend action space for slot selection |
| `src/esper/simic/features.py` | Per-slot feature extraction |
| `src/esper/simic/rewards.py` | Per-slot reward computation |
| `src/esper/simic/vectorized.py` | Multi-slot action execution |

## Summary

### The Core Principle

> **Kasmina behaves like many independent cells, but is implemented as a single morphogenetic plane for scalability.**

Each seed slot acts as if it's its own organism with full autonomy, but under the hood there's one unified coordinate system that scales to arbitrary model sizes.

### The Mental Model

- **Kasmina** = the morphogenetic plane over a task model (not "many wrappers")
- **SeedSlot** = a point on the plane where growth can occur
- **Seed** = a discrete container that can burst into a blueprint
- **Tamiyo** = the RL agent that decides where/what/when/how

### What Tamiyo Learns

The multi-location seed architecture allows Tamiyo to learn, for each region independently:

1. **Where** in the network to add complexity (slot selection)
2. **What** type of complexity to add (blueprint selection)
3. **How** to blend it in (blending algorithm selection)
4. **When** to commit (fossilisation timing based on telemetry)

### Key Design Properties

| Property | Value |
|----------|-------|
| Slots | Fully independent lifecycles; coupled only by dataflow |
| Blending | Per-seed, from Tamiyo's library |
| Telemetry | Full local state from each slot |
| Fossilisation | Final - "throw away the keys" |
| Gradient isolation | Per-slot |
| Action space | Factored multi-head policy |

This is adaptive neural architecture search where the agent discovers the structure, timing, and integration strategy of architectural modifications - all on a single morphogenetic plane that scales to arbitrary model sizes.
