# Heuristic Tamiyo Updates: Tempo Parity (Revised)

**Status:** READY
**Date:** 2025-12-19
**Goal:** Align Heuristic Tamiyo with the new 5-head action space (Tempo Lever).

---

## 1. Context Alignment
The "Blend Tempo" implementation (@docs/plans/2025-12-19-blend-tempo-implementation.md) defines the following:
- **Enum:** `TempoAction` (FAST, STANDARD, SLOW).
- **Commitment:** The tempo is chosen during the `GERMINATE` operation and stored in `SeedState`.

## 2. Updated Tasks

### Task 1: Enum & Dataclass Sync
Update `TamiyoDecision` in `src/esper/tamiyo/decisions.py` to support the 5th head.

```python
from esper.leyline.factored_actions import TempoAction

@dataclass
class TamiyoDecision:
    action: IntEnum
    target_seed_id: str | None = None
    reason: str = ""
    confidence: float = 1.0
    tempo: TempoAction = TempoAction.STANDARD  # Default to STANDARD
```

### Task 2: Heuristic Decision Logic
Update `HeuristicTamiyo._decide_germination` in `src/esper/tamiyo/heuristic.py`.

```python
def _decide_germination(self, signals: TrainingSignals) -> TamiyoDecision:
    # ... existing plateau logic ...
    if signals.metrics.plateau_epochs >= self.config.plateau_epochs_to_germinate:
        blueprint_id = self._get_next_blueprint()
        germinate_action = getattr(Action, f"GERMINATE_{blueprint_id.upper()}")
        
        return TamiyoDecision(
            action=germinate_action,
            tempo=TempoAction.STANDARD,  # Explicitly choose the baseline tempo
            reason=f"Plateau detected; starting {blueprint_id} at STANDARD tempo",
            confidence=1.0
        )
```

### Task 3: Runtime Execution
Ensure the heuristic training loop (e.g., in `src/esper/scripts/train.py`) extracts `decision.tempo` and passes it to `host.germinate_seed()`.

---

## 3. Parity Justification
By hardcoding `STANDARD` in the heuristic, we ensure that:
1. The **Heuristic Baseline** uses the same 5-head logic as the **Neural Policy**.
2. The comparison in the "Final Exam" is purely about **Structural Selection** (which blueprint) vs **Neural Adaption**, with speed held constant.