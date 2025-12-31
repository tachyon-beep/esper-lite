### Phase 3: Blueprint Embedding Module

**Goal:** Create the learned embedding for blueprints.

**Files:**

- `src/esper/tamiyo/networks/factored_lstm.py`

> **Decision:** Keep `BlueprintEmbedding` in `factored_lstm.py` since it's intimately tied to
> the network architecture and will be used directly in the network's `__init__` and `forward`.

**Implementation:**

```python
from esper.leyline import (
    NUM_BLUEPRINTS,
    BLUEPRINT_NULL_INDEX,
    DEFAULT_BLUEPRINT_EMBED_DIM,
)

class BlueprintEmbedding(nn.Module):
    """Learned blueprint embeddings for Obs V3."""

    def __init__(
        self,
        num_blueprints: int = NUM_BLUEPRINTS,
        embed_dim: int = DEFAULT_BLUEPRINT_EMBED_DIM,
    ):
        super().__init__()
        # Index 13 = null embedding for inactive slots (from leyline)
        self.embedding = nn.Embedding(num_blueprints + 1, embed_dim)

        # Small initialization per DRL expert recommendation
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Register null index as buffer: moves with module.to(device), no grad, in state_dict
        # This avoids per-forward-call tensor allocation that torch.tensor() would cause
        self.register_buffer(
            '_null_idx',
            torch.tensor(BLUEPRINT_NULL_INDEX, dtype=torch.int64)
        )

    def forward(self, blueprint_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            blueprint_indices: [batch, num_slots] with -1 for inactive, dtype=int64
        Returns:
            [batch, num_slots, embed_dim]
        """
        # _null_idx is already on correct device via module.to(device)
        safe_idx = torch.where(blueprint_indices < 0, self._null_idx, blueprint_indices)
        return self.embedding(safe_idx)
```

**Validation:**

```bash
PYTHONPATH=src python -c "
import torch
from esper.tamiyo.networks.factored_lstm import BlueprintEmbedding
emb = BlueprintEmbedding()
idx = torch.tensor([[0, 2, -1], [5, -1, 3]])  # batch=2, slots=3
out = emb(idx)
print(f'Output shape: {out.shape}')  # Should be [2, 3, 4]
"
```

---

