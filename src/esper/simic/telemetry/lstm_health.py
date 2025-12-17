"""LSTM Hidden State Health Monitoring.

Tracks hidden state statistics to detect:
- Magnitude explosion (norm > threshold)
- Magnitude vanishing (norm < threshold)
- NaN/Inf propagation

LSTMs are prone to hidden state drift over long sequences. Without monitoring,
training can fail silently as hidden states explode or vanish.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class LSTMHealthMetrics:
    """Health metrics for LSTM hidden state.

    Provides comprehensive diagnostics for detecting LSTM training issues:
    - h_norm/c_norm: Overall magnitude of hidden/cell states
    - h_max/c_max: Worst-case values (catch localized spikes)
    - has_nan/has_inf: Numerical stability checks
    """

    h_norm: float  # L2 norm of hidden state
    c_norm: float  # L2 norm of cell state
    h_max: float   # Max absolute value in h
    c_max: float   # Max absolute value in c
    has_nan: bool
    has_inf: bool

    def is_healthy(
        self,
        max_norm: float = 100.0,
        min_norm: float = 1e-6,
    ) -> bool:
        """Check if LSTM state is healthy.

        Args:
            max_norm: Upper bound for healthy norm (explosion threshold)
            min_norm: Lower bound for healthy norm (vanishing threshold)

        Returns:
            True if state is numerically stable and within magnitude bounds
        """
        return (
            not self.has_nan
            and not self.has_inf
            and self.h_norm < max_norm
            and self.c_norm < max_norm
            and self.h_norm > min_norm
            and self.c_norm > min_norm
        )

    def to_dict(self) -> dict[str, float | bool]:
        """Convert to dict for telemetry."""
        return {
            "lstm_h_norm": self.h_norm,
            "lstm_c_norm": self.c_norm,
            "lstm_h_max": self.h_max,
            "lstm_c_max": self.c_max,
            "lstm_has_nan": self.has_nan,
            "lstm_has_inf": self.has_inf,
        }


def compute_lstm_health(
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
) -> LSTMHealthMetrics | None:
    """Compute health metrics for LSTM hidden state.

    Args:
        hidden: Tuple of (h, c) tensors, or None if no hidden state
            h: [num_layers, batch, hidden_dim]
            c: [num_layers, batch, hidden_dim]

    Returns:
        LSTMHealthMetrics or None if no hidden state
    """
    if hidden is None:
        return None

    h, c = hidden

    with torch.inference_mode():
        # M14 FIX: Batch all GPU computations before CPU transfer.
        # Original had 8 .item() calls = 8 GPU-CPU syncs.
        # Now we compute all values on GPU and transfer in a single sync.

        # Compute scalar metrics on GPU
        h_norm_t = torch.linalg.vector_norm(h)
        c_norm_t = torch.linalg.vector_norm(c)
        h_max_t = h.abs().max()
        c_max_t = c.abs().max()

        # Compute boolean flags on GPU (as float for stacking)
        h_nan_t = torch.isnan(h).any().float()
        c_nan_t = torch.isnan(c).any().float()
        h_inf_t = torch.isinf(h).any().float()
        c_inf_t = torch.isinf(c).any().float()

        # Single GPU-CPU sync: stack all values and transfer together
        all_values = torch.stack([
            h_norm_t, c_norm_t, h_max_t, c_max_t,
            h_nan_t, c_nan_t, h_inf_t, c_inf_t,
        ])
        result = all_values.tolist()  # Single sync!

    h_norm, c_norm, h_max, c_max, h_nan, c_nan, h_inf, c_inf = result
    has_nan = bool(h_nan or c_nan)
    has_inf = bool(h_inf or c_inf)

    return LSTMHealthMetrics(
        h_norm=h_norm,
        c_norm=c_norm,
        h_max=h_max,
        c_max=c_max,
        has_nan=has_nan,
        has_inf=has_inf,
    )


__all__ = ["LSTMHealthMetrics", "compute_lstm_health"]
