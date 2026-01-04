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
    - h_l2_total/c_l2_total: L2 norm of full [layers, batch, hidden_dim] tensors
        NOTE: Scales with sqrt(numel) and is NOT batch-size invariant.
    - h_rms/c_rms: RMS magnitude over all elements (batch-size invariant)
    - h_env_* / c_env_*: Per-environment RMS stats (outlier detection across envs)
    - h_max/c_max: Worst-case values (catch localized spikes)
    - has_nan/has_inf: Numerical stability checks
    """

    # Capacity/load signals (scale with sqrt(numel))
    h_l2_total: float
    c_l2_total: float

    # Scale-free health signals
    h_rms: float
    c_rms: float

    # Per-env RMS stats (RMS over [layers * hidden_dim] per env)
    h_env_rms_mean: float
    h_env_rms_max: float
    c_env_rms_mean: float
    c_env_rms_max: float

    # Extremes and finiteness
    h_max: float
    c_max: float
    has_nan: bool
    has_inf: bool

    def is_healthy(
        self,
        max_rms: float = 10.0,
        min_rms: float = 1e-6,
    ) -> bool:
        """Check if LSTM state is healthy.

        Args:
            max_rms: Upper bound for healthy RMS (explosion/saturation proxy)
            min_rms: Lower bound for healthy RMS (vanishing threshold)

        Returns:
            True if state is numerically stable and within magnitude bounds
        """
        return (
            not self.has_nan
            and not self.has_inf
            and self.h_rms < max_rms
            and self.c_rms < max_rms
            and self.h_rms > min_rms
            and self.c_rms > min_rms
        )

    def to_dict(self) -> dict[str, float | bool]:
        """Convert to dict for telemetry."""
        return {
            "lstm_h_l2_total": self.h_l2_total,
            "lstm_c_l2_total": self.c_l2_total,
            "lstm_h_rms": self.h_rms,
            "lstm_c_rms": self.c_rms,
            "lstm_h_env_rms_mean": self.h_env_rms_mean,
            "lstm_h_env_rms_max": self.h_env_rms_max,
            "lstm_c_env_rms_mean": self.c_env_rms_mean,
            "lstm_c_env_rms_max": self.c_env_rms_max,
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

        batch_size = h.shape[1]
        env_dim = h.shape[0] * h.shape[2]  # layers * hidden_dim per env
        inv_sqrt_numel = 1.0 / (float(h.numel()) ** 0.5)
        inv_sqrt_env_dim = 1.0 / (float(env_dim) ** 0.5)

        # Capacity/load: L2 over full tensor (scales with sqrt(numel))
        h_l2_total_t = torch.linalg.vector_norm(h)
        c_l2_total_t = torch.linalg.vector_norm(c)

        # Scale-free: RMS over all elements
        h_rms_t = h_l2_total_t * inv_sqrt_numel
        c_rms_t = c_l2_total_t * inv_sqrt_numel

        # Per-env RMS (outlier detection across envs)
        # h_env: [batch, layers*hidden_dim]
        h_env = h.permute(1, 0, 2).reshape(batch_size, -1)
        c_env = c.permute(1, 0, 2).reshape(batch_size, -1)
        h_env_l2 = torch.linalg.vector_norm(h_env, dim=1)
        c_env_l2 = torch.linalg.vector_norm(c_env, dim=1)

        h_env_rms = h_env_l2 * inv_sqrt_env_dim
        c_env_rms = c_env_l2 * inv_sqrt_env_dim
        h_env_rms_mean_t = h_env_rms.mean()
        h_env_rms_max_t = h_env_rms.max()
        c_env_rms_mean_t = c_env_rms.mean()
        c_env_rms_max_t = c_env_rms.max()

        # Extremes
        h_max_t = h.abs().max()
        c_max_t = c.abs().max()

        # Compute boolean flags on GPU (as float for stacking)
        h_nan_t = torch.isnan(h).any().float()
        c_nan_t = torch.isnan(c).any().float()
        h_inf_t = torch.isinf(h).any().float()
        c_inf_t = torch.isinf(c).any().float()

        # Single GPU-CPU sync: stack all values and transfer together
        all_values = torch.stack([
            h_l2_total_t,
            c_l2_total_t,
            h_rms_t,
            c_rms_t,
            h_env_rms_mean_t,
            h_env_rms_max_t,
            c_env_rms_mean_t,
            c_env_rms_max_t,
            h_max_t,
            c_max_t,
            h_nan_t,
            c_nan_t,
            h_inf_t,
            c_inf_t,
        ])
        result = all_values.tolist()  # Single sync!

    (
        h_l2_total,
        c_l2_total,
        h_rms,
        c_rms,
        h_env_rms_mean,
        h_env_rms_max,
        c_env_rms_mean,
        c_env_rms_max,
        h_max,
        c_max,
        h_nan,
        c_nan,
        h_inf,
        c_inf,
    ) = result
    has_nan = bool(h_nan or c_nan)
    has_inf = bool(h_inf or c_inf)

    return LSTMHealthMetrics(
        h_l2_total=h_l2_total,
        c_l2_total=c_l2_total,
        h_rms=h_rms,
        c_rms=c_rms,
        h_env_rms_mean=h_env_rms_mean,
        h_env_rms_max=h_env_rms_max,
        c_env_rms_mean=c_env_rms_mean,
        c_env_rms_max=c_env_rms_max,
        h_max=h_max,
        c_max=c_max,
        has_nan=has_nan,
        has_inf=has_inf,
    )


__all__ = ["LSTMHealthMetrics", "compute_lstm_health"]
