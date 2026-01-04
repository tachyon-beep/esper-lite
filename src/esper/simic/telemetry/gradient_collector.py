"""Lightweight Gradient Collector for Seed Telemetry.

Collects gradient statistics for seed parameters without the full
overhead of DiagnosticTracker. Designed for per-epoch collection
during comparison and training loops.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterator, TypedDict

import torch
import torch.nn as nn

from esper.leyline import DEFAULT_MAX_GRAD_NORM

# H14: PPO-tuned gradient thresholds
# These defaults are calibrated for PPO with gradient clipping at DEFAULT_MAX_GRAD_NORM (0.5).
# Collection happens BEFORE clipping, so we detect:
# - Vanishing: gradients too small to provide learning signal
# - Exploding: gradients that will be heavily clipped (10x clip norm)
#
# Why 10x clip norm for "exploding"?
# - At 10x clip, gradient direction is preserved but magnitude scaled down 10x
# - This is informative (heavy clipping) but not catastrophic
# - True explosions (100x+) indicate numerical instability
DEFAULT_VANISHING_THRESHOLD = 1e-7  # Very small but non-zero
DEFAULT_EXPLODING_THRESHOLD = 10.0 * DEFAULT_MAX_GRAD_NORM  # = 5.0 with default clip


class GradientHealthStats(TypedDict):
    """Materialized gradient health statistics for telemetry sync."""

    gradient_norm: float
    gradient_health: float
    has_vanishing: bool
    has_exploding: bool


@dataclass(slots=True)
class GradientHealthMetrics:
    """Enhanced gradient health metrics for Ops Normal monitoring.

    Extends basic gradient stats with layer-wise information
    and numerical stability indicators.
    """

    # Basic stats (existing)
    gradient_norm: float
    gradient_health: float
    has_vanishing: bool
    has_exploding: bool

    # Layer-wise summary
    min_layer_norm: float
    max_layer_norm: float
    norm_ratio: float  # max/min - high ratio indicates imbalance

    # Quality indicators
    zero_grad_fraction: float
    nan_count: int
    inf_count: int

    def is_healthy(
        self,
        max_norm_ratio: float = 1000.0,
        max_zero_fraction: float = 0.5,
    ) -> bool:
        """Check if gradients indicate healthy training.

        Returns:
            True if gradients are healthy
        """
        return (
            self.nan_count == 0
            and self.inf_count == 0
            and self.norm_ratio < max_norm_ratio
            and self.zero_grad_fraction < max_zero_fraction
            and not self.has_exploding
        )

    def to_dict(self) -> dict[str, float | int | bool]:
        """Convert to dict for TelemetryEvent data field."""
        return asdict(self)


class SeedGradientCollector:
    """Lightweight gradient statistics collector for seed telemetry.

    Unlike DiagnosticTracker, this collector:
    - Does not use hooks (called explicitly after backward)
    - Computes only essential stats (norm, health, vanishing, exploding)
    - Is stateless (no history)
    - Uses vectorized operations for performance

    H14: Thresholds are PPO-tuned by default (see module-level constants).
    Collection happens BEFORE gradient clipping, so "exploding" means
    "will be heavily clipped" rather than "catastrophic".
    """

    def __init__(
        self,
        vanishing_threshold: float = DEFAULT_VANISHING_THRESHOLD,
        exploding_threshold: float = DEFAULT_EXPLODING_THRESHOLD,
    ):
        """Initialize collector with detection thresholds.

        Args:
            vanishing_threshold: Gradient norm below this is considered vanishing.
                Default: 1e-7 (very small but non-zero).
            exploding_threshold: Gradient norm above this is considered exploding.
                Default: 10x clip norm (5.0 with DEFAULT_MAX_GRAD_NORM=0.5).
                At this level, gradients will be scaled down 10x by clipping.
        """
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold

    def collect(self, parameters: Iterator[nn.Parameter]) -> GradientHealthStats:
        """Collect gradient statistics from parameters (sync version).

        Call this after loss.backward() to gather gradient stats.
        WARNING: Calls .item() which forces CPU-GPU sync. Use collect_async()
        inside CUDA stream contexts to avoid blocking.

        Args:
            parameters: Iterator of parameters (e.g., model.parameters())

        Returns:
            GradientHealthStats with gradient_norm, gradient_health, has_vanishing, has_exploding
        """
        async_stats = self.collect_async(parameters)
        return materialize_grad_stats(async_stats)

    def collect_async(self, parameters: Iterator[nn.Parameter]) -> dict[str, bool | int | float | torch.Tensor]:
        """Collect gradient statistics as tensors (async-safe version).

        Returns tensors instead of floats to avoid .item() sync inside CUDA streams.
        Call materialize_grad_stats() AFTER stream.synchronize() to get final values.

        Args:
            parameters: Iterator of parameters (e.g., model.parameters())

        Returns:
            Dict with tensor values (call materialize_grad_stats to convert to Python types)
        """
        # Filter params with grads
        grads = [p.grad for p in parameters if p.grad is not None]

        if not grads:
            return {
                '_empty': True,
                'gradient_norm': 0.0,
                'gradient_health': 1.0,
                'has_vanishing': False,
                'has_exploding': False,
            }

        # [PyTorch 2.0+] _foreach_norm is a stable internal API used by clip_grad_norm_.
        # This is a fused CUDA kernel that computes all norms in a single kernel launch,
        # avoiding Python iteration overhead. If this breaks in future versions, fall back
        # to: [g.norm(2) for g in grads] (slower, O(n) kernels).
        #
        # B7-PT-01: Convert to float64 BEFORE _foreach_norm to prevent internal overflow.
        # _foreach_norm squares values internally, so 1e20^2 = 1e40 overflows float32.
        grads_double = [g.double() for g in grads]
        per_param_norms = torch._foreach_norm(grads_double, ord=2)

        # Stack for vectorized comparisons (now in float64)
        all_norms = torch.stack(per_param_norms)
        n_grads = len(grads)

        # Total norm via Pythagorean theorem (already float64, safe to square)
        total_squared_norm = (all_norms ** 2).sum()

        return {
            '_empty': False,
            '_n_grads': n_grads,
            '_total_squared_norm': total_squared_norm,
            '_all_norms': all_norms,
            '_n_vanishing': (all_norms < self.vanishing_threshold).sum(),
            '_n_exploding': (all_norms > self.exploding_threshold).sum(),
        }


def materialize_grad_stats(async_stats: dict[str, bool | int | float | torch.Tensor]) -> GradientHealthStats:
    """Convert async gradient stats tensors to Python values.

    Call this AFTER stream.synchronize() to safely extract .item() values.

    Args:
        async_stats: Dict from collect_async() with tensor values

    Returns:
        GradientHealthStats with Python float/bool values ready for telemetry
    """
    if async_stats['_empty']:
        # Already materialized (empty case returns Python values directly)
        return {
            'gradient_norm': float(async_stats['gradient_norm']),
            'gradient_health': float(async_stats['gradient_health']),
            'has_vanishing': bool(async_stats['has_vanishing']),
            'has_exploding': bool(async_stats['has_exploding']),
        }

    # Extract tensor values - safe to call .item() after stream sync
    n_grads_val = async_stats['_n_grads']
    assert isinstance(n_grads_val, int)
    n_grads = n_grads_val

    total_squared_val = async_stats['_total_squared_norm']
    assert isinstance(total_squared_val, torch.Tensor)
    total_squared_norm = total_squared_val.item()

    n_vanishing_val = async_stats['_n_vanishing']
    assert isinstance(n_vanishing_val, torch.Tensor)
    n_vanishing = int(n_vanishing_val.item())

    n_exploding_val = async_stats['_n_exploding']
    assert isinstance(n_exploding_val, torch.Tensor)
    n_exploding = int(n_exploding_val.item())

    # Compute global L2 norm (matches torch.nn.utils.clip_grad_norm_ semantics)
    # NOTE: Previous code divided by n_grads, producing values incomparable to
    # clipping thresholds. Global L2 = sqrt(sum(||g_i||²)) without division.
    gradient_norm = total_squared_norm ** 0.5

    # Compute health score (0-1, higher is healthier)
    vanishing_ratio = n_vanishing / n_grads
    exploding_ratio = n_exploding / n_grads

    health = 1.0
    health -= vanishing_ratio * 0.5  # Penalize vanishing
    health -= exploding_ratio * 0.8  # Penalize exploding more
    health = max(0.0, min(1.0, health))

    return {
        'gradient_norm': gradient_norm,
        'gradient_health': health,
        'has_vanishing': n_vanishing > 0,
        'has_exploding': n_exploding > 0,
    }


def collect_seed_gradients(
    seed_parameters: Iterator[nn.Parameter],
    vanishing_threshold: float = DEFAULT_VANISHING_THRESHOLD,
    exploding_threshold: float = DEFAULT_EXPLODING_THRESHOLD,
    return_enhanced: bool = False,
) -> dict[str, float | bool] | GradientHealthMetrics:
    """Convenience function to collect gradient stats (vectorized).

    Uses torch._foreach_norm for efficient batch norm computation,
    minimizing GPU-CPU synchronization points.

    Args:
        seed_parameters: Iterator of seed parameters
        vanishing_threshold: Threshold for vanishing detection
        exploding_threshold: Threshold for exploding detection
        return_enhanced: If True, return GradientHealthMetrics dataclass

    Returns:
        Dict with gradient statistics, or GradientHealthMetrics if return_enhanced
    """
    # Convert to list to allow multiple passes
    params = list(seed_parameters)
    grads = [p.grad for p in params if p.grad is not None]

    if not grads:
        if return_enhanced:
            return GradientHealthMetrics(
                gradient_norm=0.0,
                gradient_health=1.0,
                has_vanishing=False,
                has_exploding=False,
                min_layer_norm=0.0,
                max_layer_norm=0.0,
                norm_ratio=1.0,
                zero_grad_fraction=0.0,
                nan_count=0,
                inf_count=0,
            )
        return {
            'gradient_norm': 0.0,
            'gradient_health': 1.0,
            'has_vanishing': False,
            'has_exploding': False,
        }

    n_grads = len(grads)

    # [PyTorch 2.0+] _foreach_norm is a stable internal API used by clip_grad_norm_.
    # This avoids O(n) .item() calls that cause GPU-CPU sync per gradient.
    # If this breaks in future versions, fall back to: [g.norm(2) for g in grads]
    #
    # B7-PT-01: Convert to float64 BEFORE _foreach_norm to prevent internal overflow.
    grads_double = [g.double() for g in grads]
    per_param_norms = torch._foreach_norm(grads_double, ord=2)
    all_norms = torch.stack(per_param_norms)

    # Compute all stats on GPU before any .item() calls (already float64)
    total_squared = (all_norms ** 2).sum()
    min_norm_t = all_norms.min()
    max_norm_t = all_norms.max()
    n_vanishing_t = (all_norms < vanishing_threshold).sum()
    n_exploding_t = (all_norms > exploding_threshold).sum()

    # C6 FIX: Vectorized quality checks to avoid torch.compile graph breaks.
    # Trade-off: allocates O(total_params) temporary tensor, but eliminates
    # data-dependent loop that caused TorchDynamo to insert graph breaks.
    # Since this is telemetry (not hot path), memory overhead is acceptable.
    #
    # PERF NOTE: For models with 100M+ params, this allocation may spike VRAM.
    # Future optimization: compute zero/nan/inf counts per-param then sum:
    #   counts = [(g == 0).sum() for g in grads]; total_zero = sum(counts)
    # Would trade O(1) kernel for O(n) but eliminate large temp tensor.
    all_grads_flat = torch.cat([g.view(-1) for g in grads])
    total_elements = all_grads_flat.numel()
    zero_count_t = (all_grads_flat == 0).sum()
    nan_count_t = torch.isnan(all_grads_flat).sum()
    inf_count_t = torch.isinf(all_grads_flat).sum()

    # SINGLE sync point: stack all scalar tensors and materialize with one .tolist()
    stats_tensor = torch.stack([
        total_squared,
        min_norm_t,
        max_norm_t,
        n_vanishing_t.float(),
        n_exploding_t.float(),
        zero_count_t.float(),
        nan_count_t.float(),
        inf_count_t.float(),
    ])
    stats = stats_tensor.tolist()

    total_squared_val, min_norm, max_norm, n_vanishing, n_exploding, zero_count, nan_count, inf_count = stats
    # Global L2 norm (matches torch.nn.utils.clip_grad_norm_ semantics)
    # NOTE: Previous code divided by n_grads, producing values incomparable to
    # clipping thresholds. Global L2 = sqrt(sum(||g_i||²)) without division.
    gradient_norm = total_squared_val ** 0.5
    n_vanishing = int(n_vanishing)
    n_exploding = int(n_exploding)
    zero_fraction = zero_count / max(total_elements, 1)
    nan_count = int(nan_count)
    inf_count = int(inf_count)

    # Compute derived stats
    norm_ratio = max_norm / max(min_norm, 1e-10)

    # Health score
    vanishing_ratio = n_vanishing / n_grads
    exploding_ratio = n_exploding / n_grads
    health = 1.0
    health -= vanishing_ratio * 0.5
    health -= exploding_ratio * 0.8
    health = max(0.0, min(1.0, health))

    if return_enhanced:
        return GradientHealthMetrics(
            gradient_norm=gradient_norm,
            gradient_health=health,
            has_vanishing=n_vanishing > 0,
            has_exploding=n_exploding > 0,
            min_layer_norm=min_norm,
            max_layer_norm=max_norm,
            norm_ratio=norm_ratio,
            zero_grad_fraction=zero_fraction,
            nan_count=nan_count,
            inf_count=inf_count,
        )

    return {
        'gradient_norm': gradient_norm,
        'gradient_health': health,
        'has_vanishing': n_vanishing > 0,
        'has_exploding': n_exploding > 0,
    }


def collect_seed_gradients_async(
    seed_parameters: Iterator[nn.Parameter],
    vanishing_threshold: float = DEFAULT_VANISHING_THRESHOLD,
    exploding_threshold: float = DEFAULT_EXPLODING_THRESHOLD,
) -> dict[str, bool | int | float | torch.Tensor]:
    """Collect gradient stats as tensors (async-safe version).

    Returns tensors instead of floats to avoid .item() sync inside CUDA streams.
    Call materialize_grad_stats() AFTER stream.synchronize() to get final values.

    Args:
        seed_parameters: Iterator of seed parameters
        vanishing_threshold: Threshold for vanishing detection
        exploding_threshold: Threshold for exploding detection

    Returns:
        Dict with tensor values (call materialize_grad_stats to convert to Python types)
    """
    collector = SeedGradientCollector(
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
    )
    return collector.collect_async(seed_parameters)


# =============================================================================
# Dual Gradient Collection (Host + Seed) for G2 Gate
# =============================================================================

@dataclass(slots=True)
class DualGradientStats:
    """Gradient statistics for both host and seed networks.

    Used to compute the seed_gradient_norm_ratio metric for the G2 gate,
    which detects if a seed is actively learning vs. riding host improvements.
    """
    host_grad_norm: float
    host_param_count: int
    seed_grad_norm: float
    seed_param_count: int

    @property
    def normalized_ratio(self) -> float:
        """Parameter-normalized gradient ratio (seed intensity / host intensity).

        Formula: (seed_norm / host_norm) * sqrt(host_params / seed_params)

        This normalizes by sqrt(param_count) because for i.i.d. gradient elements
        with variance σ², the expected L2 norm is σ√n. Normalizing removes the
        parameter count bias so small seeds aren't penalized for having fewer params.

        Returns:
            Ratio >= 0. Values around 0.05-0.2 indicate healthy seed activity.
            Values < 0.01 suggest the seed is dormant or riding host improvements.
        """
        eps = 1e-8
        if self.host_grad_norm < eps or self.seed_grad_norm < eps:
            return 0.0
        if self.host_param_count == 0 or self.seed_param_count == 0:
            return 0.0

        # Normalize by sqrt(param_count) to remove scale bias
        seed_intensity = self.seed_grad_norm / (self.seed_param_count ** 0.5)
        host_intensity = self.host_grad_norm / (self.host_param_count ** 0.5)

        return float(seed_intensity / (host_intensity + eps))


def materialize_dual_grad_stats(async_stats: dict[str, torch.Tensor | int]) -> DualGradientStats:
    """Convert async dual gradient stats to DualGradientStats.

    Call this AFTER stream.synchronize() to safely extract .item() values.

    Args:
        async_stats: Dict containing host/seed squared norms and param counts

    Returns:
        DualGradientStats with computed norms and param counts
    """
    host_squared_val = async_stats['_host_squared_sum']
    seed_squared_val = async_stats['_seed_squared_sum']

    # Handle tensor vs float (depends on whether collected in stream)
    if isinstance(host_squared_val, torch.Tensor):
        host_squared = float(host_squared_val.item())
    else:
        host_squared = float(host_squared_val)

    if isinstance(seed_squared_val, torch.Tensor):
        seed_squared = float(seed_squared_val.item())
    else:
        seed_squared = float(seed_squared_val)

    host_param_count_val = async_stats['_host_param_count']
    seed_param_count_val = async_stats['_seed_param_count']
    assert isinstance(host_param_count_val, int)
    assert isinstance(seed_param_count_val, int)

    return DualGradientStats(
        host_grad_norm=host_squared ** 0.5,
        host_param_count=host_param_count_val,
        seed_grad_norm=seed_squared ** 0.5,
        seed_param_count=seed_param_count_val,
    )


def collect_host_gradients_async(
    host_parameters: Iterator[nn.Parameter],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | int]:
    """Collect gradient stats for host network only (async-safe).

    Use this when computing host gradients once and reusing across multiple seeds.

    IMPORTANT: Returned tensors are on GPU and require synchronization before
    calling .item(). Use torch.cuda.synchronize() or stream.synchronize()
    before materialize_dual_grad_stats().

    Args:
        host_parameters: Iterator over host network parameters
        device: Device for zero tensors when no gradients exist

    Returns:
        Dict with _host_squared_sum (Tensor), _host_param_count (int)
    """
    host_grads = [p.grad for p in host_parameters if p.grad is not None]
    if host_grads:
        # [PyTorch 2.0+] _foreach_norm is stable internal API (used by clip_grad_norm_)
        # B7-PT-01: Convert to float64 BEFORE _foreach_norm to prevent internal overflow
        host_grads_double = [g.double() for g in host_grads]
        host_norms = torch._foreach_norm(host_grads_double, ord=2)
        host_squared_sum = torch.stack(host_norms).pow(2).sum()
        host_param_count = sum(g.numel() for g in host_grads)
    else:
        # Return zero scalar tensor (not float) to maintain type consistency for async pattern
        host_squared_sum = torch.zeros((), device=device)
        host_param_count = 0

    return {
        "_host_squared_sum": host_squared_sum,
        "_host_param_count": host_param_count,
    }


def collect_seed_gradients_only_async(
    seed_parameters: Iterator[nn.Parameter],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | int]:
    """Collect gradient stats for seed network only (async-safe).

    Use with collect_host_gradients_async when host stats are precomputed.

    IMPORTANT: Returned tensors are on GPU and require synchronization before
    calling .item(). Use torch.cuda.synchronize() or stream.synchronize()
    before materialize_dual_grad_stats().

    Args:
        seed_parameters: Iterator over seed network parameters
        device: Device for zero tensors when no gradients exist

    Returns:
        Dict with _seed_squared_sum (Tensor), _seed_param_count (int)
    """
    seed_grads = [p.grad for p in seed_parameters if p.grad is not None]
    if seed_grads:
        # [PyTorch 2.0+] _foreach_norm is stable internal API (used by clip_grad_norm_)
        # B7-PT-01: Convert to float64 BEFORE _foreach_norm to prevent internal overflow
        seed_grads_double = [g.double() for g in seed_grads]
        seed_norms = torch._foreach_norm(seed_grads_double, ord=2)
        seed_squared_sum = torch.stack(seed_norms).pow(2).sum()
        seed_param_count = sum(g.numel() for g in seed_grads)
    else:
        # Return zero scalar tensor (not float) to maintain type consistency for async pattern
        seed_squared_sum = torch.zeros((), device=device)
        seed_param_count = 0

    return {
        "_seed_squared_sum": seed_squared_sum,
        "_seed_param_count": seed_param_count,
    }


def collect_dual_gradients_async(
    host_parameters: Iterator[nn.Parameter],
    seed_parameters: Iterator[nn.Parameter],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | int]:
    """Collect gradient stats for both host and seed networks (async-safe).

    Returns tensors instead of floats to avoid .item() sync inside CUDA streams.
    Call materialize_dual_grad_stats() AFTER stream.synchronize() to get final values.

    Note: If collecting for multiple seeds, use collect_host_gradients_async() once
    and collect_seed_gradients_only_async() per seed to avoid recomputing host norms.

    Args:
        host_parameters: Iterator over host network parameters
        seed_parameters: Iterator over seed network parameters
        device: Device for zero tensors when no gradients exist

    Returns:
        Dict with _host_squared_sum, _host_param_count, _seed_squared_sum, _seed_param_count
    """
    host_stats = collect_host_gradients_async(host_parameters, device)
    seed_stats = collect_seed_gradients_only_async(seed_parameters, device)

    return {
        "_host_squared_sum": host_stats["_host_squared_sum"],
        "_host_param_count": host_stats["_host_param_count"],
        "_seed_squared_sum": seed_stats["_seed_squared_sum"],
        "_seed_param_count": seed_stats["_seed_param_count"],
    }


__all__ = [
    # H14: PPO-tuned threshold constants
    "DEFAULT_VANISHING_THRESHOLD",
    "DEFAULT_EXPLODING_THRESHOLD",
    # Classes, TypedDicts, and functions
    "GradientHealthStats",
    "GradientHealthMetrics",
    "SeedGradientCollector",
    "materialize_grad_stats",
    "collect_seed_gradients",
    "collect_seed_gradients_async",
    "DualGradientStats",
    "materialize_dual_grad_stats",
    "collect_host_gradients_async",
    "collect_seed_gradients_only_async",
    "collect_dual_gradients_async",
]
