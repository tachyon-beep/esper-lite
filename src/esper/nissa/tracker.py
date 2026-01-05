"""Diagnostic Telemetry for Rich Training Insights.

Collects gradient health, loss landscape, per-class metrics, and more
based on TelemetryConfig. Generates human/LLM-readable narratives.

Usage:
    config = TelemetryConfig.from_profile("diagnostic")
    tracker = DiagnosticTracker(model, config)

    # During training
    tracker.on_backward()  # Call after loss.backward()
    snapshot = tracker.end_epoch(train_loss, val_loss, val_acc, per_class_acc)

    # Get readable summary
    print(tracker.generate_narrative(snapshot))
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from esper.nissa.config import TelemetryConfig


@dataclass
class GradientStats:
    """Statistics for a single layer's gradients."""
    layer_name: str
    norm: float = 0.0
    std: float = 0.0
    mean: float = 0.0
    vanishing_pct: float = 0.0  # % of near-zero gradients
    exploding_pct: float = 0.0  # % of very large gradients
    percentiles: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer_name,
            "norm": self.norm,
            "std": self.std,
            "mean": self.mean,
            "vanishing_pct": self.vanishing_pct,
            "exploding_pct": self.exploding_pct,
            "percentiles": self.percentiles,
        }


@dataclass
class GradientHealth:
    """Aggregate gradient health indicators."""
    overall_norm: float = 0.0
    norm_variance: float = 0.0
    vanishing_layers: int = 0
    exploding_layers: int = 0
    health_score: float = 1.0  # 0-1, higher is healthier

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_norm": self.overall_norm,
            "norm_variance": self.norm_variance,
            "vanishing_layers": self.vanishing_layers,
            "exploding_layers": self.exploding_layers,
            "health_score": self.health_score,
        }


@dataclass
class EpochSnapshot:
    """Rich snapshot of training state at end of epoch."""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    train_val_gap: float = 0.0

    # Gradient info (if enabled)
    gradient_stats: list[GradientStats] = field(default_factory=list)
    gradient_health: GradientHealth | None = None

    # Per-class metrics (if enabled)
    per_class_accuracy: dict[int, float] = field(default_factory=dict)
    per_class_loss: dict[int, float] = field(default_factory=dict)
    class_variance: float = 0.0

    # Loss landscape (if enabled)
    sharpness: float | None = None
    loss_noise: float | None = None

    # Weight stats (if enabled)
    weight_norms: dict[str, float] = field(default_factory=dict)

    # Narrative
    narrative: str = ""
    red_flags: list[str] = field(default_factory=list)
    opportunities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "train_val_gap": self.train_val_gap,
            "narrative": self.narrative,
            "red_flags": self.red_flags,
            "opportunities": self.opportunities,
        }

        if self.gradient_stats:
            result["gradient_stats"] = [g.to_dict() for g in self.gradient_stats]
        if self.gradient_health:
            result["gradient_health"] = self.gradient_health.to_dict()
        if self.per_class_accuracy:
            result["per_class_accuracy"] = self.per_class_accuracy
        if self.per_class_loss:
            result["per_class_loss"] = self.per_class_loss
        if self.class_variance > 0:
            result["class_variance"] = self.class_variance
        if self.sharpness is not None:
            result["sharpness"] = self.sharpness
        if self.loss_noise is not None:
            result["loss_noise"] = self.loss_noise
        if self.weight_norms:
            result["weight_norms"] = self.weight_norms

        return result


class DiagnosticTracker:
    """Collects rich telemetry based on TelemetryConfig."""

    def __init__(self, model: nn.Module, config: TelemetryConfig, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = device

        # History buffer
        self.history: deque[EpochSnapshot] = deque(maxlen=config.history_length)

        # Gradient tracking state
        self._grad_stats: dict[str, GradientStats] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Percentile quantile cache (lazily populated per device/dtype)
        # Avoids recreating q_tensor on every backward pass
        self._percentile_q_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._percentile_q_values: list[float] = (
            [p / 100.0 for p in config.gradients.percentiles]
            if config.gradients.enabled and config.gradients.percentiles
            else []
        )

        # Register gradient hooks if enabled
        if config.gradients.enabled:
            self._register_gradient_hooks()

        # Loss history for noise estimation
        self._batch_losses: list[float] = []

    def _get_percentile_q(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get cached quantile tensor for the given device and dtype.

        Caches per (device, dtype) pair to handle multi-device models and
        mixed precision training without redundant tensor allocations.
        """
        key = (device, dtype)
        if key not in self._percentile_q_cache:
            self._percentile_q_cache[key] = torch.tensor(
                self._percentile_q_values, device=device, dtype=dtype
            )
        return self._percentile_q_cache[key]

    def _should_track_layer(self, name: str) -> bool:
        """Check if we should track gradients for this layer."""
        if self.config.gradients.layers == "all":
            # Track layers with 'weight' in name (skip biases, norms, etc.)
            return "weight" in name
        return any(layer in name for layer in self.config.gradients.layers)

    def _register_gradient_hooks(self) -> None:
        """Register hooks to capture gradients during backward pass."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self._should_track_layer(name):
                hook = param.register_hook(  # type: ignore[no-untyped-call]
                    lambda grad, n=name: self._record_grad(n, grad)
                )
                self._hooks.append(hook)

    # Subsampling threshold for percentile computation on large gradient tensors.
    # 10k samples give 99%+ accuracy on percentile estimates while reducing
    # O(n log n) quantile sort cost significantly for large layers.
    _PERCENTILE_SUBSAMPLE_SIZE: int = 10000

    def _record_grad(self, name: str, grad: torch.Tensor) -> None:
        """Record gradient statistics for a layer.

        Uses batched tensor ops with single .tolist() sync to avoid
        per-metric GPU synchronization in this hot path (called every backward).

        Performance optimizations:
        - Cached quantile tensor per (device, dtype) to avoid allocation churn
        - Subsampling for large gradient tensors (>10k elements) before quantile
        - Single GPU sync via batched tolist()

        Respects config flags:
        - track_norm: If False, skip norm computation (stats.norm stays 0.0)
        - track_std: If False, skip std computation (stats.std stays 0.0)
        """
        if grad is None:
            return

        cfg = self.config.gradients
        grad_flat = grad.flatten()

        # Batch all scalar computations into a single tensor for one sync
        # Track which metrics we're computing for correct unpacking
        stats_tensors: list[torch.Tensor] = []
        metric_keys: list[str] = []

        if cfg.track_norm:
            stats_tensors.append(grad.norm())
            metric_keys.append("norm")
        if cfg.track_std:
            stats_tensors.append(grad.std())
            metric_keys.append("std")

        # Mean is always tracked (no config flag for it)
        stats_tensors.append(grad.mean())
        metric_keys.append("mean")

        # Vanishing/exploding detection uses abs - compute only if needed
        if cfg.detect_vanishing or cfg.detect_exploding:
            grad_abs = grad_flat.abs()
            if cfg.detect_vanishing:
                stats_tensors.append((grad_abs < cfg.vanishing_threshold).float().mean())
                metric_keys.append("vanishing_pct")
            if cfg.detect_exploding:
                stats_tensors.append((grad_abs > cfg.exploding_threshold).float().mean())
                metric_keys.append("exploding_pct")

        # Percentiles with subsampling for large tensors and cached q_tensor
        # torch.quantile with a tensor of q values does ONE sort instead of N sorts
        percentile_keys: list[int] = list(cfg.percentiles) if cfg.percentiles else []
        percentile_results: torch.Tensor | None = None
        if cfg.percentiles:
            # Subsample large tensors for efficiency (10k samples gives valid estimates)
            if grad_flat.numel() > self._PERCENTILE_SUBSAMPLE_SIZE:
                indices = torch.randperm(
                    grad_flat.numel(), device=grad.device
                )[: self._PERCENTILE_SUBSAMPLE_SIZE]
                sample = grad_flat[indices]
            else:
                sample = grad_flat

            # Use native dtype if float, else cast to float32
            if sample.dtype in (torch.float32, torch.float16, torch.bfloat16):
                sample_for_quantile = sample
            else:
                sample_for_quantile = sample.float()

            q_tensor = self._get_percentile_q(grad.device, sample_for_quantile.dtype)
            percentile_results = torch.quantile(sample_for_quantile, q_tensor)

        # Single GPU sync for all stats - concatenate scalars with percentile vector
        if stats_tensors and percentile_results is not None:
            all_values = torch.cat([torch.stack(stats_tensors), percentile_results]).tolist()
        elif stats_tensors:
            all_values = torch.stack(stats_tensors).tolist()
        elif percentile_results is not None:
            all_values = percentile_results.tolist()
        else:
            all_values = []

        # Unpack values using tracked metric keys
        stats = GradientStats(layer_name=name)
        idx = 0
        for key in metric_keys:
            if key == "norm":
                stats.norm = all_values[idx]
            elif key == "std":
                stats.std = all_values[idx]
            elif key == "mean":
                stats.mean = all_values[idx]
            elif key == "vanishing_pct":
                stats.vanishing_pct = all_values[idx]
            elif key == "exploding_pct":
                stats.exploding_pct = all_values[idx]
            idx += 1

        for p in percentile_keys:
            stats.percentiles[p] = all_values[idx]
            idx += 1

        self._grad_stats[name] = stats

    def on_batch_loss(self, loss: float) -> None:
        """Record batch loss for noise estimation."""
        self._batch_losses.append(loss)

    def on_backward(self) -> None:
        """Call after loss.backward() to ensure gradients are captured.

        Note: Gradient hooks fire automatically during backward(),
        but this method can be used for any post-backward processing.
        """
        pass  # Hooks handle gradient capture

    def _compute_gradient_health(self) -> GradientHealth:
        """Compute aggregate gradient health from per-layer stats."""
        if not self._grad_stats:
            return GradientHealth()

        norms = [s.norm for s in self._grad_stats.values()]
        vanishing_pcts = [s.vanishing_pct for s in self._grad_stats.values()]
        exploding_pcts = [s.exploding_pct for s in self._grad_stats.values()]

        health = GradientHealth()
        health.overall_norm = float(np.mean(norms))
        health.norm_variance = float(np.var(norms))
        health.vanishing_layers = sum(1 for v in vanishing_pcts if v > 0.5)
        health.exploding_layers = sum(1 for e in exploding_pcts if e > 0.01)

        # Health score: penalize vanishing/exploding, high variance
        score = 1.0
        if health.vanishing_layers > 0:
            score -= 0.2 * health.vanishing_layers
        if health.exploding_layers > 0:
            score -= 0.3 * health.exploding_layers
        if health.norm_variance > 10:
            score -= 0.1
        if health.overall_norm < 1e-5:
            score -= 0.3
        if health.overall_norm > 100:
            score -= 0.2
        health.health_score = max(0.0, min(1.0, score))

        return health

    def _estimate_sharpness(self, val_loader: Any, criterion: Any) -> float | None:
        """Estimate loss landscape sharpness via perturbation.

        Higher sharpness = sharper minimum = less generalizable.

        Note: This method temporarily sets model.eval() for inference but
        restores the original training mode on exit. Telemetry collection
        should be side-effect free with respect to training state.

        Respects config flags:
        - enabled: If False, skip all loss landscape analysis
        - estimate_sharpness: If False, skip sharpness estimation specifically
        """
        cfg = self.config.loss_landscape
        if not cfg.enabled or not cfg.estimate_sharpness:
            return None

        # Preserve and restore training mode to avoid leaking state into training loop
        was_training = self.model.training
        self.model.eval()

        try:
            # Get baseline loss
            baseline_loss = self._compute_val_loss(val_loader, criterion)

            # Perturb weights and measure loss changes
            perturbations = []
            original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            for _ in range(cfg.perturbation_samples):
                # Add random perturbation to weights
                with torch.no_grad():
                    for param in self.model.parameters():
                        noise = torch.randn_like(param) * cfg.perturbation_scale
                        param.add_(noise)

                # Measure perturbed loss
                perturbed_loss = self._compute_val_loss(val_loader, criterion)
                perturbations.append(abs(perturbed_loss - baseline_loss))

                # Restore original weights
                self.model.load_state_dict(original_state)

            return float(np.mean(perturbations)) / cfg.perturbation_scale
        finally:
            self.model.train(was_training)

    def _compute_val_loss(self, val_loader: Any, criterion: Any) -> float:
        """Compute validation loss."""
        total_loss = torch.tensor(0.0, device=self.device)
        n_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss
                n_batches += 1
                if n_batches >= 10:  # Sample for speed
                    break

        return total_loss.item() / max(n_batches, 1)

    def _compute_weight_norms(self) -> dict[str, float]:
        """Compute weight norms per layer (single sync for all layers)."""
        if not self.config.track_weight_norms:
            return {}

        # Collect all norms as tensors first
        names = []
        norm_tensors = []
        for name, param in self.model.named_parameters():
            if "weight" in name:
                names.append(name)
                norm_tensors.append(param.norm())

        if not norm_tensors:
            return {}

        # Single sync for all weight norms
        all_norms = torch.stack(norm_tensors).tolist()
        return dict(zip(names, all_norms))

    def end_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        per_class_accuracy: dict[int, float] | None = None,
        per_class_loss: dict[int, float] | None = None,
        val_loader: Any = None,
        criterion: Any = None,
    ) -> EpochSnapshot:
        """Record end-of-epoch metrics and return rich snapshot."""
        snapshot = EpochSnapshot(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            train_val_gap=val_loss - train_loss,
        )

        # Gradient stats
        if self.config.gradients.enabled and self._grad_stats:
            snapshot.gradient_stats = list(self._grad_stats.values())
            snapshot.gradient_health = self._compute_gradient_health()
            self._grad_stats.clear()

        # Per-class metrics
        if self.config.per_class.enabled and per_class_accuracy:
            snapshot.per_class_accuracy = per_class_accuracy
            snapshot.class_variance = float(np.var(list(per_class_accuracy.values())))
            if per_class_loss:
                snapshot.per_class_loss = per_class_loss

        # Loss landscape
        if self.config.loss_landscape.enabled and val_loader and criterion:
            snapshot.sharpness = self._estimate_sharpness(val_loader, criterion)

        # Loss noise (batch-to-batch variance)
        if self._batch_losses:
            snapshot.loss_noise = float(np.std(self._batch_losses))
            self._batch_losses.clear()

        # Weight norms
        snapshot.weight_norms = self._compute_weight_norms()

        # Generate narrative and flags
        snapshot.narrative = self.generate_narrative(snapshot)
        snapshot.red_flags = self._detect_red_flags(snapshot)
        snapshot.opportunities = self._detect_opportunities(snapshot)

        # Add to history
        self.history.append(snapshot)

        return snapshot

    def generate_narrative(self, snapshot: EpochSnapshot) -> str:
        """Generate human/LLM-readable summary of current state."""
        parts = []

        # Loss trajectory (compare current snapshot to most recent in history)
        # Note: snapshot is NOT in history yet when this is called from end_epoch()
        if self.history:
            prev = self.history[-1]
            loss_delta = prev.val_loss - snapshot.val_loss
            if loss_delta > 0.01:
                parts.append(f"Loss improving ({prev.val_loss:.3f} -> {snapshot.val_loss:.3f})")
            elif loss_delta < -0.01:
                parts.append(f"Loss degrading ({prev.val_loss:.3f} -> {snapshot.val_loss:.3f})")
            else:
                plateau_len = self._plateau_length(snapshot.val_loss)
                if plateau_len >= 3:
                    parts.append(f"Loss plateaued for {plateau_len} epochs at ~{snapshot.val_loss:.3f}")

        # Gradient health
        if snapshot.gradient_health:
            gh = snapshot.gradient_health
            if gh.vanishing_layers > 0:
                parts.append(f"Vanishing gradients in {gh.vanishing_layers} layers")
            if gh.exploding_layers > 0:
                parts.append(f"Exploding gradients in {gh.exploding_layers} layers")
            if gh.health_score >= 0.8:
                parts.append("Gradient health good")

        # Class imbalance
        if snapshot.per_class_accuracy:
            accs = list(snapshot.per_class_accuracy.values())
            worst = min(accs)
            best = max(accs)
            if best - worst > 20:
                worst_class = min(snapshot.per_class_accuracy.items(), key=lambda x: x[1])[0]
                best_class = max(snapshot.per_class_accuracy.items(), key=lambda x: x[1])[0]
                parts.append(f"Class imbalance: {worst_class}={worst:.0f}% vs {best_class}={best:.0f}%")

        # Overfitting
        if snapshot.train_val_gap > 0.5:
            parts.append(f"Possible overfitting (train-val gap: {snapshot.train_val_gap:.2f})")

        return " | ".join(parts) if parts else "Training normally"

    def _plateau_length(self, current_loss: float | None = None) -> int:
        """Count epochs in current plateau (including current epoch).

        Args:
            current_loss: The current epoch's validation loss. If provided,
                counts from current backward. If None, uses history[-1].

        Returns:
            Number of consecutive epochs with similar loss values.
            Returns 0 if no history and no current_loss provided.
        """
        threshold = 0.005

        # Determine reference loss: current_loss if given, else history[-1]
        if current_loss is not None:
            ref_loss = current_loss
            count = 1  # Start at 1 to include current epoch
            search_history = list(self.history)  # Compare against all history
        elif self.history:
            ref_loss = self.history[-1].val_loss
            count = 1  # Start at 1 to include most recent
            search_history = list(self.history)[:-1]  # Compare against all but last
        else:
            return 0  # No reference point

        # Count consecutive epochs with similar loss going backwards
        for snap in reversed(search_history):
            if abs(snap.val_loss - ref_loss) < threshold:
                count += 1
            else:
                break

        return count

    @property
    def plateau_detected(self) -> bool:
        """Check if training is in a plateau (3+ epochs with no improvement)."""
        return self._plateau_length() >= 3

    def _detect_red_flags(self, snapshot: EpochSnapshot) -> list[str]:
        """Detect issues that might need attention."""
        flags = []

        if self._plateau_length(snapshot.val_loss) >= 3:
            flags.append("sustained_plateau")

        if snapshot.gradient_health:
            if snapshot.gradient_health.vanishing_layers > 0:
                flags.append("vanishing_gradients")
            if snapshot.gradient_health.exploding_layers > 0:
                flags.append("exploding_gradients")
            if snapshot.gradient_health.health_score < 0.5:
                flags.append("poor_gradient_health")

        if snapshot.train_val_gap > 0.5:
            flags.append("overfitting")

        if snapshot.class_variance > 200:
            flags.append("severe_class_imbalance")

        if snapshot.sharpness and snapshot.sharpness > 1.0:
            flags.append("sharp_minimum")

        return flags

    def _detect_opportunities(self, snapshot: EpochSnapshot) -> list[str]:
        """Detect favorable conditions for intervention."""
        opportunities = []

        if snapshot.gradient_health and snapshot.gradient_health.health_score > 0.8:
            opportunities.append("healthy_gradients")

        if len(self.history) >= 5:
            # Stable training
            recent_losses = [h.val_loss for h in list(self.history)[-5:]]
            if max(recent_losses) - min(recent_losses) < 0.1:
                opportunities.append("stable_training")

        if snapshot.val_accuracy < 50:
            opportunities.append("early_stage_flexibility")

        return opportunities

    def get_decision_brief(self, action: str, reason: str = "") -> dict[str, Any]:
        """Generate structured brief for a decision."""
        if not self.history:
            return {"action": action, "error": "No history available"}

        snapshot = self.history[-1]

        return {
            "action": action,
            "epoch": snapshot.epoch,
            "narrative": snapshot.narrative,
            "red_flags": snapshot.red_flags,
            "opportunities": snapshot.opportunities,
            "heuristic_reason": reason,
            "gradient_health": snapshot.gradient_health.to_dict() if snapshot.gradient_health else None,
            "key_metrics": {
                "val_loss": snapshot.val_loss,
                "val_accuracy": snapshot.val_accuracy,
                "train_val_gap": snapshot.train_val_gap,
                "plateau_length": self._plateau_length(),
            }
        }

    def cleanup(self) -> None:
        """Remove gradient hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self.cleanup()


# Export
__all__ = [
    "DiagnosticTracker",
    "EpochSnapshot",
    "GradientStats",
    "GradientHealth",
]
