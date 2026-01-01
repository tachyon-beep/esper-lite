"""Telemetry Configuration for Rich Diagnostics.

Pydantic models for configuring what telemetry to collect during training.
Supports profiles (minimal/standard/diagnostic/research) with overrides.

Usage:
    # Load a profile
    config = TelemetryConfig.from_profile("diagnostic")

    # Load with overrides
    config = TelemetryConfig.from_profile("standard", {"history_length": 15})

    # Load from custom YAML file
    config = TelemetryConfig.from_yaml("my_config.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator

from esper.leyline import OBS_V3_NON_BLUEPRINT_DIM


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class GradientConfig(BaseModel):
    """Configuration for gradient tracking."""

    enabled: bool = False
    layers: list[str] | Literal["all"] = "all"
    track_norm: bool = True
    track_std: bool = True
    percentiles: list[int] = Field(default=[1, 50, 99])
    detect_vanishing: bool = True
    detect_exploding: bool = True
    vanishing_threshold: float = Field(default=1e-7, gt=0)
    exploding_threshold: float = Field(default=1e3, gt=0)
    full_histogram: bool = False

    @field_validator("percentiles")
    @classmethod
    def validate_percentiles(cls, v: list[int]) -> list[int]:
        for p in v:
            if not 0 <= p <= 100:
                raise ValueError(f"Percentile {p} must be between 0 and 100")
        return sorted(v)


class LossLandscapeConfig(BaseModel):
    """Configuration for loss landscape analysis."""

    enabled: bool = False
    perturbation_samples: int = Field(default=5, ge=1, le=20)
    perturbation_scale: float = Field(default=0.01, gt=0, lt=1)
    estimate_sharpness: bool = True


class PerClassConfig(BaseModel):
    """Configuration for per-class metrics (e.g., CIFAR-10 classes)."""

    enabled: bool = False
    track_accuracy: bool = True
    track_loss: bool = False
    track_confusion: bool = False  # Full NxN confusion matrix


class TelemetryConfig(BaseModel):
    """Main telemetry configuration - validated mercilessly.

    Attributes:
        profile_name: Name of the profile this config was loaded from.
        history_length: Number of epochs of history to track.
        gradients: Gradient tracking configuration.
        loss_landscape: Loss landscape analysis configuration.
        per_class: Per-class metrics configuration.
        track_weight_norms: Track weight norms per layer.
        track_activation_stats: Track activation statistics.
    """

    profile_name: str = "custom"
    history_length: int = Field(default=10, ge=5, le=100)

    gradients: GradientConfig = Field(default_factory=GradientConfig)
    loss_landscape: LossLandscapeConfig = Field(default_factory=LossLandscapeConfig)
    per_class: PerClassConfig = Field(default_factory=PerClassConfig)

    track_weight_norms: bool = False
    track_activation_stats: bool = False

    @classmethod
    def from_yaml(cls, path: Path | str, overrides: dict[str, Any] | None = None) -> TelemetryConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.
            overrides: Optional dict of values to override.

        Returns:
            Validated TelemetryConfig instance.

        Raises:
            ValueError: If the YAML file is empty, malformed, or not a mapping.
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Telemetry YAML must be a mapping, got {type(data).__name__}: {path}"
            )

        if overrides:
            data = deep_merge(data, overrides)

        return cls(**data)

    @classmethod
    def from_profile(cls, name: str, overrides: dict[str, Any] | None = None) -> TelemetryConfig:
        """Load a built-in profile by name.

        Args:
            name: Profile name (minimal, standard, diagnostic, research).
            overrides: Optional dict of values to override.

        Returns:
            Validated TelemetryConfig instance.

        Raises:
            ValueError: If profile name is not recognized or profiles.yaml is malformed.
            FileNotFoundError: If profiles.yaml does not exist.
        """
        profiles_path = Path(__file__).parent / "profiles.yaml"

        if not profiles_path.exists():
            raise FileNotFoundError(
                f"Profiles file not found: {profiles_path}. "
                "Please create profiles.yaml in the nissa package."
            )

        with open(profiles_path) as f:
            all_profiles = yaml.safe_load(f)

        # Validate YAML structure explicitly - fail fast with clear messages
        if not isinstance(all_profiles, dict):
            raise ValueError(
                f"profiles.yaml must be a mapping, got {type(all_profiles).__name__}"
            )

        if "profiles" not in all_profiles:
            raise ValueError(
                f"profiles.yaml is missing required 'profiles' key. "
                f"Found keys: {list(all_profiles.keys())}"
            )

        profiles = all_profiles["profiles"]
        if not isinstance(profiles, dict):
            raise ValueError(
                f"'profiles' key must be a mapping, got {type(profiles).__name__}"
            )

        if name not in profiles:
            available = list(profiles.keys())
            raise ValueError(f"Unknown profile: {name}. Available: {available}")

        data = profiles[name].copy()
        data["profile_name"] = name

        if overrides:
            data = deep_merge(data, overrides)

        return cls(**data)

    @classmethod
    def minimal(cls) -> TelemetryConfig:
        """Shortcut for minimal profile."""
        return cls.from_profile("minimal")

    @classmethod
    def standard(cls) -> TelemetryConfig:
        """Shortcut for standard profile."""
        return cls.from_profile("standard")

    @classmethod
    def diagnostic(cls) -> TelemetryConfig:
        """Shortcut for diagnostic profile."""
        return cls.from_profile("diagnostic")

    @classmethod
    def research(cls) -> TelemetryConfig:
        """Shortcut for research profile."""
        return cls.from_profile("research")

    def feature_count_estimate(self, num_classes: int = 10) -> int:
        """Estimate total feature count for this configuration.

        Args:
            num_classes: Number of output classes for per-class metrics.
                Defaults to 10 (CIFAR-10). Pass the actual class count
                for other tasks (e.g., 100 for CIFAR-100).

        Returns:
            Estimated feature count based on configuration.
        """
        # Base observation dims from leyline (Obs V3 for 3-slot config = 113 dims)
        count = OBS_V3_NON_BLUEPRINT_DIM

        if self.gradients.enabled:
            # Assume ~4 tracked layers when "all", otherwise use explicit list
            n_layers = 4 if self.gradients.layers == "all" else len(self.gradients.layers)
            stats_per_layer = 2  # norm, std
            if self.gradients.percentiles:
                stats_per_layer += len(self.gradients.percentiles)
            count += n_layers * stats_per_layer
            count += 4  # grad_health summary stats

        if self.per_class.enabled:
            count += num_classes  # Per-class accuracy
            if self.per_class.track_loss:
                count += num_classes
            if self.per_class.track_confusion:
                count += num_classes * num_classes  # NxN confusion matrix

        if self.loss_landscape.enabled:
            count += 3  # sharpness, curvature estimate, noise

        count += (self.history_length - 5) * 2  # Extra history beyond base 5

        return count

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def summary(self) -> str:
        """Human-readable summary of the configuration."""
        lines = [
            f"TelemetryConfig (profile: {self.profile_name})",
            f"  History length: {self.history_length}",
            f"  Estimated features: ~{self.feature_count_estimate()}",
            f"  Gradients: {'enabled' if self.gradients.enabled else 'disabled'}",
        ]
        if self.gradients.enabled:
            layers = "all" if self.gradients.layers == "all" else len(self.gradients.layers)
            lines.append(f"    Layers: {layers}, percentiles: {self.gradients.percentiles}")

        lines.append(f"  Per-class metrics: {'enabled' if self.per_class.enabled else 'disabled'}")
        lines.append(f"  Loss landscape: {'enabled' if self.loss_landscape.enabled else 'disabled'}")

        return "\n".join(lines)


# Re-export for convenience
__all__ = [
    "TelemetryConfig",
    "GradientConfig",
    "LossLandscapeConfig",
    "PerClassConfig",
    "deep_merge",
]
