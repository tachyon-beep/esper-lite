"""Policy validation harness for Simic (TKT-403).

Implements a lightweight validation pipeline that checks Simic training
metrics before a policy update is promoted to Tamiyo. The implementation
captures the intent of `docs/design/detailed_design/old/04-simic-unified-design.md`
while remaining prototype-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class ValidationConfig:
    """Thresholds applied to Simic training metrics."""

    min_average_reward: float = 0.0
    max_policy_loss: float = 1.0
    max_value_loss: float = 1.0
    max_param_loss: float = 1.0
    min_entropy: float = 0.0


@dataclass(slots=True)
class ValidationResult:
    """Outcome of the validation suite."""

    passed: bool
    reasons: list[str]
    metrics: dict[str, float]

    def as_dict(self) -> dict[str, float | bool | Iterable[str]]:
        """Return a serialisable view suitable for telemetry."""

        payload: dict[str, float | bool | Iterable[str]] = {
            "passed": self.passed,
            "reasons": list(self.reasons),
        }
        payload.update(self.metrics)
        return payload


class PolicyValidator:
    """Evaluate Simic training metrics against configured thresholds."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self._config = config or ValidationConfig()

    def validate(self, metrics: dict[str, float]) -> ValidationResult:
        """Validate metrics produced by the Simic trainer."""

        reasons: list[str] = []
        cfg = self._config

        reward = metrics.get("average_reward", 0.0)
        if reward < cfg.min_average_reward:
            reasons.append(f"average_reward {reward:.4f} below min {cfg.min_average_reward:.4f}")

        policy_loss = metrics.get("policy_loss", 0.0)
        if policy_loss > cfg.max_policy_loss:
            reasons.append(f"policy_loss {policy_loss:.4f} above max {cfg.max_policy_loss:.4f}")

        value_loss = metrics.get("value_loss", 0.0)
        if value_loss > cfg.max_value_loss:
            reasons.append(f"value_loss {value_loss:.4f} above max {cfg.max_value_loss:.4f}")

        param_loss = metrics.get("param_loss", 0.0)
        if param_loss > cfg.max_param_loss:
            reasons.append(f"param_loss {param_loss:.4f} above max {cfg.max_param_loss:.4f}")

        entropy = metrics.get("policy_entropy", 0.0)
        if entropy < cfg.min_entropy:
            reasons.append(f"policy_entropy {entropy:.4f} below min {cfg.min_entropy:.4f}")

        passed = not reasons
        return ValidationResult(passed=passed, reasons=reasons, metrics=metrics)


__all__ = ["ValidationConfig", "ValidationResult", "PolicyValidator"]
