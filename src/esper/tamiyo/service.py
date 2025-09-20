"""Tamiyo service wrapper combining policy inference and risk gating."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from esper.leyline import leyline_pb2

from .policy import TamiyoPolicy, TamiyoPolicyConfig


@dataclass(slots=True)
class RiskConfig:
    """Configuration for Tamiyo risk thresholds."""

    max_loss_spike: float = 0.15
    conservative_mode: bool = False


class TamiyoService:
    """High-level Tamiyo orchestration component."""

    def __init__(
        self,
        policy: TamiyoPolicy | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self._policy = policy or TamiyoPolicy(TamiyoPolicyConfig())
        self._risk = risk_config or RiskConfig()

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        """Evaluate epoch state and apply risk gating."""

        command = self._policy.select_action(state)
        if self._risk.conservative_mode:
            command.command_type = leyline_pb2.COMMAND_PAUSE
            command.annotations["conservative_mode"] = "true"
        return command

    def generate_field_report(
        self,
        command: leyline_pb2.AdaptationCommand,
        outcome: leyline_pb2.FieldReportOutcome,
        metrics_delta: dict[str, float],
        *,
        training_run_id: str,
        seed_id: str,
        blueprint_id: str,
        notes: str | None = None,
    ) -> leyline_pb2.FieldReport:
        """Produce a field report entry for downstream ingestion (Simic)."""

        report = leyline_pb2.FieldReport(
            version=1,
            report_id=str(uuid4()),
            command_id=command.command_id,
            training_run_id=training_run_id,
            seed_id=seed_id,
            blueprint_id=blueprint_id,
            outcome=outcome,
            observation_window_epochs=1,
            tamiyo_policy_version="policy-stub",
            notes=notes or "",
        )
        for key, value in metrics_delta.items():
            report.metrics[key] = value
        report.issued_at.CopyFrom(command.issued_at)
        return report

    def update_policy(self, new_policy: TamiyoPolicy) -> None:
        """Hot-swap the in-memory policy."""

        self._policy = new_policy

    def set_conservative_mode(self, enabled: bool) -> None:
        """Toggle conservative mode (breaker support)."""

        self._risk.conservative_mode = enabled


__all__ = ["TamiyoService", "RiskConfig"]
