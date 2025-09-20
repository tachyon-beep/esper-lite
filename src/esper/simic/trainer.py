"""Simic offline trainer with PPO-style updates and optional LoRA adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING

import torch
from torch import nn, optim
from torch.nn import functional as F

from esper.leyline import leyline_pb2

from esper.core.telemetry import TelemetryMetric, build_telemetry_packet
from esper.core import TelemetryEvent

from .replay import FieldReportReplayBuffer, SimicExperience

if TYPE_CHECKING:
    from esper.oona import OonaClient


@dataclass(slots=True)
class SimicTrainerConfig:
    """Configuration for the Simic training loop."""

    learning_rate: float = 5e-4
    epochs: int = 5
    batch_size: int = 64
    hidden_size: int = 64
    dropout: float = 0.1
    gamma: float = 0.99
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: float = 8.0


class SimicTrainer:
    """Offline trainer coordinating Tamiyo policy updates."""

    def __init__(
        self,
        policy: nn.Module | None,
        buffer: FieldReportReplayBuffer,
        config: SimicTrainerConfig | None = None,
    ) -> None:
        self._buffer = buffer
        self._config = config or SimicTrainerConfig()
        self._policy = policy or _build_policy_network(self._config)
        self._optimizer = optim.Adam(self._policy.parameters(), lr=self._config.learning_rate)
        self._policy_updates: list[leyline_pb2.PolicyUpdate] = []
        self._device = next(self._policy.parameters()).device
        self._policy.to(self._device)
        self._last_loss: float = 0.0
        self._last_reward: float = 0.0
        self._last_value_loss: float = 0.0
        self._iterations: int = 0

    def run_training(self) -> None:
        """Execute PPO-like training on buffered field reports."""

        self._policy.train()
        for _ in range(self._config.epochs):
            batch = self._buffer.sample_batch(self._config.batch_size)
            if batch["reward"].numel() == 0:
                continue
            loss = self._compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self._config.grad_clip)
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)
            self._last_loss = float(loss.detach().cpu())
            self._last_reward = float(batch["reward"].mean().detach().cpu())
            self._iterations += 1

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        rewards = batch["reward"].to(self._device)
        loss_deltas = batch["loss_delta"].to(self._device)
        outcome_success = batch["outcome_success"].to(self._device)

        inputs = self._prepare_inputs(loss_deltas, outcome_success)
        outputs = self._policy(inputs)
        if isinstance(outputs, tuple):
            logits, values = outputs
        else:
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(-1)
            logits = outputs[..., 0]
            value_col = outputs[..., 1] if outputs.shape[-1] > 1 else logits
            values = value_col.squeeze(-1)
        logits = logits.squeeze(-1)
        values = values.squeeze(-1)
        dist = torch.distributions.Bernoulli(logits=torch.clamp(logits, -20, 20))
        actions = outcome_success
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        advantages = rewards - values.detach()
        policy_loss = -(advantages * log_probs).mean()
        value_loss = F.mse_loss(values, rewards)

        loss = policy_loss + self._config.value_coef * value_loss - self._config.entropy_coef * entropy
        self._last_value_loss = float(value_loss.detach().cpu())
        return loss

    def _prepare_inputs(self, loss_deltas: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
        input_dim = self._infer_input_dim()
        inputs = torch.zeros((loss_deltas.shape[0], input_dim), device=self._device)
        inputs[:, 0] = loss_deltas
        if input_dim > 1:
            inputs[:, 1] = outcomes
        return inputs

    def _infer_input_dim(self) -> int:
        for module in self._policy.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        raise RuntimeError("Unable to infer policy input dimension")

    def create_policy_update(
        self,
        *,
        policy_id: str,
        training_run_id: str,
        policy_version: str,
        policy_state: dict | None = None,
    ) -> leyline_pb2.PolicyUpdate:
        """Create a policy update protobuf for downstream consumption."""

        update = leyline_pb2.PolicyUpdate(
            version=1,
            policy_id=policy_id,
            training_run_id=training_run_id,
            tamiyo_policy_version=policy_version,
        )
        update.issued_at.FromDatetime(datetime.now(tz=UTC))
        state_dict = policy_state or self._policy.state_dict()
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        update.payload = buffer.getvalue()
        self._policy_updates.append(update)
        return update

    async def publish_policy_updates(self, oona: OonaClient) -> None:
        """Publish any generated policy updates via Oona."""

        for update in self._policy_updates:
            await oona.publish_policy_update(update)

    @property
    def policy_updates(self) -> list[leyline_pb2.PolicyUpdate]:
        """Expose buffered policy updates."""

        return list(self._policy_updates)

    @property
    def last_loss(self) -> float:
        """Return the last observed training loss."""

        return self._last_loss

    def build_metrics_packet(self, *, training_run_id: str) -> leyline_pb2.TelemetryPacket:
        metrics = [
            TelemetryMetric("simic.training.loss", self._last_loss, unit="loss"),
            TelemetryMetric("simic.training.reward", self._last_reward, unit="reward"),
            TelemetryMetric("simic.value.loss", self._last_value_loss, unit="loss"),
            TelemetryMetric("simic.training.iterations", float(self._iterations), unit="count"),
        ]
        packet = build_telemetry_packet(
            packet_id=f"simic-metrics-{training_run_id}",
            source="simic",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=metrics,
            events=[
                TelemetryEvent(
                    description="Simic PPO update",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                    attributes={"training_run_id": training_run_id},
                )
            ],
        )
        return packet

    async def publish_metrics(self, oona: OonaClient, *, training_run_id: str) -> None:
        packet = self.build_metrics_packet(training_run_id=training_run_id)
        await oona.publish_telemetry(packet)


__all__ = ["SimicTrainer", "SimicTrainerConfig"]


def _build_policy_network(config: SimicTrainerConfig) -> nn.Module:
    return _PolicyNetwork(
        input_dim=2,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
    )


class _LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        enable_lora: bool = False,
        rank: int = 4,
        alpha: float = 8.0,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        if enable_lora and rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.scaling = alpha / rank
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.scaling = 0.0

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        result = super().forward(input)
        if self.lora_A is not None and self.lora_B is not None:
            lora_update = (input @ self.lora_A.t()) @ self.lora_B.t()
            result = result + self.scaling * lora_update
        return result


class _PolicyNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        dropout: float,
        use_lora: bool,
        lora_rank: int,
        lora_alpha: float,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _LoRALinear(input_dim, hidden_size, enable_lora=use_lora, rank=lora_rank, alpha=lora_alpha),
            nn.Tanh(),
            nn.Dropout(dropout),
            _LoRALinear(hidden_size, hidden_size, enable_lora=use_lora, rank=lora_rank, alpha=lora_alpha),
            nn.Tanh(),
        )
        self.policy_head = _LoRALinear(hidden_size, 1, enable_lora=use_lora, rank=lora_rank, alpha=lora_alpha)
        self.value_head = _LoRALinear(hidden_size, 1, enable_lora=use_lora, rank=lora_rank, alpha=lora_alpha)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(inputs)
        logits = self.policy_head(features).squeeze(-1)
        value = self.value_head(features).squeeze(-1)
        return logits, value
