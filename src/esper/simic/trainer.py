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
    feature_dim: int = 32
    gae_lambda: float = 0.95
    action_classes: int = 3
    param_coef: float = 0.1
    embedding_dim: int = 16
    seed_vocab: int = 1024
    blueprint_vocab: int = 1024
    metric_window: int = 16
    use_metric_attention: bool = True
    metric_attention_heads: int = 2


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
        self._buffer.feature_dim = self._config.feature_dim
        self._buffer.metric_window = self._config.metric_window
        self._buffer.seed_vocab = self._config.seed_vocab
        self._buffer.blueprint_vocab = self._config.blueprint_vocab
        self._policy = policy or _build_policy_network(self._config)
        self._optimizer = optim.Adam(self._policy.parameters(), lr=self._config.learning_rate)
        self._policy_updates: list[leyline_pb2.PolicyUpdate] = []
        self._device = next(self._policy.parameters()).device
        self._policy.to(self._device)
        self._last_loss: float = 0.0
        self._last_reward: float = 0.0
        self._last_value_loss: float = 0.0
        self._iterations: int = 0
        self._last_policy_loss: float = 0.0
        self._last_param_loss: float = 0.0
        self._last_entropy: float = 0.0

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
        features = batch["features"].to(self._device)
        outcome_success = batch["outcome_success"].to(self._device)
        metric_sequence = batch["metric_sequence"].to(self._device)
        seed_index = batch["seed_index"].to(self._device)
        blueprint_index = batch["blueprint_index"].to(self._device)
        discounted_returns = _compute_discounted_returns(rewards, self._config.gamma)

        outputs = self._policy(features, metric_sequence, seed_index, blueprint_index)
        if isinstance(outputs, tuple):
            action_logits, param_pred, values = outputs
        else:
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            action_logits = outputs[..., : self._config.action_classes]
            remaining = outputs[..., self._config.action_classes :]
            if remaining.shape[-1] >= 2:
                param_pred = remaining[..., :1]
                values = remaining[..., 1:2]
            else:
                param_pred = remaining[..., :1]
                values = remaining[..., :1]
        values = values.squeeze(-1)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        actions = _derive_actions(outcome_success, rewards)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()

        advantages = discounted_returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        policy_loss = -(advantages * log_probs).mean()
        value_loss = F.mse_loss(values, discounted_returns)
        param_loss = F.mse_loss(param_pred.squeeze(-1), batch["loss_delta"].to(self._device))

        loss = (
            policy_loss
            + self._config.value_coef * value_loss
            + self._config.param_coef * param_loss
            - self._config.entropy_coef * entropy
        )
        self._last_value_loss = float(value_loss.detach().cpu())
        self._last_policy_loss = float(policy_loss.detach().cpu())
        self._last_param_loss = float(param_loss.detach().cpu())
        self._last_entropy = float(entropy.detach().cpu())
        return loss

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
            TelemetryMetric("simic.policy.loss", self._last_policy_loss, unit="loss"),
            TelemetryMetric("simic.param.loss", self._last_param_loss, unit="loss"),
            TelemetryMetric("simic.policy.entropy", self._last_entropy, unit="nats"),
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
        feature_dim=config.feature_dim,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        action_classes=config.action_classes,
        embedding_dim=config.embedding_dim,
        seed_vocab=config.seed_vocab,
        blueprint_vocab=config.blueprint_vocab,
        metric_window=config.metric_window,
        use_metric_attention=config.use_metric_attention,
        metric_attention_heads=config.metric_attention_heads,
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
        feature_dim: int,
        hidden_size: int,
        dropout: float,
        use_lora: bool,
        lora_rank: int,
        lora_alpha: float,
        action_classes: int,
        embedding_dim: int,
        seed_vocab: int,
        blueprint_vocab: int,
        metric_window: int,
        use_metric_attention: bool,
        metric_attention_heads: int,
    ) -> None:
        super().__init__()
        self.use_metric_attention = use_metric_attention
        self.metric_window = metric_window
        self.embedding_dim = embedding_dim
        self.seed_embedding = nn.Embedding(seed_vocab, embedding_dim) if seed_vocab > 0 else None
        self.blueprint_embedding = nn.Embedding(blueprint_vocab, embedding_dim) if blueprint_vocab > 0 else None
        if use_metric_attention:
            self.metric_encoder = nn.Linear(1, embedding_dim)
            self.metric_attention = nn.MultiheadAttention(
                embedding_dim,
                metric_attention_heads,
                batch_first=True,
            )
            metric_context_dim = embedding_dim
        else:
            self.metric_encoder = nn.Linear(metric_window, embedding_dim)
            self.metric_attention = None
            metric_context_dim = embedding_dim

        total_input_dim = feature_dim
        if self.seed_embedding is not None:
            total_input_dim += embedding_dim
        if self.blueprint_embedding is not None:
            total_input_dim += embedding_dim
        total_input_dim += metric_context_dim

        self.backbone = nn.Sequential(
            _LoRALinear(total_input_dim, hidden_size, enable_lora=use_lora, rank=lora_rank, alpha=lora_alpha),
            nn.Tanh(),
            nn.Dropout(dropout),
            _LoRALinear(hidden_size, hidden_size, enable_lora=use_lora, rank=lora_rank, alpha=lora_alpha),
            nn.Tanh(),
        )
        self.policy_head = _LoRALinear(hidden_size, action_classes, enable_lora=use_lora, rank=lora_rank, alpha=lora_alpha)
        self.param_head = _LoRALinear(hidden_size, 1, enable_lora=use_lora, rank=lora_rank, alpha=lora_alpha)
        self.value_head = _LoRALinear(hidden_size, 1, enable_lora=use_lora, rank=lora_rank, alpha=lora_alpha)

    def forward(
        self,
        numeric_features: torch.Tensor,
        metric_sequence: torch.Tensor,
        seed_index: torch.Tensor,
        blueprint_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pieces = [numeric_features]

        if self.seed_embedding is not None:
            pieces.append(self.seed_embedding(seed_index))
        if self.blueprint_embedding is not None:
            pieces.append(self.blueprint_embedding(blueprint_index))

        if self.use_metric_attention:
            seq = metric_sequence.unsqueeze(-1)
            seq_embed = self.metric_encoder(seq)
            context, _ = self.metric_attention(seq_embed, seq_embed, seq_embed)
            metric_context = context.mean(dim=1)
        else:
            metric_context = self.metric_encoder(metric_sequence)

        pieces.append(metric_context)
        fused = torch.cat(pieces, dim=1)
        features = self.backbone(fused)
        action_logits = self.policy_head(features)
        param = self.param_head(features)
        value = self.value_head(features)
        return action_logits, param, value


def _compute_discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = 0.0
    for idx in reversed(range(rewards.shape[0])):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def _derive_actions(outcome_success: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    actions = torch.zeros_like(outcome_success, dtype=torch.long)
    actions[outcome_success < 0.5] = 2  # pause/emergency
    actions[(outcome_success >= 0.5) & (rewards < 0)] = 1  # adjust blueprint/optimizer
    return actions
