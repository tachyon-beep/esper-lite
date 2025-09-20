"""Tamiyo policy scaffolding using Leyline contracts.

The stub policy consumes a generated `SystemStatePacket` protobuf message and emits
an `AdaptationCommand` while leaving hooks for future GNN integration as described
in `docs/design/detailed_design/03-tamiyo.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4
from pathlib import Path

import torch
from torch import Tensor, nn

from esper.leyline import leyline_pb2
from esper.simic.replay import MAX_METRIC_FEATURES
from esper.simic.registry import EmbeddingRegistry, EmbeddingRegistryConfig


@dataclass(slots=True)
class TamiyoPolicyConfig:
    """Configuration for the Tamiyo policy network."""

    feature_dim: int = 32
    hidden_dim: int = 64
    action_classes: int = 3
    dropout: float = 0.1
    embedding_dim: int = 16
    seed_registry: EmbeddingRegistry | None = None
    blueprint_registry: EmbeddingRegistry | None = None
    registry_path: str = "var/tamiyo"
    seed_vocab: int = 1024
    blueprint_vocab: int = 1024


class TamiyoPolicy(nn.Module):
    """Stub neural policy implemented as a feed-forward network."""

    def __init__(self, config: TamiyoPolicyConfig | None = None) -> None:
        super().__init__()
        cfg = config or TamiyoPolicyConfig()
        self._config = cfg
        Path(cfg.registry_path).mkdir(parents=True, exist_ok=True)
        self._seed_registry = cfg.seed_registry or EmbeddingRegistry(
            EmbeddingRegistryConfig(Path(cfg.registry_path) / "seed_registry.json", cfg.seed_vocab)
        )
        self._blueprint_registry = cfg.blueprint_registry or EmbeddingRegistry(
            EmbeddingRegistryConfig(Path(cfg.registry_path) / "blueprint_registry.json", cfg.blueprint_vocab)
        )
        self._numeric_dim = cfg.feature_dim
        self._seed_embedding = nn.Embedding(cfg.seed_vocab, cfg.embedding_dim)
        self._blueprint_embedding = nn.Embedding(cfg.blueprint_vocab, cfg.embedding_dim)
        input_dim = cfg.feature_dim + cfg.embedding_dim * 2
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
        )
        self._action_head = nn.Linear(cfg.hidden_dim, cfg.action_classes)
        self._param_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(
        self,
        numeric_features: Tensor,
        seed_indices: Tensor,
        blueprint_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pieces = [numeric_features]
        pieces.append(self._seed_embedding(seed_indices.to(torch.long)).squeeze(1))
        pieces.append(self._blueprint_embedding(blueprint_indices.to(torch.long)).squeeze(1))
        fused = torch.cat(pieces, dim=1)
        encoded = self._encoder(fused)
        logits = self._action_head(encoded)
        param = self._param_head(encoded)
        return logits, param

    def select_action(self, packet: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        """Map a system state packet into an adaptation command."""

        numeric, seed_idx, blueprint_idx = self._encode_packet(packet)
        logits, param = self.forward(numeric, seed_idx, blueprint_idx)
        action = torch.argmax(logits, dim=-1).item()
        param_delta = float(param.squeeze().detach())

        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id=str(uuid4()),
            issued_by="tamiyo",
        )
        command.issued_at.FromNanoseconds(packet.timestamp_ns or 0)

        if action == 0:  # graft
            command.command_type = leyline_pb2.COMMAND_SEED
            command.target_seed_id = "seed-1"
            command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
            command.seed_operation.blueprint_id = "bp-demo"
            command.seed_operation.parameters["alpha"] = 0.1 + param_delta
        elif action == 1:  # adjust optimizer
            command.command_type = leyline_pb2.COMMAND_OPTIMIZER
            command.optimizer_adjustment.optimizer_id = "sgd"
            command.optimizer_adjustment.hyperparameters["lr_delta"] = param_delta
        else:  # pause / conservative mode
            command.command_type = leyline_pb2.COMMAND_PAUSE
            command.annotations["reason"] = "simic-policy"
            command.annotations["param_delta"] = str(param_delta)

        return command

    def _encode_packet(
        self, packet: leyline_pb2.SystemStatePacket
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode the packet into numeric features and categorical indices."""

        numeric = torch.zeros(1, self._config.feature_dim, dtype=torch.float32)
        metrics = list(packet.training_metrics.values())
        limit = min(len(metrics), MAX_METRIC_FEATURES)
        if limit:
            numeric[0, :limit] = torch.tensor(metrics[:limit], dtype=torch.float32)
        if limit + 1 < numeric.shape[1]:
            numeric[0, limit] = float(packet.current_epoch)
        seed_idx = torch.tensor([[self._seed_registry.get(packet.training_run_id)]], dtype=torch.long)
        blueprint_idx = torch.tensor([[self._blueprint_registry.get(packet.training_run_id)]], dtype=torch.long)
        return numeric, seed_idx, blueprint_idx

    @staticmethod
    def encode_tags(packet: leyline_pb2.SystemStatePacket) -> dict[str, str]:
        """Produce context tags for telemetry and policy introspection."""

        return {
            "epoch": str(packet.current_epoch),
            "run_id": packet.training_run_id,
            "packet_id": packet.packet_id,
        }


__all__ = ["TamiyoPolicy", "TamiyoPolicyConfig"]
