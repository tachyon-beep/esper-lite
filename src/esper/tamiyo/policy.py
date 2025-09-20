"""Tamiyo policy scaffolding using Leyline contracts.

The stub policy consumes a generated `SystemStatePacket` protobuf message and emits
an `AdaptationCommand` while leaving hooks for future GNN integration as described
in `docs/design/detailed_design/03-tamiyo.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import torch
from torch import Tensor, nn

from esper.leyline import leyline_pb2


@dataclass(slots=True)
class TamiyoPolicyConfig:
    """Configuration for the Tamiyo policy network."""

    input_dim: int = 32
    hidden_dim: int = 64
    output_dim: int = 8


class TamiyoPolicy(nn.Module):
    """Stub neural policy implemented as a feed-forward network."""

    def __init__(self, config: TamiyoPolicyConfig | None = None) -> None:
        super().__init__()
        cfg = config or TamiyoPolicyConfig()
        self._config = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(self, state_vector: Tensor) -> Tensor:
        return self.net(state_vector)

    def select_action(self, packet: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        """Map a system state packet into an adaptation command."""

        features = self._encode_packet(packet)
        _ = self.forward(features)

        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_id=str(uuid4()),
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-1",
            execution_deadline_ms=18,
            issued_by="tamiyo",
        )
        command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        command.seed_operation.parameters["alpha"] = 0.1
        command.issued_at.FromNanoseconds(packet.timestamp_ns or 0)
        return command

    def _encode_packet(self, packet: leyline_pb2.SystemStatePacket) -> Tensor:
        """Encode the packet into a fixed-size tensor."""

        metrics = list(packet.training_metrics.values())
        features = torch.zeros(self._config.input_dim, dtype=torch.float32)
        limit = min(len(metrics), self._config.input_dim)
        if limit:
            features[:limit] = torch.tensor(metrics[:limit], dtype=torch.float32)
        return features

    @staticmethod
    def encode_tags(packet: leyline_pb2.SystemStatePacket) -> dict[str, str]:
        """Produce context tags for telemetry and policy introspection."""

        return {
            "epoch": str(packet.current_epoch),
            "run_id": packet.training_run_id,
            "packet_id": packet.packet_id,
        }


__all__ = ["TamiyoPolicy", "TamiyoPolicyConfig"]
