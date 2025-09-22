"""Tamiyo heterogenous GNN implementation (GraphSAGE → GAT stack).

This module codifies the architecture described in
`docs/prototype-delta/tamiyo/GNN-WP1.md`. The implementation intentionally keeps
the layer shapes modest so the prototype remains lightweight while still
providing the structural hooks required by the detailed design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:  # pragma: no cover - optional dependency
    from torch_geometric.nn import GATConv, HeteroConv, SAGEConv
except ImportError as exc:  # pragma: no cover - integration guard
    raise ModuleNotFoundError(
        "PyTorch Geometric is required for Tamiyo GNN support. Install esper-lite[tamiyo-gnn]."
    ) from exc


@dataclass(slots=True)
class TamiyoGNNConfig:
    """Model hyperparameters."""

    global_input_dim: int = 16
    seed_input_dim: int = 12
    blueprint_input_dim: int = 8
    layer_input_dim: int = 8
    activation_input_dim: int = 6
    parameter_input_dim: int = 6
    hidden_dim: int = 96
    dropout: float = 0.2
    attention_heads: int = 4
    policy_classes: int = 5
    blending_classes: int = 3
    schedule_output_dim: int = 2
    edge_feature_dim: int = 3
    risk_classes: int = 5


class TamiyoGNN(nn.Module):
    """GraphSAGE → GAT heterogenous network powering Tamiyo decisions."""

    def __init__(self, config: TamiyoGNNConfig) -> None:
        super().__init__()
        self._cfg = config

        self._node_encoders = nn.ModuleDict(
            {
                "global": nn.Sequential(
                    nn.Linear(config.global_input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(config.hidden_dim),
                ),
                "seed": nn.Sequential(
                    nn.Linear(config.seed_input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(config.hidden_dim),
                ),
                "blueprint": nn.Sequential(
                    nn.Linear(config.blueprint_input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(config.hidden_dim),
                ),
                "layer": nn.Sequential(
                    nn.Linear(config.layer_input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(config.hidden_dim),
                ),
                "activation": nn.Sequential(
                    nn.Linear(config.activation_input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(config.hidden_dim),
                ),
                "parameter": nn.Sequential(
                    nn.Linear(config.parameter_input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(config.hidden_dim),
                ),
            }
        )

        relations = [
            ("global", "influences", "seed"),
            ("seed", "reports", "global"),
            ("global", "annotates", "blueprint"),
            ("blueprint", "monitored_by", "global"),
            ("seed", "peer", "seed"),
            ("global", "operates", "layer"),
            ("layer", "feedback", "global"),
            ("layer", "activates", "activation"),
            ("activation", "affects", "layer"),
            ("activation", "configures", "parameter"),
            ("parameter", "modulates", "activation"),
            ("seed", "allowed", "parameter"),
            ("parameter", "associated", "seed"),
            ("blueprint", "composes", "layer"),
            ("layer", "belongs_to", "blueprint"),
            ("blueprint", "energizes", "activation"),
            ("parameter", "targets", "blueprint"),
        ]

        self._sage_layers = nn.ModuleList(
            [
                HeteroConv(
                    {
                        relation: SAGEConv((-1, -1), config.hidden_dim)
                        for relation in relations
                    },
                    aggr="mean",
                )
            ]
        )

        self._gat_layers = nn.ModuleList(
            [
                HeteroConv(
                    {
                        relation: GATConv(
                            (-1, -1),
                            config.hidden_dim,
                            heads=config.attention_heads,
                            concat=False,
                            add_self_loops=False,
                            edge_dim=config.edge_feature_dim,
                        )
                        for relation in relations
                    },
                    aggr="mean",
                )
            ]
        )

        self._policy_head = nn.Linear(config.hidden_dim, config.policy_classes)
        with torch.no_grad():
            nn.init.constant_(self._policy_head.weight, 0.0)
            nn.init.constant_(self._policy_head.bias, -0.5)
            self._policy_head.bias[0] = 0.5
        self._param_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self._blending_head = nn.Linear(config.hidden_dim, config.blending_classes)
        self._schedule_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.schedule_output_dim),
        )
        self._seed_score_head = nn.Linear(config.hidden_dim, 1)
        self._blueprint_score_head = nn.Linear(config.hidden_dim, 1)
        self._breaker_head = nn.Linear(config.hidden_dim, 2)
        self._risk_head = nn.Linear(config.hidden_dim, config.risk_classes)
        self._value_head = nn.Linear(config.hidden_dim, 1)

        self._dropout = config.dropout

    def forward(self, data) -> dict[str, Tensor]:  # type: ignore[override]
        """Run the heterogenous graph through the policy network."""

        x_dict: Dict[str, Tensor] = data.x_dict
        edge_index_dict: Dict[tuple[str, str, str], Tensor] = data.edge_index_dict
        edge_attr_dict: Dict[tuple[str, str, str], Tensor] = {}
        for relation in edge_index_dict:
            edge_store = data[relation]
            edge_attr = getattr(edge_store, "edge_attr", None)
            if edge_attr is not None:
                edge_attr_dict[relation] = edge_attr.to(next(iter(x_dict.values())).device)

        encoded = {key: self._node_encoders[key](x) for key, x in x_dict.items() if key in self._node_encoders}
        for layer in self._sage_layers:
            encoded = layer(encoded, edge_index_dict)
            encoded = {key: F.gelu(tensor) for key, tensor in encoded.items()}
            encoded = {key: F.dropout(tensor, p=self._dropout, training=self.training) for key, tensor in encoded.items()}

        for layer in self._gat_layers:
            encoded = layer(encoded, edge_index_dict, edge_attr_dict=edge_attr_dict)
            encoded = {key: F.gelu(tensor) for key, tensor in encoded.items()}
            encoded = {key: F.dropout(tensor, p=self._dropout, training=self.training) for key, tensor in encoded.items()}

        global_embedding = encoded.get("global")
        if global_embedding is None:
            raise RuntimeError("TamiyoGNN expected a 'global' node representation")

        graph_embedding = global_embedding.mean(dim=0, keepdim=True)

        outputs = {
            "policy_logits": self._policy_head(graph_embedding),
            "param_delta": torch.tanh(self._param_head(graph_embedding)),
            "blending_logits": self._blending_head(graph_embedding),
            "risk_logits": self._risk_head(graph_embedding),
            "value": self._value_head(graph_embedding),
            "schedule_params": torch.tanh(self._schedule_head(graph_embedding)),
            "graph_embedding": graph_embedding,
        }
        if "seed" in encoded:
            outputs["seed_scores"] = self._seed_score_head(encoded["seed"]).squeeze(-1)
        if "blueprint" in encoded:
            outputs["blueprint_scores"] = self._blueprint_score_head(encoded["blueprint"]).squeeze(-1)
        outputs["breaker_logits"] = self._breaker_head(graph_embedding)
        return outputs


__all__ = ["TamiyoGNN", "TamiyoGNNConfig"]
