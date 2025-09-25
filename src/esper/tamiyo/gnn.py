"""Tamiyo heterogenous GNN implementation (GraphSAGE → GAT stack).

This module codifies the architecture described in
`docs/prototype-delta/tamiyo/GNN-WP1.md`. The implementation follows the
four-layer progression (two GraphSAGE layers followed by two GAT layers),
emits structured heads for policy, risk, and telemetry outputs, and exposes the
residual hooks required by the detailed design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv


@dataclass(slots=True)
class TamiyoGNNConfig:
    """Model hyperparameters."""

    global_input_dim: int = 16
    seed_input_dim: int = 14
    blueprint_input_dim: int = 14
    layer_input_dim: int = 12
    activation_input_dim: int = 8
    parameter_input_dim: int = 10
    sage_hidden_dim: int = 256
    gat_hidden_dim: int = 128
    dropout: float = 0.2
    attention_heads: int = 4
    policy_classes: int = 32
    blending_classes: int = 3
    schedule_output_dim: int = 2
    edge_feature_dim: int = 3
    risk_classes: int = 5
    param_vector_dim: int = 4


class TamiyoGNN(nn.Module):
    """GraphSAGE → GAT heterogenous network powering Tamiyo decisions."""

    def __init__(self, config: TamiyoGNNConfig) -> None:
        super().__init__()
        self._cfg = config

        encoder_dim = config.sage_hidden_dim
        self._node_encoders = nn.ModuleDict(
            {
                key: self._build_encoder(input_dim, encoder_dim)
                for key, input_dim in {
                    "global": config.global_input_dim,
                    "seed": config.seed_input_dim,
                    "blueprint": config.blueprint_input_dim,
                    "layer": config.layer_input_dim,
                    "activation": config.activation_input_dim,
                    "parameter": config.parameter_input_dim,
                }.items()
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
            ("layer", "connects", "layer"),
            ("seed", "monitors", "layer"),
            ("layer", "monitored_by", "seed"),
            ("layer", "feeds", "activation"),
        ]

        self._relations: tuple[tuple[str, str, str], ...] = tuple(relations)
        self._sage_layers = nn.ModuleList(
            [
                self._build_sage_layer(config.sage_hidden_dim, relations),
                self._build_sage_layer(config.sage_hidden_dim, relations),
            ]
        )

        self._gat_layers = nn.ModuleList(
            [
                self._build_gat_layer(
                    config.gat_hidden_dim,
                    relations,
                    config.attention_heads,
                    config.edge_feature_dim,
                ),
                self._build_gat_layer(
                    config.gat_hidden_dim,
                    relations,
                    config.attention_heads,
                    config.edge_feature_dim,
                ),
            ]
        )

        head_input_dim = config.gat_hidden_dim
        self._policy_head = nn.Linear(head_input_dim, config.policy_classes)
        with torch.no_grad():
            nn.init.constant_(self._policy_head.weight, 0.0)
            nn.init.constant_(self._policy_head.bias, -0.5)
            self._policy_head.bias[0] = 0.5
        self._param_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.GELU(),
            nn.Linear(head_input_dim // 2, config.param_vector_dim),
        )
        self._blending_head = nn.Linear(head_input_dim, config.blending_classes)
        self._schedule_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.GELU(),
            nn.Linear(head_input_dim // 2, config.schedule_output_dim),
        )
        self._seed_score_head = nn.Linear(head_input_dim, 1)
        self._blueprint_score_head = nn.Linear(head_input_dim, 1)
        self._breaker_head = nn.Linear(head_input_dim, 2)
        self._risk_head = nn.Linear(head_input_dim, config.risk_classes)
        self._value_head = nn.Linear(head_input_dim, 1)

        self._dropout = config.dropout
        self._gat_hidden_dim = config.gat_hidden_dim

    def forward(self, data) -> dict[str, Tensor]:  # type: ignore[override]
        """Run the heterogenous graph through the policy network."""

        x_dict: Dict[str, Tensor] = data.x_dict
        edge_index_dict: Dict[tuple[str, str, str], Tensor] = data.edge_index_dict
        edge_attr_dict: Dict[tuple[str, str, str], Tensor] = {}
        for relation in self._relations:
            if relation not in edge_index_dict:
                continue
            edge_store = data[relation]
            edge_attr = getattr(edge_store, "edge_attr", None)
            if edge_attr is not None:
                edge_attr_dict[relation] = edge_attr.to(next(iter(x_dict.values())).device)

        encoded = {
            key: self._node_encoders[key](x)
            for key, x in x_dict.items()
            if key in self._node_encoders
        }
        for layer in self._sage_layers:
            residual = {key: tensor for key, tensor in encoded.items()}
            encoded = layer(encoded, edge_index_dict)
            encoded = self._apply_activation(encoded, residual)

        for layer in self._gat_layers:
            residual = {key: tensor for key, tensor in encoded.items()}
            encoded = layer(encoded, edge_index_dict, edge_attr_dict=edge_attr_dict)
            encoded = self._apply_activation(encoded, residual)

        global_embedding = encoded.get("global")
        if global_embedding is None:
            raise RuntimeError("TamiyoGNN expected a 'global' node representation")

        graph_embedding = global_embedding.mean(dim=0, keepdim=True)

        outputs = {
            "policy_logits": self._policy_head(graph_embedding),
            "policy_params": torch.tanh(self._param_head(graph_embedding)),
            "blending_logits": self._blending_head(graph_embedding),
            "risk_logits": self._risk_head(graph_embedding),
            "value": self._value_head(graph_embedding),
            "schedule_params": torch.tanh(self._schedule_head(graph_embedding)),
            "graph_embedding": graph_embedding,
        }
        outputs["param_delta"] = outputs["policy_params"][..., 0:1]
        if "seed" in encoded:
            outputs["seed_scores"] = self._seed_score_head(encoded["seed"]).squeeze(-1)
        if "blueprint" in encoded:
            outputs["blueprint_scores"] = self._blueprint_score_head(encoded["blueprint"]).squeeze(
                -1
            )
        outputs["breaker_logits"] = self._breaker_head(graph_embedding)
        return outputs

    def _apply_activation(
        self,
        encoded: Mapping[str, Tensor],
        residual: Mapping[str, Tensor],
    ) -> Dict[str, Tensor]:
        activated: Dict[str, Tensor] = {}
        for key, tensor in encoded.items():
            out = torch.nn.functional.gelu(tensor)  # pylint: disable=not-callable
            skip = residual.get(key)
            if skip is not None and skip.shape == out.shape:
                out = out + skip
            activated[key] = F.dropout(out, p=self._dropout, training=self.training)
        return activated

    @staticmethod
    def _build_encoder(input_dim: int, output_dim: int) -> nn.Sequential:
        layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )
        for module in layer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        return layer

    @staticmethod
    def _build_sage_layer(
        hidden_dim: int,
        relations: Iterable[tuple[str, str, str]],
    ) -> HeteroConv:
        convs = {relation: SAGEConv((-1, -1), hidden_dim) for relation in relations}
        return HeteroConv(convs, aggr="mean")

    @staticmethod
    def _build_gat_layer(
        hidden_dim: int,
        relations: Iterable[tuple[str, str, str]],
        heads: int,
        edge_dim: int,
    ) -> HeteroConv:
        convs = {
            relation: GATConv(
                (-1, -1),
                hidden_dim,
                heads=heads,
                concat=False,
                add_self_loops=False,
                edge_dim=edge_dim,
            )
            for relation in relations
        }
        return HeteroConv(convs, aggr="mean")


__all__ = ["TamiyoGNN", "TamiyoGNNConfig"]
