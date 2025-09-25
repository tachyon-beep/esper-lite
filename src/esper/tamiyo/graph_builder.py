"""Utilities to translate step-level state into heterogenous graphs for Tamiyo.

This implements the graph construction helpers required by
`docs/prototype-delta/tamiyo/GNN-WP1.md`. The builder keeps the payload lean and
falls back to safe defaults when certain metrics are unavailable so the policy
never crashes mid-run.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from esper.leyline import leyline_pb2
from esper.simic.registry import EmbeddingRegistry

_DEFAULT_NORMALISATION: Mapping[str, tuple[float, float]] = {
    "loss": (0.8, 0.3),
    "validation_loss": (0.8, 0.3),
    "training_loss": (0.8, 0.3),
    "gradient_norm": (1.0, 0.5),
    "samples_per_s": (4500.0, 500.0),
    "hook_latency_ms": (12.0, 6.0),
    "seed_learning_rate": (0.01, 0.005),
    "seed_risk": (0.4, 0.2),
    "seed_age": (10.0, 4.0),
    "epoch_progress": (0.5, 0.25),
    "layer_latency_ms": (6.0, 2.0),
    "layer_parameter_count": (1024.0, 512.0),
    "layer_weight_norm": (1.0, 0.4),
    "layer_gradient_norm": (0.8, 0.4),
    "activation_saturation": (0.5, 0.2),
    "activation_gradient_flow": (0.8, 0.2),
    "activation_cost": (128.0, 64.0),
    "activation_nonlinearity": (0.5, 0.2),
    "parameter_min": (0.0, 0.5),
    "parameter_max": (1.0, 0.5),
    "parameter_span": (0.5, 0.4),
    "parameter_default": (0.5, 0.4),
    "blueprint_param_log": (math.log1p(2048.0), 1.0),
    "blueprint_candidate_score": (0.5, 0.25),
    "global_epoch": (10.0, 5.0),
    "global_step": (100.0, 40.0),
}


class _FeatureNormalizer:
    """Exponentially-weighted normaliser with persistence."""

    def __init__(self, path: Path | None, alpha: float = 0.1) -> None:
        self._path = path
        self._alpha = float(alpha)
        self._stats: dict[str, dict[str, float]] = {}
        self._dirty = False

        for key, (mean, std) in _DEFAULT_NORMALISATION.items():
            var = float(std) ** 2 if std > 0 else 1.0
            self._stats[key] = {"mean": float(mean), "var": max(var, 1e-6)}

        if path and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if not isinstance(value, Mapping):
                        continue
                    mean = float(value.get("mean", 0.0))
                    var = float(value.get("var", value.get("std", 1.0) ** 2))
                    self._stats[str(key)] = {"mean": mean, "var": max(var, 1e-6)}

    def normalize(self, name: str, raw: float) -> float:
        self._update(name, raw)
        stats = self._stats.setdefault(name, {"mean": float(raw), "var": 1.0})
        mean = stats["mean"]
        std = math.sqrt(max(stats["var"], 1e-6))
        if std <= 1e-6:
            return float(raw - mean)
        return float((raw - mean) / std)

    def _update(self, name: str, raw: float) -> None:
        stats = self._stats.setdefault(name, {"mean": float(raw), "var": 1.0})
        mean = stats["mean"]
        var = stats["var"]
        alpha = self._alpha
        delta = float(raw) - mean
        mean = mean + alpha * delta
        var = (1.0 - alpha) * (var + alpha * delta * delta)
        stats["mean"] = mean
        stats["var"] = max(var, 1e-6)
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty or self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                key: {"mean": stats["mean"], "var": stats["var"]}
                for key, stats in self._stats.items()
            }
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._dirty = False
        except Exception:
            # Persistence failures should not break inference paths
            pass


@dataclass(slots=True)
class TamiyoGraphBuilderConfig:
    """Configuration payload for `TamiyoGraphBuilder`."""

    global_feature_dim: int = 16
    seed_feature_dim: int = 14
    blueprint_feature_dim: int = 14
    layer_feature_dim: int = 12
    activation_feature_dim: int = 8
    parameter_feature_dim: int = 10
    edge_feature_dim: int = 3
    # Extended registries for categorical encodings
    layer_type_registry: EmbeddingRegistry | None = None
    activation_type_registry: EmbeddingRegistry | None = None
    optimizer_family_registry: EmbeddingRegistry | None = None
    hazard_class_registry: EmbeddingRegistry | None = None
    # Vocab sizes for normalization
    layer_vocab: int = 1024
    activation_vocab: int = 1024
    optimizer_vocab: int = 256
    hazard_vocab: int = 256
    normalizer_path: Path = Path("var/tamiyo/gnn_norms.json")
    seed_vocab: int = 1024
    blueprint_vocab: int = 1024
    max_layers: int = 4
    max_activations: int = 2
    max_parameters: int = 2
    seed_registry: EmbeddingRegistry | None = None
    blueprint_registry: EmbeddingRegistry | None = None
    blueprint_metadata_provider: Callable[[str], Mapping[str, float | str | bool | int]] | None = (
        None
    )


class TamiyoGraphBuilder:
    """Build heterogenous graphs from step-level Leyline packets."""

    def __init__(self, config: TamiyoGraphBuilderConfig) -> None:
        self._cfg = config
        self._normalizer = _FeatureNormalizer(config.normalizer_path)
        self._seed_registry = config.seed_registry
        self._blueprint_registry = config.blueprint_registry
        self._layer_type_registry = config.layer_type_registry
        self._activation_type_registry = config.activation_type_registry
        self._optimizer_family_registry = config.optimizer_family_registry
        self._hazard_class_registry = config.hazard_class_registry
        self._metadata_provider = config.blueprint_metadata_provider

    def _lookup_blueprint_metadata(
        self, blueprint_id: str
    ) -> Mapping[str, float | str | bool | int]:
        if not blueprint_id or self._metadata_provider is None:
            return {}
        try:
            metadata = self._metadata_provider(blueprint_id)
        except Exception:
            return {}
        return metadata or {}

    def build(self, packet: leyline_pb2.SystemStatePacket) -> HeteroData:
        data = HeteroData()
        blueprint_id = packet.packet_id or packet.training_run_id or "bp-unknown"
        metadata = self._lookup_blueprint_metadata(blueprint_id)
        graph_meta = metadata.get("graph", {}) if isinstance(metadata.get("graph"), Mapping) else {}
        coverage = _CoverageTracker()

        global_features = self._build_global_features(packet, coverage)
        seed_features, seed_ids, seed_caps, seed_scores = self._build_seed_features(
            packet.seed_states,
            metadata,
            graph_meta,
            coverage,
        )
        blueprint_features, blueprint_ids, blueprint_scores = self._build_blueprint_features(
            packet,
            metadata,
            graph_meta,
            coverage,
        )
        layer_features, layer_ids = self._build_layer_features(
            packet, metadata, graph_meta, coverage
        )
        activation_features, activation_ids = self._build_activation_features(
            packet, metadata, graph_meta, coverage
        )
        parameter_features, parameter_ids = self._build_parameter_features(
            packet, metadata, graph_meta, coverage
        )

        data["global"].x = global_features
        data["seed"].x = seed_features
        data["blueprint"].x = blueprint_features
        data["layer"].x = layer_features
        data["activation"].x = activation_features
        data["parameter"].x = parameter_features

        data["seed"].node_ids = seed_ids
        data["blueprint"].node_ids = blueprint_ids
        data["layer"].node_ids = layer_ids
        data["activation"].node_ids = activation_ids
        data["parameter"].node_ids = parameter_ids
        data["seed"].capabilities = seed_caps
        data["seed"].candidate_scores = seed_scores
        data["blueprint"].candidate_scores = blueprint_scores

        self._populate_edges(
            data,
            packet,
            metadata,
            graph_meta,
            seed_caps,
            len(layer_ids),
            len(activation_ids),
            len(parameter_ids),
            coverage,
        )
        self._normalizer.flush()
        cov_map = coverage.summary()
        data.feature_coverage = cov_map
        # WP15: expose typed per-family coverage
        stats = coverage.stats()
        # Accumulate present/total for weighted ratios
        present_totals: dict[str, list[int]] = {}

        def _accumulate(group_key: str, present: int, total: int) -> None:
            present_totals.setdefault(group_key, [0, 0])
            pt = present_totals[group_key]
            pt[0] += present
            pt[1] += total

        groups_init: dict[str, None] = {
            "node.seed": [],
            "node.layer": [],
            "node.activation": [],
            "node.parameter": [],
            "node.blueprint": [],
            "node.global": [],
            "edges.layer_connects": [],
            "edges.seed_monitors": [],
            "edges.layer_feeds": [],
            "edges.layer_activates": [],
            "edges.activation_configures": [],
            "edges.parameter_modulates": [],
            "edges.blueprint_composes": [],
            "edges.parameter_targets": [],
            "edges.global_influences": [],
            "edges.seed_reports": [],
            "edges.global_operates": [],
            "edges.layer_feedback": [],
        }
        for key, (present, total) in stats.items():
            if total <= 0:
                continue
            if key.startswith("seed."):
                _accumulate("node.seed", present, total)
            elif key.startswith("layer."):
                _accumulate("node.layer", present, total)
            elif key.startswith("activation."):
                _accumulate("node.activation", present, total)
            elif key.startswith("parameter."):
                _accumulate("node.parameter", present, total)
            elif key.startswith("blueprint."):
                _accumulate("node.blueprint", present, total)
            elif key.startswith("global."):
                _accumulate("node.global", present, total)
            elif key in groups_init:
                _accumulate(key, present, total)
        types_weighted: dict[str, float] = {}
        for gkey, (p_sum, t_sum) in present_totals.items():
            if t_sum > 0:
                types_weighted[gkey] = float(p_sum) / float(t_sum)
        data.feature_coverage_types = types_weighted
        return data

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def _build_global_features(
        self,
        packet: leyline_pb2.SystemStatePacket,
        coverage: _CoverageTracker,
    ) -> Tensor:
        dim = self._cfg.global_feature_dim
        feats = torch.zeros((1, dim), dtype=torch.float32)
        metrics = dict(packet.training_metrics)
        idx = 0
        keys = (
            "loss",
            "validation_loss",
            "training_loss",
            "gradient_norm",
            "samples_per_s",
            "hook_latency_ms",
        )
        for key in keys:
            raw_value = metrics.get(key)
            present = raw_value is not None
            value = float(raw_value or 0.0)
            feats[0, idx] = self._normalizer.normalize(key, value if present else 0.0)
            feats[0, idx + 1] = 1.0 if present else 0.0
            coverage.observe(f"global.{key}", present)
            idx += 2
        feats[0, idx] = self._normalizer.normalize("global_epoch", float(packet.current_epoch))
        coverage.observe("global.epoch", True)
        idx += 1
        feats[0, idx] = self._normalizer.normalize(
            "global_step", math.log1p(float(packet.global_step))
        )
        coverage.observe("global.step", True)
        idx += 1
        feats[0, idx] = math.tanh(float(len(packet.seed_states)))
        coverage.observe("global.seed_count", True)
        idx += 1
        epochs_total = float(metrics.get("epochs_total", 0.0))
        if epochs_total > 0.0:
            epoch_progress = float(packet.current_epoch) / max(1.0, epochs_total)
        else:
            epoch_progress = 0.0
        feats[0, idx] = self._normalizer.normalize("epoch_progress", epoch_progress)
        coverage.observe("global.epoch_progress", epochs_total > 0.0)
        idx += 1
        # Optional optimizer family categorical (requires extra capacity)
        # Source hint: metadata["optimizer_family"] or metadata["optimizer"]["family"]
        optimizer_family = None
        try:
            # Primary: from blueprint metadata provider
            md = self._lookup_blueprint_metadata(packet.packet_id or packet.training_run_id or "")
            if isinstance(md, Mapping):
                optimizer_family = md.get("optimizer_family")
                if optimizer_family is None and isinstance(md.get("optimizer"), Mapping):
                    optimizer_family = md.get("optimizer", {}).get("family")  # type: ignore[index]
            # Optional mirror via training_metrics (index): if present use directly
            if optimizer_family is None and "optimizer_family_index" in metrics:
                val = int(metrics.get("optimizer_family_index", 0.0) or 0.0)
                if idx < dim:
                    feats[0, idx] = float(val) / max(1.0, float(self._cfg.optimizer_vocab))
                    coverage.observe("global.optimizer_family", True)
                    if idx + 1 < dim:
                        feats[0, idx + 1] = 1.0
                    optimizer_family = None  # handled
        except Exception:
            optimizer_family = None
        if optimizer_family and idx < dim and self._optimizer_family_registry is not None:
            try:
                ridx = self._optimizer_family_registry.get(str(optimizer_family))
                feats[0, idx] = float(ridx) / max(1.0, float(self._cfg.optimizer_vocab))
                coverage.observe("global.optimizer_family", True)
                if idx + 1 < dim:
                    feats[0, idx + 1] = 1.0
            except Exception:
                pass
        return feats

    def _build_seed_features(
        self,
        seeds: Iterable[leyline_pb2.SeedState],
        metadata: Mapping[str, float | str | bool | int],
        graph_meta: Mapping[str, object],
        coverage: _CoverageTracker,
    ) -> tuple[Tensor, list[str], list[dict[str, float]], Tensor]:
        seed_list = list(seeds)
        if not seed_list:
            return (
                torch.zeros((0, self._cfg.seed_feature_dim), dtype=torch.float32),
                [],
                [],
                torch.zeros(0, dtype=torch.float32),
            )
        features = torch.zeros((len(seed_list), self._cfg.seed_feature_dim), dtype=torch.float32)
        max_stage = float(max(leyline_pb2.SeedLifecycleStage.values())) or 1.0
        seed_ids: list[str] = []
        capabilities: list[dict[str, float]] = []
        fallback_scores: list[float] = []
        capability_meta = graph_meta.get("capabilities") if isinstance(graph_meta, Mapping) else {}
        allowed_methods: set[str] = set()
        if isinstance(capability_meta, Mapping):
            methods = capability_meta.get("allowed_blending_methods")
            if isinstance(methods, (list, tuple)):
                allowed_methods = {str(name) for name in methods}
        for index, seed in enumerate(seed_list):
            cursor = 0
            metrics_map = {}
            try:
                metrics_map = dict(getattr(seed, "metrics", {}))
            except Exception:
                metrics_map = {}
            # Proto3 scalars don't have presence; treat non-zero as present
            lr_present = bool(getattr(seed, "learning_rate", 0.0))
            features[index, cursor] = self._normalizer.normalize(
                "seed_learning_rate",
                seed.learning_rate if lr_present else 0.0,
            )
            features[index, cursor + 1] = 1.0 if lr_present else 0.0
            coverage.observe("seed.learning_rate", lr_present)
            cursor += 2

            risk_present = bool(getattr(seed, "risk_score", 0.0))
            features[index, cursor] = self._normalizer.normalize(
                "seed_risk",
                seed.risk_score if risk_present else 0.0,
            )
            features[index, cursor + 1] = 1.0 if risk_present else 0.0
            coverage.observe("seed.risk", risk_present)
            cursor += 2

            grad_present = bool(getattr(seed, "gradient_norm", 0.0))
            features[index, cursor] = self._normalizer.normalize(
                "gradient_norm",
                seed.gradient_norm if grad_present else 0.0,
            )
            features[index, cursor + 1] = 1.0 if grad_present else 0.0
            coverage.observe("seed.gradient_norm", grad_present)
            cursor += 2

            age_present = bool(getattr(seed, "age_epochs", 0))
            features[index, cursor] = self._normalizer.normalize(
                "seed_age",
                float(seed.age_epochs) if age_present else 0.0,
            )
            features[index, cursor + 1] = 1.0 if age_present else 0.0
            coverage.observe("seed.age", age_present)
            cursor += 2

            stage_norm = float(seed.stage) / max_stage
            features[index, cursor] = stage_norm
            features[index, cursor + 1] = 1.0
            coverage.observe("seed.stage", True)
            cursor += 2

            depth_present = bool(seed.layer_depth)
            depth_norm = float(seed.layer_depth) / max(1.0, float(self._cfg.max_layers))
            features[index, cursor] = depth_norm if depth_present else 0.0
            features[index, cursor + 1] = 1.0 if depth_present else 0.0
            coverage.observe("seed.layer_depth", depth_present)
            cursor += 2

            seed_id = seed.seed_id or f"seed-{index}"
            seed_ids.append(seed_id)
            registry = self._seed_registry
            if registry is not None and cursor < features.size(1):
                reg_idx = registry.get(seed_id)
                features[index, cursor] = float(reg_idx) / max(1.0, float(self._cfg.seed_vocab))
                coverage.observe("seed.embedding", True)
            cursor += 1
            # Blend allowance: prefer explicit seed metric if provided; else stage or blueprint capability
            blend_allowed = (
                1.0
                if float(metrics_map.get("blend_allowed", 0.0)) > 0.0
                else (
                    1.0 if seed.stage >= leyline_pb2.SeedLifecycleStage.SEED_STAGE_BLENDING else 0.0
                )
            )
            if allowed_methods:
                blend_allowed = 1.0
            if cursor < features.size(1):
                features[index, cursor] = blend_allowed
            coverage.observe("seed.blend_allowed", True)
            capabilities.append(
                {
                    "stage": stage_norm,
                    "risk": float(seed.risk_score),
                    "blend_allowed": blend_allowed,
                    "layer_depth": float(seed.layer_depth),
                    # Expose optional WP9 enrichments as capabilities for downstream use
                    "alpha": float(metrics_map.get("alpha", 0.0) or 0.0),
                    "alpha_steps": float(metrics_map.get("alpha_steps", 0.0) or 0.0),
                    "alpha_total_steps": float(metrics_map.get("alpha_total_steps", 0.0) or 0.0),
                    "alpha_temperature": float(metrics_map.get("alpha_temperature", 0.0) or 0.0),
                    "risk_tolerance": float(metrics_map.get("risk_tolerance", 0.0) or 0.0),
                }
            )
            fallback_score = (
                blend_allowed * 2.0
                + max(0.0, 1.0 - float(seed.risk_score))
                + (float(seed.stage) / max_stage)
            )
            fallback_scores.append(float(fallback_score))
        scores_tensor = torch.tensor(fallback_scores, dtype=torch.float32)
        return features, seed_ids, capabilities, scores_tensor

    def _build_blueprint_features(
        self,
        packet: leyline_pb2.SystemStatePacket,
        metadata: Mapping[str, float | str | bool | int],
        graph_meta: Mapping[str, object],
        coverage: _CoverageTracker,
    ) -> tuple[Tensor, list[str], Tensor]:
        blueprint_id = packet.packet_id or packet.training_run_id or "bp-unknown"
        dim = self._cfg.blueprint_feature_dim
        features = torch.zeros((1, dim), dtype=torch.float32)
        registry = self._blueprint_registry
        if registry is not None and dim > 0:
            idx = registry.get(blueprint_id)
            features[0, 0] = float(idx) / max(1.0, float(self._cfg.blueprint_vocab))
            coverage.observe("blueprint.embedding", True)

        def _set(col: int, value: float) -> None:
            if col < dim:
                features[0, col] = value

        val_present = bool(packet.validation_loss)
        _set(
            1,
            self._normalizer.normalize(
                "validation_loss", float(packet.validation_loss) if val_present else 0.0
            ),
        )
        _set(2, 1.0 if val_present else 0.0)
        coverage.observe("blueprint.validation_loss", val_present)

        train_present = bool(packet.training_loss)
        _set(
            3,
            self._normalizer.normalize(
                "training_loss", float(packet.training_loss) if train_present else 0.0
            ),
        )
        _set(4, 1.0 if train_present else 0.0)
        coverage.observe("blueprint.training_loss", train_present)
        risk = float(metadata.get("risk", 0.0) or 0.0)
        stage = float(metadata.get("stage", 0.0) or 0.0)
        tier_index = float(metadata.get("tier_index", 0))
        param_count = float(metadata.get("parameter_count", 0.0) or 0.0)
        if isinstance(graph_meta, Mapping):
            parameters = graph_meta.get("parameters")
            if isinstance(parameters, list) and parameters:
                param_count = float(len(parameters))
        quarantine = 1.0 if metadata.get("quarantine_only") else 0.0
        approval = 1.0 if metadata.get("approval_required") else 0.0

        _set(5, risk)
        coverage.observe("blueprint.risk", True)
        _set(6, stage / 10.0)
        coverage.observe("blueprint.stage", True)
        log_param = math.log1p(param_count)
        _set(7, self._normalizer.normalize("blueprint_param_log", log_param))
        _set(8, 1.0 if param_count > 0 else 0.0)
        coverage.observe("blueprint.parameter_count", param_count > 0)
        _set(9, tier_index / 10.0)
        coverage.observe("blueprint.tier", True)
        _set(10, quarantine)
        _set(11, approval)
        coverage.observe("blueprint.flags", True)
        candidate_score = float(metadata.get("candidate_score", 0.0))
        _set(12, self._normalizer.normalize("blueprint_candidate_score", candidate_score))
        _set(13, 1.0 if metadata.get("candidate_score") is not None else 0.0)
        coverage.observe("blueprint.candidate_score", metadata.get("candidate_score") is not None)
        # Optional hazard class categorical if capacity allows
        hazard_class = None
        if isinstance(metadata, Mapping):
            hazard_class = metadata.get("hazard_class") or metadata.get("hazard")
        if hazard_class and dim > 14 and self._hazard_class_registry is not None:
            try:
                hidx = self._hazard_class_registry.get(str(hazard_class))
                _set(14, float(hidx) / max(1.0, float(self._cfg.hazard_vocab)))
                if dim > 15:
                    _set(15, 1.0)
                coverage.observe("blueprint.hazard_class", True)
            except Exception:
                pass
        score_tensor = torch.tensor([candidate_score], dtype=torch.float32)
        return features, [blueprint_id], score_tensor

    def _build_layer_features(
        self,
        packet: leyline_pb2.SystemStatePacket,
        metadata: Mapping[str, float | str | bool | int],
        graph_meta: Mapping[str, object],
        coverage: _CoverageTracker,
    ) -> tuple[Tensor, list[str]]:
        raw_layers = []
        if isinstance(graph_meta, Mapping):
            maybe_layers = graph_meta.get("layers")
            if isinstance(maybe_layers, list):
                raw_layers = [entry for entry in maybe_layers if isinstance(entry, Mapping)]
        count = min(self._cfg.max_layers, len(raw_layers)) if raw_layers else self._cfg.max_layers
        count = max(1, count)
        dim = self._cfg.layer_feature_dim
        features = torch.zeros((count, dim), dtype=torch.float32)
        layer_ids: list[str] = []
        total_layers = max(len(raw_layers), count)
        for idx in range(count):
            descriptor = raw_layers[idx] if idx < len(raw_layers) else {}
            depth = float(descriptor.get("depth", idx))
            depth_norm = (depth + 1.0) / max(1.0, float(total_layers))
            latency = float(descriptor.get("latency_ms", 0.0) or 0.0)
            param_count = float(descriptor.get("parameter_count", 0.0) or 0.0)
            dropout = float(descriptor.get("dropout_rate", 0.0) or 0.0)
            weight_norm = float(descriptor.get("weight_norm", 0.0) or 0.0)
            gradient_norm = float(descriptor.get("gradient_norm", 0.0) or 0.0)
            layer_type = str(descriptor.get("type", "unknown"))
            activation_type = str(
                descriptor.get("activation", descriptor.get("activation_type", "unknown"))
            )
            if dim > 0:
                features[idx, 0] = depth_norm
                coverage.observe("layer.depth", True)
            if dim > 1:
                features[idx, 1] = 1.0 if descriptor.get("depth") is not None else 0.0
            if dim > 2:
                features[idx, 2] = self._normalizer.normalize("layer_latency_ms", latency)
                coverage.observe("layer.latency", bool(latency))
            if dim > 3:
                features[idx, 3] = 1.0 if descriptor.get("latency_ms") is not None else 0.0
            if dim > 4:
                features[idx, 4] = self._normalizer.normalize(
                    "layer_parameter_count", math.log1p(param_count)
                )
                coverage.observe("layer.param_count", param_count > 0)
            if dim > 5:
                features[idx, 5] = 1.0 if descriptor.get("parameter_count") is not None else 0.0
            if dim > 6:
                features[idx, 6] = float(max(0.0, min(1.0, dropout)))
                coverage.observe("layer.dropout", descriptor.get("dropout_rate") is not None)
            if dim > 7:
                if self._layer_type_registry is not None:
                    idx_val = self._layer_type_registry.get(layer_type)
                    features[idx, 7] = float(idx_val) / max(1.0, float(self._cfg.layer_vocab))
                else:
                    features[idx, 7] = self._encode_category(layer_type)
                coverage.observe("layer.type", descriptor.get("type") is not None)
            if dim > 8:
                if self._activation_type_registry is not None:
                    aidx = self._activation_type_registry.get(activation_type)
                    features[idx, 8] = float(aidx) / max(1.0, float(self._cfg.activation_vocab))
                else:
                    features[idx, 8] = self._encode_category(activation_type)
                coverage.observe(
                    "layer.activation",
                    descriptor.get("activation") is not None
                    or descriptor.get("activation_type") is not None,
                )
            # Optional categorical presence masks when feature dim is extended
            if dim > 12:
                features[idx, 12] = 1.0 if descriptor.get("type") is not None else 0.0
            if dim > 13:
                features[idx, 13] = (
                    1.0
                    if (
                        descriptor.get("activation") is not None
                        or descriptor.get("activation_type") is not None
                    )
                    else 0.0
                )
            if dim > 9:
                features[idx, 9] = self._normalizer.normalize("layer_weight_norm", weight_norm)
                coverage.observe("layer.weight_norm", bool(weight_norm))
            if dim > 10:
                features[idx, 10] = self._normalizer.normalize("layer_gradient_norm", gradient_norm)
                coverage.observe("layer.gradient_norm", bool(gradient_norm))
            if dim > 11:
                features[idx, 11] = 1.0 if descriptor else 0.0
            layer_id = str(descriptor.get("layer_id", f"{packet.training_run_id or 'run'}-L{idx}"))
            layer_ids.append(layer_id)
        return features, layer_ids

    def _build_activation_features(
        self,
        packet: leyline_pb2.SystemStatePacket,
        metadata: Mapping[str, float | str | bool | int],
        graph_meta: Mapping[str, object],
        coverage: _CoverageTracker,
    ) -> tuple[Tensor, list[str]]:
        raw_activations = []
        if isinstance(graph_meta, Mapping):
            maybe_acts = graph_meta.get("activations")
            if isinstance(maybe_acts, list):
                raw_activations = [entry for entry in maybe_acts if isinstance(entry, Mapping)]
        count = (
            min(self._cfg.max_activations, len(raw_activations))
            if raw_activations
            else self._cfg.max_activations
        )
        count = max(1, count)
        dim = self._cfg.activation_feature_dim
        features = torch.zeros((count, dim), dtype=torch.float32)
        activation_ids: list[str] = []
        for idx in range(count):
            descriptor = raw_activations[idx] if idx < len(raw_activations) else {}
            activation_type = str(descriptor.get("type", "unknown"))
            saturation = float(descriptor.get("saturation_rate", 0.0) or 0.0)
            gradient_flow = float(descriptor.get("gradient_flow", 0.0) or 0.0)
            cost = float(descriptor.get("computational_cost", 0.0) or 0.0)
            dominance = float(descriptor.get("nonlinearity_strength", 0.0) or 0.0)
            if dim > 0:
                if self._activation_type_registry is not None:
                    aidx = self._activation_type_registry.get(activation_type)
                    features[idx, 0] = float(aidx) / max(1.0, float(self._cfg.activation_vocab))
                else:
                    features[idx, 0] = self._encode_category(activation_type)
                coverage.observe("activation.type", True)
            if dim > 1:
                features[idx, 1] = 1.0 if descriptor else 0.0
            if dim > 2:
                features[idx, 2] = self._normalizer.normalize("activation_saturation", saturation)
                coverage.observe("activation.saturation", bool(saturation))
            if dim > 3:
                features[idx, 3] = 1.0 if descriptor.get("saturation_rate") is not None else 0.0
            if dim > 4:
                features[idx, 4] = self._normalizer.normalize(
                    "activation_gradient_flow", gradient_flow
                )
                coverage.observe("activation.gradient_flow", bool(gradient_flow))
            if dim > 5:
                features[idx, 5] = self._normalizer.normalize("activation_cost", math.log1p(cost))
                coverage.observe("activation.cost", bool(cost))
            if dim > 6:
                features[idx, 6] = self._normalizer.normalize("activation_nonlinearity", dominance)
            if dim > 7:
                features[idx, 7] = float(idx) / max(1.0, float(count - 1 or 1))
            # Optional explicit activation type presence mask when feature dim is extended
            if dim > 8:
                features[idx, 8] = 1.0 if descriptor.get("type") is not None else 0.0
            activation_ids.append(
                str(descriptor.get("activation_id", f"{packet.training_run_id or 'run'}-A{idx}"))
            )
        return features, activation_ids

    def _build_parameter_features(
        self,
        packet: leyline_pb2.SystemStatePacket,
        metadata: Mapping[str, float | str | bool | int],
        graph_meta: Mapping[str, object],
        coverage: _CoverageTracker,
    ) -> tuple[Tensor, list[str]]:
        descriptors = []
        if isinstance(graph_meta, Mapping):
            maybe_parameters = graph_meta.get("parameters")
            if isinstance(maybe_parameters, list):
                descriptors = [entry for entry in maybe_parameters if isinstance(entry, Mapping)]
        allowed = (
            metadata.get("allowed_parameters", {})
            if isinstance(metadata.get("allowed_parameters"), Mapping)
            else {}
        )
        if not descriptors and isinstance(allowed, Mapping):
            for name, bounds in allowed.items():
                descriptors.append(
                    {
                        "name": name,
                        "min": float(bounds.get("min", 0.0)),
                        "max": float(bounds.get("max", 0.0)),
                        "span": float(
                            bounds.get(
                                "span",
                                float(bounds.get("max", 0.0)) - float(bounds.get("min", 0.0)),
                            )
                        ),
                        "default": 0.5
                        * (float(bounds.get("min", 0.0)) + float(bounds.get("max", 0.0))),
                    }
                )
        if not descriptors:
            descriptors.append(
                {"name": "alpha", "min": 0.0, "max": 1.0, "span": 1.0, "default": 0.5}
            )

        count = min(self._cfg.max_parameters, max(1, len(descriptors)))
        dim = self._cfg.parameter_feature_dim
        features = torch.zeros((count, dim), dtype=torch.float32)
        parameter_ids: list[str] = []
        risk = float(metadata.get("risk", 0.0) or 0.0)
        stage = float(metadata.get("stage", 0.0) or 0.0)
        for idx in range(count):
            descriptor = descriptors[idx] if idx < len(descriptors) else descriptors[-1]
            name = str(descriptor.get("name", f"param-{idx}"))
            min_v = float(descriptor.get("min", 0.0) or 0.0)
            max_v = float(descriptor.get("max", 0.0) or 0.0)
            span = float(descriptor.get("span", max_v - min_v))
            default = float(descriptor.get("default", (min_v + max_v) * 0.5))
            if dim > 0:
                features[idx, 0] = self._normalizer.normalize("parameter_min", min_v)
                coverage.observe("parameter.min", descriptor.get("min") is not None)
            if dim > 1:
                features[idx, 1] = 1.0 if descriptor.get("min") is not None else 0.0
            if dim > 2:
                features[idx, 2] = self._normalizer.normalize("parameter_max", max_v)
                coverage.observe("parameter.max", descriptor.get("max") is not None)
            if dim > 3:
                features[idx, 3] = 1.0 if descriptor.get("max") is not None else 0.0
            if dim > 4:
                features[idx, 4] = self._normalizer.normalize("parameter_span", span)
                coverage.observe("parameter.span", descriptor.get("span") is not None)
            if dim > 5:
                features[idx, 5] = 1.0 if descriptor.get("span") is not None else 0.0
            if dim > 6:
                features[idx, 6] = self._normalizer.normalize("parameter_default", default)
                coverage.observe("parameter.default", descriptor.get("default") is not None)
            if dim > 7:
                features[idx, 7] = 1.0 if descriptor.get("default") is not None else 0.0
            if dim > 8:
                features[idx, 8] = risk
            if dim > 9:
                features[idx, 9] = stage / 10.0
            parameter_ids.append(f"{packet.training_run_id or 'run'}-{name}")
        return features, parameter_ids

    def _populate_edges(
        self,
        data: HeteroData,
        packet: leyline_pb2.SystemStatePacket,
        metadata: Mapping[str, float | str | bool | int],
        graph_meta: Mapping[str, object],
        seed_capabilities: Sequence[dict[str, float]],
        layer_count: int,
        activation_count: int,
        parameter_count: int,
        coverage: _CoverageTracker,
    ) -> None:
        edge_dim = self._cfg.edge_feature_dim
        risk = float(metadata.get("risk", 0.0) or 0.0)
        seed_count = data["seed"].x.size(0)
        blueprint_count = data["blueprint"].x.size(0)

        def _set_edge(
            relation: tuple[str, str, str],
            src: torch.Tensor,
            dst: torch.Tensor,
            attrs: Sequence[Sequence[float]] | torch.Tensor | None = None,
        ) -> None:
            if src.numel() == 0:
                data[relation].edge_index = torch.zeros((2, 0), dtype=torch.long)
                data[relation].edge_attr = torch.zeros((0, edge_dim), dtype=torch.float32)
                return
            edge_index = torch.stack((src, dst))
            if attrs is None:
                edge_attr = torch.zeros((src.numel(), edge_dim), dtype=torch.float32)
            else:
                if isinstance(attrs, torch.Tensor):
                    edge_attr = attrs.to(dtype=torch.float32)
                else:
                    edge_attr = torch.tensor(attrs, dtype=torch.float32)
                if edge_attr.ndim == 1:
                    edge_attr = edge_attr.unsqueeze(0)
                # Ensure row count matches number of edges (broadcast if needed)
                if edge_attr.size(0) != src.numel():
                    edge_attr = edge_attr.expand(src.numel(), min(edge_dim, edge_attr.size(1)))
                # Pad or slice to match configured edge_dim
                if edge_attr.size(1) < edge_dim:
                    pad = torch.zeros(
                        (edge_attr.size(0), edge_dim - edge_attr.size(1)), dtype=torch.float32
                    )
                    edge_attr = torch.cat([edge_attr, pad], dim=1)
            data[relation].edge_index = edge_index
            data[relation].edge_attr = edge_attr[:, :edge_dim]

        # global ↔ seed edges
        if seed_count:
            global_src = torch.zeros(seed_count, dtype=torch.long)
            seed_dst = torch.arange(seed_count, dtype=torch.long)
            attrs = [
                [1.0, cap.get("stage", 0.0), cap.get("risk", 0.0)] for cap in seed_capabilities
            ]
            _set_edge(("global", "influences", "seed"), global_src, seed_dst, attrs)
            _set_edge(("seed", "reports", "global"), seed_dst, global_src, attrs)
            coverage.observe("edges.global_influences", True)
            coverage.observe("edges.seed_reports", True)
        else:
            _set_edge(
                ("global", "influences", "seed"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            _set_edge(
                ("seed", "reports", "global"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            coverage.observe("edges.global_influences", False)
            coverage.observe("edges.seed_reports", False)

        # global ↔ blueprint
        if blueprint_count:
            blueprint_indices = torch.arange(blueprint_count, dtype=torch.long)
            attrs = [[1.0, float(metadata.get("stage", 0.0) or 0.0), risk]] * blueprint_count
            _set_edge(
                ("global", "annotates", "blueprint"),
                torch.zeros(blueprint_count, dtype=torch.long),
                blueprint_indices,
                attrs,
            )
            _set_edge(
                ("blueprint", "monitored_by", "global"),
                blueprint_indices,
                torch.zeros(blueprint_count, dtype=torch.long),
                attrs,
            )
        else:
            _set_edge(
                ("global", "annotates", "blueprint"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            _set_edge(
                ("blueprint", "monitored_by", "global"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )

        # seed peer chain
        if seed_count > 1:
            src = torch.arange(seed_count - 1, dtype=torch.long)
            dst = torch.arange(1, seed_count, dtype=torch.long)
            attrs = [[0.5, 0.0, 0.0] for _ in range(seed_count - 1)]
            _set_edge(("seed", "peer", "seed"), src, dst, attrs)
        else:
            _set_edge(
                ("seed", "peer", "seed"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )

        # global ↔ layer
        layer_indices = torch.arange(layer_count, dtype=torch.long)
        layers_depth = [(idx + 1) / max(1, layer_count) for idx in range(layer_count)]
        if layer_count:
            depth_t = torch.tensor(layers_depth, dtype=torch.float32)
            attrs_layers_t = torch.stack(
                [
                    torch.ones(layer_count, dtype=torch.float32),
                    depth_t,
                    torch.full((layer_count,), risk, dtype=torch.float32),
                ],
                dim=1,
            )
            _set_edge(
                ("global", "operates", "layer"),
                torch.zeros(layer_count, dtype=torch.long),
                layer_indices,
                attrs_layers_t,
            )
            _set_edge(
                ("layer", "feedback", "global"),
                layer_indices,
                torch.zeros(layer_count, dtype=torch.long),
                attrs_layers_t,
            )
        else:
            _set_edge(
                ("global", "operates", "layer"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            _set_edge(
                ("layer", "feedback", "global"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
        coverage.observe("edges.global_operates", bool(layer_count))
        coverage.observe("edges.layer_feedback", bool(layer_count))

        # layer ↔ activation
        activation_indices = torch.arange(activation_count, dtype=torch.long)
        activation_ratio = [(idx + 1) / max(1, activation_count) for idx in range(activation_count)]
        act_ratio_t = (
            torch.tensor(activation_ratio, dtype=torch.float32)
            if activation_count
            else torch.zeros(0, dtype=torch.float32)
        )
        if layer_count and activation_count:
            src = layer_indices.repeat_interleave(activation_count)
            dst = activation_indices.repeat(layer_count)
            depth_t = torch.tensor(layers_depth, dtype=torch.float32)
            depth_b = depth_t.view(-1, 1).repeat(1, activation_count).reshape(-1)
            act_b = act_ratio_t.view(1, -1).repeat(layer_count, 1).reshape(-1)
            attrs_la_t = torch.stack([depth_b, act_b, torch.ones_like(depth_b)], dim=1)
            _set_edge(("layer", "activates", "activation"), src, dst, attrs_la_t)
            _set_edge(("activation", "affects", "layer"), dst, src, attrs_la_t)
            coverage.observe("edges.layer_activates", True)
        else:
            _set_edge(
                ("layer", "activates", "activation"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            _set_edge(
                ("activation", "affects", "layer"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            coverage.observe("edges.layer_activates", False)

        # activation ↔ parameter
        parameter_indices = torch.arange(parameter_count, dtype=torch.long)
        param_ratio = [(idx + 1) / max(1, parameter_count) for idx in range(parameter_count)]
        param_ratio_t = (
            torch.tensor(param_ratio, dtype=torch.float32)
            if parameter_count
            else torch.zeros(0, dtype=torch.float32)
        )
        if activation_count and parameter_count:
            src = activation_indices.repeat_interleave(parameter_count)
            dst = parameter_indices.repeat(activation_count)
            act_b = act_ratio_t.view(-1, 1).repeat(1, parameter_count).reshape(-1)
            par_b = param_ratio_t.view(1, -1).repeat(activation_count, 1).reshape(-1)
            attrs_ap_t = torch.stack([act_b, par_b, torch.ones_like(act_b)], dim=1)
            _set_edge(("activation", "configures", "parameter"), src, dst, attrs_ap_t)
            _set_edge(("parameter", "modulates", "activation"), dst, src, attrs_ap_t)
            coverage.observe("edges.activation_configures", True)
            coverage.observe("edges.parameter_modulates", True)
        else:
            _set_edge(
                ("activation", "configures", "parameter"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            _set_edge(
                ("parameter", "modulates", "activation"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            coverage.observe("edges.activation_configures", False)
            coverage.observe("edges.parameter_modulates", False)

        # seed ↔ parameter capability edges (WP10): build only when allowances present; mask otherwise
        if seed_count and parameter_count:
            allowed_seed_indices = [
                idx
                for idx, cap in enumerate(seed_capabilities)
                if float(cap.get("blend_allowed", 0.0)) > 0.0
            ]
            # Derive per-seed allowed parameter indices (names and/or indices supported via extras)
            param_node_ids = list(getattr(data["parameter"], "node_ids", []))
            param_names = [
                (pid.rsplit("-", 1)[1] if isinstance(pid, str) and "-" in pid else str(pid))
                for pid in param_node_ids
            ]
            capabilities_block = (
                graph_meta.get("capabilities") if isinstance(graph_meta, Mapping) else None
            )
            allow_by_seed_id: dict[str, set[int]] = {}
            allow_by_seed_index: dict[int, set[int]] = {}
            if isinstance(capabilities_block, Mapping):
                by_id = capabilities_block.get("allowed_parameters_by_seed_id")
                if isinstance(by_id, Mapping):
                    for sid, entries in by_id.items():
                        indices: set[int] = set()
                        if isinstance(entries, list):
                            for e in entries:
                                if isinstance(e, str):
                                    try:
                                        idx = param_names.index(e)
                                        indices.add(idx)
                                    except ValueError:
                                        continue
                                else:
                                    try:
                                        ei = int(e)
                                        if 0 <= ei < parameter_count:
                                            indices.add(ei)
                                    except Exception:
                                        continue
                        allow_by_seed_id[str(sid)] = indices
                by_index = capabilities_block.get("allowed_parameters_by_seed_index")
                if isinstance(by_index, Mapping):
                    for s, entries in by_index.items():
                        try:
                            sidx = int(s)
                        except Exception:
                            continue
                        indices: set[int] = set()
                        if isinstance(entries, list):
                            for e in entries:
                                try:
                                    ei = int(e)
                                    if 0 <= ei < parameter_count:
                                        indices.add(ei)
                                except Exception:
                                    continue
                        allow_by_seed_index[sidx] = indices
            if allowed_seed_indices:
                src_list: list[int] = []
                dst_list: list[int] = []
                attrs_sp: list[list[float]] = []
                for sidx in allowed_seed_indices:
                    cap = seed_capabilities[sidx] if 0 <= sidx < len(seed_capabilities) else {}
                    blend_allowed = float(cap.get("blend_allowed", 0.0))
                    stage_norm = float(cap.get("stage", 0.0))
                    # Determine allowed parameters for this seed
                    seed_id_list = list(getattr(data["seed"], "node_ids", []))
                    sid = seed_id_list[sidx] if 0 <= sidx < len(seed_id_list) else None
                    allowed_indices: set[int] | None = None
                    if sid is not None and sid in allow_by_seed_id:
                        allowed_indices = set(allow_by_seed_id[sid])
                    elif sidx in allow_by_seed_index:
                        allowed_indices = set(allow_by_seed_index[sidx])
                    # Default: if no explicit list, all parameters are allowed for an allowed seed
                    if allowed_indices is None:
                        allowed_indices = set(range(parameter_count))
                    # Expose per-parameter allow mask in seed capabilities (0/1 per parameter name)
                    try:
                        caps_mut = cap  # same object referenced by data["seed"].capabilities
                        for pidx, pname in enumerate(param_names):
                            key = f"allowed_param_{pname}"
                            caps_mut[key] = 1.0 if pidx in allowed_indices else 0.0
                    except Exception:
                        pass
                    for pidx in sorted(allowed_indices):
                        src_list.append(sidx)
                        dst_list.append(pidx)
                        attrs_sp.append([blend_allowed, 1.0, stage_norm])
                src = (
                    torch.tensor(src_list, dtype=torch.long)
                    if src_list
                    else torch.zeros(0, dtype=torch.long)
                )
                dst = (
                    torch.tensor(dst_list, dtype=torch.long)
                    if dst_list
                    else torch.zeros(0, dtype=torch.long)
                )
                _set_edge(
                    ("seed", "allowed", "parameter"), src, dst, attrs_sp if attrs_sp else None
                )
                _set_edge(
                    ("parameter", "associated", "seed"), dst, src, attrs_sp if attrs_sp else None
                )
                coverage.observe("edges.seed_param_allowed", bool(src_list))
            else:
                _set_edge(
                    ("seed", "allowed", "parameter"),
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                )
                _set_edge(
                    ("parameter", "associated", "seed"),
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                )
                coverage.observe("edges.seed_param_allowed", False)
        else:
            _set_edge(
                ("seed", "allowed", "parameter"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            _set_edge(
                ("parameter", "associated", "seed"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            coverage.observe("edges.seed_param_allowed", False)

        # blueprint ↔ layer
        if blueprint_count and layer_count:
            src = torch.zeros(layer_count, dtype=torch.long)
            dst = layer_indices
            attrs_bl = [[risk, layers_depth[i], 1.0] for i in range(layer_count)]
            _set_edge(("blueprint", "composes", "layer"), src, dst, attrs_bl)
            _set_edge(("layer", "belongs_to", "blueprint"), dst, src, attrs_bl)
        else:
            _set_edge(
                ("blueprint", "composes", "layer"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            _set_edge(
                ("layer", "belongs_to", "blueprint"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
        coverage.observe("edges.blueprint_composes", bool(layer_count and blueprint_count))

        # blueprint ↔ activation
        if blueprint_count and activation_count:
            src = torch.zeros(activation_count, dtype=torch.long)
            dst = activation_indices
            attrs_ba_t = torch.stack(
                [
                    torch.full((activation_count,), risk, dtype=torch.float32),
                    act_ratio_t,
                    torch.ones(activation_count, dtype=torch.float32),
                ],
                dim=1,
            )
            _set_edge(("blueprint", "energizes", "activation"), src, dst, attrs_ba_t)
        else:
            _set_edge(
                ("blueprint", "energizes", "activation"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )

        # parameter -> blueprint
        if blueprint_count and parameter_count:
            src = parameter_indices
            dst = torch.zeros(parameter_count, dtype=torch.long)
            attrs_pb_t = torch.stack(
                [
                    param_ratio_t,
                    torch.full((parameter_count,), risk, dtype=torch.float32),
                    torch.ones(parameter_count, dtype=torch.float32),
                ],
                dim=1,
            )
            _set_edge(("parameter", "targets", "blueprint"), src, dst, attrs_pb_t)
        else:
            _set_edge(
                ("parameter", "targets", "blueprint"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
        coverage.observe("edges.parameter_targets", bool(parameter_count and blueprint_count))

        # layer connectivity from extras adjacency, fallback to simple chain
        pairs: list[tuple[int, int]] = []
        if isinstance(graph_meta, Mapping):
            adj = graph_meta.get("adjacency")
            layer_pairs = None
            if isinstance(adj, Mapping):
                layer_pairs = adj.get("layer")
            elif isinstance(adj, list):
                layer_pairs = adj
            if isinstance(layer_pairs, list):
                for p in layer_pairs:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        s = int(p[0])
                        d = int(p[1])
                        if 0 <= s < layer_count and 0 <= d < layer_count:
                            pairs.append((s, d))
        if pairs:
            src = torch.tensor([s for s, _ in pairs], dtype=torch.long)
            dst = torch.tensor([d for _, d in pairs], dtype=torch.long)
            depth_t = torch.tensor(layers_depth, dtype=torch.float32)
            attrs_ll_t = torch.stack(
                [
                    depth_t[src],
                    depth_t[dst],
                    torch.full((src.numel(),), risk, dtype=torch.float32),
                ],
                dim=1,
            )
            _set_edge(("layer", "connects", "layer"), src, dst, attrs_ll_t)
            coverage.observe("edges.layer_connects", True)
        elif layer_count > 1:
            src = torch.arange(layer_count - 1, dtype=torch.long)
            dst = torch.arange(1, layer_count, dtype=torch.long)
            depth_t = torch.tensor(layers_depth, dtype=torch.float32)
            attrs_ll_t = torch.stack(
                [
                    depth_t[src],
                    depth_t[dst],
                    torch.full((src.numel(),), risk, dtype=torch.float32),
                ],
                dim=1,
            )
            _set_edge(("layer", "connects", "layer"), src, dst, attrs_ll_t)
            coverage.observe("edges.layer_connects", True)
        else:
            _set_edge(
                ("layer", "connects", "layer"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            coverage.observe("edges.layer_connects", False)

        # seed monitors layer; prefer explicit extras, fallback to layer_depth heuristic
        if seed_count and layer_count:
            src_indices: list[int] = []
            dst_indices: list[int] = []
            attrs_sl: list[list[float]] = []
            # Parse optional explicit monitors from extras
            explicit_pairs: list[tuple[int, int, float | None]] = []
            seed_ids_list: list[str] = list(getattr(data["seed"], "node_ids", []))
            monitors_block = graph_meta.get("monitors") if isinstance(graph_meta, Mapping) else None
            if isinstance(graph_meta, Mapping):
                monitored_layers = graph_meta.get("monitored_layers")
                if isinstance(monitored_layers, list):
                    for entry in monitored_layers:
                        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            s = int(entry[0])
                            d = int(entry[1])
                            w = float(entry[2]) if len(entry) >= 3 else None
                            if 0 <= s < seed_count and 0 <= d < layer_count:
                                explicit_pairs.append((s, d, w))
            if isinstance(monitors_block, Mapping):
                by_seed_id = monitors_block.get("by_seed_id")
                if isinstance(by_seed_id, Mapping):
                    for sid, layers in by_seed_id.items():
                        try:
                            sidx = seed_ids_list.index(str(sid))
                        except ValueError:
                            continue
                        if isinstance(layers, list):
                            for d in layers:
                                try:
                                    di = int(d)
                                except Exception:
                                    continue
                                if 0 <= di < layer_count:
                                    explicit_pairs.append((sidx, di, None))
                by_seed_index = monitors_block.get("by_seed_index")
                if isinstance(by_seed_index, Mapping):
                    for s, layers in by_seed_index.items():
                        try:
                            sidx = int(s)
                        except Exception:
                            continue
                        if not (0 <= sidx < seed_count):
                            continue
                        if isinstance(layers, list):
                            for d in layers:
                                try:
                                    di = int(d)
                                except Exception:
                                    continue
                                if 0 <= di < layer_count:
                                    explicit_pairs.append((sidx, di, None))
            thresholds = {}
            if isinstance(monitors_block, Mapping):
                th = monitors_block.get("thresholds") or monitors_block.get("monitors_thresholds")
                if isinstance(th, Mapping):
                    thresholds = dict(th)

            def _passes_threshold(sidx: int) -> bool:
                cap = seed_capabilities[sidx] if 0 <= sidx < len(seed_capabilities) else {}
                try:
                    if "risk_max" in thresholds and float(cap.get("risk", 0.0)) > float(
                        thresholds["risk_max"]
                    ):
                        return False
                except Exception:
                    pass
                try:
                    if "stage_min" in thresholds and float(cap.get("stage", 0.0)) < float(
                        thresholds["stage_min"]
                    ):
                        return False
                except Exception:
                    pass
                return True

            if explicit_pairs:
                for sidx, didx, w in explicit_pairs:
                    if not _passes_threshold(sidx):
                        continue
                    cap = seed_capabilities[sidx] if 0 <= sidx < len(seed_capabilities) else {}
                    src_indices.append(sidx)
                    dst_indices.append(didx)
                    last_attr = (
                        float(w) if w is not None else (layers_depth[didx] if layer_count else 0.0)
                    )
                    attrs_sl.append([cap.get("stage", 0.0), cap.get("risk", 0.0), last_attr])
            else:
                for idx, cap in enumerate(seed_capabilities):
                    if not _passes_threshold(idx):
                        continue
                    depth_raw = int(round(cap.get("layer_depth", 0.0)))
                    depth = max(0, min(layer_count - 1, depth_raw))
                    src_indices.append(idx)
                    dst_indices.append(depth)
                    attrs_sl.append(
                        [
                            cap.get("stage", 0.0),
                            cap.get("risk", 0.0),
                            layers_depth[depth] if layer_count else 0.0,
                        ]
                    )
            if src_indices:
                seed_tensor = torch.tensor(src_indices, dtype=torch.long)
                layer_tensor = torch.tensor(dst_indices, dtype=torch.long)
                _set_edge(("seed", "monitors", "layer"), seed_tensor, layer_tensor, attrs_sl)
                _set_edge(("layer", "monitored_by", "seed"), layer_tensor, seed_tensor, attrs_sl)
                coverage.observe("edges.seed_monitors", True)
            else:
                _set_edge(
                    ("seed", "monitors", "layer"),
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                )
                _set_edge(
                    ("layer", "monitored_by", "seed"),
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                )
                coverage.observe("edges.seed_monitors", False)
        else:
            _set_edge(
                ("seed", "monitors", "layer"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            _set_edge(
                ("layer", "monitored_by", "seed"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            coverage.observe("edges.seed_monitors", False)

        # layer feeds activation (alias for activates)
        if layer_count and activation_count:
            src = layer_indices.repeat_interleave(activation_count)
            dst = activation_indices.repeat(layer_count)
            # Vectorized attribute construction to reduce Python overhead
            depth_t = torch.tensor(layers_depth, dtype=torch.float32)
            act_t = torch.tensor(activation_ratio, dtype=torch.float32)
            depth_b = depth_t.view(-1, 1).repeat(1, activation_count).reshape(-1)
            act_b = act_t.view(1, -1).repeat(layer_count, 1).reshape(-1)
            attrs_feed = torch.stack([depth_b, act_b, torch.ones_like(depth_b)], dim=1).tolist()
            _set_edge(("layer", "feeds", "activation"), src, dst, attrs_feed)
            coverage.observe("edges.layer_feeds", True)
        else:
            _set_edge(
                ("layer", "feeds", "activation"),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
            coverage.observe("edges.layer_feeds", False)

    @staticmethod
    def _encode_category(value: str) -> float:
        encoded = value.encode("utf-8", errors="ignore")
        if not encoded:
            return 0.0
        digest = hashlib.blake2s(encoded, digest_size=4).digest()
        return int.from_bytes(digest, "big") / 0xFFFFFFFF


class _CoverageTracker:
    """Track feature coverage for telemetry purposes."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._present: Counter[str] = Counter()

    def observe(self, key: str, present: bool) -> None:
        self._counts[key] += 1
        if present:
            self._present[key] += 1

    def summary(self) -> dict[str, float]:
        return {
            key: float(self._present.get(key, 0)) / float(total)
            for key, total in self._counts.items()
            if total > 0
        }

    def stats(self) -> dict[str, tuple[int, int]]:
        """Return raw (present, total) counts per feature key for weighted aggregation."""
        return {
            key: (int(self._present.get(key, 0)), int(total)) for key, total in self._counts.items()
        }


__all__ = ["TamiyoGraphBuilder", "TamiyoGraphBuilderConfig"]
