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
    normalizer_path: Path = Path("var/tamiyo/gnn_norms.json")
    seed_vocab: int = 1024
    blueprint_vocab: int = 1024
    max_layers: int = 4
    max_activations: int = 2
    max_parameters: int = 2
    seed_registry: EmbeddingRegistry | None = None
    blueprint_registry: EmbeddingRegistry | None = None
    blueprint_metadata_provider: Callable[[str], Mapping[str, float | str | bool | int]] | None = None


class TamiyoGraphBuilder:
    """Build heterogenous graphs from step-level Leyline packets."""

    def __init__(self, config: TamiyoGraphBuilderConfig) -> None:
        self._cfg = config
        self._normalizer = _FeatureNormalizer(config.normalizer_path)
        self._seed_registry = config.seed_registry
        self._blueprint_registry = config.blueprint_registry
        self._metadata_provider = config.blueprint_metadata_provider

    def _lookup_blueprint_metadata(self, blueprint_id: str) -> Mapping[str, float | str | bool | int]:
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
        layer_features, layer_ids = self._build_layer_features(packet, metadata, graph_meta, coverage)
        activation_features, activation_ids = self._build_activation_features(packet, metadata, graph_meta, coverage)
        parameter_features, parameter_ids = self._build_parameter_features(packet, metadata, graph_meta, coverage)

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
            seed_caps,
            len(layer_ids),
            len(activation_ids),
            len(parameter_ids),
        )
        self._normalizer.flush()
        data.feature_coverage = coverage.summary()
        return data

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def _build_global_features(
        self,
        packet: leyline_pb2.SystemStatePacket,
        coverage: _CoverageTracker,
    ) -> Tensor:
        feats = torch.zeros((1, self._cfg.global_feature_dim), dtype=torch.float32)
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
        feats[0, idx] = self._normalizer.normalize("global_step", math.log1p(float(packet.global_step)))
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
            blend_allowed = 1.0 if seed.stage >= leyline_pb2.SeedLifecycleStage.SEED_STAGE_BLENDING else 0.0
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
        _set(1, self._normalizer.normalize("validation_loss", float(packet.validation_loss) if val_present else 0.0))
        _set(2, 1.0 if val_present else 0.0)
        coverage.observe("blueprint.validation_loss", val_present)

        train_present = bool(packet.training_loss)
        _set(3, self._normalizer.normalize("training_loss", float(packet.training_loss) if train_present else 0.0))
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
            activation_type = str(descriptor.get("activation", descriptor.get("activation_type", "unknown")))
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
                features[idx, 4] = self._normalizer.normalize("layer_parameter_count", math.log1p(param_count))
                coverage.observe("layer.param_count", param_count > 0)
            if dim > 5:
                features[idx, 5] = 1.0 if descriptor.get("parameter_count") is not None else 0.0
            if dim > 6:
                features[idx, 6] = float(max(0.0, min(1.0, dropout)))
                coverage.observe("layer.dropout", descriptor.get("dropout_rate") is not None)
            if dim > 7:
                features[idx, 7] = self._encode_category(layer_type)
                coverage.observe("layer.type", descriptor.get("type") is not None)
            if dim > 8:
                features[idx, 8] = self._encode_category(activation_type)
                coverage.observe(
                    "layer.activation",
                    descriptor.get("activation") is not None or descriptor.get("activation_type") is not None,
                )
            # Optional categorical presence masks when feature dim is extended
            if dim > 12:
                features[idx, 12] = 1.0 if descriptor.get("type") is not None else 0.0
            if dim > 13:
                features[idx, 13] = (
                    1.0
                    if (descriptor.get("activation") is not None or descriptor.get("activation_type") is not None)
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
        count = min(self._cfg.max_activations, len(raw_activations)) if raw_activations else self._cfg.max_activations
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
                features[idx, 4] = self._normalizer.normalize("activation_gradient_flow", gradient_flow)
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
            activation_ids.append(str(descriptor.get("activation_id", f"{packet.training_run_id or 'run'}-A{idx}")))
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
        allowed = metadata.get("allowed_parameters", {}) if isinstance(metadata.get("allowed_parameters"), Mapping) else {}
        if not descriptors and isinstance(allowed, Mapping):
            for name, bounds in allowed.items():
                descriptors.append(
                    {
                        "name": name,
                        "min": float(bounds.get("min", 0.0)),
                        "max": float(bounds.get("max", 0.0)),
                        "span": float(bounds.get("span", float(bounds.get("max", 0.0)) - float(bounds.get("min", 0.0)))),
                        "default": 0.5 * (float(bounds.get("min", 0.0)) + float(bounds.get("max", 0.0))),
                    }
                )
        if not descriptors:
            descriptors.append({"name": "alpha", "min": 0.0, "max": 1.0, "span": 1.0, "default": 0.5})

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
        seed_capabilities: Sequence[dict[str, float]],
        layer_count: int,
        activation_count: int,
        parameter_count: int,
    ) -> None:
        edge_dim = self._cfg.edge_feature_dim
        risk = float(metadata.get("risk", 0.0) or 0.0)
        seed_count = data["seed"].x.size(0)
        blueprint_count = data["blueprint"].x.size(0)

        def _set_edge(
            relation: tuple[str, str, str],
            src: torch.Tensor,
            dst: torch.Tensor,
            attrs: Sequence[Sequence[float]] | None = None,
        ) -> None:
            if src.numel() == 0:
                data[relation].edge_index = torch.zeros((2, 0), dtype=torch.long)
                data[relation].edge_attr = torch.zeros((0, edge_dim), dtype=torch.float32)
                return
            edge_index = torch.stack((src, dst))
            if attrs is None:
                edge_attr = torch.zeros((src.numel(), edge_dim), dtype=torch.float32)
            else:
                edge_attr = torch.tensor(attrs, dtype=torch.float32)
                if edge_attr.ndim == 1:
                    edge_attr = edge_attr.unsqueeze(0)
                if edge_attr.size(0) != src.numel():
                    edge_attr = edge_attr.expand(src.numel(), edge_dim)
            data[relation].edge_index = edge_index
            data[relation].edge_attr = edge_attr[:, :edge_dim]

        # global ↔ seed edges
        if seed_count:
            global_src = torch.zeros(seed_count, dtype=torch.long)
            seed_dst = torch.arange(seed_count, dtype=torch.long)
            attrs = [[1.0, cap.get("stage", 0.0), cap.get("risk", 0.0)] for cap in seed_capabilities]
            _set_edge(("global", "influences", "seed"), global_src, seed_dst, attrs)
            _set_edge(("seed", "reports", "global"), seed_dst, global_src, attrs)
        else:
            _set_edge(("global", "influences", "seed"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
            _set_edge(("seed", "reports", "global"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

        # global ↔ blueprint
        if blueprint_count:
            blueprint_indices = torch.arange(blueprint_count, dtype=torch.long)
            attrs = [[1.0, float(metadata.get("stage", 0.0) or 0.0), risk]] * blueprint_count
            _set_edge(("global", "annotates", "blueprint"), torch.zeros(blueprint_count, dtype=torch.long), blueprint_indices, attrs)
            _set_edge(("blueprint", "monitored_by", "global"), blueprint_indices, torch.zeros(blueprint_count, dtype=torch.long), attrs)
        else:
            _set_edge(("global", "annotates", "blueprint"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
            _set_edge(("blueprint", "monitored_by", "global"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

        # seed peer chain
        if seed_count > 1:
            src = torch.arange(seed_count - 1, dtype=torch.long)
            dst = torch.arange(1, seed_count, dtype=torch.long)
            attrs = [[0.5, 0.0, 0.0] for _ in range(seed_count - 1)]
            _set_edge(("seed", "peer", "seed"), src, dst, attrs)
        else:
            _set_edge(("seed", "peer", "seed"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

        # global ↔ layer
        layer_indices = torch.arange(layer_count, dtype=torch.long)
        layers_depth = [(idx + 1) / max(1, layer_count) for idx in range(layer_count)]
        attrs_layers = [[1.0, depth, risk] for depth in layers_depth]
        _set_edge(("global", "operates", "layer"), torch.zeros(layer_count, dtype=torch.long), layer_indices, attrs_layers)
        _set_edge(("layer", "feedback", "global"), layer_indices, torch.zeros(layer_count, dtype=torch.long), attrs_layers)

        # layer ↔ activation
        activation_indices = torch.arange(activation_count, dtype=torch.long)
        activation_ratio = [(idx + 1) / max(1, activation_count) for idx in range(activation_count)]
        if layer_count and activation_count:
            src = layer_indices.repeat_interleave(activation_count)
            dst = activation_indices.repeat(layer_count)
            attrs_la = [[layers_depth[i], activation_ratio[j], 1.0] for i in range(layer_count) for j in range(activation_count)]
            _set_edge(("layer", "activates", "activation"), src, dst, attrs_la)
            _set_edge(("activation", "affects", "layer"), dst, src, attrs_la)
        else:
            _set_edge(("layer", "activates", "activation"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
            _set_edge(("activation", "affects", "layer"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

        # activation ↔ parameter
        parameter_indices = torch.arange(parameter_count, dtype=torch.long)
        param_ratio = [(idx + 1) / max(1, parameter_count) for idx in range(parameter_count)]
        if activation_count and parameter_count:
            src = activation_indices.repeat_interleave(parameter_count)
            dst = parameter_indices.repeat(activation_count)
            attrs_ap = [[activation_ratio[i], param_ratio[j], 1.0] for i in range(activation_count) for j in range(parameter_count)]
            _set_edge(("activation", "configures", "parameter"), src, dst, attrs_ap)
            _set_edge(("parameter", "modulates", "activation"), dst, src, attrs_ap)
        else:
            _set_edge(("activation", "configures", "parameter"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
            _set_edge(("parameter", "modulates", "activation"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

        # seed ↔ parameter capability edges
        if seed_count and parameter_count:
            src = torch.arange(seed_count, dtype=torch.long).repeat_interleave(parameter_count)
            dst = torch.arange(parameter_count, dtype=torch.long).repeat(seed_count)
            attrs_sp = []
            for cap in seed_capabilities:
                blend_allowed = cap.get("blend_allowed", 0.0)
                stage_norm = cap.get("stage", 0.0)
                for _ in range(parameter_count):
                    attrs_sp.append([blend_allowed, 1.0, stage_norm])
            _set_edge(("seed", "allowed", "parameter"), src, dst, attrs_sp)
            _set_edge(("parameter", "associated", "seed"), dst, src, attrs_sp)
        else:
            _set_edge(("seed", "allowed", "parameter"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
            _set_edge(("parameter", "associated", "seed"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

        # blueprint ↔ layer
        if blueprint_count and layer_count:
            src = torch.zeros(layer_count, dtype=torch.long)
            dst = layer_indices
            attrs_bl = [[risk, layers_depth[i], 1.0] for i in range(layer_count)]
            _set_edge(("blueprint", "composes", "layer"), src, dst, attrs_bl)
            _set_edge(("layer", "belongs_to", "blueprint"), dst, src, attrs_bl)
        else:
            _set_edge(("blueprint", "composes", "layer"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
            _set_edge(("layer", "belongs_to", "blueprint"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

        # blueprint ↔ activation
        if blueprint_count and activation_count:
            src = torch.zeros(activation_count, dtype=torch.long)
            dst = activation_indices
            attrs_ba = [[risk, activation_ratio[i], 1.0] for i in range(activation_count)]
            _set_edge(("blueprint", "energizes", "activation"), src, dst, attrs_ba)
        else:
            _set_edge(("blueprint", "energizes", "activation"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

        # parameter -> blueprint
        if blueprint_count and parameter_count:
            src = parameter_indices
            dst = torch.zeros(parameter_count, dtype=torch.long)
            attrs_pb = [[param_ratio[i], risk, 1.0] for i in range(parameter_count)]
            _set_edge(("parameter", "targets", "blueprint"), src, dst, attrs_pb)
        else:
            _set_edge(("parameter", "targets", "blueprint"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

        # layer connectivity chain
        if layer_count > 1:
            src = torch.arange(layer_count - 1, dtype=torch.long)
            dst = torch.arange(1, layer_count, dtype=torch.long)
            attrs_ll = [
                [layers_depth[s], layers_depth[d], risk]
                for s, d in zip(src.tolist(), dst.tolist(), strict=False)
            ]
            _set_edge(("layer", "connects", "layer"), src, dst, attrs_ll)
        else:
            _set_edge(("layer", "connects", "layer"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

        # seed monitors layer (approximate by layer depth)
        if seed_count and layer_count:
            src_indices = []
            dst_indices = []
            attrs_sl = []
            for idx, cap in enumerate(seed_capabilities):
                depth_raw = int(round(cap.get("layer_depth", 0.0)))
                depth = max(0, min(layer_count - 1, depth_raw))
                src_indices.append(idx)
                dst_indices.append(depth)
                attrs_sl.append([
                    cap.get("stage", 0.0),
                    cap.get("risk", 0.0),
                    layers_depth[depth] if layer_count else 0.0,
                ])
            if src_indices:
                seed_tensor = torch.tensor(src_indices, dtype=torch.long)
                layer_tensor = torch.tensor(dst_indices, dtype=torch.long)
                _set_edge(("seed", "monitors", "layer"), seed_tensor, layer_tensor, attrs_sl)
                _set_edge(("layer", "monitored_by", "seed"), layer_tensor, seed_tensor, attrs_sl)
            else:
                _set_edge(("seed", "monitors", "layer"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
                _set_edge(("layer", "monitored_by", "seed"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
        else:
            _set_edge(("seed", "monitors", "layer"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
            _set_edge(("layer", "monitored_by", "seed"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

        # layer feeds activation (alias for activates)
        if layer_count and activation_count:
            src = layer_indices.repeat_interleave(activation_count)
            dst = activation_indices.repeat(layer_count)
            attrs_feed = [[layers_depth[i], activation_ratio[j], 1.0] for i in range(layer_count) for j in range(activation_count)]
            _set_edge(("layer", "feeds", "activation"), src, dst, attrs_feed)
        else:
            _set_edge(("layer", "feeds", "activation"), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))


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


__all__ = ["TamiyoGraphBuilder", "TamiyoGraphBuilderConfig"]
