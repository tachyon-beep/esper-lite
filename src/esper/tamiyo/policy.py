"""Tamiyo hetero-GNN policy implementation."""

from __future__ import annotations

import contextlib
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

import torch
from torch import nn

from esper.leyline import leyline_pb2
from esper.simic.registry import EmbeddingRegistry, EmbeddingRegistryConfig

from .graph_builder import TamiyoGraphBuilder, TamiyoGraphBuilderConfig
from .gnn import TamiyoGNN, TamiyoGNNConfig


logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = Path("var/tamiyo")
_DEFAULT_ARCH_VERSION = "tamiyo_gnn_v1"
_DEFAULT_BLENDING_METHODS: tuple[str, ...] = ("linear", "cosine", "warmup_hold")


@dataclass(slots=True)
class TamiyoPolicyConfig:
    """Configuration envelope for the Tamiyo GNN policy."""

    registry_path: Path = _DEFAULT_REGISTRY_PATH
    seed_vocab: int = 1024
    blueprint_vocab: int = 1024
    device: str = "cpu"
    dropout: float = 0.2
    attention_heads: int = 4
    enable_compile: bool = False
    enable_autocast: bool = True
    architecture_version: str = _DEFAULT_ARCH_VERSION
    blending_methods: tuple[str, ...] = _DEFAULT_BLENDING_METHODS
    normalizer_path: Path = Path("var/tamiyo/gnn_norms.json")
    seed_registry: EmbeddingRegistry | None = None
    blueprint_registry: EmbeddingRegistry | None = None
    schedule_registry: EmbeddingRegistry | None = None
    # Extended categorical registries for parity with Simic
    layer_registry: EmbeddingRegistry | None = None
    activation_registry: EmbeddingRegistry | None = None
    optimizer_registry: EmbeddingRegistry | None = None
    hazard_registry: EmbeddingRegistry | None = None
    max_layers: int = 3
    max_activations: int = 1
    max_parameters: int = 1
    layer_feature_dim: int = 12
    activation_feature_dim: int = 8
    parameter_feature_dim: int = 10
    edge_feature_dim: int = 3
    schedule_output_dim: int = 2
    policy_classes: int = 32
    risk_classes: int = 5
    sage_hidden_dim: int = 256
    gat_hidden_dim: int = 128
    param_vector_dim: int = 4


class TamiyoPolicy(nn.Module):
    """GNN-powered Tamiyo policy.

    The schedule head emits fractions of Kasmina's blending window; values are
    clamped to [0.0, 1.0] and written back into the command alongside
    `blending_schedule_units=fraction_0_1`. See
    `docs/prototype-delta/tamiyo/README.md#blending-schedule-semantics`.
    """

    def __init__(self, config: TamiyoPolicyConfig | None = None) -> None:
        super().__init__()
        cfg = config or TamiyoPolicyConfig()
        self._config = cfg

        cfg.registry_path.mkdir(parents=True, exist_ok=True)

        self._seed_registry = cfg.seed_registry or EmbeddingRegistry(
            EmbeddingRegistryConfig(cfg.registry_path / "seed_registry.json", cfg.seed_vocab)
        )
        self._blueprint_registry = cfg.blueprint_registry or EmbeddingRegistry(
            EmbeddingRegistryConfig(cfg.registry_path / "blueprint_registry.json", cfg.blueprint_vocab)
        )
        # Schedule/enum registry for categorical stability (e.g., blending methods)
        self._schedule_registry = cfg.schedule_registry or EmbeddingRegistry(
            EmbeddingRegistryConfig(cfg.registry_path / "schedule_registry.json", max(64, len(cfg.blending_methods) + 16))
        )
        # Pre-seed known schedules for deterministic indices
        for name in cfg.blending_methods:
            try:
                _ = self._schedule_registry.get(str(name))
            except Exception:
                pass

        # Extended registries for categorical stability
        self._layer_registry = cfg.layer_registry or EmbeddingRegistry(
            EmbeddingRegistryConfig(cfg.registry_path / "layer_type_registry.json", 1024)
        )
        self._activation_registry = cfg.activation_registry or EmbeddingRegistry(
            EmbeddingRegistryConfig(cfg.registry_path / "activation_type_registry.json", 1024)
        )
        self._optimizer_registry = cfg.optimizer_registry or EmbeddingRegistry(
            EmbeddingRegistryConfig(cfg.registry_path / "optimizer_family_registry.json", 256)
        )
        self._hazard_registry = cfg.hazard_registry or EmbeddingRegistry(
            EmbeddingRegistryConfig(cfg.registry_path / "hazard_class_registry.json", 256)
        )

        builder_cfg = TamiyoGraphBuilderConfig(
            normalizer_path=cfg.normalizer_path,
            seed_registry=self._seed_registry,
            blueprint_registry=self._blueprint_registry,
            layer_type_registry=self._layer_registry,
            activation_type_registry=self._activation_registry,
            seed_vocab=cfg.seed_vocab,
            blueprint_vocab=cfg.blueprint_vocab,
            max_layers=cfg.max_layers,
            max_activations=cfg.max_activations,
            max_parameters=cfg.max_parameters,
            layer_feature_dim=cfg.layer_feature_dim,
            activation_feature_dim=cfg.activation_feature_dim,
            parameter_feature_dim=cfg.parameter_feature_dim,
            edge_feature_dim=cfg.edge_feature_dim,
            blueprint_metadata_provider=self._lookup_blueprint_metadata,
        )
        self._graph_builder = TamiyoGraphBuilder(builder_cfg)

        gnn_cfg = TamiyoGNNConfig(
            global_input_dim=builder_cfg.global_feature_dim,
            seed_input_dim=builder_cfg.seed_feature_dim,
            blueprint_input_dim=builder_cfg.blueprint_feature_dim,
            layer_input_dim=builder_cfg.layer_feature_dim,
            activation_input_dim=builder_cfg.activation_feature_dim,
            parameter_input_dim=builder_cfg.parameter_feature_dim,
            sage_hidden_dim=cfg.sage_hidden_dim,
            gat_hidden_dim=cfg.gat_hidden_dim,
            dropout=cfg.dropout,
            attention_heads=cfg.attention_heads,
            policy_classes=cfg.policy_classes,
            blending_classes=len(cfg.blending_methods),
            edge_feature_dim=cfg.edge_feature_dim,
            schedule_output_dim=cfg.schedule_output_dim,
            risk_classes=cfg.risk_classes,
            param_vector_dim=cfg.param_vector_dim,
        )
        self._gnn = TamiyoGNN(gnn_cfg)
        self._device = torch.device(cfg.device)
        self._gnn.to(self._device)

        self._autocast_enabled = cfg.enable_autocast
        self._compiled_model: nn.Module | None = None
        self._compile_enabled = False
        self._blueprint_metadata: dict[str, dict[str, float | str | bool | int]] = {}

        if cfg.enable_compile:
            try:  # pragma: no cover - depends on backend support
                self._compiled_model = torch.compile(self._gnn, dynamic=True, mode="reduce-overhead")
                self._compile_enabled = True
            except Exception as exc:  # pragma: no cover - compile fallback path
                logger.info("tamiyo_gnn_compile_disabled", extra={"reason": str(exc)})
                self._compiled_model = None

        try:
            torch.set_float32_matmul_precision("high")
            if torch.cuda.is_available():  # pragma: no cover - hardware dependent
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:  # pragma: no cover - defensive guard
            pass

        self._architecture_version = cfg.architecture_version
        self._blending_methods = cfg.blending_methods
        self._last_action: dict[str, float | str] = {}
        self._last_feature_coverage: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_action(self, packet: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        graph = self._graph_builder.build(packet)
        coverage = dict(getattr(graph, "feature_coverage", {}))
        seed_candidates = list(getattr(graph["seed"], "node_ids", []))
        seed_capabilities = list(getattr(graph["seed"], "capabilities", []))
        seed_fallback_scores = getattr(graph["seed"], "candidate_scores", None)
        blueprint_candidates = list(getattr(graph["blueprint"], "node_ids", []))
        blueprint_fallback_scores = getattr(graph["blueprint"], "candidate_scores", None)

        # Fast-path: no seed candidates → construct a pause command without
        # running the GNN to meet tight step budgets.
        if not seed_candidates:
            action_idx = 2  # pause fallback
            param_delta = 0.0
            blending_method = self._blending_methods[0] if self._blending_methods else "linear"
            schedule_values: tuple[float, float] = (0.0, 0.0)
            command = self._build_command(
                action_idx,
                param_delta,
                blending_method,
                "",
                "",
                packet,
                schedule_values,
                None,
            )
            # Populate annotations and last_action with safe defaults
            command.annotations.setdefault("policy_action", str(action_idx))
            command.annotations.setdefault("policy_param_delta", f"{param_delta:.6f}")
            command.annotations.setdefault("policy_version", self._architecture_version)
            command.annotations.setdefault("policy_value_estimate", f"{0.0:.6f}")
            command.annotations.setdefault("policy_risk_index", "0")
            command.annotations.setdefault("policy_risk_score", f"{0.0:.6f}")

            self._last_action = {
                "action": float(action_idx),
                "param_delta": param_delta,
                "policy_param_vector": (),
                "blending_method": blending_method,
                "blending_index": 0.0,
                "value_estimate": 0.0,
                "risk_index": 0.0,
                "risk_score": 0.0,
                "compile_enabled": 1.0 if self._compile_enabled else 0.0,
                "target_seed": "",
                "blueprint_id": "",
                "blending_schedule_start": schedule_values[0],
                "blending_schedule_end": schedule_values[1],
                "selected_seed_index": -1.0,
                "selected_seed_score": 0.0,
                "selected_blueprint_index": -1.0,
            }
            self._last_feature_coverage = coverage
            return command
        if self._device.type == "cuda" and hasattr(graph, "pin_memory"):
            with contextlib.suppress(Exception):
                graph = graph.pin_memory()
        non_blocking = self._device.type == "cuda"
        graph = graph.to(self._device, non_blocking=non_blocking)
        module: nn.Module = self._compiled_model or self._gnn

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._autocast_enabled and self._device.type == "cuda"
            else nullcontext()
        )

        with torch.inference_mode():
            with autocast_ctx:
                try:
                    outputs = module(graph)
                except Exception as exc:  # pragma: no cover - backend specific
                    if self._compiled_model is not None:
                        logger.info("tamiyo_gnn_compile_runtime_failure", extra={"reason": str(exc)})
                        self._compiled_model = None
                        self._compile_enabled = False
                        module = self._gnn
                        outputs = module(graph)
                    else:
                        raise

        policy_logits = outputs["policy_logits"].softmax(dim=-1)
        action_idx = int(policy_logits.argmax(dim=-1))
        param_vector = outputs["policy_params"].detach().cpu().view(-1)
        param_delta = float(outputs["param_delta"].squeeze())

        seed_scores = outputs.get("seed_scores")
        blueprint_scores = outputs.get("blueprint_scores")
        breaker_logits = outputs.get("breaker_logits")
        blending_logits = outputs["blending_logits"].softmax(dim=-1)
        blending_idx = int(blending_logits.argmax(dim=-1))
        blending_method = self._blending_methods[blending_idx % len(self._blending_methods)]

        value_estimate = float(outputs["value"].squeeze())
        risk_logits = outputs["risk_logits"].softmax(dim=-1)
        risk_idx = int(risk_logits.argmax(dim=-1))
        risk_score = float(risk_logits[0, risk_idx])
        schedule_params = outputs.get("schedule_params")
        schedule_values: tuple[float, float] = (0.0, 0.0)
        if schedule_params is not None:
            flattened = schedule_params.squeeze()
            if flattened.ndim == 1 and flattened.numel() >= 2:
                raw_start = float(flattened[0])
                raw_end = float(flattened[1])
                norm_start = 0.5 * (raw_start + 1.0)
                norm_end = 0.5 * (raw_end + 1.0)
                norm_start = max(0.0, min(1.0, norm_start))
                norm_end = max(0.0, min(1.0, norm_end))
                if norm_end < norm_start:
                    norm_start, norm_end = norm_end, norm_start
                schedule_values = (norm_start, norm_end)

        selected_seed, selected_seed_idx, selected_seed_score = self._select_seed(
            seed_scores,
            seed_fallback_scores,
            seed_candidates,
            seed_capabilities,
        )
        selected_blueprint, selected_blueprint_idx = self._select_blueprint(
            blueprint_scores,
            blueprint_fallback_scores,
            blueprint_candidates,
        )

        command = self._build_command(
            action_idx,
            param_delta,
            blending_method,
            selected_seed,
            selected_blueprint,
            packet,
            schedule_values,
            breaker_logits,
        )

        command.annotations["policy_action"] = str(action_idx)
        command.annotations["policy_param_delta"] = f"{param_delta:.6f}"
        command.annotations["policy_version"] = self._architecture_version
        command.annotations["blending_method"] = blending_method
        command.annotations["policy_value_estimate"] = f"{value_estimate:.6f}"
        command.annotations["policy_risk_index"] = str(risk_idx)
        command.annotations["policy_risk_score"] = f"{risk_score:.6f}"
        # Attach feature coverage summary for downstream degraded-input reactions
        if coverage:
            avg_cov = float(sum(coverage.values()) / max(1, len(coverage)))
            command.annotations.setdefault("feature_coverage", f"{avg_cov:.3f}")
        if selected_seed:
            command.annotations.setdefault("selected_seed", selected_seed)
        if selected_blueprint:
            command.annotations.setdefault("selected_blueprint", selected_blueprint)

        self._last_action = {
            "action": float(action_idx),
            "param_delta": param_delta,
            "policy_param_vector": tuple(float(value) for value in param_vector.tolist()),
            "blending_method": blending_method,
            "blending_index": float(blending_idx),
            "value_estimate": value_estimate,
            "risk_index": float(risk_idx),
            "risk_score": risk_score,
            "compile_enabled": 1.0 if self._compile_enabled else 0.0,
            "target_seed": command.target_seed_id,
            "blueprint_id": command.seed_operation.blueprint_id if command.HasField("seed_operation") else "",
            "blending_schedule_start": schedule_values[0],
            "blending_schedule_end": schedule_values[1],
            "selected_seed_index": float(selected_seed_idx),
            "selected_seed_score": selected_seed_score,
            "selected_blueprint_index": float(selected_blueprint_idx),
        }
        self._last_feature_coverage = coverage
        return command

    @property
    def last_action(self) -> dict[str, float | str]:
        return dict(self._last_action)

    @property
    def architecture_version(self) -> str:
        return self._architecture_version

    @property
    def compile_enabled(self) -> bool:
        return self._compile_enabled

    @property
    def feature_coverage(self) -> dict[str, float]:
        return dict(self._last_feature_coverage)

    def update_blueprint_metadata(
        self, metadata: Mapping[str, Mapping[str, float | str | bool | int]]
    ) -> None:
        for key, value in metadata.items():
            self._blueprint_metadata[key] = dict(value)

    def _lookup_blueprint_metadata(self, blueprint_id: str) -> Mapping[str, float | str | bool | int]:
        return self._blueprint_metadata.get(blueprint_id, {})

    @staticmethod
    def encode_tags(packet: leyline_pb2.SystemStatePacket) -> dict[str, str]:
        return {
            "epoch": str(packet.current_epoch),
            "run_id": packet.training_run_id,
            "packet_id": packet.packet_id,
        }

    def validate_state_dict(self, state_dict: Mapping[str, object]) -> None:
        clean_state, metadata = self._split_state_dict(state_dict)
        version = str(metadata.get("architecture_version", "")) if metadata else ""
        if version and version != self._architecture_version:
            msg = (
                "Tamiyo policy checkpoint version mismatch: "
                f"expected {self._architecture_version}, got {version}"
            )
            raise ValueError(msg)

        # Registry parity check (optional metadata)
        registries_meta = metadata.get("registries", {}) if metadata else {}
        if isinstance(registries_meta, dict) and registries_meta:
            # Compute digests from current on-disk state to catch out-of-band changes
            def _fresh_digest(reg: EmbeddingRegistry | None) -> str:
                try:
                    cfg = getattr(reg, "_config", None)
                    path = getattr(cfg, "path", None)
                    if path is None:
                        return getattr(reg, "digest", lambda: "")()
                    from esper.simic.registry import EmbeddingRegistry as _ER, EmbeddingRegistryConfig as _ERC
                    return _ER(_ERC(path, getattr(cfg, "max_size", 1024))).digest()
                except Exception:
                    return getattr(reg, "digest", lambda: "")()

            expected = {
                "layer_type": _fresh_digest(self._layer_registry),
                "activation_type": _fresh_digest(self._activation_registry),
                "optimizer_family": _fresh_digest(self._optimizer_registry),
                "hazard_class": _fresh_digest(self._hazard_registry),
            }
            mismatches = [
                name
                for name, dig in registries_meta.items()
                if expected.get(name, "") and dig != expected.get(name, "")
            ]
            if mismatches:
                raise ValueError(f"Tamiyo registry mismatch: {mismatches}")

        reference = super().state_dict()
        missing = [key for key in reference.keys() if key not in clean_state]
        mismatched: list[str] = []
        for key, tensor in reference.items():
            candidate = clean_state.get(key)
            if candidate is None:
                continue
            try:
                if isinstance(candidate, torch.Tensor) and isinstance(tensor, torch.Tensor):
                    # Some lazy tensors may not have a defined shape until first forward; skip in that case
                    cand_shape = getattr(candidate, "shape", None)
                    ref_shape = getattr(tensor, "shape", None)
                    if cand_shape is not None and ref_shape is not None and cand_shape != ref_shape:
                        mismatched.append(key)
            except RuntimeError:
                # Skip shape checks for uninitialized parameters/buffers
                continue
        if missing or mismatched:
            raise ValueError(
                "Tamiyo policy checkpoint incompatible; missing="
                f"{missing}, mismatched={mismatched}"
            )

    def state_dict(self, *args: object, **kwargs: object) -> dict:  # type: ignore[override]
        payload = super().state_dict(*args, **kwargs)
        payload["_metadata"] = {
            "architecture_version": self._architecture_version,
            "registries": {
                "layer_type": getattr(self._layer_registry, "digest", lambda: "")(),
                "activation_type": getattr(self._activation_registry, "digest", lambda: "")(),
                "optimizer_family": getattr(self._optimizer_registry, "digest", lambda: "")(),
                "hazard_class": getattr(self._hazard_registry, "digest", lambda: "")(),
            },
        }
        return payload

    def load_state_dict(self, state_dict: Mapping[str, object], strict: bool = True) -> object:  # type: ignore[override]
        self.validate_state_dict(state_dict)
        clean_state, _ = self._split_state_dict(state_dict)
        return super().load_state_dict(clean_state, strict=strict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _select_seed(
        self,
        seed_scores: torch.Tensor | None,
        fallback_scores: torch.Tensor | None,
        candidates: list[str],
        capabilities: list[dict[str, float]],
    ) -> tuple[str, int, float]:
        if not candidates:
            return "", -1, 0.0
        score_source: torch.Tensor | None = None
        if seed_scores is not None and seed_scores.numel() >= len(candidates):
            score_source = seed_scores.detach().cpu()
        elif fallback_scores is not None and fallback_scores.numel() >= len(candidates):
            score_source = fallback_scores.detach().cpu()

        valid_indices = [idx for idx, cid in enumerate(candidates) if cid]
        if not valid_indices:
            return "", -1, 0.0

        best_index = valid_indices[0]
        best_score = float("-inf")

        if score_source is not None:
            best_index = max(valid_indices, key=lambda idx: float(score_source[idx]))
            best_score = float(score_source[best_index])
        elif capabilities:
            for idx in valid_indices:
                caps = capabilities[idx] if idx < len(capabilities) else {}
                score = caps.get("blend_allowed", 0.0) * 2.0 + (1.0 - caps.get("risk", 0.0))
                if score > best_score:
                    best_score = score
                    best_index = idx

        if not math.isfinite(best_score):
            best_score = 0.0
        selected_seed = candidates[best_index]
        return selected_seed, int(best_index), best_score

    def _select_blueprint(
        self,
        blueprint_scores: torch.Tensor | None,
        fallback_scores: torch.Tensor | None,
        candidates: list[str],
    ) -> tuple[str, int]:
        if not candidates:
            return "", -1
        score_source: torch.Tensor | None = None
        if blueprint_scores is not None and blueprint_scores.numel() >= len(candidates):
            score_source = blueprint_scores.detach().cpu()
        elif fallback_scores is not None and fallback_scores.numel() >= len(candidates):
            score_source = fallback_scores.detach().cpu()

        valid_indices = [idx for idx, cid in enumerate(candidates) if cid]
        if not valid_indices:
            return "", -1

        if score_source is not None:
            idx = max(valid_indices, key=lambda i: float(score_source[i]))
            return candidates[idx], idx

        return candidates[valid_indices[0]], valid_indices[0]

    def _build_command(
        self,
        action_idx: int,
        param_delta: float,
        blending_method: str,
        selected_seed: str,
        selected_blueprint: str,
        packet: leyline_pb2.SystemStatePacket,
        schedule_values: tuple[float, float],
        breaker_logits: torch.Tensor | None,
    ) -> leyline_pb2.AdaptationCommand:
        command = leyline_pb2.AdaptationCommand(version=1, issued_by="tamiyo")
        command.issued_at.GetCurrentTime()

        if action_idx == 0 and not selected_seed:
            action_idx = 2  # treat as pause fallback

        if action_idx == 0:  # seed graft
            command.command_type = leyline_pb2.COMMAND_SEED
            command.target_seed_id = selected_seed or "seed-1"
            command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
            command.seed_operation.blueprint_id = selected_blueprint or (packet.packet_id or packet.training_run_id or "bp-demo")
            command.seed_operation.parameters["alpha"] = max(0.0, 0.1 + param_delta)
            command.seed_operation.parameters["blending_method_index"] = float(self._blending_methods.index(blending_method))
            command.seed_operation.parameters["blending_schedule_start"] = schedule_values[0]
            command.seed_operation.parameters["blending_schedule_end"] = schedule_values[1]
        elif action_idx == 1:  # optimizer tweak
            command.command_type = leyline_pb2.COMMAND_OPTIMIZER
            command.optimizer_adjustment.optimizer_id = "sgd"
            command.optimizer_adjustment.hyperparameters["lr_delta"] = param_delta
        elif action_idx == 3:  # breaker open
            command.command_type = leyline_pb2.COMMAND_CIRCUIT_BREAKER
            command.circuit_breaker.desired_state = leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_OPEN
            command.circuit_breaker.rationale = "policy_action"
        elif action_idx == 4:  # breaker close
            command.command_type = leyline_pb2.COMMAND_CIRCUIT_BREAKER
            command.circuit_breaker.desired_state = leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_CLOSED
            command.circuit_breaker.rationale = "policy_action"
        else:  # pause fallback
            command.command_type = leyline_pb2.COMMAND_PAUSE
            command.annotations["reason"] = "policy"
            if not selected_seed:
                command.annotations.setdefault("risk_reason", "no_seed_candidates")

        if breaker_logits is not None:
            logits = breaker_logits.detach().cpu().reshape(-1)
            if logits.numel() >= 2:
                open_score = float(torch.sigmoid(logits[0]))
                close_score = float(torch.sigmoid(logits[1]))
                command.annotations.setdefault("breaker_open_score", f"{open_score:.4f}")
                command.annotations.setdefault("breaker_close_score", f"{close_score:.4f}")

        command.annotations.setdefault("blending_schedule_units", "fraction_0_1")
        command.annotations.setdefault("blending_schedule_start", f"{schedule_values[0]:.4f}")
        command.annotations.setdefault("blending_schedule_end", f"{schedule_values[1]:.4f}")

        return command

    @staticmethod
    def _split_state_dict(
        state_dict: Mapping[str, object]
    ) -> tuple[Mapping[str, object], dict[str, object]]:
        if isinstance(state_dict, dict) and "_metadata" in state_dict:
            metadata = state_dict["_metadata"]
            cleaned = state_dict.copy() if hasattr(state_dict, "copy") else dict(state_dict)
            cleaned.pop("_metadata", None)
            meta_dict = metadata if isinstance(metadata, dict) else {}
            return cleaned, dict(meta_dict)
        return state_dict, {}


__all__ = ["TamiyoPolicy", "TamiyoPolicyConfig"]
