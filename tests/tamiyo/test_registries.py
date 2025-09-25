from __future__ import annotations

from pathlib import Path

import pytest

from esper.leyline import leyline_pb2
from esper.simic.registry import EmbeddingRegistry, EmbeddingRegistryConfig
from esper.tamiyo import TamiyoPolicy, TamiyoPolicyConfig
from esper.tamiyo.graph_builder import TamiyoGraphBuilder, TamiyoGraphBuilderConfig


def test_embedding_registry_stability(tmp_path: Path) -> None:
    path = tmp_path / "act.json"
    reg = EmbeddingRegistry(EmbeddingRegistryConfig(path, 16))
    a = reg.get("relu")
    b = reg.get("gelu")
    reg2 = EmbeddingRegistry(EmbeddingRegistryConfig(path, 16))
    assert reg2.get("relu") == a
    assert reg2.get("gelu") == b


def test_policy_checkpoint_registry_metadata(tmp_path: Path) -> None:
    cfg = TamiyoPolicyConfig(registry_path=tmp_path)
    policy = TamiyoPolicy(cfg)
    sd = policy.state_dict()
    meta = sd.get("_metadata", {})
    assert "registries" in meta
    regs = meta["registries"]
    assert all(
        k in regs for k in ("layer_type", "activation_type", "optimizer_family", "hazard_class")
    )


def test_policy_rejects_mismatched_registry(tmp_path: Path) -> None:
    cfg = TamiyoPolicyConfig(registry_path=tmp_path)
    policy = TamiyoPolicy(cfg)
    sd = policy.state_dict()
    # Mutate activation registry file to change digest
    act_path = tmp_path / "activation_type_registry.json"
    act_path.write_text('{"custom": 999}', encoding="utf-8")
    with pytest.raises(ValueError):
        policy.validate_state_dict(sd)


def test_optimizer_and_hazard_registry_integration(tmp_path: Path) -> None:
    # Build builder with extended dims to include optimizer family and hazard class
    cfg = TamiyoGraphBuilderConfig(
        normalizer_path=tmp_path / "norms.json",
        layer_type_registry=EmbeddingRegistry(EmbeddingRegistryConfig(tmp_path / "layer.json", 64)),
        activation_type_registry=EmbeddingRegistry(
            EmbeddingRegistryConfig(tmp_path / "act.json", 64)
        ),
        optimizer_family_registry=EmbeddingRegistry(
            EmbeddingRegistryConfig(tmp_path / "opt.json", 16)
        ),
        hazard_class_registry=EmbeddingRegistry(EmbeddingRegistryConfig(tmp_path / "haz.json", 16)),
        global_feature_dim=18,
        blueprint_feature_dim=16,
    )
    builder = TamiyoGraphBuilder(cfg)
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-opt")
    # A seed to ensure non-empty graph
    s = packet.seed_states.add()
    s.seed_id = "s1"

    # Provide metadata with optimizer family and hazard class
    def provider(_bp: str) -> dict:
        return {"optimizer_family": "sgd", "hazard_class": "critical"}

    builder._metadata_provider = provider  # type: ignore[attr-defined]
    graph = builder.build(packet)
    gv = graph["global"].x[0]
    bv = graph["blueprint"].x[0]
    # Optimizer index should be present at position 16 (0-based) with extended dim
    assert float(gv[16]) > 0.0
    # Hazard class index should be present at position 14 (0-based) with extended dim
    assert float(bv[14]) > 0.0
