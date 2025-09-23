from __future__ import annotations

from pathlib import Path

import pytest

from esper.simic.registry import EmbeddingRegistry, EmbeddingRegistryConfig
from esper.tamiyo import TamiyoPolicy, TamiyoPolicyConfig


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
    assert all(k in regs for k in ("layer_type", "activation_type", "optimizer_family", "hazard_class"))


def test_policy_rejects_mismatched_registry(tmp_path: Path) -> None:
    cfg = TamiyoPolicyConfig(registry_path=tmp_path)
    policy = TamiyoPolicy(cfg)
    sd = policy.state_dict()
    # Mutate activation registry file to change digest
    act_path = tmp_path / "activation_type_registry.json"
    act_path.write_text('{"custom": 999}', encoding="utf-8")
    with pytest.raises(ValueError):
        policy.validate_state_dict(sd)
