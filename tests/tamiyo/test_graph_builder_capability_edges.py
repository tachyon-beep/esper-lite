from __future__ import annotations

import torch

from esper.leyline import leyline_pb2
from esper.tamiyo import TamiyoGraphBuilder, TamiyoGraphBuilderConfig


def _packet_with_seeds_two_params(*, allow_a: bool, allow_b: bool) -> tuple[leyline_pb2.SystemStatePacket, dict]:
    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-cap")
    seed_a = packet.seed_states.add()
    seed_a.seed_id = "seed-A"
    seed_a.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_BLENDING if allow_a else leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
    seed_a.learning_rate = 0.01
    seed_b = packet.seed_states.add()
    seed_b.seed_id = "seed-B"
    seed_b.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_BLENDING if allow_b else leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
    seed_b.learning_rate = 0.02
    # Blueprint extras: two parameters
    metadata = {
        "graph": {
            "parameters": [
                {"name": "alpha", "min": 0.0, "max": 1.0, "span": 1.0, "default": 0.5},
                {"name": "beta", "min": -0.5, "max": 0.5, "span": 1.0, "default": 0.0},
            ]
        }
    }
    return packet, metadata


def test_capability_edges_masked_when_no_allowances() -> None:
    builder = TamiyoGraphBuilder(TamiyoGraphBuilderConfig())
    packet, metadata = _packet_with_seeds_two_params(allow_a=False, allow_b=False)
    # No global allowed methods; rely solely on stages (both not blending)
    def provider(_bp: str) -> dict:  # pragma: no cover - simple mapping
        return metadata
    builder._metadata_provider = provider  # type: ignore[attr-defined]
    graph = builder.build(packet)
    edge = graph[("seed", "allowed", "parameter")]
    assert edge.edge_index.shape[1] == 0
    assert edge.edge_attr.shape[0] == 0
    coverage = getattr(graph, "feature_coverage", {})
    assert coverage.get("edges.seed_param_allowed", 0.0) == 0.0


def test_capability_edges_present_only_for_allowed() -> None:
    builder = TamiyoGraphBuilder(TamiyoGraphBuilderConfig())
    # Only seed-A allowed (BLENDING)
    packet, metadata = _packet_with_seeds_two_params(allow_a=True, allow_b=False)
    def provider(_bp: str) -> dict:
        return metadata
    builder._metadata_provider = provider  # type: ignore[attr-defined]
    graph = builder.build(packet)
    edge = graph[("seed", "allowed", "parameter")]
    # Two parameters, one allowed seed -> 2 edges
    assert edge.edge_index.shape[1] == 2
    assert edge.edge_attr.shape[0] == 2
    # Deterministic reverse edges mirror count
    rev = graph[("parameter", "associated", "seed")]
    assert rev.edge_index.shape[1] == 2
    # Determinism: rebuild and compare
    graph2 = builder.build(packet)
    edge2 = graph2[("seed", "allowed", "parameter")]
    assert torch.equal(edge.edge_index, edge2.edge_index)


def test_capability_edges_filter_by_per_seed_allowed_names_and_capabilities_mask() -> None:
    builder = TamiyoGraphBuilder(TamiyoGraphBuilderConfig())
    packet, metadata = _packet_with_seeds_two_params(allow_a=True, allow_b=True)
    # Restrict seed-A to only 'alpha'; seed-B to none, via by-seed-id names
    metadata["graph"]["capabilities"] = {
        "allowed_parameters_by_seed_id": {
            "seed-A": ["alpha"],
            "seed-B": [],
        }
    }
    def provider(_bp: str) -> dict:
        return metadata
    builder._metadata_provider = provider  # type: ignore[attr-defined]
    graph = builder.build(packet)
    edge = graph[("seed", "allowed", "parameter")]
    # Only one edge: seed-A -> alpha
    assert edge.edge_index.shape[1] == 1
    # Capabilities include per-parameter mask
    caps = list(getattr(graph["seed"], "capabilities", []))
    cap_a = caps[0]
    cap_b = caps[1]
    assert cap_a.get("allowed_param_alpha", 0.0) == 1.0
    assert cap_a.get("allowed_param_beta", 0.0) == 0.0
    assert cap_b.get("allowed_param_alpha", 0.0) == 0.0
    assert cap_b.get("allowed_param_beta", 0.0) == 0.0


def test_capability_edges_filter_by_seed_index() -> None:
    builder = TamiyoGraphBuilder(TamiyoGraphBuilderConfig())
    packet, metadata = _packet_with_seeds_two_params(allow_a=True, allow_b=True)
    # Restrict seed index 1 (seed-B) to only parameter index 1 (beta); seed-A defaults to all
    metadata["graph"]["capabilities"] = {
        "allowed_parameters_by_seed_index": {
            "1": [1]
        }
    }
    def provider(_bp: str) -> dict:
        return metadata
    builder._metadata_provider = provider  # type: ignore[attr-defined]
    graph = builder.build(packet)
    edge = graph[("seed", "allowed", "parameter")]
    # Seed-A -> alpha,beta (2 edges); Seed-B -> beta (1 edge) → total 3
    assert edge.edge_index.shape[1] == 3
