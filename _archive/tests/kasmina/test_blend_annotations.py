from __future__ import annotations

import json

import torch
import pytest

from torch import nn

from esper.core.dependency_guard import DependencyViolationError
from esper.kasmina import KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext, sign

_SIGNING_CONTEXT = SignatureContext(secret=b"kasmina-test-secret")


def _make_seed_command(seed_id: str, blueprint_id: str) -> leyline_pb2.AdaptationCommand:
    cmd = leyline_pb2.AdaptationCommand(
        version=1,
        command_id=f"cmd-{seed_id}",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id=seed_id,
    )
    cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    cmd.seed_operation.blueprint_id = blueprint_id
    cmd.issued_at.GetCurrentTime()
    cmd.annotations["training_run_id"] = "run-test"
    cmd.annotations["mesh_host_layers"] = json.dumps(["weight", "bias"])
    return cmd


def test_confidence_mode_annotations_parsed_and_emitted() -> None:
    mgr = KasminaSeedManager(
        runtime=type("R", (), {"fetch_kernel": lambda *_: (nn.Identity(), 0.0)})(),
        signing_context=_SIGNING_CONTEXT,
    )
    mgr.register_host_model(nn.Linear(1, 1))
    cmd = _make_seed_command("seed-conf", "BP-ANN")
    cmd.annotations["blend_mode"] = "CONFIDENCE"
    cmd.annotations["blend_mode_source"] = "override"
    cmd.annotations["gate_k"] = "4.0"
    cmd.annotations["gate_tau"] = "0.2"
    cmd.annotations["alpha_lo"] = "0.1"
    cmd.annotations["alpha_hi"] = "0.9"
    # Re-sign with annotations present (remove old signature first)
    if "signature" in cmd.annotations:
        del cmd.annotations["signature"]
    cmd.annotations["signature"] = sign(cmd.SerializeToString(deterministic=True), _SIGNING_CONTEXT)
    mgr.handle_command(cmd)
    ctx = mgr.seeds().get("seed-conf")
    if ctx is None:
        mgr.finalize_step(step_index=101)
        pkt = mgr.telemetry_packets[-1]
        rejected = any(e.description == "command_rejected" for e in pkt.events)
        assert not rejected, "command verification failed unexpectedly"
    assert ctx is not None and ctx.blend_config is not None
    assert ctx.blend_config.mode == "CONFIDENCE"
    assert ctx.blend_config.gate_k == 4.0
    assert ctx.blend_config.gate_tau == 0.2
    assert ctx.blend_config.alpha_lo == 0.1
    assert ctx.blend_config.alpha_hi == 0.9
    # Telemetry event includes mode and source
    mgr.finalize_step(step_index=1)
    pkt = mgr.telemetry_packets[-1]
    ev = [e for e in pkt.events if e.description == "blend_config"][-1]
    assert ev.attributes.get("mode") == "CONFIDENCE"
    assert ev.attributes.get("source") == "override"


def test_confidence_blend_requires_logits() -> None:
    mgr = KasminaSeedManager(
        runtime=type("R", (), {"fetch_kernel": lambda *_: (nn.Identity(), 0.0)})(),
        signing_context=_SIGNING_CONTEXT,
    )
    mgr.register_host_model(nn.Linear(1, 1))
    cmd = _make_seed_command("seed-missing", "BP-ANN")
    cmd.annotations["blend_mode"] = "CONFIDENCE"
    cmd.annotations["confidence_logits_required"] = "true"
    cmd.annotations["gate_k"] = "4.0"
    cmd.annotations["gate_tau"] = "0.2"
    cmd.annotations["alpha_lo"] = "0.0"
    cmd.annotations["alpha_hi"] = "1.0"
    if "signature" in cmd.annotations:
        del cmd.annotations["signature"]
    cmd.annotations["signature"] = sign(cmd.SerializeToString(deterministic=True), _SIGNING_CONTEXT)
    with pytest.raises(DependencyViolationError) as exc:
        mgr.handle_command(cmd)
    assert "missing_confidence_logits" in str(exc.value)
    assert "seed-missing" not in mgr.seeds()
    assert not mgr.telemetry_packets


def test_confidence_blend_gate_computed_for_logits_tensor() -> None:
    mgr = KasminaSeedManager(
        runtime=type("R", (), {"fetch_kernel": lambda *_: (nn.Identity(), 0.0)})(),
        signing_context=_SIGNING_CONTEXT,
    )
    mgr.register_host_model(nn.Linear(1, 1))
    cmd = _make_seed_command("seed-logits", "BP-ANN")
    cmd.annotations["blend_mode"] = "CONFIDENCE"
    cmd.annotations["confidence_logits_required"] = "true"
    cmd.annotations["gate_k"] = "4.0"
    cmd.annotations["gate_tau"] = "0.1"
    cmd.annotations["confidence_logits"] = json.dumps([2.0, 1.0, -1.0])
    if "signature" in cmd.annotations:
        del cmd.annotations["signature"]
    cmd.annotations["signature"] = sign(cmd.SerializeToString(deterministic=True), _SIGNING_CONTEXT)
    mgr.handle_command(cmd)
    host = torch.zeros(2, 3)
    # Seed logits: batch=2, classes=3
    seed_logits = torch.tensor([[2.0, 1.0, -1.0], [0.1, 0.05, 0.0]])
    context = mgr.seeds().get("seed-logits")
    if context is None:
        mgr.finalize_step(step_index=104)
        pkt = mgr.telemetry_packets[-1]
        rejected = any(e.description == "command_rejected" for e in pkt.events)
        assert not rejected, "command verification failed unexpectedly"
        context = mgr.seeds().get("seed-logits")
    blended = mgr.blend(host, seed_logits, seed_id="seed-logits")
    assert blended.shape == host.shape
    assert context is not None
    assert not any(
        event.description == "confidence_gate_missing_logits" for event in context.pending_events
    )


def test_channel_mode_annotations_parse_alpha_vec() -> None:
    mgr = KasminaSeedManager(
        runtime=type("R", (), {"fetch_kernel": lambda *_: (nn.Identity(), 0.0)})(),
        signing_context=_SIGNING_CONTEXT,
    )
    mgr.register_host_model(nn.Linear(1, 1))
    cmd = _make_seed_command("seed-chan", "BP-ANN")
    vec = [0.2, 0.6, 1.0]
    cmd.annotations["blend_mode"] = "CHANNEL"
    cmd.annotations["blend_mode_source"] = "heuristic"
    cmd.annotations["alpha_vec"] = json.dumps(vec)
    cmd.annotations["alpha_vec_len"] = str(len(vec))
    if "signature" in cmd.annotations:
        del cmd.annotations["signature"]
    cmd.annotations["signature"] = sign(cmd.SerializeToString(deterministic=True), _SIGNING_CONTEXT)
    mgr.handle_command(cmd)
    ctx = mgr.seeds().get("seed-chan")
    if ctx is None:
        mgr.finalize_step(step_index=102)
        pkt = mgr.telemetry_packets[-1]
        rejected = any(e.description == "command_rejected" for e in pkt.events)
        assert not rejected, "command verification failed unexpectedly"
    assert ctx is not None and ctx.blend_config is not None
    assert ctx.blend_config.mode == "CHANNEL"
    mgr.finalize_step(step_index=2)
    pkt = mgr.telemetry_packets[-1]
    ev = [e for e in pkt.events if e.description == "blend_config"][-1]
    assert ev.attributes.get("alpha_vec_len") == str(len(vec))


def test_channel_mode_missing_alpha_vec_rejected() -> None:
    mgr = KasminaSeedManager(
        runtime=type("R", (), {"fetch_kernel": lambda *_: (nn.Identity(), 0.0)})(),
        signing_context=_SIGNING_CONTEXT,
    )
    mgr.register_host_model(nn.Linear(1, 1))
    cmd = _make_seed_command("seed-chan-missing", "BP-ANN")
    cmd.annotations["blend_mode"] = "CHANNEL"
    if "signature" in cmd.annotations:
        del cmd.annotations["signature"]
    cmd.annotations["signature"] = sign(cmd.SerializeToString(deterministic=True), _SIGNING_CONTEXT)
    with pytest.raises(DependencyViolationError):
        mgr.handle_command(cmd)


def test_channel_mode_rejects_oversized_alpha_vec() -> None:
    mgr = KasminaSeedManager(
        runtime=type("R", (), {"fetch_kernel": lambda *_: (nn.Identity(), 0.0)})(),
        signing_context=_SIGNING_CONTEXT,
    )
    mgr.register_host_model(nn.Linear(1, 1))
    cmd = _make_seed_command("seed-chan-oversize", "BP-ANN")
    cmd.annotations["blend_mode"] = "CHANNEL"
    vec = [0.5] * 65
    cmd.annotations["alpha_vec"] = json.dumps(vec)
    if "signature" in cmd.annotations:
        del cmd.annotations["signature"]
    cmd.annotations["signature"] = sign(cmd.SerializeToString(deterministic=True), _SIGNING_CONTEXT)
    with pytest.raises(DependencyViolationError):
        mgr.handle_command(cmd)


def test_unknown_mode_falls_back_to_convex() -> None:
    mgr = KasminaSeedManager(
        runtime=type("R", (), {"fetch_kernel": lambda *_: (nn.Identity(), 0.0)})(),
        signing_context=_SIGNING_CONTEXT,
    )
    mgr.register_host_model(nn.Linear(1, 1))
    cmd = _make_seed_command("seed-unk", "BP-ANN")
    cmd.annotations["blend_mode"] = "FOO"
    if "signature" in cmd.annotations:
        del cmd.annotations["signature"]
    cmd.annotations["signature"] = sign(cmd.SerializeToString(), _SIGNING_CONTEXT)
    mgr.handle_command(cmd)
    mgr.finalize_step(step_index=3)
    context = mgr.seeds().get("seed-unk")
    if context is not None and context.blend_config is not None:
        assert context.blend_config.mode == "CONVEX"


def test_absent_annotations_keeps_default() -> None:
    mgr = KasminaSeedManager(
        runtime=type("R", (), {"fetch_kernel": lambda *_: (nn.Identity(), 0.0)})(),
        signing_context=_SIGNING_CONTEXT,
    )
    mgr.register_host_model(nn.Linear(1, 1))
    cmd = _make_seed_command("seed-none", "BP-ANN")
    # No blend annotations added — sign deterministically
    cmd.annotations["signature"] = sign(cmd.SerializeToString(deterministic=True), _SIGNING_CONTEXT)
    mgr.handle_command(cmd)
    ctx = mgr.seeds().get("seed-none")
    # No config present → convex default applies and no blend_config event expected
    assert ctx is not None and ctx.blend_config is None
    mgr.finalize_step(step_index=4)
    pkt = mgr.telemetry_packets[-1]
    assert all(e.description != "blend_config" for e in pkt.events)
