from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from esper.core.dependency_guard import DependencyViolationError
from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext, sign
from esper.tolaria import TamiyoClient, TolariaTrainer, TrainingLoopConfig
from esper.kasmina import KasminaSeedManager

_SIGNING_CONTEXT = SignatureContext(secret=b"tamiyo-kasmina-annotations")

pytestmark = pytest.mark.soak


def _make_dataset() -> DataLoader:
    data = TensorDataset(torch.randn(2, 4), torch.randint(0, 2, (2,)))
    return DataLoader(data, batch_size=2)


class _RuntimeStub:
    def fetch_kernel(self, blueprint_id: str) -> tuple[nn.Module, float]:
        return nn.Identity(), 0.0


def _sign(command: leyline_pb2.AdaptationCommand) -> None:
    command.issued_at.GetCurrentTime()
    command.annotations["signature"] = sign(command.SerializeToString(deterministic=True), _SIGNING_CONTEXT)


class _TamiyoConfidenceMissing(TamiyoClient):
    def evaluate_epoch(self, _state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        cmd = leyline_pb2.AdaptationCommand(
            version=1,
            command_id="cmd-missing-confidence",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-confidence",
        )
        cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        cmd.seed_operation.blueprint_id = "BP-CONF"
        cmd.annotations["training_run_id"] = "tamiyo-run"
        cmd.annotations["mesh_host_layers"] = json.dumps(["weight", "bias"])
        cmd.annotations["blend_mode"] = "CONFIDENCE"
        cmd.annotations["confidence_logits_required"] = "true"
        _sign(cmd)
        return cmd


class _TamiyoChannelMissing(TamiyoClient):
    def evaluate_epoch(self, _state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        cmd = leyline_pb2.AdaptationCommand(
            version=1,
            command_id="cmd-missing-channel",
            command_type=leyline_pb2.COMMAND_SEED,
            target_seed_id="seed-channel",
        )
        cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
        cmd.seed_operation.blueprint_id = "BP-CHAN"
        cmd.annotations["training_run_id"] = "tamiyo-run"
        cmd.annotations["mesh_host_layers"] = json.dumps(["weight", "bias"])
        cmd.annotations["blend_mode"] = "CHANNEL"
        _sign(cmd)
        return cmd


def _build_trainer(tamiyo: TamiyoClient) -> TolariaTrainer:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    kasmina = KasminaSeedManager(
        runtime=_RuntimeStub(),
        signing_context=_SIGNING_CONTEXT,
    )
    kasmina.register_host_model(model)
    kasmina.register_optimizer(optimizer)
    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=_make_dataset(),
        tamiyo=tamiyo,
        kasmina=kasmina,
        config=TrainingLoopConfig(
            max_epochs=1,
            gradient_accumulation_steps=1,
            enable_graphs=False,
            tamiyo_timeout_s=0.0,
        ),
    )
    return trainer


def test_tamiyo_confidence_missing_logits_rejected() -> None:
    trainer = _build_trainer(_TamiyoConfidenceMissing())
    command = trainer._call_tamiyo(state=leyline_pb2.SystemStatePacket(), use_step=False)
    with pytest.raises(DependencyViolationError) as exc:
        trainer._apply_kasmina_command(command)
    trainer.close()
    assert "missing_confidence_logits" in str(exc.value)
    assert "seed-confidence" not in trainer._kasmina.seeds()


def test_tamiyo_channel_missing_alpha_rejected() -> None:
    trainer = _build_trainer(_TamiyoChannelMissing())
    command = trainer._call_tamiyo(state=leyline_pb2.SystemStatePacket(), use_step=False)
    with pytest.raises(DependencyViolationError) as exc:
        trainer._apply_kasmina_command(command)
    trainer.close()
    assert "channel blend" in str(exc.value)
    context = trainer._kasmina.seeds().get("seed-channel")
    assert context is not None
    assert context.kernel is None
