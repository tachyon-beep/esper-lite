"""Topology-aware gradient isolation contracts for SeedSlot transitions."""

from __future__ import annotations

import pytest
import torch

from esper.kasmina.slot import SeedSlot
from esper.leyline import SeedStage
from esper.leyline.alpha import AlphaAlgorithm
from esper.tamiyo.policy.features import TaskConfig


def _task_config(topology: str) -> TaskConfig:
    return TaskConfig(
        task_type="classification",
        topology=topology,
        baseline_loss=2.3,
        target_loss=0.5,
        typical_loss_delta_std=0.1,
        max_epochs=100,
        blending_steps=4,
        train_to_blend_fraction=0.1,
    )


def _slot_ready_for_blending(
    *,
    topology: str,
    blueprint_id: str,
    alpha_algorithm: AlphaAlgorithm = AlphaAlgorithm.ADD,
    blend_algorithm_id: str = "linear",
) -> SeedSlot:
    slot = SeedSlot(
        slot_id="r0c0",
        channels=8,
        device="cpu",
        task_config=_task_config(topology),
    )
    slot.germinate(
        blueprint_id=blueprint_id,
        seed_id=f"{topology}-{blueprint_id}",
        alpha_algorithm=alpha_algorithm,
        blend_algorithm_id=blend_algorithm_id,
    )

    training_result = slot.advance_stage(SeedStage.TRAINING)
    assert training_result.passed

    slot.state.metrics.record_accuracy(0.30)
    for step in range(10):
        slot.state.metrics.record_accuracy(0.36 + step * 0.06)
    slot.state.metrics.seed_gradient_norm_ratio = 1.0
    return slot


def _nonzero_seed_grad(slot: SeedSlot) -> bool:
    assert slot.seed is not None
    return any(
        param.grad is not None and param.grad.detach().abs().sum().item() > 0.0
        for param in slot.seed.parameters()
        if param.requires_grad
    )


@pytest.mark.parametrize(
    ("alpha_algorithm", "blend_algorithm_id", "blueprint_id"),
    [
        (AlphaAlgorithm.ADD, "linear", "conv_light"),
        (AlphaAlgorithm.MULTIPLY, "linear", "norm"),
        (AlphaAlgorithm.GATE, "gated", "conv_light"),
    ],
)
def test_training_stage_ste_preserves_host_output_and_trains_seed(
    alpha_algorithm: AlphaAlgorithm,
    blend_algorithm_id: str,
    blueprint_id: str,
) -> None:
    """TRAINING keeps activations host-identical while still sending seed gradients."""
    torch.manual_seed(0)
    slot = _slot_ready_for_blending(
        topology="cnn",
        blueprint_id=blueprint_id,
        alpha_algorithm=alpha_algorithm,
        blend_algorithm_id=blend_algorithm_id,
    )

    host = torch.randn(2, 8, 6, 6, requires_grad=True)
    output = slot(host)
    torch.testing.assert_close(output, host)

    output.square().sum().backward()

    torch.testing.assert_close(host.grad, 2.0 * host.detach())
    assert _nonzero_seed_grad(slot)


def test_cnn_blending_keeps_seed_feedback_out_of_host_gradients() -> None:
    """CNN BLENDING keeps isolation: host gradients see only the direct blend path."""
    torch.manual_seed(1)
    slot = _slot_ready_for_blending(topology="cnn", blueprint_id="conv_light")

    result = slot.advance_stage(SeedStage.BLENDING)
    assert result.passed
    assert slot.isolate_gradients is True

    slot.set_alpha(0.5)
    host = torch.randn(2, 8, 6, 6, requires_grad=True)
    output = slot(host)
    output.sum().backward()

    torch.testing.assert_close(host.grad, torch.full_like(host, 0.5))
    assert _nonzero_seed_grad(slot)


def test_transformer_blending_allows_seed_feedback_into_host_gradients() -> None:
    """Transformer BLENDING disables isolation so host and seed co-adapt."""
    torch.manual_seed(2)
    slot = _slot_ready_for_blending(topology="transformer", blueprint_id="mlp_small")

    result = slot.advance_stage(SeedStage.BLENDING)
    assert result.passed
    assert slot.isolate_gradients is False

    slot.set_alpha(0.5)
    host = torch.randn(2, 4, 8, requires_grad=True)
    output = slot(host)
    output.sum().backward()

    direct_blend_only = torch.full_like(host, 0.5)
    assert not torch.allclose(host.grad, direct_blend_only)
    assert _nonzero_seed_grad(slot)
