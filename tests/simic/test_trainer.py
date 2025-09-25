from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta

import pytest
import torch

from esper.leyline import leyline_pb2
from esper.simic import FieldReportReplayBuffer, SimicTrainer, SimicTrainerConfig


def _make_report(
    loss_delta: float, outcome: int = leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS
) -> leyline_pb2.FieldReport:
    report = leyline_pb2.FieldReport(
        version=1,
        report_id="fr",
        command_id="cmd",
        training_run_id="run",
        seed_id="seed-1",
        blueprint_id="bp",
        outcome=outcome,
        observation_window_epochs=1,
        tamiyo_policy_version="policy",
    )
    report.metrics["loss_delta"] = loss_delta
    return report


def test_trainer_updates_policy_parameters() -> None:
    buffer = FieldReportReplayBuffer(capacity=32)
    for idx in range(16):
        buffer.add(_make_report(loss_delta=-0.1 * idx))

    config = SimicTrainerConfig(epochs=3, batch_size=8, hidden_size=16)
    trainer = SimicTrainer(policy=None, buffer=buffer, config=config)
    initial_state = copy.deepcopy(trainer._policy.state_dict())

    trainer.run_training()

    updated_state = trainer._policy.state_dict()
    assert any(not torch.equal(initial_state[key], updated_state[key]) for key in initial_state)
    assert trainer.last_loss != 0.0

    update = trainer.create_policy_update(
        policy_id="tamiyo-policy",
        training_run_id="run",
        policy_version="v2",
    )
    assert update.payload

    metrics = trainer.build_metrics_packet(training_run_id="run")
    metric_names = {metric.name for metric in metrics.metrics}
    assert "simic.training.loss" in metric_names
    assert "simic.training.iterations" in metric_names
    assert "simic.policy.loss" in metric_names
    assert "simic.param.loss" in metric_names
    assert "simic.validation.pass" in metric_names
    assert trainer.validation_result is not None and trainer.validation_result.passed


def test_trainer_supports_lora_enabled() -> None:
    buffer = FieldReportReplayBuffer(capacity=16)
    for _ in range(8):
        buffer.add(_make_report(loss_delta=-0.2))

    config = SimicTrainerConfig(epochs=1, batch_size=4, use_lora=True, hidden_size=8, lora_rank=2)
    trainer = SimicTrainer(policy=None, buffer=buffer, config=config)
    trainer.run_training()

    lora_params = [name for name, _ in trainer._policy.named_parameters() if "lora" in name]
    assert lora_params, "LoRA parameters should be present when use_lora=True"


def test_trainer_configures_replay_ttl() -> None:
    buffer = FieldReportReplayBuffer(capacity=4, ttl=timedelta(hours=12))
    config = SimicTrainerConfig(replay_ttl_hours=6)
    SimicTrainer(policy=None, buffer=buffer, config=config)
    assert buffer.ttl == timedelta(hours=6)


def test_trainer_blocks_policy_update_on_validation_failure() -> None:
    buffer = FieldReportReplayBuffer(capacity=16)
    for idx in range(8):
        buffer.add(_make_report(loss_delta=-0.01 * idx))

    config = SimicTrainerConfig(validation_min_reward=0.5, epochs=2, batch_size=4)
    trainer = SimicTrainer(policy=None, buffer=buffer, config=config)
    trainer.run_training()

    with pytest.raises(RuntimeError) as exc_info:
        trainer.create_policy_update(
            policy_id="tamiyo-policy",
            training_run_id="run",
            policy_version="v3",
        )
    assert "Policy update blocked" in str(exc_info.value)

    packet = trainer.build_metrics_packet(training_run_id="run")
    reasons_event = next(
        (event for event in packet.events if "validation" in event.description.lower()),
        None,
    )
    assert reasons_event is not None
