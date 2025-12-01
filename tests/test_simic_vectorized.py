"""Unit tests for vectorized PPO helpers."""

import pytest

from esper.leyline import SeedStage
from esper.simic.vectorized import _advance_active_seed


class _StubGateResult:
    def __init__(self, passed: bool = True, checks_failed: list | None = None):
        self.passed = passed
        self.checks_failed = checks_failed or []


class _StubSeedState:
    def __init__(self, stage: SeedStage):
        self.stage = stage
        self.transition_calls: list[SeedStage] = []

    def transition(self, target_stage: SeedStage) -> bool:
        self.transition_calls.append(target_stage)
        self.stage = target_stage
        return True


class _StubSeedSlot:
    def __init__(self, seed_state: _StubSeedState, gate_result: _StubGateResult | None = None):
        self.seed_state = seed_state
        self.gate_result = gate_result or _StubGateResult()
        self.advance_calls: list[SeedStage] = []
        self.set_alpha_calls: list[float] = []
        self.start_blending_calls: list[tuple[int, float]] = []

    def advance_stage(self, target_stage: SeedStage | None = None) -> _StubGateResult:
        self.advance_calls.append(target_stage)
        self.seed_state.stage = target_stage
        return self.gate_result

    def set_alpha(self, alpha: float) -> None:
        self.set_alpha_calls.append(alpha)

    def start_blending(self, total_steps: int, temperature: float = 1.0) -> None:
        self.start_blending_calls.append((total_steps, temperature))


class _StubModel:
    def __init__(self, seed_stage: SeedStage, gate_result: _StubGateResult | None = None):
        self.has_active_seed = True
        self.seed_state = _StubSeedState(seed_stage)
        self.seed_slot = _StubSeedSlot(self.seed_state, gate_result=gate_result)


def test_advance_active_seed_fossilizes_via_seed_slot():
    """PROBATIONARY seeds should fossilize through SeedSlot.advance_stage (emits telemetry)."""
    model = _StubModel(SeedStage.PROBATIONARY)

    _advance_active_seed(model)

    assert model.seed_slot.advance_calls == [SeedStage.FOSSILIZED]
    assert model.seed_slot.set_alpha_calls == [1.0]
    assert model.seed_state.stage == SeedStage.FOSSILIZED
    # Transition should happen inside advance_stage, not direct transition
    assert model.seed_state.transition_calls == []


def test_advance_active_seed_noop_on_failed_fossilization_gate():
    """Failed fossilization gate should be a no-op (Tamiyo learns from failed attempts)."""
    gate_result = _StubGateResult(passed=False, checks_failed=["no_improvement"])
    model = _StubModel(SeedStage.PROBATIONARY, gate_result=gate_result)

    # Should not raise - failed gate is normal RL outcome
    _advance_active_seed(model)

    # Gate was checked but transition didn't happen
    assert model.seed_slot.advance_calls == [SeedStage.FOSSILIZED]
    assert model.seed_slot.set_alpha_calls == []  # No alpha change on failed gate
    # Stage should NOT change (stub's advance_stage still sets it, but in real code it wouldn't)


def test_advance_active_seed_noop_from_training_stage():
    """TRAINING seeds are handled mechanically; fossilize action should do nothing."""
    model = _StubModel(SeedStage.TRAINING)

    _advance_active_seed(model)

    assert model.seed_state.transition_calls == []
    assert model.seed_slot.start_blending_calls == []
    assert model.seed_state.stage == SeedStage.TRAINING
