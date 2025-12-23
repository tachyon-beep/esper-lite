"""Tests for DDP gate synchronization in SeedSlot.

These tests verify the rank-0 broadcast mechanism for gate decisions,
which prevents BUG-030 (architecture divergence from unsynced lifecycle transitions).
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import torch

from esper.kasmina.slot import SeedSlot
from esper.leyline import GateLevel, GateResult


@pytest.fixture
def slot() -> SeedSlot:
    """Create a minimal SeedSlot for testing."""
    return SeedSlot(
        slot_id="test_slot",
        channels=64,
        device="cpu",
    )


class TestSyncGateDecisionNoDDP:
    """Test _sync_gate_decision when DDP is not available/initialized."""

    def test_passthrough_when_dist_not_available(self, slot: SeedSlot) -> None:
        """When torch.distributed is not available, return gate_result unchanged."""
        gate_result = GateResult(
            gate=GateLevel.G1,
            passed=True,
            score=0.8,
            checks_passed=["gradient_stability"],
            checks_failed=[],
            message="G1 passed",
        )

        with patch("torch.distributed.is_available", return_value=False):
            result = slot._sync_gate_decision(gate_result)

        assert result is gate_result  # Same object, not a copy

    def test_passthrough_when_dist_not_initialized(self, slot: SeedSlot) -> None:
        """When torch.distributed is available but not initialized, return unchanged."""
        gate_result = GateResult(
            gate=GateLevel.G2,
            passed=False,
            score=0.3,
            checks_passed=[],
            checks_failed=["insufficient_improvement"],
            message="G2 failed",
        )

        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=False):
            result = slot._sync_gate_decision(gate_result)

        assert result is gate_result


class TestSyncGateDecisionWithDDP:
    """Test _sync_gate_decision with mocked DDP operations."""

    def test_rank0_result_broadcast_to_all_ranks(self, slot: SeedSlot) -> None:
        """Rank 0's gate result should be broadcast to all ranks."""
        rank0_result = GateResult(
            gate=GateLevel.G1,
            passed=True,
            score=0.9,
            checks_passed=["gradient_stability", "loss_decreasing"],
            checks_failed=[],
            message="G1 passed on rank 0",
        )

        # Simulate what broadcast_object_list does: puts rank 0's data in the list
        def mock_broadcast(object_list: list, src: int) -> None:
            # Simulate: rank 0 data is already in list, other ranks receive it
            pass  # In real DDP, this would populate object_list[0] on other ranks

        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_rank", return_value=0), \
             patch("torch.distributed.broadcast_object_list", side_effect=mock_broadcast):

            result = slot._sync_gate_decision(rank0_result)

        # Rank 0 should get back a reconstructed GateResult with same values
        assert result.gate == GateLevel.G1
        assert result.passed is True
        assert result.score == 0.9
        assert result.checks_passed == ["gradient_stability", "loss_decreasing"]
        assert result.checks_failed == []
        assert result.message == "G1 passed on rank 0"

    def test_non_rank0_receives_broadcast_result(self, slot: SeedSlot) -> None:
        """Non-rank-0 nodes should receive rank 0's decision, ignoring local result."""
        local_result = GateResult(
            gate=GateLevel.G1,
            passed=False,  # Local evaluation failed
            score=0.2,
            checks_passed=[],
            checks_failed=["gradient_unstable"],
            message="G1 failed locally",
        )

        # Rank 0's data that will be "broadcast"
        rank0_data = {
            "gate": GateLevel.G1.value,
            "passed": True,  # Rank 0 passed
            "score": 0.9,
            "checks_passed": ["gradient_stability"],
            "checks_failed": [],
            "message": "G1 passed on rank 0",
        }

        def mock_broadcast(object_list: list, src: int) -> None:
            # Simulate receiving rank 0's data on non-rank-0 node
            object_list[0] = rank0_data

        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_rank", return_value=1), \
             patch("torch.distributed.broadcast_object_list", side_effect=mock_broadcast):

            result = slot._sync_gate_decision(local_result)

        # Should use rank 0's values, not local
        assert result.passed is True  # Rank 0's decision
        assert result.score == 0.9
        assert "gradient_stability" in result.checks_passed
        # Should have divergence note since local was False but synced is True
        assert "local=False" in result.message
        assert "synced from rank0=True" in result.message

    def test_no_divergence_note_when_results_agree(self, slot: SeedSlot) -> None:
        """No divergence note when local and synced results agree."""
        local_result = GateResult(
            gate=GateLevel.G1,
            passed=True,  # Same as rank 0
            score=0.85,
            checks_passed=["gradient_stability"],
            checks_failed=[],
            message="G1 passed locally",
        )

        rank0_data = {
            "gate": GateLevel.G1.value,
            "passed": True,
            "score": 0.9,
            "checks_passed": ["gradient_stability"],
            "checks_failed": [],
            "message": "G1 passed on rank 0",
        }

        def mock_broadcast(object_list: list, src: int) -> None:
            object_list[0] = rank0_data

        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_rank", return_value=1), \
             patch("torch.distributed.broadcast_object_list", side_effect=mock_broadcast):

            result = slot._sync_gate_decision(local_result)

        # No divergence note when both agree on passed=True
        assert "local=" not in result.message
        assert "synced from rank0" not in result.message


class TestAdvanceStageWithDDPSync:
    """Test that advance_stage() calls _sync_gate_decision()."""

    def test_advance_stage_calls_sync_gate_decision(self, slot: SeedSlot) -> None:
        """advance_stage() should call _sync_gate_decision() after gate check."""
        # Germinate a seed first using blueprint_id
        slot.germinate(blueprint_id="norm")

        # Track if _sync_gate_decision was called
        original_sync = slot._sync_gate_decision
        sync_called = False
        sync_input = None

        def track_sync(gate_result: GateResult) -> GateResult:
            nonlocal sync_called, sync_input
            sync_called = True
            sync_input = gate_result
            return original_sync(gate_result)

        slot._sync_gate_decision = track_sync

        # Try to advance (will fail gate but should still call sync)
        slot.advance_stage()

        assert sync_called, "_sync_gate_decision was not called by advance_stage()"
        assert sync_input is not None
        assert isinstance(sync_input, GateResult)
