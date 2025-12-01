"""Tests for sanity check logging."""

import logging
import pytest
import torch


def test_reward_magnitude_warning(caplog):
    """Large reward magnitude triggers warning."""
    from esper.simic.sanity import check_reward_magnitude

    caplog.set_level(logging.WARNING)

    check_reward_magnitude(2.5, epoch=1, max_epochs=25)
    assert not caplog.records

    check_reward_magnitude(15.0, epoch=1, max_epochs=25)
    assert any("reward magnitude" in r.message.lower() for r in caplog.records)


def test_params_ratio_logging():
    """Params ratio is logged for debugging."""
    from esper.simic.sanity import log_params_ratio

    log_params_ratio(total_params=50000, host_params=100000, epoch=5)


def test_shape_guard_assertion():
    """Shape guard raises on mismatch."""
    from esper.simic.sanity import assert_slot_shape

    x = torch.randn(2, 64, 8, 8)
    assert_slot_shape(x, expected_dim=64, topology="cnn")

    with pytest.raises(AssertionError):
        assert_slot_shape(x, expected_dim=128, topology="cnn")
