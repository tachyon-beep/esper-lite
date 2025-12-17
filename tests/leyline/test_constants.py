"""Tests for leyline constants."""

import torch
from esper.leyline import MASKED_LOGIT_VALUE


def test_masked_logit_value_properties():
    """MASKED_LOGIT_VALUE should be a large negative value for action masking."""
    # Should be negative (to suppress probability)
    assert MASKED_LOGIT_VALUE < 0
    # Should be large enough to effectively zero out probability
    assert MASKED_LOGIT_VALUE <= -1e3
    # Should not be so large that it causes numerical issues
    assert MASKED_LOGIT_VALUE >= -1e5


def test_masked_logit_value_safe_for_fp16():
    """MASKED_LOGIT_VALUE should not overflow in FP16 softmax."""
    logits = torch.tensor([0.0, MASKED_LOGIT_VALUE], dtype=torch.float16)
    probs = torch.softmax(logits, dim=0)
    assert probs[0].item() > 0.99  # Masked action should have ~0 probability
    assert not torch.isnan(probs).any()
    assert not torch.isinf(probs).any()
