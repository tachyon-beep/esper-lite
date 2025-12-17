"""Tests for ratio explosion diagnostics."""

import torch

from esper.simic.telemetry import RatioExplosionDiagnostic


class TestRatioExplosionDiagnostic:
    """Tests for RatioExplosionDiagnostic."""

    def test_create_from_tensors(self):
        """Can create diagnostic from tensors."""
        ratio = torch.tensor([0.5, 1.0, 1.5, 6.0, 0.05])
        old_log_probs = torch.tensor([-1.0, -0.5, -0.8, -0.3, -2.0])
        new_log_probs = torch.tensor([-1.2, -0.5, -0.4, 1.5, -5.0])
        states = torch.randn(5, 10)
        actions = torch.tensor([0, 1, 2, 1, 0])
        action_masks = torch.ones(5, 4)

        diag = RatioExplosionDiagnostic.from_batch(
            ratio=ratio,
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            actions=actions,
            max_threshold=5.0,
            min_threshold=0.1,
            states=states,
            action_masks=action_masks,
        )

        assert len(diag.worst_ratio_indices) == 2  # 6.0 > 5.0, 0.05 < 0.1
        assert diag.logit_diff_max > 0

    def test_to_dict_serializable(self):
        """Diagnostic can be serialized to dict."""
        diag = RatioExplosionDiagnostic(
            worst_ratio_indices=[3, 4],
            worst_ratio_values=[6.0, 0.05],
            worst_ratio_actions=[1, 0],
            logit_diff_mean=0.5,
            logit_diff_max=2.0,
        )
        d = diag.to_dict()
        assert "worst_ratio_indices" in d
        assert d["logit_diff_max"] == 2.0

    def test_empty_tensors_handled_gracefully(self):
        """Empty tensors return valid diagnostic without crashing.

        Edge case: when valid_mask selects zero transitions, all input tensors
        are empty. PyTorch's max() raises RuntimeError on empty tensors, and
        mean() returns nan. The fix returns a valid "no problems" diagnostic.
        """
        empty_ratio = torch.tensor([])
        empty_log_probs = torch.tensor([])
        empty_actions = torch.tensor([], dtype=torch.long)

        diag = RatioExplosionDiagnostic.from_batch(
            ratio=empty_ratio,
            old_log_probs=empty_log_probs,
            new_log_probs=empty_log_probs,
            actions=empty_actions,
        )

        # Should return valid diagnostic with empty lists and zero stats
        assert diag.worst_ratio_indices == []
        assert diag.worst_ratio_values == []
        assert diag.worst_ratio_actions == []
        assert diag.logit_diff_mean == 0.0
        assert diag.logit_diff_max == 0.0

        # Should be serializable
        d = diag.to_dict()
        assert d["logit_diff_mean"] == 0.0
