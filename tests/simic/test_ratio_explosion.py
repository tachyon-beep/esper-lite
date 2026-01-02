"""Tests for ratio explosion diagnostics."""

import torch

from esper.simic.telemetry import RatioExplosionDiagnostic
from esper.simic.telemetry.debug_telemetry import _MAX_DIAGNOSTIC_EXEMPLARS


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


class TestNaNRatioDetection:
    """Tests for NaN/Inf ratio detection.

    Before the fix, bad_mask only checked threshold violations. NaN comparisons
    are always False, so NaN ratios were silently ignored. The fix explicitly
    checks for non-finite values.
    """

    def test_nan_ratio_is_flagged(self):
        """Ratios containing NaN should be reported as problematic."""
        ratio = torch.tensor([1.0, float('nan'), 1.5, 0.8])
        old_log_probs = torch.tensor([-1.0, -0.5, -0.8, -0.3])
        new_log_probs = torch.tensor([-1.0, -0.5, -0.8, -0.3])
        actions = torch.tensor([0, 1, 2, 3])

        diag = RatioExplosionDiagnostic.from_batch(
            ratio=ratio,
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            actions=actions,
        )

        # NaN at index 1 should be flagged
        assert 1 in diag.worst_ratio_indices
        assert diag.num_nonfinite == 1
        assert diag.num_bad_total >= 1

    def test_inf_ratio_is_flagged(self):
        """Ratios containing Inf should be reported as problematic."""
        ratio = torch.tensor([1.0, float('inf'), float('-inf'), 0.8])
        old_log_probs = torch.tensor([-1.0, -0.5, -0.8, -0.3])
        new_log_probs = torch.tensor([-1.0, -0.5, -0.8, -0.3])
        actions = torch.tensor([0, 1, 2, 3])

        diag = RatioExplosionDiagnostic.from_batch(
            ratio=ratio,
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            actions=actions,
        )

        # Both inf values should be flagged
        assert 1 in diag.worst_ratio_indices
        assert 2 in diag.worst_ratio_indices
        assert diag.num_nonfinite == 2

    def test_num_bad_total_includes_threshold_and_nonfinite(self):
        """num_bad_total should count both threshold violations and non-finite."""
        ratio = torch.tensor([1.0, float('nan'), 10.0, 0.01])  # 1 NaN + 2 threshold
        old_log_probs = torch.zeros(4)
        new_log_probs = torch.zeros(4)
        actions = torch.tensor([0, 1, 2, 3])

        diag = RatioExplosionDiagnostic.from_batch(
            ratio=ratio,
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            actions=actions,
            max_threshold=5.0,
            min_threshold=0.1,
        )

        assert diag.num_nonfinite == 1  # Just the NaN
        assert diag.num_bad_total == 3  # NaN + ratio>5 + ratio<0.1


class TestPayloadCapping:
    """Tests for diagnostic payload size limits.

    In widespread instability, all ratios may be problematic. Without capping,
    the diagnostic could contain millions of indices/values, causing memory
    and telemetry explosion.
    """

    def test_indices_capped_to_max_exemplars(self):
        """Large batches should be capped to _MAX_DIAGNOSTIC_EXEMPLARS."""
        # Create more bad indices than the cap
        n = _MAX_DIAGNOSTIC_EXEMPLARS + 500
        ratio = torch.ones(n) * 10.0  # All above threshold
        old_log_probs = torch.zeros(n)
        new_log_probs = torch.zeros(n)
        actions = torch.zeros(n, dtype=torch.long)

        diag = RatioExplosionDiagnostic.from_batch(
            ratio=ratio,
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            actions=actions,
            max_threshold=5.0,
        )

        # Lists should be capped
        assert len(diag.worst_ratio_indices) == _MAX_DIAGNOSTIC_EXEMPLARS
        assert len(diag.worst_ratio_values) == _MAX_DIAGNOSTIC_EXEMPLARS
        assert len(diag.worst_ratio_actions) == _MAX_DIAGNOSTIC_EXEMPLARS

        # But total count should reflect actual number
        assert diag.num_bad_total == n

    def test_small_batches_not_affected(self):
        """Batches smaller than cap should report all indices."""
        n = 5
        ratio = torch.ones(n) * 10.0  # All above threshold
        old_log_probs = torch.zeros(n)
        new_log_probs = torch.zeros(n)
        actions = torch.zeros(n, dtype=torch.long)

        diag = RatioExplosionDiagnostic.from_batch(
            ratio=ratio,
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            actions=actions,
            max_threshold=5.0,
        )

        # All 5 should be reported
        assert len(diag.worst_ratio_indices) == 5
        assert diag.num_bad_total == 5
