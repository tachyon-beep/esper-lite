"""Tests for value function metrics computation."""

import torch

from esper.simic.telemetry.value_metrics import compute_value_function_metrics


class TestComputeValueFunctionMetrics:
    """Tests for compute_value_function_metrics()."""

    def test_computes_td_error_stats(self):
        """Should compute TD error mean, std, and bellman error."""
        td_errors = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([10.0, 20.0, 30.0, 40.0])
        returns = torch.tensor([11.0, 22.0, 33.0, 44.0])

        metrics = compute_value_function_metrics(td_errors, values, returns)

        assert abs(metrics["td_error_mean"] - 2.5) < 0.01
        assert metrics["td_error_std"] > 0
        # Bellman = mean(|delta|) = mean([1, 2, 3, 4]) = 2.5
        assert abs(metrics["bellman_error"] - 2.5) < 0.01

    def test_computes_v_return_correlation(self):
        """Should compute Pearson correlation between values and returns."""
        # Perfect positive correlation
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        returns = torch.tensor([10.0, 20.0, 30.0, 40.0])
        td_errors = torch.zeros(4)

        metrics = compute_value_function_metrics(td_errors, values, returns)

        # Perfect correlation = 1.0
        assert abs(metrics["v_return_correlation"] - 1.0) < 0.01

    def test_computes_return_percentiles(self):
        """Should compute p10, p50, p90 of returns."""
        returns = torch.arange(100, dtype=torch.float32)
        values = torch.zeros(100)
        td_errors = torch.zeros(100)

        metrics = compute_value_function_metrics(td_errors, values, returns)

        # p10 ~ 10, p50 ~ 50, p90 ~ 90
        assert 8 < metrics["return_p10"] < 12
        assert 48 < metrics["return_p50"] < 52
        assert 88 < metrics["return_p90"] < 92

    def test_computes_return_variance_and_skewness(self):
        """Should compute variance and skewness of returns."""
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        values = torch.zeros(5)
        td_errors = torch.zeros(5)

        metrics = compute_value_function_metrics(td_errors, values, returns)

        assert metrics["return_variance"] > 0
        # Symmetric distribution has ~0 skewness
        assert abs(metrics["return_skewness"]) < 0.5

    def test_handles_empty_input(self):
        """Should return zeros for empty tensors."""
        td_errors = torch.tensor([])
        values = torch.tensor([])
        returns = torch.tensor([])

        metrics = compute_value_function_metrics(td_errors, values, returns)

        assert metrics["td_error_mean"] == 0.0
        assert metrics["v_return_correlation"] == 0.0

    def test_handles_constant_values(self):
        """Should handle zero std gracefully (constant values)."""
        td_errors = torch.tensor([1.0, 1.0, 1.0, 1.0])
        values = torch.tensor([5.0, 5.0, 5.0, 5.0])  # Zero std
        returns = torch.tensor([10.0, 10.0, 10.0, 10.0])  # Zero std

        metrics = compute_value_function_metrics(td_errors, values, returns)

        # Correlation undefined when std=0, should return 0.0
        assert metrics["v_return_correlation"] == 0.0
        # Skewness undefined when std=0, should return 0.0
        assert metrics["return_skewness"] == 0.0
