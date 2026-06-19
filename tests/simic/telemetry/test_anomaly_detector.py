"""Tests for AnomalyDetector scheduling logic."""

from __future__ import annotations

from esper.simic.telemetry import AnomalyDetector


def test_ev_threshold_non_decreasing_over_training() -> None:
    """EV threshold should get stricter later in training (or stay equal)."""
    detector = AnomalyDetector()

    early = detector.get_ev_threshold(current_episode=100, total_episodes=1000)
    late = detector.get_ev_threshold(current_episode=900, total_episodes=1000)

    assert isinstance(early, float)
    assert isinstance(late, float)
    assert late >= early


class TestCheckValueFunctionRobustAnchored:
    """Robust-anchored value-collapse gate (EV-telemetry-robustness, locked owner decision).

    GATE = ROBUST-ANCHORED ONLY. value_collapse fires SOLELY on the scale-anchored robust
    signal: value_loss (PRIMARY) OR bellman_error (SECONDARY safety-net) over their
    calibrated thresholds (both 5.0). explained_variance / value_nrmse /
    v_return_correlation / ev_low_return_variance are PURE DIAGNOSTICS — carried in
    telemetry but NEVER a gate trigger. The EV arm has been removed entirely.
    """

    def test_check_value_function_ignores_artifactual_low_ev(self) -> None:
        """Artifactual low-EV with healthy robust signals must NOT fire (EV is diagnostic-only)."""
        detector = AnomalyDetector()
        report = detector.check_value_function(
            explained_variance=-3.0,  # crater EV — diagnostic only, must not gate
            current_episode=80,
            total_episodes=100,
            bellman_error=0.5,   # healthy (< bellman_error_threshold 5.0)
            value_loss=0.099,    # healthy (< value_loss_threshold 5.0)
            value_nrmse=0.3,     # diagnostic
            v_return_correlation=0.9,  # diagnostic
            ev_low_return_variance=True,  # denominator artifact flagged (diagnostic)
        )
        assert report.has_anomaly is False
        assert "value_collapse" not in report.anomaly_types

    def test_check_value_function_fires_on_genuine_collapse(self) -> None:
        """Genuine collapse fires via the robust signal (bad value_loss and bellman_error)."""
        detector = AnomalyDetector()
        report = detector.check_value_function(
            explained_variance=-1.0,
            current_episode=80,
            total_episodes=100,
            bellman_error=20.0,  # bad (> bellman_error_threshold 5.0)
            value_loss=10.0,     # bad (> value_loss_threshold 5.0)
            value_nrmse=1.5,
            v_return_correlation=-0.1,
            ev_low_return_variance=False,
        )
        assert report.has_anomaly is True
        assert "value_collapse" in report.anomaly_types

    def test_check_value_function_fires_on_low_var_collapse_via_robust_arm(self) -> None:
        """Genuine collapse on a low-variance batch STILL fires (EV flag is diagnostic-only)."""
        detector = AnomalyDetector()
        report = detector.check_value_function(
            explained_variance=0.5,  # EV is diagnostic only — irrelevant to firing
            current_episode=80,
            total_episodes=100,
            bellman_error=25.0,  # bad robust signal
            value_loss=10.0,     # bad robust signal
            # Diagnostic signals are floor/sentinel-stabilized on a flagged update; set them
            # to "healthy-looking" to prove they do not gate the firing decision either way.
            value_nrmse=0.2,
            v_return_correlation=0.0,  # 0.0 sentinel on low-r_std batch
            ev_low_return_variance=True,  # diagnostic flag — must NOT suppress the robust fire
        )
        assert report.has_anomaly is True
        assert "value_collapse" in report.anomaly_types

    def test_check_value_function_does_not_fire_on_bad_ev_with_healthy_robust(self) -> None:
        """Bad EV (unflagged) with healthy robust signals must NOT fire.

        EV is a pure diagnostic: a crater EV never drives value_collapse on its own.
        Locks against any re-introduction of an EV firing arm.
        """
        detector = AnomalyDetector()
        report = detector.check_value_function(
            explained_variance=-2.0,  # EV is bad — diagnostic only
            current_episode=80,
            total_episodes=100,
            bellman_error=0.5,   # healthy
            value_loss=0.1,      # healthy
            value_nrmse=0.3,
            v_return_correlation=0.8,
            ev_low_return_variance=False,
        )
        assert report.has_anomaly is False
        assert "value_collapse" not in report.anomaly_types

    def test_check_value_function_fires_on_value_loss_alone(self) -> None:
        """value_loss alone (PRIMARY) over threshold trips the robust gate."""
        detector = AnomalyDetector()
        report = detector.check_value_function(
            explained_variance=0.9,
            current_episode=80,
            total_episodes=100,
            bellman_error=0.5,   # OK
            value_loss=6.0,      # bad (> value_loss_threshold 5.0)
            value_nrmse=0.3,
            v_return_correlation=0.9,
            ev_low_return_variance=False,
        )
        assert report.has_anomaly is True
        assert "value_collapse" in report.anomaly_types

    def test_check_value_function_fires_on_bellman_error_alone(self) -> None:
        """bellman_error alone (SECONDARY safety-net) over threshold trips the robust gate."""
        detector = AnomalyDetector()
        report = detector.check_value_function(
            explained_variance=0.9,
            current_episode=80,
            total_episodes=100,
            bellman_error=6.0,   # bad (> bellman_error_threshold 5.0)
            value_loss=0.099,    # OK
            value_nrmse=0.3,
            v_return_correlation=0.9,
            ev_low_return_variance=False,
        )
        assert report.has_anomaly is True
        assert "value_collapse" in report.anomaly_types

    def test_check_all_threads_robust_signals_artifact_suppressed(self) -> None:
        """check_all must forward the robust signals; artifact stays suppressed end-to-end."""
        detector = AnomalyDetector()
        report = detector.check_all(
            ratio_max=1.1,
            ratio_min=0.9,
            explained_variance=-3.0,
            current_episode=80,
            total_episodes=100,
            value_collapse_applicable=True,
            bellman_error=0.5,
            value_loss=0.099,
            value_nrmse=0.3,
            v_return_correlation=0.9,
            ev_low_return_variance=True,
        )
        assert "value_collapse" not in report.anomaly_types

    def test_check_all_threads_robust_signals_collapse_fires(self) -> None:
        """check_all forwards robust signals; genuine low-var collapse fires via robust signal."""
        detector = AnomalyDetector()
        report = detector.check_all(
            ratio_max=1.1,
            ratio_min=0.9,
            explained_variance=0.5,
            current_episode=80,
            total_episodes=100,
            value_collapse_applicable=True,
            bellman_error=25.0,
            value_loss=10.0,
            value_nrmse=0.2,
            v_return_correlation=0.0,
            ev_low_return_variance=True,
        )
        assert report.has_anomaly is True
        assert "value_collapse" in report.anomaly_types


class TestCheckLstmHealth:
    """Tests for LSTM hidden state health checking (B7-DRL-04)."""

    def test_healthy_state_no_anomaly(self) -> None:
        """Healthy LSTM state should not trigger anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=5.0,
            c_rms=5.0,
            h_env_rms_max=5.0,
            c_env_rms_max=5.0,
            has_nan=False,
            has_inf=False,
        )
        assert not report.has_anomaly
        assert report.anomaly_types == []

    def test_nan_triggers_anomaly(self) -> None:
        """NaN in hidden state should trigger lstm_nan anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=5.0,
            c_rms=5.0,
            h_env_rms_max=5.0,
            c_env_rms_max=5.0,
            has_nan=True,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_nan" in report.anomaly_types

    def test_inf_triggers_anomaly(self) -> None:
        """Inf in hidden state should trigger lstm_inf anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=5.0,
            c_rms=5.0,
            h_env_rms_max=5.0,
            c_env_rms_max=5.0,
            has_nan=False,
            has_inf=True,
        )
        assert report.has_anomaly
        assert "lstm_inf" in report.anomaly_types

    def test_h_explosion_triggers_anomaly(self) -> None:
        """h_rms above threshold should trigger lstm_h_explosion anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=15.0,  # > 10.0 default
            c_rms=5.0,
            h_env_rms_max=15.0,
            c_env_rms_max=5.0,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_h_explosion" in report.anomaly_types

    def test_c_explosion_triggers_anomaly(self) -> None:
        """c_rms above threshold should trigger lstm_c_explosion anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=5.0,
            c_rms=15.0,  # > 10.0 default
            h_env_rms_max=5.0,
            c_env_rms_max=15.0,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_c_explosion" in report.anomaly_types

    def test_h_vanishing_triggers_anomaly(self) -> None:
        """h_rms below threshold should trigger lstm_h_vanishing anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=1e-8,  # < 1e-6 default
            c_rms=5.0,
            h_env_rms_max=1e-8,
            c_env_rms_max=5.0,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_h_vanishing" in report.anomaly_types

    def test_c_vanishing_triggers_anomaly(self) -> None:
        """c_rms below threshold should trigger lstm_c_vanishing anomaly."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=5.0,
            c_rms=1e-8,  # < 1e-6 default
            h_env_rms_max=5.0,
            c_env_rms_max=1e-8,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_c_vanishing" in report.anomaly_types

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should be respected."""
        detector = AnomalyDetector(
            lstm_max_rms=5.0,
            lstm_min_rms=1e-4,
        )
        # 7.5 is under default (10.0) but over custom (5.0)
        report = detector.check_lstm_health(
            h_rms=7.5,
            c_rms=5.0,
            h_env_rms_max=7.5,
            c_env_rms_max=5.0,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly
        assert "lstm_h_explosion" in report.anomaly_types

    def test_multiple_anomalies(self) -> None:
        """Multiple LSTM issues should all be reported."""
        detector = AnomalyDetector()
        report = detector.check_lstm_health(
            h_rms=15.0,  # Explosion
            c_rms=1e-8,   # Vanishing
            h_env_rms_max=15.0,
            c_env_rms_max=1e-8,
            has_nan=True,  # NaN
            has_inf=False,
        )
        assert report.has_anomaly
        assert len(report.anomaly_types) >= 3
        assert "lstm_nan" in report.anomaly_types
        assert "lstm_h_explosion" in report.anomaly_types
        assert "lstm_c_vanishing" in report.anomaly_types


class TestPerHeadEntropyCollapse:
    """Tests for per-head entropy collapse detection with hysteresis."""

    def test_no_warning_on_single_collapse(self) -> None:
        """Single collapse should not trigger warning (hysteresis)."""
        detector = AnomalyDetector()
        head_entropies = {"blueprint": 0.01}  # Below 0.05

        report = detector.check_per_head_entropy_collapse(head_entropies)

        # First collapse - no warning yet (need N consecutive)
        assert not report.has_anomaly

    def test_warning_after_consecutive_collapses(self) -> None:
        """Warning should fire after N=3 consecutive collapses."""
        detector = AnomalyDetector()
        head_entropies = {"blueprint": 0.01}

        # First two - no warning
        detector.check_per_head_entropy_collapse(head_entropies)
        detector.check_per_head_entropy_collapse(head_entropies)

        # Third - warning fires
        report = detector.check_per_head_entropy_collapse(head_entropies)
        assert "entropy_collapse_blueprint" in report.anomaly_types

    def test_no_detection_when_healthy(self) -> None:
        """Should not flag healthy heads."""
        detector = AnomalyDetector()

        head_entropies = {
            "op": 0.5,
            "blueprint": 0.3,
            "slot": 0.4,
        }

        report = detector.check_per_head_entropy_collapse(head_entropies)

        assert not report.has_anomaly

    def test_nan_entropy_triggers_anomaly_immediately(self) -> None:
        """NaN entropy must not bypass per-head collapse detection."""
        detector = AnomalyDetector()

        report = detector.check_per_head_entropy_collapse({"blueprint": float("nan")})

        assert report.has_anomaly
        assert "entropy_nan_inf_blueprint" in report.anomaly_types
        assert "entropy=nan" in report.details["entropy_nan_inf_blueprint"]

    def test_recovery_requires_margin(self) -> None:
        """Recovery should require entropy above threshold * 1.5."""
        detector = AnomalyDetector()

        # Build up streak of 2 collapses
        detector.check_per_head_entropy_collapse({"blueprint": 0.01})
        detector.check_per_head_entropy_collapse({"blueprint": 0.01})

        # Barely above threshold (0.06 > 0.05) - should NOT clear streak
        detector.check_per_head_entropy_collapse({"blueprint": 0.06})

        # Back to collapse - should continue streak and fire (now 3rd)
        report = detector.check_per_head_entropy_collapse({"blueprint": 0.01})
        assert "entropy_collapse_blueprint" in report.anomaly_types

        # Well above threshold (0.10 > 0.075) - should clear
        detector.check_per_head_entropy_collapse({"blueprint": 0.10})

        # New collapse - should start fresh (only 1 in streak, no warning)
        report = detector.check_per_head_entropy_collapse({"blueprint": 0.01})
        assert not report.has_anomaly

    def test_uses_per_head_thresholds(self) -> None:
        """Different heads should have different collapse thresholds."""
        detector = AnomalyDetector()

        # op threshold is 0.08, blueprint is 0.05
        head_entropies = {
            "op": 0.07,        # Below 0.08 -> collapse streak
            "blueprint": 0.06,  # Above 0.05 -> OK
        }

        # Need 3 consecutive to trigger
        for _ in range(3):
            report = detector.check_per_head_entropy_collapse(head_entropies)

        assert "entropy_collapse_op" in report.anomaly_types
        assert "entropy_collapse_blueprint" not in report.anomaly_types
