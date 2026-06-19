"""Tests for value function metrics in PPO update.

TELE-220 to TELE-228: Value function diagnostic metrics should be computed
and returned during PPO update() calls. These metrics enable monitoring of
value function health and training progress.
"""

import pytest
import torch

from esper.leyline import (
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)
from esper.leyline.slot_config import SlotConfig
from esper.simic.agent import PPOAgent
from esper.tamiyo.policy import create_policy
from esper.tamiyo.policy.features import get_feature_size


@pytest.fixture
def ppo_agent():
    """Create a minimal PPO agent for testing."""
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    return PPOAgent(
        policy=policy,
        slot_config=slot_config,
        device="cpu",
        num_envs=2,
        max_steps_per_env=8,
        target_kl=None,  # Disable early stopping for predictable test behavior
    )


def _fill_buffer_with_rollout(agent: PPOAgent, varied_rewards: bool = False) -> None:
    """Fill the agent's buffer with a minimal rollout for testing.

    Args:
        agent: PPO agent with buffer to fill
        varied_rewards: If True, use varied rewards to create non-trivial statistics
    """
    device = torch.device(agent.device)
    state_dim = get_feature_size(agent.slot_config)
    hidden = agent.policy.network.get_initial_hidden(1, device)

    for env_id in range(agent.num_envs):
        agent.buffer.start_episode(env_id)
        for step in range(agent.max_steps_per_env):
            state = torch.randn(1, state_dim, device=device)
            masks = {
                "slot": torch.ones(1, agent.slot_config.num_slots, dtype=torch.bool, device=device),
                "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
                "style": torch.ones(1, NUM_STYLES, dtype=torch.bool, device=device),
                "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool, device=device),
                "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
                "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
                "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
                "op": torch.ones(1, NUM_OPS, dtype=torch.bool, device=device),
            }
            pre_hidden = hidden
            bp_indices = torch.zeros(1, agent.slot_config.num_slots, dtype=torch.long, device=device)
            result = agent.policy.network.get_action(
                state,
                bp_indices,
                hidden,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
            )
            hidden = result.hidden

            # Varied rewards create non-trivial statistics for percentile tests
            if varied_rewards:
                reward = float(step + env_id)  # 0, 1, 2, ... varying rewards
            else:
                reward = 1.0

            agent.buffer.add(
                env_id=env_id,
                state=state.squeeze(0),
                slot_action=result.actions["slot"].item(),
                blueprint_action=result.actions["blueprint"].item(),
                style_action=result.actions["style"].item(),
                tempo_action=result.actions["tempo"].item(),
                alpha_target_action=result.actions["alpha_target"].item(),
                alpha_speed_action=result.actions["alpha_speed"].item(),
                alpha_curve_action=result.actions["alpha_curve"].item(),
                op_action=result.actions["op"].item(),
                effective_op_action=result.actions["op"].item(),
                slot_log_prob=result.log_probs["slot"].item(),
                blueprint_log_prob=result.log_probs["blueprint"].item(),
                style_log_prob=result.log_probs["style"].item(),
                tempo_log_prob=result.log_probs["tempo"].item(),
                alpha_target_log_prob=result.log_probs["alpha_target"].item(),
                alpha_speed_log_prob=result.log_probs["alpha_speed"].item(),
                alpha_curve_log_prob=result.log_probs["alpha_curve"].item(),
                op_log_prob=result.log_probs["op"].item(),
                value=result.values.item(),
                reward=reward,
                done=step == agent.max_steps_per_env - 1,
                truncated=False,
                slot_mask=masks["slot"].squeeze(0),
                blueprint_mask=masks["blueprint"].squeeze(0),
                style_mask=masks["style"].squeeze(0),
                tempo_mask=masks["tempo"].squeeze(0),
                alpha_target_mask=masks["alpha_target"].squeeze(0),
                alpha_speed_mask=masks["alpha_speed"].squeeze(0),
                alpha_curve_mask=masks["alpha_curve"].squeeze(0),
                op_mask=masks["op"].squeeze(0),
                hidden_h=pre_hidden[0],
                hidden_c=pre_hidden[1],
                bootstrap_value=0.0,
                blueprint_indices=bp_indices.squeeze(0),
            )
        agent.buffer.end_episode(env_id)


class TestTELE220to228ValueFunctionMetrics:
    """TELE-220 to TELE-228: Value function metrics in PPO update."""

    def test_value_function_metrics_in_update_result(self, ppo_agent):
        """update() should return all 9 value function metrics."""
        _fill_buffer_with_rollout(ppo_agent)

        metrics = ppo_agent.update()

        # All 9 metrics should be present (TELE-220 to TELE-228)
        assert "v_return_correlation" in metrics, "TELE-220: v_return_correlation missing"
        assert "td_error_mean" in metrics, "TELE-221: td_error_mean missing"
        assert "td_error_std" in metrics, "TELE-222: td_error_std missing"
        assert "bellman_error" in metrics, "TELE-223: bellman_error missing"
        assert "return_p10" in metrics, "TELE-224: return_p10 missing"
        assert "return_p50" in metrics, "TELE-225: return_p50 missing"
        assert "return_p90" in metrics, "TELE-226: return_p90 missing"
        assert "return_variance" in metrics, "TELE-227: return_variance missing"
        assert "return_skewness" in metrics, "TELE-228: return_skewness missing"

    def test_v_return_correlation_in_valid_range(self, ppo_agent):
        """V-return correlation should be in [-1, 1]."""
        _fill_buffer_with_rollout(ppo_agent, varied_rewards=True)

        metrics = ppo_agent.update()

        assert -1.0 <= metrics["v_return_correlation"] <= 1.0, \
            f"v_return_correlation {metrics['v_return_correlation']} out of [-1, 1]"

    def test_return_percentiles_ordered(self, ppo_agent):
        """Return percentiles should be ordered: p10 <= p50 <= p90."""
        _fill_buffer_with_rollout(ppo_agent, varied_rewards=True)

        metrics = ppo_agent.update()

        assert metrics["return_p10"] <= metrics["return_p50"], \
            f"p10 ({metrics['return_p10']}) > p50 ({metrics['return_p50']})"
        assert metrics["return_p50"] <= metrics["return_p90"], \
            f"p50 ({metrics['return_p50']}) > p90 ({metrics['return_p90']})"

    def test_td_error_mean_is_finite(self, ppo_agent):
        """TD error mean should be finite."""
        _fill_buffer_with_rollout(ppo_agent)

        metrics = ppo_agent.update()

        import math
        assert math.isfinite(metrics["td_error_mean"]), \
            f"td_error_mean is not finite: {metrics['td_error_mean']}"

    def test_bellman_error_non_negative(self, ppo_agent):
        """Bellman error (mean absolute TD error) should be non-negative."""
        _fill_buffer_with_rollout(ppo_agent)

        metrics = ppo_agent.update()

        assert metrics["bellman_error"] >= 0, \
            f"bellman_error should be non-negative: {metrics['bellman_error']}"

    def test_return_variance_non_negative(self, ppo_agent):
        """Return variance should be non-negative."""
        _fill_buffer_with_rollout(ppo_agent)

        metrics = ppo_agent.update()

        assert metrics["return_variance"] >= 0, \
            f"return_variance should be non-negative: {metrics['return_variance']}"

    def test_empty_buffer_returns_empty_dict(self, ppo_agent):
        """When buffer is empty, update should return empty dict."""
        # Don't fill buffer - leave it empty
        metrics = ppo_agent.update()

        assert metrics == {}, "Empty buffer should return empty metrics dict"


class TestComputeFlooredExplainedVariance:
    """W7 helper seam: leyline/value_metrics.compute_floored_explained_variance.

    Pure, testable EV/value-fit computation extracted from PPOAgent.update().
    Bessel correction=1 (the shipped EV default).
    """

    def _ev_pre_floor(self, raw_values: torch.Tensor, valid_returns: torch.Tensor) -> float:
        """The pre-floor EV formula (the OLD inlined path) as an inline witness."""
        residual_var = (valid_returns - raw_values).var()
        var_returns = valid_returns.var()
        return float(1.0 - residual_var / var_returns)

    def test_ev_floored_on_low_return_variance(self):
        """Tiny var_returns << floor + small residual: pre-floor EV << -1, floored EV bounded + flagged."""
        from esper.leyline.value_metrics import compute_floored_explained_variance

        # Tiny return spread (var_returns ~ 0.0167 << floor 1.0) but a residual whose variance
        # EXCEEDS var_returns -> pre-floor EV is a large negative outlier.
        valid_returns = torch.tensor([5.0, 5.1, 5.2, 5.3], dtype=torch.float32)
        raw_values = torch.tensor([5.5, 4.6, 5.7, 4.8], dtype=torch.float32)

        assert float(valid_returns.var()) < 1.0, "fixture must be below the floor"
        pre_floor_ev = self._ev_pre_floor(raw_values, valid_returns)
        assert pre_floor_ev < -1.0, (
            f"witness: pre-floor EV should blow out (<< -1), got {pre_floor_ev}"
        )

        explained_variance, value_nrmse, ev_low_return_variance, ev_return_variance = (
            compute_floored_explained_variance(raw_values, valid_returns, floor=1.0)
        )

        assert float(explained_variance) > -2.0, (
            f"floored EV should be bounded (> -2), got {float(explained_variance)}"
        )
        assert ev_low_return_variance is True
        assert float(ev_return_variance) < 1.0  # the true (un-floored) variance

    def test_ev_unchanged_on_healthy_variance(self):
        """var_returns in the 49-169 band: floored EV within 1e-6 of pre-floor (clamp is a no-op), flag False."""
        from esper.leyline.value_metrics import compute_floored_explained_variance

        # Return std ~10 -> var_returns ~100 (mid healthy band), small residual
        torch.manual_seed(0)
        valid_returns = torch.randn(64, dtype=torch.float32) * 10.0
        raw_values = valid_returns + torch.randn(64, dtype=torch.float32) * 2.0

        assert 49.0 <= float(valid_returns.var()) <= 169.0, "fixture must sit in healthy band"

        pre_floor_ev = self._ev_pre_floor(raw_values, valid_returns)

        explained_variance, _, ev_low_return_variance, _ = (
            compute_floored_explained_variance(raw_values, valid_returns, floor=1.0)
        )

        assert abs(float(explained_variance) - pre_floor_ev) < 1e-6, (
            "above the floor the clamp must be a no-op"
        )
        assert ev_low_return_variance is False

    def test_ev_no_special_case_zero(self):
        """Near-zero var_returns: floored EV is a CONTINUOUS value, NOT exactly 0.0; flag True.

        Proves the old `1e-8 -> torch.tensor(0.0)` sentinel is gone.
        """
        from esper.leyline.value_metrics import compute_floored_explained_variance

        # Nearly-flat returns (std ~1e-4) but a real, non-zero residual.
        valid_returns = torch.tensor([2.0, 2.0001, 1.9999, 2.0], dtype=torch.float32)
        raw_values = torch.tensor([2.5, 1.5, 2.5, 1.5], dtype=torch.float32)

        explained_variance, _, ev_low_return_variance, _ = (
            compute_floored_explained_variance(raw_values, valid_returns, floor=1.0)
        )

        ev = float(explained_variance)
        assert ev != 0.0, "floored EV must be continuous, not the old 0.0 sentinel"
        assert ev_low_return_variance is True

    def test_ev_floor_default_matches_calibration(self, ppo_agent):
        """Meta-assert: PPOAgent default ev_return_variance_floor == locked Step-0 value (1.0)."""
        assert ppo_agent.ev_return_variance_floor == 1.0

    def test_ev_degenerate_batch_raises_loudly(self, ppo_agent, monkeypatch):
        """B3: a valid_mask selecting <=1 element is a HARD BUG -> RuntimeError, NO EV/value_nrmse/ev_* metric.

        Proves the floor is never a clamp(NaN) sanitizer: the :636-640 non-finite return-stat
        guard fires before any EV value can reach telemetry. No NaN-by-convention path, no
        ev_low_return_variance=True-on-degenerate convention.
        """
        _fill_buffer_with_rollout(ppo_agent)

        # Force the valid mask to select a single timestep -> valid_returns.var() is NaN
        # (PyTorch correction=1) and the return-stat guard must raise.
        from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer

        original_get = TamiyoRolloutBuffer.get_batched_sequences

        def _degenerate_get(self, *args, **kwargs):
            data = original_get(self, *args, **kwargs)
            mask = torch.zeros_like(data["valid_mask"])
            # Select exactly one valid element.
            mask.view(-1)[0] = True
            data["valid_mask"] = mask
            return data

        monkeypatch.setattr(TamiyoRolloutBuffer, "get_batched_sequences", _degenerate_get)

        with pytest.raises(RuntimeError, match="Non-finite returns"):
            ppo_agent.update()


class TestValueNrmse:
    """W7 helper value_nrmse: floor-stabilized secondary value-fit signal (Step 2)."""

    def test_value_nrmse_finite_and_monotone(self):
        """value_nrmse finite, >=0, strictly increasing in injected residual magnitude."""
        from esper.leyline.value_metrics import compute_floored_explained_variance

        torch.manual_seed(1)
        valid_returns = torch.randn(64, dtype=torch.float32) * 10.0
        # A fixed VARYING residual direction; scaling it grows residual_var (numerator).
        residual_dir = torch.randn(64, dtype=torch.float32)

        prev = -1.0
        for scale in (0.5, 1.0, 2.0, 4.0):
            raw_values = valid_returns + residual_dir * scale
            _, value_nrmse, _, _ = compute_floored_explained_variance(
                raw_values, valid_returns, floor=1.0
            )
            v = float(value_nrmse)
            assert v == v, "value_nrmse must be finite (not NaN)"
            assert v >= 0.0
            assert v > prev, "value_nrmse must strictly increase with residual magnitude"
            prev = v

    def test_value_nrmse_bounded_on_low_variance(self):
        """On the Step-1 low-var batch, value_nrmse is finite and small (low value_loss), not a crater."""
        from esper.leyline.value_metrics import compute_floored_explained_variance

        valid_returns = torch.tensor([5.0, 5.1, 5.2, 5.3], dtype=torch.float32)
        raw_values = valid_returns + torch.tensor([0.1, -0.1, 0.1, -0.1], dtype=torch.float32)

        _, value_nrmse, ev_low_return_variance, _ = compute_floored_explained_variance(
            raw_values, valid_returns, floor=1.0
        )

        v = float(value_nrmse)
        assert v == v and v >= 0.0
        assert v < 0.5, f"value_nrmse should stay small on a genuinely-good low-var fit, got {v}"
        assert ev_low_return_variance is True

    def test_value_nrmse_degenerate_batch_raises_loudly(self, ppo_agent, monkeypatch):
        """numel<=1: value_nrmse NEVER computed/emitted; the degenerate batch raises at :636-640 first."""
        _fill_buffer_with_rollout(ppo_agent)

        from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer

        original_get = TamiyoRolloutBuffer.get_batched_sequences

        def _degenerate_get(self, *args, **kwargs):
            data = original_get(self, *args, **kwargs)
            data["valid_mask"] = torch.zeros_like(data["valid_mask"])  # numel == 0 valid
            return data

        monkeypatch.setattr(TamiyoRolloutBuffer, "get_batched_sequences", _degenerate_get)

        with pytest.raises(RuntimeError, match="Non-finite returns"):
            ppo_agent.update()


class TestEVRobustnessEmit:
    """Step 3: PPOUpdateMetrics contract + end-to-end emit through PPOAgent.update()."""

    def test_ppo_update_metrics_carries_ev_robustness_fields(self):
        """PPOUpdateMetrics TypedDict exposes the four new keys."""
        from esper.simic.agent.types import PPOUpdateMetrics

        annotations = PPOUpdateMetrics.__annotations__
        assert "value_nrmse" in annotations
        assert "ev_low_return_variance" in annotations
        assert "ev_return_variance" in annotations
        assert "ev_low_return_variance_count" in annotations

    def test_ppo_update_emits_ev_robustness_fields(self, ppo_agent):
        """End-to-end: update() returns the four keys with correct types (helper wired into update)."""
        _fill_buffer_with_rollout(ppo_agent, varied_rewards=True)

        metrics = ppo_agent.update()

        assert "value_nrmse" in metrics
        assert "ev_return_variance" in metrics
        assert "ev_low_return_variance" in metrics
        assert "ev_low_return_variance_count" in metrics

        assert isinstance(metrics["value_nrmse"], float)
        assert isinstance(metrics["ev_return_variance"], float)
        assert isinstance(metrics["ev_low_return_variance"], bool)
        assert isinstance(metrics["ev_low_return_variance_count"], int)


class TestExplainedVarianceScaling:
    """Test that explained_variance is computed on consistent scales.

    Regression test for: EV was computed mixing normalized-scale values
    (from buffer) with raw-scale returns, producing incorrect results.
    """

    def test_explained_variance_in_valid_range(self, ppo_agent):
        """explained_variance should be in reasonable range [-1, 1] typically.

        With properly scaled computation, EV should not be wildly negative
        or exceed 1.0 (which would indicate scale mismatch issues).
        """
        _fill_buffer_with_rollout(ppo_agent, varied_rewards=True)

        metrics = ppo_agent.update()

        ev = metrics["explained_variance"]
        # EV should be in [-1, 1] range for properly scaled computation
        # Extremely negative values (< -10) indicate scale mismatch
        assert ev >= -10.0, (
            f"explained_variance {ev} is extremely negative, "
            "suggesting scale mismatch between values and returns"
        )
        assert ev <= 1.0 + 1e-6, (
            f"explained_variance {ev} exceeds 1.0, "
            "which is mathematically impossible for correctly computed EV"
        )

    def test_explained_variance_with_active_normalizer(self, ppo_agent):
        """EV computed correctly when value normalizer has non-trivial scale.

        This test forces the value normalizer into a state with large std,
        then verifies EV is still in valid range (not corrupted by scale mismatch).
        """
        # Warm up the value normalizer with large-scale returns
        large_returns = torch.tensor([100.0, 200.0, 300.0, 400.0])
        for _ in range(10):  # Ensure we exceed warmup threshold (32 samples)
            ppo_agent.value_normalizer.update(large_returns)

        # Verify normalizer has non-trivial scale
        scale = ppo_agent.value_normalizer.get_scale()
        assert scale > 10.0, f"Expected large normalizer scale, got {scale}"

        # Fill buffer and update
        _fill_buffer_with_rollout(ppo_agent, varied_rewards=True)
        metrics = ppo_agent.update()

        ev = metrics["explained_variance"]
        # Even with large normalizer scale, EV should be in valid range
        # if values are properly denormalized before EV computation
        assert ev >= -10.0, (
            f"explained_variance {ev} is extremely negative with active normalizer "
            f"(scale={scale}), suggesting values not denormalized"
        )
        assert ev <= 1.0 + 1e-6, (
            f"explained_variance {ev} exceeds 1.0 with active normalizer"
        )


class TestComputeFlooredAuxExplainedVariance:
    """SLICE C: leyline/value_metrics.compute_floored_aux_explained_variance.

    Aux-EV mirror of the main floored-EV helper, on the contribution-target scale.
    The floor is DATA-RELATIVE per update: ``max(floor_min, floor_fraction * var(targets))``
    (the ``0.05`` fraction / ``0.01`` absolute-min recipe from the calibration agent).
    Like the main EV floor it is DIAGNOSTIC-ONLY (flag + aux value_nrmse denominator);
    it is NEVER a gate trigger.

    Numerical conventions mirror the main helper: the EV denominator clamp uses
    ``Tensor.var()`` (Bessel correction=1, the shipped EV convention); the data-relative
    FLOOR magnitude uses ``var(correction=0)`` (population variance — a stable estimate of
    the data scale, per the recipe's ``tgt.var(unbiased=False)``).
    """

    def _aux_ev_pre_floor(self, pred_flat: torch.Tensor, target_flat: torch.Tensor) -> float:
        """The OLD inlined aux-EV path (bare ``clamp(min=0.01)``) as an inline witness."""
        target_var = target_flat.var().clamp(min=0.01)
        residual_var = (pred_flat - target_flat).var()
        return float(1.0 - residual_var / target_var)

    def test_aux_ev_floored_on_low_target_variance(self):
        """Tiny target spread (below the data-relative floor): aux-EV is bounded, flagged, and lifted above the bare clamp."""
        from esper.leyline.value_metrics import compute_floored_aux_explained_variance

        # Healthy-scale contribution targets establish the data-relative floor (0.05*var0),
        # but THIS batch's targets have a tiny spread (var ~ 0.00042) << that floor, so the
        # bare clamp(min=0.01) over-states the EV blowout; the data-relative floor caps it harder.
        target_flat = torch.tensor([0.10, 0.12, 0.11, 0.13], dtype=torch.float32)
        pred_flat = torch.tensor([0.6, -0.4, 0.7, -0.5], dtype=torch.float32)

        true_var = float(target_flat.var())
        assert true_var < 0.01, "fixture target var must be below the abs-min floor"
        pre_floor = self._aux_ev_pre_floor(pred_flat, target_flat)
        assert pre_floor < -1.0, f"witness: pre-floor aux-EV should blow out, got {pre_floor}"

        # Drive the data-relative floor well above 0.01 (and above true_var) via floor_fraction
        # so the floor genuinely lifts the EV above the bare clamp(min=0.01) value.
        ev, aux_value_nrmse, aux_low, aux_var = compute_floored_aux_explained_variance(
            pred_flat, target_flat, floor_fraction=100.0, floor_min=0.01
        )

        assert float(ev) > float(pre_floor), "data-relative floor must lift EV above the bare clamp(min=0.01)"
        assert float(ev) > -50.0, f"floored aux-EV should be bounded, got {float(ev)}"
        assert aux_low is True
        assert float(aux_var) == true_var  # un-floored true variance

    def test_aux_ev_unchanged_on_healthy_target_variance(self):
        """Healthy contribution-target spread: floored aux-EV == bare-clamp aux-EV (data floor is a no-op), flag False."""
        from esper.leyline.value_metrics import compute_floored_aux_explained_variance

        torch.manual_seed(0)
        # Contribution targets on a healthy scale (std ~3 -> var ~9). 0.05*9 = 0.45 floor; the
        # true var (9) is well above it -> clamp is a no-op and EV is numerically unchanged.
        target_flat = torch.randn(64, dtype=torch.float32) * 3.0
        pred_flat = target_flat + torch.randn(64, dtype=torch.float32) * 0.5

        floor = max(0.01, 0.05 * float(target_flat.var(correction=0)))
        assert float(target_flat.var()) > floor, "fixture must sit above the data-relative floor"
        pre_floor = self._aux_ev_pre_floor(pred_flat, target_flat)

        ev, _, aux_low, _ = compute_floored_aux_explained_variance(
            pred_flat, target_flat, floor_fraction=0.05, floor_min=0.01
        )

        assert abs(float(ev) - pre_floor) < 1e-6, "above the floor the clamp must be a no-op"
        assert aux_low is False

    def test_aux_value_nrmse_finite_and_monotone(self):
        """aux value_nrmse finite, >=0, strictly increasing in injected residual magnitude."""
        from esper.leyline.value_metrics import compute_floored_aux_explained_variance

        torch.manual_seed(2)
        target_flat = torch.randn(64, dtype=torch.float32) * 3.0
        residual_dir = torch.randn(64, dtype=torch.float32)

        prev = -1.0
        for scale in (0.25, 0.5, 1.0, 2.0):
            pred_flat = target_flat + residual_dir * scale
            _, aux_value_nrmse, _, _ = compute_floored_aux_explained_variance(
                target_flat=target_flat, pred_flat=pred_flat, floor_fraction=0.05, floor_min=0.01
            )
            v = float(aux_value_nrmse)
            assert v == v, "aux value_nrmse must be finite"
            assert v >= 0.0
            assert v > prev, "aux value_nrmse must strictly increase with residual magnitude"
            prev = v

    def test_aux_ev_data_relative_floor_dominates_abs_min(self):
        """When 0.05*var(targets) exceeds 0.01, the data-relative term is the active floor."""
        from esper.leyline.value_metrics import compute_floored_aux_explained_variance

        # Large-scale targets (var ~ 400) -> 0.05*400 = 20 >> 0.01. A batch whose true var is
        # below 20 (but above 0.01) must flag low under the data-relative floor — the abs-min
        # alone (0.01) would NOT flag it. This is the contribution-scale-aware behavior.
        target_flat = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)  # var ~ 1.67
        pred_flat = torch.tensor([1.1, 2.2, 2.8, 4.1], dtype=torch.float32)

        true_var = float(target_flat.var())
        assert true_var > 0.01, "fixture true var exceeds the abs-min floor"
        # ...but we drive the data-relative floor above it via floor_fraction.
        ev, _, aux_low, _ = compute_floored_aux_explained_variance(
            pred_flat, target_flat, floor_fraction=5.0, floor_min=0.01
        )
        # 5.0 * var0(targets) ~ 6.25 > true_var ~1.67 -> flag True under the data-relative floor.
        assert aux_low is True


class TestAuxEVRobustnessConfig:
    """SLICE C: PPOAgent ctor kwarg + types/reducer registration for the aux floor."""

    def test_aux_ev_floor_default_is_locked_recipe(self, ppo_agent):
        """Default aux floor params == the locked 0.05 fraction / 0.01 abs-min recipe."""
        assert ppo_agent.aux_ev_return_variance_floor_fraction == 0.05
        assert ppo_agent.aux_ev_return_variance_floor_min == 0.01

    def test_aux_ev_floor_kwarg_overridable(self):
        """Aux floor params are ctor kwargs (future empirical refinement without code surgery)."""
        from esper.leyline.slot_config import SlotConfig
        from esper.simic.agent import PPOAgent
        from esper.tamiyo.policy import create_policy

        slot_config = SlotConfig.default()
        policy = create_policy(
            policy_type="lstm", slot_config=slot_config, device="cpu", compile_mode="off"
        )
        agent = PPOAgent(
            policy=policy,
            slot_config=slot_config,
            device="cpu",
            num_envs=2,
            max_steps_per_env=8,
            target_kl=None,
            aux_ev_return_variance_floor_fraction=0.1,
            aux_ev_return_variance_floor_min=0.02,
        )
        assert agent.aux_ev_return_variance_floor_fraction == 0.1
        assert agent.aux_ev_return_variance_floor_min == 0.02

    def test_aux_ev_low_return_variance_in_types(self):
        """PPOUpdateMetrics TypedDict exposes the aux flag."""
        from esper.simic.agent.types import PPOUpdateMetrics

        assert "aux_ev_low_return_variance" in PPOUpdateMetrics.__annotations__

    def test_aux_ev_low_return_variance_registered_any_reducer(self):
        """The aux bool flag is OR-reduced across the vectorized batch (mirrors main flag)."""
        from esper.simic.training.vectorized import _PPO_ANY_REDUCED_METRICS

        assert "aux_ev_low_return_variance" in _PPO_ANY_REDUCED_METRICS


def _fill_buffer_with_contributions(
    agent: PPOAgent, contribution_scale: float
) -> None:
    """Fill the buffer with a rollout carrying FRESH contribution targets every timestep.

    Drives the aux-EV block (``fresh_mask.any()`` True, numel>=2) so the floored aux helper
    actually runs inside ``update()``. ``contribution_scale`` sets the per-slot target spread:
    a large scale -> healthy target variance (flag False); a tiny scale -> below the floor
    (flag True).
    """
    device = torch.device(agent.device)
    state_dim = get_feature_size(agent.slot_config)
    num_slots = agent.slot_config.num_slots
    hidden = agent.policy.network.get_initial_hidden(1, device)
    torch.manual_seed(123)

    for env_id in range(agent.num_envs):
        agent.buffer.start_episode(env_id)
        for step in range(agent.max_steps_per_env):
            state = torch.randn(1, state_dim, device=device)
            masks = {
                "slot": torch.ones(1, num_slots, dtype=torch.bool, device=device),
                "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
                "style": torch.ones(1, NUM_STYLES, dtype=torch.bool, device=device),
                "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool, device=device),
                "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
                "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
                "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
                "op": torch.ones(1, NUM_OPS, dtype=torch.bool, device=device),
            }
            pre_hidden = hidden
            bp_indices = torch.zeros(1, num_slots, dtype=torch.long, device=device)
            result = agent.policy.network.get_action(
                state, bp_indices, hidden,
                slot_mask=masks["slot"], blueprint_mask=masks["blueprint"],
                style_mask=masks["style"], tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"], alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"], op_mask=masks["op"],
            )
            hidden = result.hidden

            # Fresh contribution targets every step (varying across slots/steps).
            targets = torch.randn(num_slots, device=device) * contribution_scale
            mask = torch.ones(num_slots, dtype=torch.bool, device=device)

            agent.buffer.add(
                env_id=env_id, state=state.squeeze(0),
                slot_action=result.actions["slot"].item(),
                blueprint_action=result.actions["blueprint"].item(),
                style_action=result.actions["style"].item(),
                tempo_action=result.actions["tempo"].item(),
                alpha_target_action=result.actions["alpha_target"].item(),
                alpha_speed_action=result.actions["alpha_speed"].item(),
                alpha_curve_action=result.actions["alpha_curve"].item(),
                op_action=result.actions["op"].item(),
                effective_op_action=result.actions["op"].item(),
                slot_log_prob=result.log_probs["slot"].item(),
                blueprint_log_prob=result.log_probs["blueprint"].item(),
                style_log_prob=result.log_probs["style"].item(),
                tempo_log_prob=result.log_probs["tempo"].item(),
                alpha_target_log_prob=result.log_probs["alpha_target"].item(),
                alpha_speed_log_prob=result.log_probs["alpha_speed"].item(),
                alpha_curve_log_prob=result.log_probs["alpha_curve"].item(),
                op_log_prob=result.log_probs["op"].item(),
                value=result.values.item(),
                reward=float(step + env_id),
                done=step == agent.max_steps_per_env - 1,
                truncated=False,
                slot_mask=masks["slot"].squeeze(0),
                blueprint_mask=masks["blueprint"].squeeze(0),
                style_mask=masks["style"].squeeze(0),
                tempo_mask=masks["tempo"].squeeze(0),
                alpha_target_mask=masks["alpha_target"].squeeze(0),
                alpha_speed_mask=masks["alpha_speed"].squeeze(0),
                alpha_curve_mask=masks["alpha_curve"].squeeze(0),
                op_mask=masks["op"].squeeze(0),
                hidden_h=pre_hidden[0], hidden_c=pre_hidden[1],
                bootstrap_value=0.0,
                blueprint_indices=bp_indices.squeeze(0),
                contribution_targets=targets,
                contribution_mask=mask,
                has_fresh_contribution=True,
            )
        agent.buffer.end_episode(env_id)


class TestAuxEVRobustnessEmit:
    """SLICE C end-to-end: the floored aux helper runs in update() and emits through metrics."""

    def test_update_emits_aux_ev_robustness_fields(self, ppo_agent):
        """With fresh contributions, update() returns the aux EV-robustness keys with correct types."""
        assert ppo_agent.enable_contribution_aux, "fixture must have aux enabled"
        _fill_buffer_with_contributions(ppo_agent, contribution_scale=3.0)

        metrics = ppo_agent.update()

        assert "aux_explained_variance" in metrics
        assert "aux_value_nrmse" in metrics
        assert "aux_ev_low_return_variance" in metrics
        assert isinstance(metrics["aux_value_nrmse"], float)
        assert isinstance(metrics["aux_ev_low_return_variance"], bool)
        import math as _math
        assert _math.isfinite(metrics["aux_explained_variance"])
        assert _math.isfinite(metrics["aux_value_nrmse"])
        assert metrics["aux_value_nrmse"] >= 0.0

    def test_aux_flag_false_on_healthy_contribution_variance(self, ppo_agent):
        """Healthy contribution-target spread -> aux flag is False (floor is a no-op)."""
        _fill_buffer_with_contributions(ppo_agent, contribution_scale=5.0)

        metrics = ppo_agent.update()

        assert metrics["aux_ev_low_return_variance"] is False

    def test_aux_flag_true_on_low_contribution_variance(self, ppo_agent):
        """Tiny contribution-target spread (below the data-relative floor) -> aux flag True (DIAGNOSTIC)."""
        # contribution_scale ~ 1e-3 -> target var ~ 1e-6 << max(0.01, 0.05*var0) floor.
        _fill_buffer_with_contributions(ppo_agent, contribution_scale=1e-3)

        metrics = ppo_agent.update()

        assert metrics["aux_ev_low_return_variance"] is True
        # DIAGNOSTIC-ONLY guarantee: the aux flag must NOT change the robust-anchored value
        # gate -- it is not a gate input. value_loss/bellman_error are the only gate triggers.
        import math as _math
        assert _math.isfinite(metrics["aux_explained_variance"]), "floored aux-EV stays finite"
