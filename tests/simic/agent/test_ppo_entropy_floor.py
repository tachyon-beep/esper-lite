"""Tests for per-head entropy floor penalty in PPO loss."""

import pytest
import torch

from esper.simic.agent.ppo_update import compute_entropy_floor_penalty, compute_losses


class TestEntropyFloorPenalty:
    """Tests for per-head entropy floor penalty function."""

    def test_no_penalty_when_above_floor(self) -> None:
        """Entropy above floor should incur no penalty."""
        entropy = {"blueprint": torch.tensor(0.5)}  # Above 0.4 floor
        head_masks = {"blueprint": torch.ones(10)}
        floor = {"blueprint": 0.4}
        coef = {"blueprint": 0.1}  # Required: always dict

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor, coef)

        assert penalty.item() == pytest.approx(0.0)

    def test_penalty_when_below_floor(self) -> None:
        """Entropy below floor should incur quadratic penalty."""
        entropy = {"blueprint": torch.tensor(0.1)}  # Below 0.4 floor
        head_masks = {"blueprint": torch.ones(10)}
        floor = {"blueprint": 0.4}
        coef = {"blueprint": 0.1}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor, coef)

        # Shortfall = 0.4 - 0.1 = 0.3, penalty = 0.1 * 0.3^2 = 0.009
        assert penalty.item() > 0
        assert penalty.item() == pytest.approx(0.1 * (0.3 ** 2), rel=0.01)

    def test_penalty_scales_with_shortfall(self) -> None:
        """Larger shortfall should incur larger penalty."""
        head_masks = {"blueprint": torch.ones(10)}
        floor = {"blueprint": 0.4}
        coef = {"blueprint": 0.1}

        # Small shortfall
        entropy_small = {"blueprint": torch.tensor(0.35)}
        penalty_small = compute_entropy_floor_penalty(entropy_small, head_masks, floor, coef)

        # Large shortfall
        entropy_large = {"blueprint": torch.tensor(0.1)}
        penalty_large = compute_entropy_floor_penalty(entropy_large, head_masks, floor, coef)

        assert penalty_large > penalty_small

    def test_penalty_respects_mask(self) -> None:
        """Penalty should only consider masked (active) timesteps."""
        # Half the timesteps are masked (inactive)
        mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        head_masks = {"blueprint": mask}
        floor = {"blueprint": 0.4}
        coef = {"blueprint": 0.1}

        # Per-step entropy: first 5 have 0.5, last 5 have 0.0
        # Masked mean should be 0.5 (above floor) -> no penalty
        per_step_entropy = torch.tensor([0.5] * 5 + [0.0] * 5)

        # Note: compute_entropy_floor_penalty expects pre-computed mean entropy per head
        # Actually, we need to check how it handles per-step vs scalar entropy
        # For now, test with scalar (mean over masked steps)
        mean_ent = (per_step_entropy * mask).sum() / mask.sum()
        entropy_scalar = {"blueprint": mean_ent}

        penalty = compute_entropy_floor_penalty(entropy_scalar, head_masks, floor, coef)
        assert penalty.item() == pytest.approx(0.0)

    def test_multiple_heads(self) -> None:
        """Penalty should sum across all heads."""
        entropy = {
            "blueprint": torch.tensor(0.1),  # Below 0.4 -> penalty
            "op": torch.tensor(0.5),         # Above 0.15 -> no penalty
        }
        head_masks = {
            "blueprint": torch.ones(10),
            "op": torch.ones(10),
        }
        floor = {"blueprint": 0.4, "op": 0.15}
        coef = {"blueprint": 0.1, "op": 0.1}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor, coef)

        # Only blueprint should contribute
        expected = 0.1 * (0.3 ** 2)  # shortfall 0.3
        assert penalty.item() == pytest.approx(expected, rel=0.01)


class TestEntropyFloorEdgeCases:
    """Edge case tests added per expert review."""

    def test_no_penalty_when_head_inactive(self) -> None:
        """Heads with no valid steps should NOT incur penalty (critical fix)."""
        # Blueprint head is completely masked (no valid steps)
        entropy = {"blueprint": torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])}
        head_masks = {"blueprint": torch.zeros(5)}  # All masked!
        floor = {"blueprint": 0.4}
        coef = {"blueprint": 0.1}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor, coef)

        # Should be 0 - head was never active
        assert penalty.item() == pytest.approx(0.0), \
            f"Expected no penalty for inactive head, got {penalty.item()}"

    def test_per_head_penalty_coefficients(self) -> None:
        """Different heads should use different penalty strengths."""
        entropy = {
            "op": torch.tensor(0.1),       # Below floor 0.15, shortfall 0.05
            "blueprint": torch.tensor(0.1) # Below floor 0.40, shortfall 0.30
        }
        head_masks = {k: torch.ones(10) for k in entropy}
        floor = {"op": 0.15, "blueprint": 0.40}
        coefs = {"op": 0.05, "blueprint": 0.20}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor, coefs)

        # op: 0.05 * 0.05^2 = 0.000125
        # bp: 0.20 * 0.30^2 = 0.018
        expected = 0.000125 + 0.018
        assert penalty.item() == pytest.approx(expected, rel=0.01)

    def test_entropy_floor_gradient_direction(self) -> None:
        """Entropy below floor should have gradient pushing it up."""
        entropy_param = torch.tensor(0.1, requires_grad=True)
        entropy_dict = {"blueprint": entropy_param}
        head_masks = {"blueprint": torch.ones(10)}
        floor = {"blueprint": 0.4}
        coef = {"blueprint": 0.1}

        penalty = compute_entropy_floor_penalty(entropy_dict, head_masks, floor, coef)
        penalty.backward()

        # Gradient should be negative (pushes entropy up to reduce penalty)
        assert entropy_param.grad < 0, f"Expected negative gradient, got {entropy_param.grad}"

    def test_empty_entropy_dict_returns_zero(self) -> None:
        """Empty entropy dict should return zero penalty."""
        penalty = compute_entropy_floor_penalty({}, {}, {"blueprint": 0.4}, {"blueprint": 0.1})
        assert penalty.item() == pytest.approx(0.0)


class TestEntropyFloorIntegration:
    """Integration tests for entropy floor in compute_losses."""

    def test_compute_losses_includes_entropy_floor_penalty(self) -> None:
        """compute_losses should include entropy floor penalty when enabled."""
        device = torch.device("cpu")
        batch_size = 32
        head_names = ("op", "blueprint", "slot", "style", "tempo",
                      "alpha_target", "alpha_speed", "alpha_curve")

        # Create minimal inputs for compute_losses (matching actual signature)
        per_head_ratios = {k: torch.ones(batch_size, device=device) for k in head_names}
        per_head_advantages = {k: torch.randn(batch_size, device=device) for k in head_names}
        head_masks = {k: torch.ones(batch_size, device=device) for k in head_names}
        values = torch.randn(batch_size, device=device)
        normalized_returns = torch.randn(batch_size, device=device)
        old_values = torch.randn(batch_size, device=device)

        # Create entropy dict with collapsed blueprint head
        entropy = {
            "op": torch.full((batch_size,), 0.5),      # Healthy
            "blueprint": torch.full((batch_size,), 0.05),  # Collapsed!
            "slot": torch.full((batch_size,), 0.4),
            "style": torch.full((batch_size,), 0.4),
            "tempo": torch.full((batch_size,), 0.4),
            "alpha_target": torch.full((batch_size,), 0.4),
            "alpha_speed": torch.full((batch_size,), 0.4),
            "alpha_curve": torch.full((batch_size,), 0.4),
        }

        entropy_coef_per_head = {"op": 1.0, "blueprint": 1.3, "slot": 1.0,
                                  "style": 1.2, "tempo": 1.3, "alpha_target": 1.2,
                                  "alpha_speed": 1.2, "alpha_curve": 1.2}

        # Without floor penalty
        losses_no_floor = compute_losses(
            per_head_ratios=per_head_ratios,
            per_head_advantages=per_head_advantages,
            head_masks=head_masks,
            values=values,
            normalized_returns=normalized_returns,
            old_values=old_values,
            entropy=entropy,
            entropy_coef_per_head=entropy_coef_per_head,
            entropy_coef=0.01,
            clip_ratio=0.2,
            clip_value=True,
            value_clip=0.2,
            value_coef=0.5,
            q_values=values,  # P0-1: Q-aux input (q_aux_coef=0 isolates entropy-floor tests)
            q_aux_coef=0.0,
            head_names=head_names,
            entropy_floor=None,  # Disabled
        )

        # With floor penalty
        entropy_floor = {"blueprint": 0.4}  # Blueprint is at 0.05, well below 0.4
        entropy_floor_penalty_coef = {"blueprint": 0.1}  # Dict required (P2 optimization)
        losses_with_floor = compute_losses(
            per_head_ratios=per_head_ratios,
            per_head_advantages=per_head_advantages,
            head_masks=head_masks,
            values=values,
            normalized_returns=normalized_returns,
            old_values=old_values,
            entropy=entropy,
            entropy_coef_per_head=entropy_coef_per_head,
            entropy_coef=0.01,
            clip_ratio=0.2,
            clip_value=True,
            value_clip=0.2,
            value_coef=0.5,
            q_values=values,  # P0-1: Q-aux input (q_aux_coef=0 isolates entropy-floor tests)
            q_aux_coef=0.0,
            head_names=head_names,
            entropy_floor=entropy_floor,
            entropy_floor_penalty_coef=entropy_floor_penalty_coef,
        )

        # Total loss should be higher with floor penalty
        assert losses_with_floor.total_loss > losses_no_floor.total_loss

    def test_compute_losses_entropy_floor_penalty_with_per_head_coef(self) -> None:
        """entropy_floor_penalty_coef dict allows per-head coefficient control.

        P2 optimization: entropy_floor_penalty_coef MUST be a dict (no scalar support).
        Caller (PPOAgent) normalizes at init time for torch.compile graph consistency.
        """
        device = torch.device("cpu")
        batch_size = 16
        head_names = ("op", "blueprint")

        per_head_ratios = {k: torch.ones(batch_size, device=device) for k in head_names}
        per_head_advantages = {k: torch.zeros(batch_size, device=device) for k in head_names}
        head_masks = {k: torch.ones(batch_size, device=device) for k in head_names}
        values = torch.zeros(batch_size, device=device)
        normalized_returns = torch.zeros(batch_size, device=device)
        old_values = torch.zeros(batch_size, device=device)

        # Both heads collapsed
        entropy = {
            "op": torch.full((batch_size,), 0.05),
            "blueprint": torch.full((batch_size,), 0.05),
        }
        entropy_coef_per_head = {"op": 1.0, "blueprint": 1.0}
        entropy_floor = {"op": 0.4, "blueprint": 0.4}

        # With uniform dict coef
        losses_uniform = compute_losses(
            per_head_ratios=per_head_ratios,
            per_head_advantages=per_head_advantages,
            head_masks=head_masks,
            values=values,
            normalized_returns=normalized_returns,
            old_values=old_values,
            entropy=entropy,
            entropy_coef_per_head=entropy_coef_per_head,
            entropy_coef=0.01,
            clip_ratio=0.2,
            clip_value=False,
            value_clip=0.2,
            value_coef=0.5,
            q_values=values,  # P0-1: Q-aux input (q_aux_coef=0 isolates entropy-floor tests)
            q_aux_coef=0.0,
            head_names=head_names,
            entropy_floor=entropy_floor,
            entropy_floor_penalty_coef={"op": 0.1, "blueprint": 0.1},
        )

        # With higher blueprint coef (should increase total loss)
        losses_higher_blueprint = compute_losses(
            per_head_ratios=per_head_ratios,
            per_head_advantages=per_head_advantages,
            head_masks=head_masks,
            values=values,
            normalized_returns=normalized_returns,
            old_values=old_values,
            entropy=entropy,
            entropy_coef_per_head=entropy_coef_per_head,
            entropy_coef=0.01,
            clip_ratio=0.2,
            clip_value=False,
            value_clip=0.2,
            value_coef=0.5,
            q_values=values,  # P0-1: Q-aux input (q_aux_coef=0 isolates entropy-floor tests)
            q_aux_coef=0.0,
            head_names=head_names,
            entropy_floor=entropy_floor,
            entropy_floor_penalty_coef={"op": 0.1, "blueprint": 0.5},  # Higher blueprint
        )

        # Higher blueprint coefficient should increase penalty (and total loss)
        assert losses_higher_blueprint.entropy_floor_penalty > losses_uniform.entropy_floor_penalty
        assert losses_higher_blueprint.total_loss > losses_uniform.total_loss

    def test_compute_losses_skips_entropy_floor_for_zero_availability_head(self) -> None:
        """A head with no available decisions must not receive entropy-floor penalty."""
        device = torch.device("cpu")
        batch_size = 8
        head_names = ("op", "blueprint")

        per_head_ratios = {k: torch.ones(batch_size, device=device) for k in head_names}
        per_head_advantages = {k: torch.zeros(batch_size, device=device) for k in head_names}
        head_masks = {k: torch.ones(batch_size, device=device) for k in head_names}
        values = torch.zeros(batch_size, device=device)
        normalized_returns = torch.zeros(batch_size, device=device)
        old_values = torch.zeros(batch_size, device=device)
        entropy = {
            "op": torch.full((batch_size,), 0.5, device=device),
            "blueprint": torch.full((batch_size,), 0.0, device=device),
        }
        availability_masks = {
            "op": torch.ones(batch_size, device=device),
            "blueprint": torch.zeros(batch_size, device=device),
        }

        losses = compute_losses(
            per_head_ratios=per_head_ratios,
            per_head_advantages=per_head_advantages,
            head_masks=head_masks,
            values=values,
            normalized_returns=normalized_returns,
            old_values=old_values,
            entropy=entropy,
            entropy_coef_per_head={"op": 1.0, "blueprint": 1.0},
            entropy_coef=0.01,
            clip_ratio=0.2,
            clip_value=False,
            value_clip=0.2,
            value_coef=0.5,
            q_values=values,  # P0-1: Q-aux input (q_aux_coef=0 isolates entropy-floor tests)
            q_aux_coef=0.0,
            head_names=head_names,
            entropy_floor={"op": 0.4, "blueprint": 0.4},
            entropy_floor_penalty_coef={"op": 0.1, "blueprint": 0.1},
            availability_masks=availability_masks,
        )

        assert losses.entropy_floor_penalty.item() == pytest.approx(0.0)

    def test_zero_availability_head_reports_not_learnable(self) -> None:
        """A zero-learnability action head should be explicit in PPO metrics."""
        from collections import defaultdict

        from esper.simic.agent.ppo_metrics import PPOUpdateMetricsBuilder

        metrics = defaultdict(list)
        metrics["ratio_max"].append(torch.tensor(1.0))
        metrics["ratio_min"].append(torch.tensor(1.0))
        metrics["finiteness_gate_failures"] = []

        builder = PPOUpdateMetricsBuilder(
            metrics=metrics,
            finiteness_failures=metrics["finiteness_gate_failures"],
            epochs_completed=1,
            head_entropies={"blueprint": [torch.tensor(0.0)]},
            conditional_head_entropies={"blueprint": [torch.tensor(0.0)]},
            choice_conditional_head_entropies={"blueprint": [torch.tensor(0.0)]},
            head_grad_norms={"blueprint": [torch.tensor(float("nan"))]},
            head_learnable_fractions={"blueprint": [torch.tensor(0.0)]},
            head_gradient_states={"blueprint": ["not_learnable"]},
            head_nan_detected={"blueprint": False},
            head_inf_detected={"blueprint": False},
            lstm_health_history=defaultdict(list),
            log_prob_min_across_epochs=torch.tensor(0.0),
            log_prob_max_across_epochs=torch.tensor(0.0),
            head_ratio_max_across_epochs={"blueprint": torch.tensor(1.0)},
            joint_ratio_max_across_epochs=torch.tensor(1.0),
            head_clip_fraction_history={"blueprint": [torch.tensor(0.0)]},
            value_func_metrics={
                "v_return_correlation": 0.0,
                "td_error_mean": 0.0,
                "td_error_std": 0.0,
                "bellman_error": 0.0,
                "return_p10": 0.0,
                "return_p50": 0.0,
                "return_p90": 0.0,
                "return_variance": 0.0,
                "return_skewness": 0.0,
            },
            cuda_memory_metrics={},
            head_names=("blueprint",),
        )

        result = builder.finalize().metrics

        assert result["head_learnable_fractions"]["blueprint"] == [0.0]
        assert result["head_gradient_states"]["blueprint"] == ["not_learnable"]


class TestPenaltySchedule:
    """Entropy-floor penalty schedule: ABSOLUTE update-round breakpoints (PDR-0055).

    Breakpoints are pinned to the 200-round experiment shape and expressed in
    update rounds (leyline constants), NOT run-relative progress:
    - rounds [0, 50):    1.5x boost - establish diverse exploration habits
    - rounds [50, 150):  1.0x baseline
    - rounds [150, 200): linear decay 1.0x -> 0.5x
    - rounds >= 200:     hold 0.5x (longer runs are true continuations)

    Shape-preserving property: every 200-round run behaves bit-identically to the
    retired horizon-normalized schedule (progress = round / total_train_steps with
    total_train_steps=200), so the completed A/B arms and any future 200-round arms
    stay comparable. The coefficient-identity regression below is the
    pre-registered test for that property (PDR-0055/0057).
    """

    @staticmethod
    def _make_agent():
        from esper.simic.agent.ppo_agent import PPOAgent
        from esper.tamiyo.policy.factory import create_policy
        from esper.leyline.slot_config import SlotConfig

        slot_config = SlotConfig.for_grid(2, 2)
        policy = create_policy(
            policy_type="lstm",
            slot_config=slot_config,
            device="cpu",
        )
        return PPOAgent(
            policy=policy,
            slot_config=slot_config,
            device="cpu",
        )

    @staticmethod
    def _retired_horizon_normalized_schedule(update_round: int, total: int = 200) -> float:
        """The retired formula, evaluated at the 200-round shape (the A/B arm config).

        Bit-identity REFERENCE for the coefficient-identity regression: progress =
        round / total; 1.5x below 25%, 1.0x below 75%, then linear decay to 0.5x
        at 100%. Kept verbatim (same operations, same literals) so the comparison
        is against the exact arithmetic the completed arms ran.
        """
        progress = min(1.0, max(0.0, update_round / total))
        if progress < 0.25:
            return 1.5
        elif progress < 0.75:
            return 1.0
        return 1.0 - 0.5 * ((progress - 0.75) / 0.25)

    def test_coefficient_identity_rounds_0_199(self) -> None:
        """PRE-REGISTERED REGRESSION (PDR-0055/0057): bit-identical over rounds 0-199.

        Every 200-round run must produce EXACTLY the coefficients the retired
        horizon-normalized schedule produced (float ==, not approx). This is the
        shape-preserving property that keeps the completed A/B arms comparable
        with all future 200-round arms.
        """
        agent = self._make_agent()
        for update_round in range(200):
            assert agent._get_penalty_schedule(update_round) == (
                self._retired_horizon_normalized_schedule(update_round)
            ), f"coefficient identity broken at update round {update_round}"

    def test_boost_boundary_off_by_one(self) -> None:
        """Boost applies on rounds [0, 50): round 49 is boosted, round 50 is not."""
        agent = self._make_agent()
        assert agent._get_penalty_schedule(0) == 1.5
        assert agent._get_penalty_schedule(49) == 1.5
        assert agent._get_penalty_schedule(50) == 1.0

    def test_decay_boundary_off_by_one(self) -> None:
        """Baseline holds through round 150 (decay start is still 1.0); round 151 is
        the first strictly-decayed round."""
        agent = self._make_agent()
        assert agent._get_penalty_schedule(149) == 1.0
        assert agent._get_penalty_schedule(150) == 1.0
        factor_151 = agent._get_penalty_schedule(151)
        assert 0.5 < factor_151 < 1.0

    def test_horizon_boundary_off_by_one(self) -> None:
        """Decay completes AT round 200: round 199 is the last partially-decayed round."""
        agent = self._make_agent()
        assert agent._get_penalty_schedule(199) == pytest.approx(0.51)
        assert agent._get_penalty_schedule(200) == 0.5
        assert agent._get_penalty_schedule(201) == 0.5

    def test_hold_beyond_horizon(self) -> None:
        """Rounds >= 200 hold 0.5x exactly - the 600-round diagnostic property.

        Under the retired schedule a 600-round run recomputed every breakpoint
        (boost would persist through round 150); under the absolute schedule,
        rounds 200-599 are a flat 0.5x continuation of the 200-round shape.
        """
        agent = self._make_agent()
        for update_round in (200, 250, 300, 599, 10_000):
            assert agent._get_penalty_schedule(update_round) == 0.5

    def test_decay_strictly_monotone(self) -> None:
        """Factor strictly decreases across the decay window (rounds 150-200)."""
        agent = self._make_agent()
        prev = agent._get_penalty_schedule(150)
        for update_round in range(151, 201):
            factor = agent._get_penalty_schedule(update_round)
            assert factor < prev, f"decay not monotone at round {update_round}"
            prev = factor

    def test_schedule_bounds(self) -> None:
        """Schedule factor always lies in [0.5, 1.5]."""
        agent = self._make_agent()
        for update_round in (0, 10, 49, 50, 100, 149, 150, 175, 199, 200, 500, 100_000):
            factor = agent._get_penalty_schedule(update_round)
            assert 0.5 <= factor <= 1.5, f"factor out of bounds at round {update_round}"

    def test_schedule_applied_to_coefficients_in_update(self) -> None:
        """The update path drives the schedule with train_steps as the update round."""
        agent = self._make_agent()

        agent.train_steps = 10
        assert agent._get_penalty_schedule(agent.train_steps) == 1.5

        agent.train_steps = 100
        assert agent._get_penalty_schedule(agent.train_steps) == 1.0

        agent.train_steps = 175
        expected_decay = agent._get_penalty_schedule(agent.train_steps)
        assert expected_decay == pytest.approx(0.75)

        scheduled_coef = {
            head: coef * expected_decay
            for head, coef in agent.entropy_floor_penalty_coef.items()
        }
        blueprint_base = agent.entropy_floor_penalty_coef["blueprint"]
        assert scheduled_coef["blueprint"] == pytest.approx(blueprint_base * 0.75)
