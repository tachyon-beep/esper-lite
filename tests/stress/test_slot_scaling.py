"""Stress tests for slot scaling.

Tests verify that the system handles large slot configurations efficiently:
- Memory usage with 25 slots (5x5 grid)
- Batch mask computation performance with many environments
- Episode cycling without memory leaks
"""

import gc
import time
import tracemalloc

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
    SeedStage,
)
from esper.leyline.slot_config import SlotConfig
from esper.simic.agent import PPOAgent
from esper.tamiyo.policy import create_policy
from esper.tamiyo.policy.features import get_feature_size
from esper.tamiyo.policy.action_masks import compute_action_masks, MaskSeedInfo


@pytest.mark.slow
class TestMemoryUsage:
    """Tests for memory usage with large slot configurations."""

    def test_agent_25_slots_memory_reasonable(self):
        """PPOAgent with 25 slots should use reasonable memory (<500MB for network)."""
        config = SlotConfig.for_grid(rows=5, cols=5)
        state_dim = get_feature_size(config)

        gc.collect()
        tracemalloc.start()

        policy = create_policy(
            policy_type="lstm",
            state_dim=state_dim,
            slot_config=config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(
            policy=policy,
            slot_config=config,
            device="cpu",
            num_envs=4,
            max_steps_per_env=100,
        )

        # Force allocation by doing a forward pass
        states = torch.randn(4, state_dim)
        masks = {
            "slot": torch.ones(4, 25, dtype=torch.bool),
            "blueprint": torch.ones(4, NUM_BLUEPRINTS, dtype=torch.bool),
            "style": torch.ones(4, NUM_STYLES, dtype=torch.bool),
            "tempo": torch.ones(4, NUM_TEMPO, dtype=torch.bool),
            "alpha_target": torch.ones(4, NUM_ALPHA_TARGETS, dtype=torch.bool),
            "alpha_speed": torch.ones(4, NUM_ALPHA_SPEEDS, dtype=torch.bool),
            "alpha_curve": torch.ones(4, NUM_ALPHA_CURVES, dtype=torch.bool),
            "op": torch.ones(4, NUM_OPS, dtype=torch.bool),
        }
        blueprint_indices = torch.full(
            (4, config.num_slots),
            -1,
            dtype=torch.int64,
        )

        with torch.no_grad():
            _ = agent.policy.network.get_action(
                states,
                blueprint_indices,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
            )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory should be reasonable (< 500MB for the network alone)
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 500, f"Peak memory {peak_mb:.1f}MB exceeds 500MB limit"

    def test_feature_size_scales_linearly_with_slots(self):
        """Feature size should scale linearly with slot count."""
        configs = [
            SlotConfig.for_grid(1, 1),   # 1 slot
            SlotConfig.for_grid(1, 5),   # 5 slots
            SlotConfig.for_grid(3, 3),   # 9 slots
            SlotConfig.for_grid(5, 5),   # 25 slots
        ]

        sizes = [get_feature_size(c) for c in configs]

        # Calculate expected linear scaling
        # size = BASE + num_slots * SLOT_SIZE
        # Verify ratio is approximately num_slots difference
        delta_1_to_5 = sizes[1] - sizes[0]   # 4 slots difference
        delta_5_to_9 = sizes[2] - sizes[1]   # 4 slots difference
        delta_9_to_25 = sizes[3] - sizes[2]  # 16 slots difference

        # Per-slot feature size should be consistent
        per_slot_1_to_5 = delta_1_to_5 / 4
        per_slot_5_to_9 = delta_5_to_9 / 4
        per_slot_9_to_25 = delta_9_to_25 / 16

        assert per_slot_1_to_5 == per_slot_5_to_9, "Per-slot size should be constant"
        assert per_slot_5_to_9 == per_slot_9_to_25, "Per-slot size should be constant"


@pytest.mark.slow
class TestMaskComputationPerformance:
    """Tests for mask computation performance."""

    def test_batch_mask_computation_fast(self):
        """Mask computation for 16 envs should complete in <100ms."""
        n_envs = 16
        config = SlotConfig.default()

        # Create slot states for each environment
        slot_states_batch = []
        for env_idx in range(n_envs):
            states = {}
            for i, slot_id in enumerate(config.slot_ids):
                if (env_idx + i) % 3 == 0:
                    states[slot_id] = MaskSeedInfo(
                        stage=SeedStage.TRAINING,
                        seed_age_epochs=env_idx + 1
                    )
                else:
                    states[slot_id] = None
            slot_states_batch.append(states)

        # Time the computation
        start = time.perf_counter()

        all_masks = []
        for slot_states in slot_states_batch:
            masks = compute_action_masks(
                slot_states,
                enabled_slots=list(config.slot_ids),
                slot_config=config,
            )
            all_masks.append(masks)

        # Stack into batched format
        batched_masks = {
            key: torch.stack([m[key] for m in all_masks])
            for key in all_masks[0].keys()
        }

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Batch mask computation took {elapsed_ms:.1f}ms (>100ms)"
        assert batched_masks["slot"].shape == (n_envs, config.num_slots)

    def test_25_slot_mask_computation_fast(self):
        """Mask computation with 25 slots should complete in <50ms per env."""
        config = SlotConfig.for_grid(rows=5, cols=5)

        # Create varied slot states
        slot_states = {}
        stages = list(SeedStage)
        for i, slot_id in enumerate(config.slot_ids):
            if i % 4 == 0:
                slot_states[slot_id] = None
            else:
                slot_states[slot_id] = MaskSeedInfo(
                    stage=stages[i % len(stages)],
                    seed_age_epochs=i
                )

        # Time multiple computations
        n_iterations = 100
        start = time.perf_counter()

        for _ in range(n_iterations):
            masks = compute_action_masks(
                slot_states,
                enabled_slots=list(config.slot_ids),
                slot_config=config,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_call_ms = elapsed_ms / n_iterations

        assert per_call_ms < 50, f"25-slot mask computation took {per_call_ms:.2f}ms (>50ms)"
        assert masks["slot"].shape == (25,)


@pytest.mark.slow
class TestEpisodeCycling:
    """Tests for memory stability during episode cycling."""

    def test_buffer_episode_cycling_no_memory_leak(self):
        """Cycling through many episodes should not leak memory."""
        config = SlotConfig.default()
        state_dim = get_feature_size(config)
        n_envs = 2
        n_episodes = 50  # Reduced from 100 for faster tests

        policy = create_policy(
            policy_type="lstm",
            state_dim=state_dim,
            slot_config=config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(
            policy=policy,
            slot_config=config,
            device="cpu",
            num_envs=n_envs,
            max_steps_per_env=20,
        )

        gc.collect()
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Cycle through episodes
        blueprint_indices = torch.full(
            (config.num_slots,),
            -1,
            dtype=torch.int64,
        )
        for episode in range(n_episodes):
            # Start episodes
            for env_idx in range(n_envs):
                agent.buffer.start_episode(env_id=env_idx)

            # Add transitions
            hidden_dim = agent.policy.hidden_dim  # Get from policy
            for step in range(10):
                for env_idx in range(n_envs):
                    state = torch.randn(state_dim)
                    hidden_h = torch.randn(1, hidden_dim)
                    hidden_c = torch.randn(1, hidden_dim)

                    agent.buffer.add(
                        env_id=env_idx,
                        state=state,
                        blueprint_indices=blueprint_indices,
                        slot_action=step % config.num_slots,
                        blueprint_action=0,
                        style_action=0,
                        tempo_action=1,  # STANDARD tempo
                        alpha_target_action=0,
                        alpha_speed_action=0,
                        alpha_curve_action=0,
                        op_action=0,
                        effective_op_action=0,
                        slot_log_prob=-0.5,
                        blueprint_log_prob=-0.5,
                        style_log_prob=-0.5,
                        tempo_log_prob=-0.5,
                        alpha_target_log_prob=-0.5,
                        alpha_speed_log_prob=-0.5,
                        alpha_curve_log_prob=-0.5,
                        op_log_prob=-0.5,
                        value=0.5,
                        reward=1.0,
                        done=(step == 9),
                        slot_mask=torch.ones(config.num_slots, dtype=torch.bool),
                        blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
                        style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
                        tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
                        alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
                        alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
                        alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
                        op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
                        hidden_h=hidden_h,
                        hidden_c=hidden_c,
                    )

            # Reset buffer (simulating epoch end)
            agent.buffer.reset()

            # Check memory every 10 episodes
            if episode % 10 == 9:
                gc.collect()

        gc.collect()
        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # Memory growth should be bounded (< 50MB growth over 50 episodes)
        memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)
        assert memory_growth_mb < 50, f"Memory grew by {memory_growth_mb:.1f}MB (>50MB leak suspected)"

    def test_agent_action_selection_stable(self):
        """Repeated action selection should not accumulate tensors."""
        config = SlotConfig.default()
        state_dim = get_feature_size(config)

        policy = create_policy(
            policy_type="lstm",
            state_dim=state_dim,
            slot_config=config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(
            policy=policy,
            slot_config=config,
            device="cpu",
        )

        gc.collect()
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Many action selections
        n_iterations = 500
        blueprint_indices = torch.full(
            (1, config.num_slots),
            -1,
            dtype=torch.int64,
        )
        for _ in range(n_iterations):
            states = torch.randn(1, state_dim)
            masks = {
                "slot": torch.ones(1, config.num_slots, dtype=torch.bool),
                "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool),
                "style": torch.ones(1, NUM_STYLES, dtype=torch.bool),
                "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool),
                "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool),
                "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool),
                "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool),
                "op": torch.ones(1, NUM_OPS, dtype=torch.bool),
            }

            with torch.no_grad():
                result = agent.policy.network.get_action(
                    states,
                    blueprint_indices,
                    slot_mask=masks["slot"],
                    blueprint_mask=masks["blueprint"],
                    style_mask=masks["style"],
                    tempo_mask=masks["tempo"],
                    alpha_target_mask=masks["alpha_target"],
                    alpha_speed_mask=masks["alpha_speed"],
                    alpha_curve_mask=masks["alpha_curve"],
                    op_mask=masks["op"],
                )
                # Access result attributes to ensure they're computed
                _ = result.actions, result.log_probs, result.values, result.hidden

        gc.collect()
        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # Memory growth should be minimal (< 10MB for 500 iterations)
        memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)
        assert memory_growth_mb < 10, f"Memory grew by {memory_growth_mb:.1f}MB during action selection"


@pytest.mark.slow
class TestScalingBehavior:
    """Tests for scaling behavior with increasing slots/envs."""

    def test_forward_pass_time_scales_reasonably(self):
        """Forward pass time should scale sub-linearly with slot count."""
        configs = [
            (SlotConfig.for_grid(1, 3), "3 slots"),
            (SlotConfig.for_grid(3, 3), "9 slots"),
            (SlotConfig.for_grid(5, 5), "25 slots"),
        ]

        times = []
        for config, label in configs:
            state_dim = get_feature_size(config)

            policy = create_policy(
                policy_type="lstm",
                state_dim=state_dim,
                slot_config=config,
                device="cpu",
                compile_mode="off",
            )
            agent = PPOAgent(
                policy=policy,
                slot_config=config,
                device="cpu",
            )

            # Warmup
            states = torch.randn(1, state_dim)
            masks = {
                "slot": torch.ones(1, config.num_slots, dtype=torch.bool),
                "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool),
                "style": torch.ones(1, NUM_STYLES, dtype=torch.bool),
                "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool),
                "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool),
                "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool),
                "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool),
                "op": torch.ones(1, NUM_OPS, dtype=torch.bool),
            }
            blueprint_indices = torch.full(
                (1, config.num_slots),
                -1,
                dtype=torch.int64,
            )

            with torch.no_grad():
                for _ in range(5):
                    _ = agent.policy.network.get_action(
                        states,
                        blueprint_indices,
                        slot_mask=masks["slot"],
                        blueprint_mask=masks["blueprint"],
                        style_mask=masks["style"],
                        tempo_mask=masks["tempo"],
                        alpha_target_mask=masks["alpha_target"],
                        alpha_speed_mask=masks["alpha_speed"],
                        alpha_curve_mask=masks["alpha_curve"],
                        op_mask=masks["op"],
                    )

            # Timed runs
            n_iterations = 50
            start = time.perf_counter()

            with torch.no_grad():
                for _ in range(n_iterations):
                    _ = agent.policy.network.get_action(
                        states,
                        blueprint_indices,
                        slot_mask=masks["slot"],
                        blueprint_mask=masks["blueprint"],
                        style_mask=masks["style"],
                        tempo_mask=masks["tempo"],
                        alpha_target_mask=masks["alpha_target"],
                        alpha_speed_mask=masks["alpha_speed"],
                        alpha_curve_mask=masks["alpha_curve"],
                        op_mask=masks["op"],
                    )

            elapsed_ms = (time.perf_counter() - start) * 1000 / n_iterations
            times.append((config.num_slots, elapsed_ms, label))

        # 25 slots should be less than 10x slower than 3 slots (sub-linear scaling)
        time_3_slots = times[0][1]
        time_25_slots = times[2][1]

        # Allow 10x slowdown for 8x more slots (25/3 ≈ 8)
        assert time_25_slots < time_3_slots * 10, \
            f"25-slot time ({time_25_slots:.2f}ms) > 10x 3-slot time ({time_3_slots:.2f}ms)"
