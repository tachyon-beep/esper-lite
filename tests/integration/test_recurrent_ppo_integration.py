"""Integration tests for recurrent PPO training."""

import torch
from esper.simic.ppo import PPOAgent
from esper.simic.networks import RecurrentActorCritic


class TestRecurrentIntegration:
    """End-to-end tests for recurrent policy."""

    def test_full_episode_training_loop(self):
        """Test complete training loop with recurrent policy."""
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=8,
            device='cpu',
        )

        # Simulate 2 episodes of 12 steps each
        for episode in range(2):
            agent.recurrent_buffer.start_episode(env_id=0)
            hidden = None

            for step in range(12):
                state = torch.randn(27)
                mask = torch.ones(7, dtype=torch.bool)
                mask[3] = False  # Mask one action

                action, log_prob, value, hidden = agent.get_action(state, mask, hidden)

                agent.store_recurrent_transition(
                    state=state,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=0.1 * (step + 1),
                    done=(step == 11),
                    action_mask=mask,
                    env_id=0,
                )

            agent.recurrent_buffer.end_episode(env_id=0)

        # Should have 2 episodes worth of chunks
        # 12 steps / 8 chunk = 2 chunks per episode = 4 total
        metrics = agent.update_recurrent(n_epochs=2)

        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert metrics['policy_loss'] != 0.0

    def test_multi_env_parallel_episodes(self):
        """Test parallel episodes from multiple environments."""
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=8,
            device='cpu',
        )

        # 4 parallel envs, each running 10-step episodes
        hiddens = {i: None for i in range(4)}

        for env_id in range(4):
            agent.recurrent_buffer.start_episode(env_id=env_id)

        for step in range(10):
            for env_id in range(4):
                state = torch.randn(27)
                mask = torch.ones(7, dtype=torch.bool)

                action, log_prob, value, hiddens[env_id] = agent.get_action(
                    state, mask, hiddens[env_id]
                )

                agent.store_recurrent_transition(
                    state=state,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=0.1,
                    done=(step == 9),
                    action_mask=mask,
                    env_id=env_id,
                )

        for env_id in range(4):
            agent.recurrent_buffer.end_episode(env_id=env_id)

        # Each env has 10 steps -> 2 chunks of 8 (padded)
        # 4 envs * 2 chunks = 8 chunks total
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')
        assert len(chunks) == 8

        metrics = agent.update_recurrent(n_epochs=2)
        assert 'policy_loss' in metrics

    def test_advantages_nonzero_with_rewards(self):
        """Critical: Verify GAE advantages are computed and non-zero."""
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=4,
            device='cpu',
        )

        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None

        for step in range(4):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)

            agent.store_recurrent_transition(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=float(step + 1),  # 1, 2, 3, 4
                done=(step == 3),
                action_mask=mask,
                env_id=0,
            )

        agent.recurrent_buffer.end_episode(env_id=0)
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')

        # Advantages MUST be non-zero (this was a critical bug)
        assert chunks[0]['advantages'].abs().sum() > 0

    def test_hidden_state_continuity_within_episode(self):
        """Verify hidden state evolves within episode."""
        torch.manual_seed(42)
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            device='cpu',
        )

        state = torch.randn(27)
        mask = torch.ones(7, dtype=torch.bool)

        # Get values at different points in "history"
        _, _, value_t0, hidden = agent.get_action(state, mask, None)

        # Build up context
        for _ in range(5):
            _, _, _, hidden = agent.get_action(torch.randn(27), mask, hidden)

        _, _, value_t5, _ = agent.get_action(state, mask, hidden)

        # Same state, different context -> different values
        assert not torch.isclose(torch.tensor(value_t0), torch.tensor(value_t5), atol=1e-4)

    def test_episode_boundary_resets_hidden(self):
        """Hidden state should reset at episode boundaries."""
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            device='cpu',
        )

        # Get hidden after some steps
        hidden = None
        for _ in range(5):
            _, _, _, hidden = agent.get_action(
                torch.randn(27), torch.ones(7, dtype=torch.bool), hidden
            )

        assert hidden is not None
        assert hidden[0].abs().sum() > 0  # Hidden evolved

        # Episode boundary: reset to None
        hidden = None  # This is what vectorized.py does on done=True

        # Get fresh hidden
        _, _, _, fresh_hidden = agent.get_action(
            torch.randn(27), torch.ones(7, dtype=torch.bool), hidden
        )

        # Verify fresh hidden was initialized from zeros by comparing to explicit zeros
        h_zeros, c_zeros = agent.network.get_initial_hidden(batch_size=1, device='cpu')
        assert h_zeros.abs().sum() == 0, "Initial hidden h should be zeros"
        assert c_zeros.abs().sum() == 0, "Initial hidden c should be zeros"

        # The fresh_hidden should produce same result as explicit zeros input
        test_state = torch.randn(27)
        test_mask = torch.ones(7, dtype=torch.bool)

        _, _, value_from_none, _ = agent.get_action(test_state, test_mask, None)
        _, _, value_from_zeros, _ = agent.get_action(
            test_state, test_mask, (h_zeros, c_zeros)
        )

        assert torch.isclose(
            torch.tensor(value_from_none), torch.tensor(value_from_zeros), atol=1e-6
        ), f"hidden=None should equal explicit zeros: {value_from_none} vs {value_from_zeros}"

    def test_ratio_near_one_before_update(self):
        """PPO ratio should be ~1.0 before any gradient updates (same policy).

        This is a CRITICAL test - if this fails, the hidden state handling
        is broken and the PPO ratio is biased.
        """
        torch.manual_seed(42)
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=4,
            device='cpu',
        )

        # Collect episode
        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(4):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=0.1, done=(i == 3), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        # Compute GAE and get chunks
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')
        chunk = chunks[0]

        # Recompute log_probs with SAME initial hidden (zeros for episode start)
        with torch.no_grad():
            log_probs_recomputed, _, _, _ = agent.network.evaluate_actions(
                chunk['states'],
                chunk['actions'],
                chunk['action_masks'],
                hidden=(chunk['initial_hidden_h'], chunk['initial_hidden_c']),
            )

        # Extract valid (non-padded) positions
        valid = chunk['valid_mask'].squeeze(0)
        old_log_probs = chunk['old_log_probs'].squeeze(0)[valid]
        new_log_probs = log_probs_recomputed.squeeze(0)[valid]

        # Ratio should be ~1.0 (same policy, same hidden)
        # Note: Using atol=1e-3 due to floating point accumulation in LSTM
        # (tanh/sigmoid approximations, softmax numerical stability, etc.)
        ratio = torch.exp(new_log_probs - old_log_probs)
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3), (
            f"Ratio should be ~1.0 before updates, got {ratio}"
        )

    def test_gradient_flows_to_early_timesteps(self):
        """CRITICAL: Verify BPTT propagates gradients to early timesteps.

        If this test fails, the LSTM isn't learning temporal dependencies.
        Gradients should flow from later rewards back to early policy params.

        This test verifies TWO things:
        1. GAE correctly bootstraps reward from final step to early steps (advantages non-zero)
        2. LSTM recurrent weights change (gradients flowed through time via weight_hh)
        """
        torch.manual_seed(42)
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=8,
            device='cpu',
        )

        # Collect episode with reward ONLY at LAST step
        # If BPTT works, early steps should have non-zero advantages (GAE bootstrap)
        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(8):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            # Reward structure: ONLY last step gets reward
            reward = 10.0 if i == 7 else 0.0
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=reward, done=(i == 7), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        # VERIFICATION 1: Early timesteps have non-zero advantage from GAE bootstrap
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')
        advantages = chunks[0]['advantages'].squeeze(0)

        # Early timesteps should have non-zero advantage from gamma-discounted final reward
        assert advantages[0].abs() > 0.01, (
            f"Early timestep advantage is ~0 ({advantages[0]:.4f}). "
            f"GAE not bootstrapping reward backward through time!"
        )

        # Get LSTM weights before update
        lstm_weight_hh_before = agent.network.lstm.weight_hh_l0.clone()
        lstm_weight_ih_before = agent.network.lstm.weight_ih_l0.clone()

        # Run update (need to re-add data since compute_gae consumed it for chunks)
        # Actually, update_recurrent calls compute_gae internally, so recreate episode
        agent.recurrent_buffer.clear()
        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        torch.manual_seed(42)  # Same seed for reproducibility
        for i in range(8):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            reward = 10.0 if i == 7 else 0.0
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=reward, done=(i == 7), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        metrics = agent.update_recurrent(n_epochs=1)

        # VERIFICATION 2: LSTM recurrent weights changed (BPTT worked)
        lstm_weight_hh_after = agent.network.lstm.weight_hh_l0
        weight_hh_diff = (lstm_weight_hh_after - lstm_weight_hh_before).abs().sum()

        # weight_hh connects h(t) -> h(t+1), so if it changed, gradients flowed through time
        assert weight_hh_diff > 1e-6, (
            f"LSTM weight_hh unchanged ({weight_hh_diff:.2e}). "
            f"Gradients not flowing through recurrent connection - BPTT broken!"
        )

        # VERIFICATION 3: Input weights also changed (full gradient flow)
        lstm_weight_ih_after = agent.network.lstm.weight_ih_l0
        weight_ih_diff = (lstm_weight_ih_after - lstm_weight_ih_before).abs().sum()
        assert weight_ih_diff > 1e-6, (
            f"LSTM weight_ih unchanged ({weight_ih_diff:.2e}). "
            f"Gradients not reaching input projection!"
        )

    def test_value_targets_equal_advantages_plus_values(self):
        """Verify GAE relationship: returns = advantages + old_values."""
        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=4,
            device='cpu',
        )

        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(4):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=float(i + 1), done=(i == 3), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')
        chunk = chunks[0]

        # GAE property: returns = advantages + values
        computed_returns = chunk['advantages'] + chunk['old_values']
        assert torch.allclose(computed_returns, chunk['returns'], atol=1e-5), (
            f"GAE violated: returns != advantages + values"
        )

    # === Edge Case Tests (v4) ===

    def test_single_step_episode(self):
        """Single-step episode should work correctly."""
        agent = PPOAgent(
            state_dim=27, action_dim=7, recurrent=True,
            lstm_hidden_dim=64, chunk_length=4, device='cpu',
        )

        agent.recurrent_buffer.start_episode(env_id=0)
        state = torch.randn(27)
        mask = torch.ones(7, dtype=torch.bool)
        action, log_prob, value, hidden = agent.get_action(state, mask, None)
        agent.store_recurrent_transition(
            state=state, action=action, log_prob=log_prob, value=value,
            reward=1.0, done=True, action_mask=mask, env_id=0,
        )
        agent.recurrent_buffer.end_episode(env_id=0)

        # Should produce 1 chunk (padded)
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')
        assert len(chunks) == 1
        assert chunks[0]['valid_mask'].sum() == 1  # Only 1 valid step

    def test_exact_chunk_length_episode(self):
        """Episode exactly matching chunk_length needs no padding."""
        chunk_length = 4
        agent = PPOAgent(
            state_dim=27, action_dim=7, recurrent=True,
            lstm_hidden_dim=64, chunk_length=chunk_length, device='cpu',
        )

        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(chunk_length):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=0.1, done=(i == chunk_length - 1), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')

        assert len(chunks) == 1
        assert chunks[0]['valid_mask'].all()  # All steps valid (no padding)

    def test_approx_kl_returned_in_metrics(self):
        """update_recurrent should return approx_kl for diagnostics."""
        agent = PPOAgent(
            state_dim=27, action_dim=7, recurrent=True,
            lstm_hidden_dim=64, chunk_length=4, device='cpu',
        )

        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(4):
            state = torch.randn(27)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=0.1, done=(i == 3), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        metrics = agent.update_recurrent(n_epochs=1)

        assert 'approx_kl' in metrics
        # approx_kl should be near 0 before any policy drift
        assert abs(metrics['approx_kl']) < 0.1
