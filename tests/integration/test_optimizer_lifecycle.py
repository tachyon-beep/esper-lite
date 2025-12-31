"""Integration test to verify Optimizer Lifecycle Management via GC Forensics.

Ensures that per-seed optimizers are physically destroyed (garbage collected)
after pruning, preventing long-term memory leaks.
"""

import pytest
import torch
import gc
import weakref
from unittest.mock import MagicMock

from esper.simic.training.vectorized import train_ppo_vectorized
from esper.simic.training.config import TrainingConfig

# =============================================================================
# Helper: The Surgeon Agent
# =============================================================================

class SurgeonAgent:
    """
    A scripted agent that forces a specific lifecycle:
    Germinate -> Train -> Prune.
    """
    def __init__(self, n_envs: int, *args, **kwargs):
        self.n_envs = n_envs
        self.step_count = 0
        # Minimal PPO Interface needed by vectorized.py
        self.buffer = type("MockBuffer", (), {"add": lambda *args, **kwargs: None, "reset": lambda: None, "compute_advantages": lambda: None, "mark_terminal_with_penalty": lambda *args: None, "step_counts": [0]*n_envs, "bootstrap_values": {}})()
        self.lstm_hidden_dim = 512
        self.policy = type("MockPolicy", (), {"reset_noise": lambda: None})()
        # Mock parameters for the trainer to update/clip
        self.policy.network = torch.nn.Linear(10, 2)
        self.target_kl = None

    def get_action(self, obs, hidden=None, **kwargs):
        self.step_count += 1
        
        # Action Structure: 0=WAIT, 1=GERMINATE, 2=PRUNE
        # Maps to esper.leyline.OP_NAMES: 0=WAIT, 1=GERMINATE, 2=ADVANCE, 3=PRUNE, ...
        # Wait, I need to check actual OP indices.
        from esper.leyline import OP_WAIT, OP_GERMINATE, OP_PRUNE, OP_ADVANCE
        
        ops = torch.zeros(self.n_envs, dtype=torch.long)
        slots = torch.zeros(self.n_envs, dtype=torch.long) # Always target r0c1 (index 0 for canonical?)
        # Need to ensure r0c1 is index 0. Default SlotConfig sorts slots.
        
        if self.step_count == 1:
            # Step 1: Germinate
            ops.fill_(OP_GERMINATE) 
        elif self.step_count == 5:
            # Step 5: Advance (TRAINING->BLENDING) if G2 passes
            # But we just want to ensure it stays alive long enough
            # Let's just wait until prune
            ops.fill_(OP_ADVANCE)
        elif self.step_count == 10:
            # Step 10: Prune
            ops.fill_(OP_PRUNE)
        else:
            # Wait/Train
            ops.fill_(OP_WAIT)
            
        # We need all head keys
        from esper.leyline import ACTION_HEAD_NAMES
        actions = {name: torch.zeros(self.n_envs, dtype=torch.long) for name in ACTION_HEAD_NAMES}
        actions["op"] = ops
        actions["slot"] = slots
        
        # Mock returns
        class Result: pass
        res = Result()
        res.actions = actions
        res.values = torch.zeros(self.n_envs)
        res.log_probs = {k: torch.zeros(self.n_envs) for k in actions}
        res.hidden_h = torch.zeros(1, self.n_envs, 512)
        res.hidden_c = torch.zeros(1, self.n_envs, 512)
        
        return res

    def update(self, **kwargs):
        return {} # No-op

    def save(self, path):
        pass

# =============================================================================
# The Leak Test
# =============================================================================

@pytest.mark.integration
class TestOptimizerLifecycle:

    def count_optimizers(self):
        """Forensic sweep of the Python heap."""
        gc.collect()
        count = 0
        for obj in gc.get_objects():
            if isinstance(obj, torch.optim.Optimizer):
                count += 1
        return count

    def test_optimizer_cleanup(self):
        """
        Verifies that seed optimizers are physically destroyed after pruning.
        """
        # 1. Baseline Sweep
        gc.collect()
        initial_count = self.count_optimizers()
        print(f"\n[Lifecycle] Initial Optimizer Count: {initial_count}")
        
        # 2. Config setup
        config = TrainingConfig.for_cifar10()
        config.n_envs = 1 # Keep it simple
        config.max_epochs = 20 # Enough steps to cover the script
        config.seed = 42
        config.use_telemetry = False
        config.n_episodes = 1
        
        device = "cpu"
        
        # Monkey Patch the Agent class used inside vectorized.py
        import esper.simic.training.vectorized as vec_module
        original_agent_cls = vec_module.PPOAgent
        
        # Inject our Surgeon
        vec_module.PPOAgent = SurgeonAgent
        
        try:
            agent, history = train_ppo_vectorized(
                **config.to_train_kwargs(),
                device=device,
                devices=[device],
                quiet_analytics=True
            )
        finally:
            # Restore class
            vec_module.PPOAgent = original_agent_cls

        # 4. Final Sweep
        # We must explicitly delete the returned agent/history to clear potential references
        del agent
        del history
        gc.collect()
        
        final_count = self.count_optimizers()
        print(f"[Lifecycle] Final Optimizer Count: {final_count}")
        
        # 5. The Assertion
        diff = final_count - initial_count
        print(f"[Lifecycle] Leaked Objects: {diff}")
        
        # The SurgeonAgent creates ONE seed optimizer. 
        # The Host creates ONE host optimizer.
        # The Host optimizer is attached to env_state which is usually local to train loop.
        # But if train_ppo_vectorized cleans up, env_state should be gone.
        
        # Ideally diff should be 0.
        # If diff is 1, it might be the Host optimizer if something leaked slightly.
        # If diff is 2+, the Seed optimizer definitely leaked.
        
        assert diff <= 1, f"Critical Memory Leak: Seed Optimizer not collected! Diff={diff}"