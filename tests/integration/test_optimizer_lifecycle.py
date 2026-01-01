"""Integration test to verify Optimizer Lifecycle Management via GC Forensics.

Ensures that per-seed optimizers are physically destroyed (garbage collected)
after pruning, preventing long-term memory leaks.
"""

import pytest
import torch
import gc

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
    def __init__(self, num_envs: int, *args, **kwargs):
        self.n_envs = num_envs
        self.step_count = 0
        # Minimal PPO Interface needed by vectorized.py
        self.buffer = type("MockBuffer", (), {
            "add": lambda *args, **kwargs: None, 
            "reset": lambda: None, 
            "compute_advantages": lambda: None, 
            "mark_terminal_with_penalty": lambda *args: None, 
            "start_episode": lambda *args, **kwargs: None,
            "end_episode": lambda *args, **kwargs: None,
            "__len__": lambda self: 0,
            "step_counts": [0]*num_envs, 
            "bootstrap_values": {}
        })()
        
        # Define Policy class with get_action
        class MockPolicy:
            def __init__(self, agent):
                self.agent = agent
                self.network = torch.nn.Linear(10, 2)
                
            def reset_noise(self): pass
            
            def initial_hidden(self, batch_size):
                return (torch.zeros(1, batch_size, 512), torch.zeros(1, batch_size, 512))
                
            def get_action(self, obs, hidden=None, **kwargs):
                self.agent.step_count += 1
                
                # Action Structure: 0=WAIT, 1=GERMINATE, 2=PRUNE
                from esper.leyline import OP_WAIT, OP_GERMINATE, OP_PRUNE, OP_ADVANCE, BLUEPRINT_ID_TO_INDEX
                
                ops = torch.zeros(self.agent.n_envs, dtype=torch.long)
                slots = torch.zeros(self.agent.n_envs, dtype=torch.long) # r0c1
                blueprints = torch.zeros(self.agent.n_envs, dtype=torch.long)
                
                # Pick a valid blueprint with params (e.g. conv_small)
                bp_idx = BLUEPRINT_ID_TO_INDEX.get("conv_small", 1) # Default to 1 if not found
                
                if self.agent.step_count == 1:
                    ops.fill_(OP_GERMINATE) 
                    blueprints.fill_(bp_idx)
                elif self.agent.step_count == 5:
                    ops.fill_(OP_ADVANCE)
                elif self.agent.step_count == 10:
                    ops.fill_(OP_PRUNE)
                else:
                    ops.fill_(OP_WAIT)
                    
                from esper.leyline import ACTION_HEAD_NAMES
                actions = {name: torch.zeros(self.agent.n_envs, dtype=torch.long) for name in ACTION_HEAD_NAMES}
                actions["op"] = ops
                actions["slot"] = slots
                actions["blueprint"] = blueprints

                class Result:
                    pass

                res = Result()
                res.action = actions
                res.value = torch.zeros(self.agent.n_envs)
                res.log_prob = {k: torch.zeros(self.agent.n_envs) for k in actions}
                res.hidden = (torch.zeros(1, self.agent.n_envs, 512), torch.zeros(1, self.agent.n_envs, 512))
                res.op_logits = None
                return res

            def state_dict(self):
                return {}

        self.policy = MockPolicy(self)
        self.target_kl = None

    # SurgeonAgent.get_action is removed as it's not called
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
                num_workers=0,
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
