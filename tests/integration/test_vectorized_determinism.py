"""
Integration test to verify Vectorized Environment Determinism.

Ensures that the environment, data loading, and RNG usage are strictly deterministic
when seeded. This is critical for debugging RL regressions.
"""

import pytest
import torch
import numpy as np
import random
import hashlib

from esper.simic.training.vectorized import train_ppo_vectorized
from esper.simic.training.config import TrainingConfig

# =============================================================================
# Helper: Trace Recorder
# =============================================================================

class TraceRecorder:
    def __init__(self):
        self.trace = []

    def log(self, step: int, env_id: int, obs: torch.Tensor, rewards: float, actions: dict):
        # Digest observation to save memory but capture changes
        obs_hash = hashlib.md5(obs.cpu().numpy().tobytes()).hexdigest()
        
        # Log entry
        entry = {
            "step": step,
            "env_id": env_id,
            "obs_hash": obs_hash,
            "reward": float(rewards),
            # Actions are dict of tensors/ints, convert to primitive
            "actions": {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in actions.items()}
        }
        self.trace.append(entry)

    def __eq__(self, other):
        if len(self.trace) != len(other.trace):
            return False
        for t1, t2 in zip(self.trace, other.trace):
            if t1 != t2:
                return False
        return True

    def diff(self, other):
        """Return description of first difference."""
        if len(self.trace) != len(other.trace):
            return f"Length mismatch: {len(self.trace)} vs {len(other.trace)}"
        
        for i, (t1, t2) in enumerate(zip(self.trace, other.trace)):
            if t1 != t2:
                # Find which key differs
                diffs = []
                for k in t1:
                    if t1[k] != t2[k]:
                        diffs.append(f"{k}: {t1[k]} vs {t2[k]}")
                return f"Difference at index {i}: {', '.join(diffs)}"
        return "No difference"


# =============================================================================
# Mock Agent for Deterministic Actions
# =============================================================================

class MockDeterministicAgent:
    """Agent that takes actions based purely on step count (no RNG)."""
    def __init__(self, n_envs: int):
        self.n_envs = n_envs
        self.step = 0
        # Minimal interface required by vectorized.py
        self.buffer = type("MockBuffer", (), {"add": lambda *args, **kwargs: None, "reset": lambda: None, "compute_advantages": lambda: None})()
        self.lstm_hidden_dim = 512
        self.policy = type("MockPolicy", (), {"reset_noise": lambda: None})()

    def get_action(self, obs, hidden=None, **kwargs):
        self.step += 1
        # Return deterministic fake actions
        actions = {
            "slot": torch.zeros(self.n_envs, dtype=torch.long),
            "op": torch.zeros(self.n_envs, dtype=torch.long), # WAIT
            # Add other heads if needed
        }
        # Fake values/log_probs
        values = torch.zeros(self.n_envs)
        log_probs = {k: torch.zeros(self.n_envs) for k in actions}
        
        class Result:
            pass
        res = Result()
        res.actions = actions
        res.values = values
        res.log_probs = log_probs
        res.hidden_h = torch.zeros(1, self.n_envs, 512)
        res.hidden_c = torch.zeros(1, self.n_envs, 512)
        return res

    def update(self, **kwargs):
        return {} # No-op update


# =============================================================================
# Verification Test
# =============================================================================

@pytest.mark.integration
class TestVectorizedDeterminism:
    
    def run_trace(self, seed: int, n_steps: int = 20) -> list[dict]:
        """Run a short training loop and return history trace."""
        
        # 1. Set Global Seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 2. Config
        # Use tinystories or cifar10? Cifar10 is standard.
        # Use CPU to avoid GPU non-determinism issues complicating the baseline check
        config = TrainingConfig.for_cifar10()
        config.n_envs = 2
        config.max_epochs = n_steps
        config.seed = seed
        config.use_telemetry = False # Reduce noise
        config.n_episodes = 2 # Short run
        
        # 3. Run
        # We invoke train_ppo_vectorized directly.
        # Ideally we'd inject a mock agent, but train_ppo_vectorized instantiates it internally.
        # However, if we pass `seed`, PPO initialization should be deterministic.
        # We will rely on PPO determinism rather than mocking the agent, 
        # which is a stronger test (tests the whole stack).
        
        # Force CPU device for determinism test to isolate logic from CUDA atomics
        device = "cpu"
        
        # Capture history returned by train
        agent, history = train_ppo_vectorized(
            **config.to_train_kwargs(),
            device=device,
            devices=[device] * config.n_envs,
            quiet_analytics=True
        )
        
        # The history list contains metrics per update.
        # But we want environment interaction traces.
        # Since we can't easily hook into the internal loop of train_ppo_vectorized without modifying it,
        # we will verify determinism based on the *outputs*:
        # 1. The training history metrics (rewards, losses)
        # 2. The final agent weights
        
        # Digest weights
        weights_digest = hashlib.md5(
            torch.cat([p.flatten().cpu().detach() for p in agent.policy.network.parameters()]).numpy().tobytes()
        ).hexdigest()
        
        return {
            "history": history,
            "weights_digest": weights_digest
        }

    def test_end_to_end_determinism(self):
        """Verify that two runs with same seed produce identical results."""
        seed = 42
        steps = 5  # Short run to be fast
        
        print("Starting Run 1...")
        result1 = self.run_trace(seed, n_steps=steps)
        
        print("Starting Run 2...")
        result2 = self.run_trace(seed, n_steps=steps)
        
        # 1. Check Weights
        assert result1["weights_digest"] == result2["weights_digest"], \
            "Final agent weights diverged!"
            
        # 2. Check History (Metrics)
        # History is list of dicts. Compare length first.
        assert len(result1["history"]) == len(result2["history"]), \
            "History length mismatch"
            
        # Compare each step
        for i, (h1, h2) in enumerate(zip(result1["history"], result2["history"])):
            # Compare scalar metrics
            for k in h1:
                v1 = h1[k]
                v2 = h2.get(k)
                
                # Allow slight floating point noise if any? 
                # On CPU with deterministic seeds, it should be exact or very close.
                if isinstance(v1, (float, int)):
                    assert v1 == pytest.approx(v2, abs=1e-6), \
                        f"Metric '{k}' diverged at step {i}: {v1} vs {v2}"
                elif isinstance(v1, str):
                    assert v1 == v2, f"Metric '{k}' diverged at step {i}"
                    
        print("Determinism Verified!")
