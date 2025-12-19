#!/usr/bin/env python3
"""Profile GPU synchronization in vectorized training.

Run with: PYTHONPATH=src uv run python scripts/profile_gpu_sync.py
"""
import torch
import time
from contextlib import contextmanager


@contextmanager
def sync_timer(name: str):
    """Time a block including GPU sync."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) * 1000
    print(f"{name}: {elapsed:.2f}ms")


def profile_action_extraction():
    """Compare .item() vs .cpu().numpy() for action extraction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4
    num_epochs = 25
    num_heads = 4

    actions_dict = {
        f"head_{i}": torch.randint(0, 10, (num_envs,), device=device)
        for i in range(num_heads)
    }

    # Method 1: Per-element .item()
    with sync_timer("Per-element .item()"):
        for _ in range(num_epochs):
            for env_idx in range(num_envs):
                _ = {key: actions_dict[key][env_idx].item() for key in actions_dict}

    # Method 2: Batched .cpu().numpy()
    with sync_timer("Batched .cpu().numpy()"):
        for _ in range(num_epochs):
            actions_cpu = {key: actions_dict[key].cpu().numpy() for key in actions_dict}
            for env_idx in range(num_envs):
                _ = {key: int(actions_cpu[key][env_idx]) for key in actions_cpu}


def profile_log_prob_storage():
    """Compare .item() vs tensor assignment for log_probs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4
    num_epochs = 25
    num_heads = 4

    log_probs = {
        f"head_{i}": torch.randn(num_envs, device=device)
        for i in range(num_heads)
    }

    # Target buffer (pre-allocated)
    buffer = torch.zeros(num_envs, num_epochs, num_heads, device=device)

    # Method 1: Per-element .item()
    with sync_timer("Log probs .item()"):
        for epoch in range(num_epochs):
            for env_idx in range(num_envs):
                for head_idx, key in enumerate(log_probs):
                    buffer[env_idx, epoch, head_idx] = log_probs[key][env_idx].item()

    # Method 2: Tensor assignment
    buffer2 = torch.zeros(num_envs, num_epochs, num_heads, device=device)
    with sync_timer("Log probs tensor assign"):
        for epoch in range(num_epochs):
            for env_idx in range(num_envs):
                for head_idx, key in enumerate(log_probs):
                    buffer2[env_idx, epoch, head_idx] = log_probs[key][env_idx]


if __name__ == "__main__":
    print("=== GPU Sync Profiling ===\n")

    print("Action Extraction:")
    profile_action_extraction()

    print("\nLog Prob Storage:")
    profile_log_prob_storage()

    print("\nDone!")
