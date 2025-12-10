"""Prioritized Experience Replay buffer for DQN-style algorithms.

Based on Schaul et al., 2016: "Prioritized Experience Replay"
https://arxiv.org/abs/1511.05952

Key features:
- SumTree data structure for O(log n) priority sampling
- Stratified sampling to ensure diversity in batches
- Importance sampling weights to correct for bias
- Adaptive beta annealing toward 1.0
"""

from __future__ import annotations

import numpy as np
import torch


class SumTree:
    """Binary tree structure for efficient priority-based sampling.

    The tree stores priorities in leaf nodes and maintains sum aggregates
    in internal nodes, enabling O(log n) sampling by cumulative priority.

    Structure:
        - Leaf nodes: store individual priorities
        - Internal nodes: sum of children's values
        - Root node: total sum of all priorities

    Args:
        capacity: Maximum number of experiences to store
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1  # Binary tree with capacity leaves
        self.tree = np.zeros(self.tree_size, dtype=np.float32)
        self.data_idx_map = np.zeros(capacity, dtype=np.int32)  # Maps tree_idx -> data_idx
        self.write_idx = 0  # Next position to write

    def add(self, priority: float, data_idx: int) -> None:
        """Add a new priority and data index to the tree.

        Args:
            priority: Priority value (higher = more important)
            data_idx: Index in the experience buffer
        """
        tree_idx = self.write_idx + self.capacity - 1  # Leaf node index
        self.data_idx_map[self.write_idx] = data_idx
        self.update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity

    def update(self, tree_idx: int, new_priority: float) -> None:
        """Update priority at a tree index and propagate changes upward.

        Args:
            tree_idx: Index in the tree (leaf or internal node)
            new_priority: New priority value
        """
        change = new_priority - self.tree[tree_idx]
        self.tree[tree_idx] = new_priority

        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, cumsum: float) -> int:
        """Get data index corresponding to cumulative priority.

        Args:
            cumsum: Cumulative priority value to search for.
                    Clamped to valid range [0, total) internally.

        Returns:
            Data index in the experience buffer

        Raises:
            RuntimeError: If total priority is too small to sample from
        """
        # Guard against empty/near-empty tree
        if self.total <= 1e-8:
            raise RuntimeError(
                f"Cannot sample from SumTree with near-zero total priority ({self.total})"
            )
        # Clamp cumsum to valid range for numeric stability
        cumsum = max(0.0, min(cumsum, self.total - 1e-8))
        tree_idx = 0  # Start at root

        # Traverse down the tree
        while tree_idx < self.capacity - 1:
            left_child = 2 * tree_idx + 1
            right_child = left_child + 1

            if cumsum <= self.tree[left_child]:
                tree_idx = left_child
            else:
                cumsum -= self.tree[left_child]
                tree_idx = right_child

        # tree_idx is now a leaf node
        leaf_idx = tree_idx - (self.capacity - 1)
        return self.data_idx_map[leaf_idx]

    @property
    def total(self) -> float:
        """Total sum of all priorities."""
        return self.tree[0]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with importance sampling.

    Samples experiences proportional to their TD error (priority), with
    stratified sampling for batch diversity and importance weights to
    correct for sampling bias.

    Args:
        capacity: Maximum number of experiences to store
        alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
        beta: Importance sampling weight exponent (0 = no correction, 1 = full)
        beta_increment: Amount to increase beta per sample (anneals to 1.0)
        epsilon: Small constant to ensure non-zero priorities
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # SumTree for efficient priority sampling
        self.tree = SumTree(capacity)

        # Experience storage
        self.states: list[torch.Tensor] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.next_states: list[torch.Tensor] = []
        self.dones: list[bool] = []

        self.write_pos = 0
        self.size = 0

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        priority: float,
    ) -> None:
        """Add an experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            priority: Initial priority (typically 1.0 or max TD error).
                      Negative values are converted to absolute value.
        """
        # Apply alpha to priority and add epsilon
        # Use abs() to handle negative TD errors safely
        priority = (abs(priority) + self.epsilon) ** self.alpha

        # Store experience
        if self.write_pos < len(self.states):
            self.states[self.write_pos] = state.cpu()
            self.actions[self.write_pos] = action
            self.rewards[self.write_pos] = reward
            self.next_states[self.write_pos] = next_state.cpu()
            self.dones[self.write_pos] = done
        else:
            self.states.append(state.cpu())
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state.cpu())
            self.dones.append(done)

        # Add to SumTree
        self.tree.add(priority, self.write_pos)

        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> tuple[dict, list[int], torch.Tensor]:
        """Sample a batch of experiences using stratified sampling.

        Args:
            batch_size: Number of experiences to sample
            device: Device to place batch tensors on

        Returns:
            Tuple of (batch_dict, indices, importance_weights)
            - batch_dict: Dict with keys 'states', 'actions', 'rewards', 'next_states', 'dones'
            - indices: List of sampled buffer indices
            - importance_weights: Tensor of importance sampling weights

        Raises:
            RuntimeError: If buffer is empty
            ValueError: If batch_size exceeds buffer size
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        if batch_size > self.size:
            raise ValueError(
                f"batch_size ({batch_size}) cannot exceed buffer size ({self.size})"
            )

        # Stratified sampling: divide priority range into segments
        segment_size = self.tree.total / batch_size
        indices = []
        priorities = []

        for i in range(batch_size):
            # Sample uniformly within each segment
            segment_start = segment_size * i
            segment_end = segment_size * (i + 1)
            cumsum = np.random.uniform(segment_start, segment_end)

            # Get data index from tree
            data_idx = self.tree.get(cumsum)
            indices.append(data_idx)

            # Get priority for importance weight calculation
            tree_idx = data_idx + self.tree.capacity - 1
            priority = self.tree.tree[tree_idx]
            priorities.append(priority)

        # Compute importance sampling weights with numeric stability
        priorities = np.array(priorities, dtype=np.float32)
        min_priority = max(
            self.tree.tree[self.tree.capacity - 1:self.tree.capacity - 1 + self.size].min(),
            self.epsilon,  # Ensure non-zero minimum
        )
        total = max(self.tree.total, self.epsilon)  # Guard against zero total

        max_weight = (self.size * min_priority / total) ** (-self.beta)
        weights = (self.size * priorities / total) ** (-self.beta)
        weights = np.clip(weights / max_weight, 0.0, 1.0)  # Normalize and clamp

        # Construct batch on target device
        batch = {
            'states': torch.stack([self.states[i] for i in indices]).to(device),
            'actions': torch.tensor([self.actions[i] for i in indices], dtype=torch.long, device=device),
            'rewards': torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device),
            'next_states': torch.stack([self.next_states[i] for i in indices]).to(device),
            'dones': torch.tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device),
        }

        # Increment beta toward 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Return weights as torch.Tensor on target device
        return batch, indices, torch.from_numpy(weights).to(device)

    def update_priorities(
        self,
        indices: list[int],
        new_priorities: list[float] | torch.Tensor,
    ) -> None:
        """Update priorities for sampled experiences.

        Args:
            indices: List of buffer indices to update
            new_priorities: List or tensor of new priority values (typically TD errors).
                           Negative values are converted to absolute value.
        """
        # Handle tensor input (common from TD error computation)
        if isinstance(new_priorities, torch.Tensor):
            new_priorities = new_priorities.detach().cpu().tolist()

        for data_idx, priority in zip(indices, new_priorities):
            # Apply alpha and epsilon, use abs() for negative TD errors
            priority = (abs(priority) + self.epsilon) ** self.alpha

            # Update tree
            tree_idx = data_idx + self.tree.capacity - 1
            self.tree.update(tree_idx, priority)

    def __len__(self) -> int:
        return self.size


__all__ = [
    "SumTree",
    "PrioritizedReplayBuffer",
]
