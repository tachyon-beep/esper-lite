"""Buffer data structures for RL training (PPO)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, NamedTuple

import torch


# =============================================================================
# PPO Buffers
# =============================================================================

class RolloutStep(NamedTuple):
    """Single step in a PPO rollout."""
    state: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    action_mask: torch.Tensor
    truncated: bool = False  # True if episode ended due to time limit (should bootstrap)
    bootstrap_value: float = 0.0  # Value to bootstrap from if truncated


@dataclass
class RolloutBuffer:
    """Buffer for storing PPO rollout data.

    Stores trajectory steps and computes returns/advantages using GAE.
    Action masks are stored alongside observations to ensure consistent
    masking during PPO updates (same mask that was active when action was taken).
    """
    steps: list[RolloutStep] = field(default_factory=list)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_mask: torch.Tensor,
        truncated: bool = False,
        bootstrap_value: float = 0.0,
    ) -> None:
        """Add a step to the buffer.

        Args:
            state: Observation tensor
            action: Action taken
            log_prob: Log probability of action under policy
            value: Value estimate at this state
            reward: Reward received
            done: Whether episode ended (naturally or truncated)
            action_mask: Binary mask of valid actions at this state
            truncated: Whether episode ended due to time limit (not natural termination)
            bootstrap_value: Value to bootstrap from if truncated
        """
        self.steps.append(RolloutStep(state, action, log_prob, value, reward, done, action_mask, truncated, bootstrap_value))

    def clear(self) -> None:
        """Clear all steps from the buffer."""
        self.steps = []

    def __len__(self) -> int:
        return len(self.steps)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and GAE advantages.

        Args:
            last_value: Value estimate for state after last step (0 if done)
            gamma: Discount factor
            gae_lambda: GAE lambda for bias-variance tradeoff
            device: Device to create tensors on (avoids CPU->GPU transfer)

        Returns:
            Tuple of (returns, advantages) on the specified device
        """
        n_steps = len(self.steps)
        returns = torch.zeros(n_steps, device=device)
        advantages = torch.zeros(n_steps, device=device)

        last_gae = 0.0
        next_value = last_value

        for t in reversed(range(n_steps)):
            step = self.steps[t]

            if step.done:
                # For truncated episodes, bootstrap from the estimated value
                # For naturally terminated episodes, use 0.0
                if step.truncated:
                    next_value = step.bootstrap_value
                else:
                    next_value = 0.0
                # Reset GAE at ALL episode boundaries
                # bootstrap_value handles value continuation; GAE chain should not span episodes
                last_gae = 0.0

            delta = step.reward + gamma * next_value - step.value
            advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae
            returns[t] = advantages[t] + step.value
            next_value = step.value

        return returns, advantages

    def get_batches(self, batch_size: int, device: str) -> list[tuple[dict, torch.Tensor]]:
        """Get shuffled minibatches for PPO update.

        Returns list of (batch_dict, batch_indices) tuples.
        Includes action_masks for correct masked policy evaluation.

        Performance: Pre-stacks all data once, then slices for batches to avoid
        repeated tensor creation and memory fragmentation.
        """
        n_steps = len(self.steps)
        indices = torch.randperm(n_steps)

        # Pre-stack all data once to avoid repeated tensor creation per batch
        all_states = torch.stack([s.state for s in self.steps]).to(device)
        all_actions = torch.tensor([s.action for s in self.steps], dtype=torch.long, device=device)
        all_log_probs = torch.tensor([s.log_prob for s in self.steps], dtype=torch.float32, device=device)
        all_values = torch.tensor([s.value for s in self.steps], dtype=torch.float32, device=device)
        all_masks = torch.stack([s.action_mask for s in self.steps]).to(device)

        batches = []
        for start in range(0, n_steps, batch_size):
            end = min(start + batch_size, n_steps)
            batch_idx = indices[start:end]

            # Slice pre-stacked tensors (views, not copies)
            batch = {
                'states': all_states[batch_idx],
                'actions': all_actions[batch_idx],
                'old_log_probs': all_log_probs[batch_idx],
                'values': all_values[batch_idx],
                'action_masks': all_masks[batch_idx],
            }
            batches.append((batch, batch_idx))

        return batches


# =============================================================================
# Recurrent PPO Buffers
# =============================================================================

class RecurrentRolloutStep(NamedTuple):
    """Single step in a recurrent PPO rollout.

    Note: Hidden states are NOT stored per-step (memory optimization).
    They're only stored at chunk boundaries during get_chunks().
    """
    state: torch.Tensor       # [state_dim]
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    action_mask: torch.Tensor  # [action_dim]
    truncated: bool = False  # True if episode ended due to time limit (should bootstrap)
    bootstrap_value: float = 0.0  # Value to bootstrap from if truncated


@dataclass
class RecurrentRolloutBuffer:
    """Buffer for recurrent PPO with per-environment storage.

    Key design decisions (from reviewer feedback):
    1. Per-env step lists - avoids interleaving corruption
    2. chunk_length=25 matches max_epochs default (no mid-episode chunking)
    3. GAE computed per-episode, then distributed to chunks
    4. Returns/advantages included in every chunk dict
    5. Single PPO epoch avoids hidden state drift between old/new log_probs

    Args:
        chunk_length: Length of chunks for BPTT (default 25 = max_epochs default)
        lstm_hidden_dim: Hidden dimension for zero-init at episode starts
    """
    chunk_length: int = 25
    lstm_hidden_dim: int = 128

    # Per-environment step storage (no interleaving)
    env_steps: dict[int, list[RecurrentRolloutStep]] = field(default_factory=dict)
    episode_boundaries: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    _current_episode_start: dict[int, int] = field(default_factory=dict)

    # GAE results (computed once, distributed to chunks)
    _episode_returns: dict[tuple[int, int, int], torch.Tensor] = field(default_factory=dict)
    _episode_advantages: dict[tuple[int, int, int], torch.Tensor] = field(default_factory=dict)
    _gae_computed: bool = field(default=False, init=False)

    def start_episode(self, env_id: int) -> None:
        """Mark start of a new episode for an environment."""
        if env_id not in self.env_steps:
            self.env_steps[env_id] = []
        self._current_episode_start[env_id] = len(self.env_steps[env_id])

    def end_episode(self, env_id: int) -> None:
        """Mark end of episode, recording boundary."""
        if env_id not in self._current_episode_start:
            return
        start = self._current_episode_start[env_id]
        end = len(self.env_steps[env_id])
        if env_id not in self.episode_boundaries:
            self.episode_boundaries[env_id] = []
        self.episode_boundaries[env_id].append((start, end))
        del self._current_episode_start[env_id]

    def add(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_mask: torch.Tensor,
        env_id: int,
        truncated: bool = False,
        bootstrap_value: float = 0.0,
    ) -> None:
        """Add a step to the correct environment's list.

        Note: Tensors kept on original device until get_chunks() for performance.
        The .detach() prevents gradient graph retention without forcing GPU sync.
        Transfer to target device happens in batch during get_chunks().
        """
        if env_id not in self.env_steps:
            self.env_steps[env_id] = []

        self.env_steps[env_id].append(RecurrentRolloutStep(
            state=state.detach(),  # Keep on GPU, detach from graph (no sync)
            action=action,
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=done,
            action_mask=action_mask.detach(),  # Keep on GPU, detach from graph (no sync)
            truncated=truncated,
            bootstrap_value=bootstrap_value,
        ))

    def clear(self) -> None:
        """Clear all data."""
        self.env_steps = {}
        self.episode_boundaries = {}
        self._current_episode_start = {}
        self._episode_returns = {}
        self._episode_advantages = {}
        self._gae_computed = False

    def __len__(self) -> int:
        return sum(len(steps) for steps in self.env_steps.values())

    def compute_gae(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE returns/advantages for all episodes.

        Must be called before get_chunks(). Stores results keyed by
        (env_id, episode_start, episode_end) for chunk retrieval.
        """
        for env_id, boundaries in self.episode_boundaries.items():
            steps = self.env_steps[env_id]

            for start_idx, end_idx in boundaries:
                episode_steps = steps[start_idx:end_idx]
                n_steps = len(episode_steps)

                if n_steps == 0:
                    continue

                returns = torch.zeros(n_steps)
                advantages = torch.zeros(n_steps)

                last_gae = 0.0
                next_value = 0.0  # Terminal

                for t in reversed(range(n_steps)):
                    step = episode_steps[t]
                    if step.done:
                        # For truncated episodes, bootstrap from the estimated value
                        # For naturally terminated episodes, use 0.0
                        if step.truncated:
                            next_value = step.bootstrap_value
                            # Continue GAE chain - truncation is artificial, not true terminal
                        else:
                            next_value = 0.0
                            # Reset GAE only for true terminal states
                            last_gae = 0.0

                    delta = step.reward + gamma * next_value - step.value
                    advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae
                    returns[t] = advantages[t] + step.value
                    next_value = step.value

                # Store keyed by episode identity
                key = (env_id, start_idx, end_idx)
                self._episode_returns[key] = returns
                self._episode_advantages[key] = advantages

        self._gae_computed = True

    def normalize_advantages(self) -> None:
        """Normalize advantages globally across all episodes.

        Must be called after compute_gae(). Normalizes advantages in-place
        to prevent per-batch normalization variance during PPO updates.
        """
        if not self._gae_computed:
            raise RuntimeError("Must call compute_gae() before normalize_advantages()")

        # Collect all advantages into a single tensor
        all_advantages = []
        for adv in self._episode_advantages.values():
            all_advantages.append(adv)

        if not all_advantages:
            return

        # Compute global mean and std
        all_adv_tensor = torch.cat(all_advantages)
        mean = all_adv_tensor.mean()
        std = all_adv_tensor.std()

        # Normalize each episode's advantages in-place
        for key in self._episode_advantages:
            self._episode_advantages[key] = (self._episode_advantages[key] - mean) / (std + 1e-8)

    def get_chunks(self, device: str | torch.device) -> list[dict]:
        """Get all chunks with returns/advantages included.

        Each chunk is a dict with shape [1, chunk_len, ...] for batching later.
        """
        if not self._gae_computed:
            raise RuntimeError("Must call compute_gae() before get_chunks()")

        chunks = []

        for env_id, boundaries in self.episode_boundaries.items():
            steps = self.env_steps[env_id]

            for start_idx, end_idx in boundaries:
                episode_steps = steps[start_idx:end_idx]
                episode_len = len(episode_steps)

                # Get precomputed GAE for this episode
                key = (env_id, start_idx, end_idx)
                episode_returns = self._episode_returns[key]
                episode_advantages = self._episode_advantages[key]

                # Chunk this episode
                for chunk_start in range(0, episode_len, self.chunk_length):
                    chunk_end = min(chunk_start + self.chunk_length, episode_len)
                    chunk_steps = episode_steps[chunk_start:chunk_end]
                    actual_len = len(chunk_steps)
                    pad_len = self.chunk_length - actual_len

                    # Get GAE slice for this chunk
                    chunk_returns = episode_returns[chunk_start:chunk_end]
                    chunk_advantages = episode_advantages[chunk_start:chunk_end]

                    # Pad if needed
                    if pad_len > 0:
                        pad_step = chunk_steps[-1]
                        for _ in range(pad_len):
                            chunk_steps = list(chunk_steps) + [RecurrentRolloutStep(
                                state=torch.zeros_like(pad_step.state),
                                action=0,
                                log_prob=0.0,
                                value=0.0,
                                reward=0.0,
                                done=True,
                                action_mask=pad_step.action_mask,
                            )]
                        # Pad returns/advantages with zeros
                        chunk_returns = torch.cat([
                            chunk_returns,
                            torch.zeros(pad_len),
                        ])
                        chunk_advantages = torch.cat([
                            chunk_advantages,
                            torch.zeros(pad_len),
                        ])

                    # Initial hidden state determination:
                    # - Episode start (chunk_start == 0): zeros (correct)
                    # - Mid-episode chunk: would need stored hidden from previous chunk
                    #
                    # With chunk_length=25 matching max_epochs exactly,
                    # episodes fit in a single chunk, so chunk_start is ALWAYS 0.
                    # This avoids the hidden state drift problem entirely.
                    #
                    # GRACEFUL HANDLING: If chunk_length < episode_length (mid-episode
                    # chunking), we use zeros for initial hidden and log a warning.
                    # This is suboptimal (loses temporal context at chunk boundaries)
                    # but won't crash. For best results, set chunk_length >= episode_length.
                    is_episode_start = (chunk_start == 0)
                    if not is_episode_start:
                        import warnings
                        warnings.warn(
                            f"Mid-episode chunking detected (chunk_start={chunk_start}). "
                            f"Using zeros for initial hidden state, which loses temporal "
                            f"context. For correct BPTT, set chunk_length >= episode_length "
                            f"or implement burn-in BPTT.",
                            RuntimeWarning,
                        )

                    # Stack into tensors with batch dim [1, seq, ...]
                    chunk = {
                        'states': torch.stack([s.state for s in chunk_steps]).unsqueeze(0).to(device),
                        'actions': torch.tensor([s.action for s in chunk_steps], dtype=torch.long).unsqueeze(0).to(device),
                        'old_log_probs': torch.tensor([s.log_prob for s in chunk_steps], dtype=torch.float32).unsqueeze(0).to(device),
                        'old_values': torch.tensor([s.value for s in chunk_steps], dtype=torch.float32).unsqueeze(0).to(device),
                        'action_masks': torch.stack([s.action_mask for s in chunk_steps]).unsqueeze(0).to(device),
                        'returns': chunk_returns.unsqueeze(0).to(device),
                        'advantages': chunk_advantages.unsqueeze(0).to(device),
                        'valid_mask': torch.tensor(
                            [True] * actual_len + [False] * pad_len,
                            dtype=torch.bool,
                        ).unsqueeze(0).to(device),
                        # Initial hidden: zeros for episode start, zeros for chunk boundaries (TBPTT)
                        'initial_hidden_h': torch.zeros(1, 1, self.lstm_hidden_dim, device=device),
                        'initial_hidden_c': torch.zeros(1, 1, self.lstm_hidden_dim, device=device),
                        'is_episode_start': is_episode_start,
                    }
                    chunks.append(chunk)

        return chunks

    def get_batched_chunks(
        self,
        device: str | torch.device,
        batch_size: int = 8,
    ) -> Iterator[dict]:
        """Yield batched chunks for efficient GPU processing.

        Stacks multiple chunks into batches for parallel LSTM processing.
        """
        chunks = self.get_chunks(device)

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            n_chunks = len(batch_chunks)

            # Stack along batch dimension
            batch = {
                'states': torch.cat([c['states'] for c in batch_chunks], dim=0),  # [batch, seq, state]
                'actions': torch.cat([c['actions'] for c in batch_chunks], dim=0),
                'old_log_probs': torch.cat([c['old_log_probs'] for c in batch_chunks], dim=0),
                'old_values': torch.cat([c['old_values'] for c in batch_chunks], dim=0),
                'action_masks': torch.cat([c['action_masks'] for c in batch_chunks], dim=0),
                'returns': torch.cat([c['returns'] for c in batch_chunks], dim=0),
                'advantages': torch.cat([c['advantages'] for c in batch_chunks], dim=0),
                'valid_mask': torch.cat([c['valid_mask'] for c in batch_chunks], dim=0),
                # Hidden: [num_layers, batch, hidden]
                'initial_hidden_h': torch.cat(
                    [c['initial_hidden_h'] for c in batch_chunks], dim=1
                ),
                'initial_hidden_c': torch.cat(
                    [c['initial_hidden_c'] for c in batch_chunks], dim=1
                ),
            }
            yield batch


__all__ = [
    "RolloutStep",
    "RolloutBuffer",
    "RecurrentRolloutStep",
    "RecurrentRolloutBuffer",
]
