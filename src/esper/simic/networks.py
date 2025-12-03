"""Policy Networks for Tamiyo

This module contains neural network architectures for policy learning,
including simple MLP policies for imitation learning and more complex
actor-critic architectures for RL.
"""

from __future__ import annotations

import math
from pathlib import Path

from esper.leyline.actions import build_action_enum

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Network Utilities
# =============================================================================

def _check_torch():
    """Check if torch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


# =============================================================================
# Policy Network
# =============================================================================

class PolicyNetwork:
    """Simple MLP policy for imitation learning.

    Architecture: Input(27) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(7)

    Usage:
        policy = PolicyNetwork()
        policy.train_on_episodes(episodes, epochs=100)
        action = policy.predict(snapshot)
        policy.save("policy.pt")
    """

    def __init__(self, hidden1: int = 64, hidden2: int = 32, lr: float = 0.001):
        if not _check_torch():
            raise ImportError("PyTorch required for PolicyNetwork")

        import torch
        import torch.nn as nn

        # Import here to avoid circular dependencies
        from esper.simic.episodes import TrainingSnapshot
        from esper.leyline.actions import build_action_enum

        self.TrainingSnapshot = TrainingSnapshot
        self.action_enum = build_action_enum("cnn")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Simple MLP
        self.model = nn.Sequential(
            nn.Linear(TrainingSnapshot.vector_size(), hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, len(self.action_enum)),
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Track training history
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_accuracies: list[float] = []

    def _prepare_data(
        self, episodes: list, val_split: float = 0.2
    ) -> tuple:
        """Prepare training and validation data from episodes."""
        import torch
        import random

        # Shuffle and split by episode
        episodes = episodes.copy()
        random.shuffle(episodes)
        split_idx = int(len(episodes) * (1 - val_split))
        train_eps = episodes[:split_idx]
        val_eps = episodes[split_idx:]

        def episodes_to_tensors(eps_list):
            X, y = [], []
            for ep in eps_list:
                for dp in ep.decisions:
                    X.append(dp.observation.to_vector())
                    y.append(dp.action.action.value)
            return (
                torch.tensor(X, dtype=torch.float32, device=self.device),
                torch.tensor(y, dtype=torch.long, device=self.device),
            )

        X_train, y_train = episodes_to_tensors(train_eps)
        X_val, y_val = episodes_to_tensors(val_eps)

        return X_train, y_train, X_val, y_val

    def train_on_episodes(
        self,
        episodes: list,
        epochs: int = 100,
        batch_size: int = 32,
        val_split: float = 0.2,
        verbose: bool = True,
        class_weights: bool = True,
    ) -> dict:
        """Train the policy on collected episodes.

        Args:
            episodes: List of Episode objects
            epochs: Number of training epochs
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation
            verbose: Whether to print training progress
            class_weights: If True, weight classes inversely to frequency
                          to handle imbalanced data (WAIT dominates).

        Returns:
            Dict with final training metrics
        """
        import torch

        X_train, y_train, X_val, y_val = self._prepare_data(episodes, val_split)

        # Compute class weights if requested
        if class_weights:
            class_counts = torch.bincount(y_train, minlength=len(self.action_enum)).float()
            # Inverse frequency weighting, with smoothing
            weights = 1.0 / (class_counts + 1.0)
            weights = weights / weights.sum() * len(self.action_enum)  # Normalize
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights.to(self.device))
            if verbose:
                print(f"Class weights: {dict(zip([a.name for a in self.action_enum], weights.tolist()))}")

        if verbose:
            print(f"Training on {len(X_train)} samples, validating on {len(X_val)}")
            print(f"Device: {self.device}")

        self.train_losses.clear()
        self.val_losses.clear()
        self.val_accuracies.clear()

        for epoch in range(epochs):
            # Training
            self.model.train()
            indices = torch.randperm(len(X_train))
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i + batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / n_batches
            self.train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val).item()
                predictions = val_outputs.argmax(dim=1)
                accuracy = (predictions == y_val).float().mean().item() * 100

            self.val_losses.append(val_loss)
            self.val_accuracies.append(accuracy)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={accuracy:.1f}%")

        return {
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "final_val_accuracy": self.val_accuracies[-1],
        }

    def predict(self, snapshot):
        """Predict action for a single observation."""
        import torch

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(
                [snapshot.to_vector()],
                dtype=torch.float32,
                device=self.device
            )
            output = self.model(x)
            action_idx = output.argmax(dim=1).item()

        return self.action_enum(action_idx)

    def predict_probs(self, snapshot) -> dict:
        """Predict action probabilities for a single observation.

        Args:
            snapshot: TrainingSnapshot object

        Returns:
            Dict mapping action enum to probability
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(
                [snapshot.to_vector()],
                dtype=torch.float32,
                device=self.device
            )
            output = self.model(x)
            probs = F.softmax(output, dim=1)[0]

        return {
            self.action_enum(i): probs[i].item()
            for i in range(len(self.action_enum))
        }

    def evaluate(self, episodes: list) -> dict:
        """Evaluate policy on episodes, returning accuracy and confusion matrix.

        Args:
            episodes: List of Episode objects

        Returns:
            Dict with evaluation metrics
        """
        import torch

        X, y_true = [], []
        for ep in episodes:
            for dp in ep.decisions:
                X.append(dp.observation.to_vector())
                y_true.append(dp.action.action.value)

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_true = torch.tensor(y_true, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            y_pred = outputs.argmax(dim=1)

        # Accuracy
        accuracy = (y_pred == y_true).float().mean().item() * 100

        # Confusion matrix
        n_classes = len(self.action_enum)
        confusion = [[0] * n_classes for _ in range(n_classes)]
        for true, pred in zip(y_true.tolist(), y_pred.tolist()):
            confusion[true][pred] += 1

        # Per-class accuracy
        class_acc = {}
        for i, action in enumerate(self.action_enum):
            total = sum(confusion[i])
            correct = confusion[i][i] if total > 0 else 0
            class_acc[action.name] = (correct / total * 100) if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "confusion_matrix": confusion,
            "class_accuracy": class_acc,
            "total_samples": len(y_true),
        }

    def save(self, path: str | Path) -> None:
        """Save model weights."""
        import torch
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load model weights."""
        import torch
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def print_confusion_matrix(eval_result: dict) -> None:
    """Pretty print a confusion matrix.

    Args:
        eval_result: Dict from PolicyNetwork.evaluate()
    """
    cm = eval_result["confusion_matrix"]
    actions = [a.name for a in build_action_enum("cnn")]

    # Header
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':>12}", end="")
    for name in actions:
        print(f"{name:>10}", end="")
    print()

    # Rows
    for i, name in enumerate(actions):
        print(f"{name:>12}", end="")
        for j in range(len(actions)):
            print(f"{cm[i][j]:>10}", end="")
        print()

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for name, acc in eval_result["class_accuracy"].items():
        print(f"  {name}: {acc:.1f}%")


# =============================================================================
# RL Network Architectures
# =============================================================================

if TORCH_AVAILABLE:
    import torch.nn.functional as F

    class InvalidStateMachineError(RuntimeError):
        """Raised when action mask has no valid actions (state machine bug)."""
        pass

    @torch.compiler.disable
    def _validate_action_mask(mask: torch.Tensor) -> None:
        """Validate that at least one action is valid per batch element.

        Isolated from torch.compile to prevent graph breaks in the main forward path.
        The .any() call forces CPU sync, but this safety check is worth the cost.
        """
        valid_count = mask.sum(dim=-1)
        if (valid_count == 0).any():
            raise InvalidStateMachineError(
                f"No valid actions available. Mask: {mask}. "
                "This indicates a bug in the Kasmina state machine."
            )

    class MaskedCategorical:
        """Categorical distribution with action masking and correct entropy calculation.

        Masks invalid actions by setting their logits to dtype minimum before softmax.
        Uses torch.finfo().min for float16/bfloat16 compatibility.
        Computes entropy only over valid actions to avoid penalizing restricted states.
        """

        def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
            """Initialize masked categorical distribution.

            Args:
                logits: Raw policy logits [batch, num_actions]
                mask: Binary mask, 1.0 = valid, 0.0 = invalid [batch, num_actions]

            Raises:
                InvalidStateMachineError: If any batch element has no valid actions

            Note:
                The validation check is isolated via @torch.compiler.disable to prevent
                graph breaks in the main forward path while preserving safety checks.
            """
            # Validate mask in isolated function to prevent graph break propagation
            _validate_action_mask(mask)

            self.mask = mask
            # Use dtype-appropriate minimum to avoid overflow in float16/bfloat16
            mask_value = torch.finfo(logits.dtype).min
            self.masked_logits = logits.masked_fill(mask < 0.5, mask_value)
            self._dist = Categorical(logits=self.masked_logits)

        @property
        def probs(self) -> torch.Tensor:
            """Action probabilities (masked actions have ~0 probability)."""
            return self._dist.probs

        def sample(self) -> torch.Tensor:
            """Sample actions from the masked distribution."""
            return self._dist.sample()

        def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            """Compute log probability of actions."""
            return self._dist.log_prob(actions)

        def entropy(self) -> torch.Tensor:
            """Compute normalized entropy over valid actions.

            Returns entropy normalized to [0, 1] by dividing by max entropy
            (log of number of valid actions). This makes exploration incentives
            comparable across states with different action restrictions.
            """
            # Standard entropy: -sum(p * log(p)) over valid actions only
            probs = self._dist.probs
            log_probs = self._dist.logits - self._dist.logits.logsumexp(dim=-1, keepdim=True)

            # Raw entropy over valid actions only
            raw_entropy = -(probs * log_probs * self.mask).sum(dim=-1)

            # Normalize by max entropy for this action space
            # max_entropy = log(num_valid_actions) - uniform distribution entropy
            num_valid = self.mask.sum(dim=-1).clamp(min=1)
            max_entropy = torch.log(num_valid)

            # Return normalized entropy (0 = deterministic, 1 = uniform over valid)
            return raw_entropy / max_entropy.clamp(min=1e-8)


    class ActorCritic(nn.Module):
        """Actor-Critic network for PPO with action masking support.

        Uses shared feature extraction with separate actor and critic heads.
        Action mask is applied only to actor logits, not to critic (value is mask-agnostic).
        """

        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()

            self.action_dim = action_dim

            # Shared feature extractor
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            # Actor head (policy)
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
            )

            # Critic head (value function)
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            self._init_weights()

        def _init_weights(self):
            """Initialize weights with orthogonal initialization."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                    nn.init.zeros_(module.bias)

            # Smaller init for output layers
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
            nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

        def forward(
            self,
            state: torch.Tensor,
            action_mask: torch.Tensor,
        ) -> tuple[MaskedCategorical, torch.Tensor]:
            """Forward pass returning masked action distribution and value.

            Args:
                state: Observation tensor [batch, state_dim]
                action_mask: Binary mask [batch, action_dim], 1=valid, 0=invalid

            Returns:
                dist: MaskedCategorical distribution over actions
                value: State value estimate [batch]
            """
            features = self.shared(state)
            logits = self.actor(features)
            dist = MaskedCategorical(logits=logits, mask=action_mask)
            value = self.critic(features).squeeze(-1)
            return dist, value

        def get_action(
            self,
            state: torch.Tensor,
            action_mask: torch.Tensor,
            deterministic: bool = False,
        ) -> tuple[int, float, float]:
            """Sample action from policy with masking.

            Args:
                state: Single observation [1, state_dim] or [state_dim]
                action_mask: Binary mask [1, action_dim] or [action_dim]
                deterministic: If True, return argmax instead of sampling

            Returns:
                action: Selected action index
                log_prob: Log probability of action
                value: State value estimate
            """
            # inference_mode is more efficient than no_grad (disables version tracking)
            with torch.inference_mode():
                # Ensure batch dimension
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                if action_mask.dim() == 1:
                    action_mask = action_mask.unsqueeze(0)

                dist, value = self.forward(state, action_mask)
                if deterministic:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob.item(), value.item()

        def get_action_batch(
            self,
            states: torch.Tensor,
            action_masks: torch.Tensor,
            deterministic: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Sample actions for a batch of states with masking.

            Args:
                states: Batch of observations [batch, state_dim]
                action_masks: Batch of masks [batch, action_dim]
                deterministic: If True, return argmax instead of sampling

            Returns:
                actions: Selected actions [batch]
                log_probs: Log probabilities [batch]
                values: State values [batch]
            """
            with torch.inference_mode():
                dist, values = self.forward(states, action_masks)
                if deterministic:
                    actions = dist.probs.argmax(dim=-1)
                else:
                    actions = dist.sample()
                log_probs = dist.log_prob(actions)
                return actions, log_probs, values

        def evaluate_actions(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            action_masks: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Evaluate actions for PPO update with masking.

            Args:
                states: Batch of observations [batch, state_dim]
                actions: Actions taken [batch]
                action_masks: Binary masks from rollout [batch, action_dim]

            Returns:
                log_probs: Log probabilities of actions [batch]
                values: State value estimates [batch]
                entropy: Policy entropy (masked) [batch]
            """
            dist, values = self.forward(states, action_masks)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_probs, values, entropy


    class QNetwork(nn.Module):
        """Q-network for IQL: Q(s, a) for all actions."""

        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """Returns Q-values for all actions: shape (batch, action_dim)."""
            return self.net(state)


    class VNetwork(nn.Module):
        """V-network for IQL: V(s) state value."""

        def __init__(self, state_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """Returns state value: shape (batch, 1)."""
            return self.net(state)
else:
    # Stub classes when torch is not available
    ActorCritic = None
    QNetwork = None
    VNetwork = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PolicyNetwork",
    "print_confusion_matrix",
    "ActorCritic",
    "MaskedCategorical",
    "InvalidStateMachineError",
    "QNetwork",
    "VNetwork",
]
