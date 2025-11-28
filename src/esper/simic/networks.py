"""Policy Networks for Tamiyo

This module contains neural network architectures for policy learning,
including simple MLP policies for imitation learning and more complex
actor-critic architectures for RL.
"""

from __future__ import annotations

from pathlib import Path


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
        from esper.leyline import SimicAction

        self.TrainingSnapshot = TrainingSnapshot
        self.SimicAction = SimicAction

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Simple MLP
        self.model = nn.Sequential(
            nn.Linear(TrainingSnapshot.vector_size(), hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, len(SimicAction)),
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
            class_counts = torch.bincount(y_train, minlength=len(self.SimicAction)).float()
            # Inverse frequency weighting, with smoothing
            weights = 1.0 / (class_counts + 1.0)
            weights = weights / weights.sum() * len(self.SimicAction)  # Normalize
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights.to(self.device))
            if verbose:
                print(f"Class weights: {dict(zip([a.name for a in self.SimicAction], weights.tolist()))}")

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

    def predict(self, snapshot) -> "SimicAction":
        """Predict action for a single observation.

        Args:
            snapshot: TrainingSnapshot object

        Returns:
            SimicAction enum value
        """
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

        return self.SimicAction(action_idx)

    def predict_probs(self, snapshot) -> dict:
        """Predict action probabilities for a single observation.

        Args:
            snapshot: TrainingSnapshot object

        Returns:
            Dict mapping SimicAction to probability
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
            self.SimicAction(i): probs[i].item()
            for i in range(len(self.SimicAction))
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
        n_classes = len(self.SimicAction)
        confusion = [[0] * n_classes for _ in range(n_classes)]
        for true, pred in zip(y_true.tolist(), y_pred.tolist()):
            confusion[true][pred] += 1

        # Per-class accuracy
        class_acc = {}
        for i, action in enumerate(self.SimicAction):
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
    from esper.leyline import SimicAction

    cm = eval_result["confusion_matrix"]
    actions = [a.name for a in SimicAction]

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
# Exports
# =============================================================================

__all__ = [
    "PolicyNetwork",
    "print_confusion_matrix",
]
