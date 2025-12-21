"""Karn Analytics - Research-focused queries on telemetry data.

Provides typed queries on the TelemetryStore for research analysis:
- Trajectory analysis (accuracy/loss curves)
- Seed contribution analysis (counterfactual, Shapley)
- Stage duration statistics
- Performance queries (best epoch, convergence)
- Comparative analysis

Usage:
    from esper.karn import get_collector
    from esper.karn.analytics import EpisodeAnalytics

    collector = get_collector()
    analytics = EpisodeAnalytics(collector.store)

    # Get accuracy trajectory
    trajectory = analytics.accuracy_trajectory()

    # Get seed contributions
    contributions = analytics.slot_contributions()
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from esper.karn.store import SeedStage

if TYPE_CHECKING:
    from esper.karn.store import TelemetryStore


@dataclass
class TrajectoryPoint:
    """A single point in a training trajectory."""

    epoch: int
    value: float
    slot_states: dict[str, str] = field(default_factory=dict)  # slot_id -> stage name


@dataclass
class ConvergenceInfo:
    """Information about training convergence."""

    converged: bool = False
    convergence_epoch: int | None = None
    final_accuracy: float = 0.0
    plateau_length: int = 0  # Epochs without improvement
    plateau_threshold: float = 0.001  # Min improvement to not count as plateau


@dataclass
class SlotSummary:
    """Summary statistics for a single slot."""

    slot_id: str
    total_seeds: int = 0
    fossilized: int = 0
    pruned: int = 0
    current_stage: SeedStage = SeedStage.DORMANT
    epochs_active: int = 0  # Epochs with seed present

    # Performance
    mean_contribution: float = 0.0
    max_contribution: float = 0.0
    total_improvement: float = 0.0

    # Stage durations
    mean_germination_to_foster: int = 0
    mean_germination_to_prune: int = 0

    @property
    def fossilization_rate(self) -> float:
        """Percentage of seeds that fossilized vs pruned."""
        total = self.fossilized + self.pruned
        return (self.fossilized / total * 100) if total > 0 else 0.0


@dataclass
class EpisodeSummary:
    """High-level summary of a training episode."""

    episode_id: str
    total_epochs: int = 0
    final_accuracy: float = 0.0
    best_accuracy: float = 0.0
    best_epoch: int = 0

    # Seed statistics
    total_seeds_germinated: int = 0
    total_seeds_fossilized: int = 0
    total_seeds_pruned: int = 0

    # Convergence
    convergence: ConvergenceInfo = field(default_factory=ConvergenceInfo)

    # Per-slot summaries
    slot_summaries: dict[str, SlotSummary] = field(default_factory=dict)


class EpisodeAnalytics:
    """Analytics engine for querying episode telemetry.

    Provides research-focused queries on stored telemetry data.
    All methods are read-only and do not modify the store.
    """

    def __init__(self, store: "TelemetryStore"):
        self.store = store

    # =========================================================================
    # Trajectory Analysis
    # =========================================================================

    def accuracy_trajectory(self) -> list[TrajectoryPoint]:
        """Get validation accuracy trajectory over epochs.

        Returns:
            List of (epoch, accuracy, slot_states) points
        """
        points = []
        for snap in self.store.epoch_snapshots:
            slot_states = {
                slot_id: slot.stage.name
                for slot_id, slot in snap.slots.items()
            }
            points.append(TrajectoryPoint(
                epoch=snap.epoch,
                value=snap.host.val_accuracy,
                slot_states=slot_states,
            ))
        return points

    def loss_trajectory(self) -> list[TrajectoryPoint]:
        """Get validation loss trajectory over epochs."""
        points = []
        for snap in self.store.epoch_snapshots:
            points.append(TrajectoryPoint(
                epoch=snap.epoch,
                value=snap.host.val_loss,
            ))
        return points

    def train_accuracy_trajectory(self) -> list[TrajectoryPoint]:
        """Get training accuracy trajectory over epochs."""
        return [
            TrajectoryPoint(epoch=snap.epoch, value=snap.host.train_accuracy)
            for snap in self.store.epoch_snapshots
        ]

    def gradient_norm_trajectory(self) -> list[TrajectoryPoint]:
        """Get host gradient norm trajectory over epochs."""
        return [
            TrajectoryPoint(epoch=snap.epoch, value=snap.host.host_grad_norm)
            for snap in self.store.epoch_snapshots
        ]

    # =========================================================================
    # Performance Queries
    # =========================================================================

    def best_epoch(self) -> tuple[int, float]:
        """Get epoch with best validation accuracy.

        Returns:
            (epoch, accuracy) tuple
        """
        if not self.store.epoch_snapshots:
            return (0, 0.0)

        best = max(self.store.epoch_snapshots, key=lambda s: s.host.val_accuracy)
        return (best.epoch, best.host.val_accuracy)

    def worst_epoch(self) -> tuple[int, float]:
        """Get epoch with worst validation accuracy."""
        if not self.store.epoch_snapshots:
            return (0, 0.0)

        worst = min(self.store.epoch_snapshots, key=lambda s: s.host.val_accuracy)
        return (worst.epoch, worst.host.val_accuracy)

    def accuracy_at_epoch(self, epoch: int) -> float | None:
        """Get accuracy at specific epoch."""
        for snap in self.store.epoch_snapshots:
            if snap.epoch == epoch:
                return snap.host.val_accuracy
        return None

    def detect_convergence(
        self,
        window: int = 10,
        threshold: float = 0.001,
    ) -> ConvergenceInfo:
        """Detect if training has converged.

        Args:
            window: Number of epochs to check for plateau
            threshold: Minimum improvement to not count as plateau

        Returns:
            ConvergenceInfo with convergence details
        """
        if len(self.store.epoch_snapshots) < window:
            return ConvergenceInfo(
                converged=False,
                final_accuracy=self.store.epoch_snapshots[-1].host.val_accuracy
                if self.store.epoch_snapshots else 0.0,
            )

        # Check recent epochs for plateau
        recent = list(self.store.epoch_snapshots)[-window:]
        accuracies = [s.host.val_accuracy for s in recent]

        max_acc = max(accuracies)
        min_acc = min(accuracies)
        spread = max_acc - min_acc

        if spread < threshold:
            # Find when plateau started
            plateau_start = recent[0].epoch
            for i, snap in enumerate(self.store.epoch_snapshots):
                if snap.host.val_accuracy >= max_acc - threshold:
                    plateau_start = snap.epoch
                    break

            return ConvergenceInfo(
                converged=True,
                convergence_epoch=plateau_start,
                final_accuracy=accuracies[-1],
                plateau_length=len(recent),
                plateau_threshold=threshold,
            )

        return ConvergenceInfo(
            converged=False,
            final_accuracy=accuracies[-1],
        )

    # =========================================================================
    # Seed Contribution Analysis
    # =========================================================================

    def slot_contributions(self) -> dict[str, list[float]]:
        """Get counterfactual contributions per slot over time.

        Returns:
            Dict mapping slot_id to list of contributions per epoch
        """
        contributions: dict[str, list[float]] = {}

        for snap in self.store.epoch_snapshots:
            for slot_id, slot in snap.slots.items():
                if slot_id not in contributions:
                    contributions[slot_id] = []
                contributions[slot_id].append(
                    slot.counterfactual_contribution or 0.0
                )

        return contributions

    def slot_summary(self, slot_id: str) -> SlotSummary:
        """Get detailed summary for a specific slot.

        Args:
            slot_id: The slot to summarize

        Returns:
            SlotSummary with statistics
        """
        summary = SlotSummary(slot_id=slot_id)
        contributions = []

        for snap in self.store.epoch_snapshots:
            if slot_id not in snap.slots:
                continue

            slot = snap.slots[slot_id]
            summary.current_stage = slot.stage

            # Track contributions
            if slot.counterfactual_contribution:
                contributions.append(slot.counterfactual_contribution)

            # Count active epochs
            if slot.stage not in (SeedStage.DORMANT, SeedStage.PRUNED):
                summary.epochs_active += 1

            # Track lifecycle events
            if slot.stage == SeedStage.GERMINATED and slot.epochs_in_stage == 0:
                summary.total_seeds += 1
            elif slot.stage == SeedStage.FOSSILIZED and slot.epochs_in_stage == 0:
                summary.fossilized += 1
            elif slot.stage == SeedStage.PRUNED and slot.epochs_in_stage == 0:
                summary.pruned += 1

        # Compute statistics
        if contributions:
            summary.mean_contribution = statistics.mean(contributions)
            summary.max_contribution = max(contributions)
            summary.total_improvement = sum(contributions)

        return summary

    def all_slot_summaries(self) -> dict[str, SlotSummary]:
        """Get summaries for all slots."""
        slot_ids = set()
        for snap in self.store.epoch_snapshots:
            slot_ids.update(snap.slots.keys())

        return {slot_id: self.slot_summary(slot_id) for slot_id in slot_ids}

    # =========================================================================
    # Stage Duration Analysis
    # =========================================================================

    def stage_durations(self, slot_id: str) -> dict[str, list[int]]:
        """Get duration in each stage for a slot.

        Returns:
            Dict mapping stage name to list of durations (epochs)
        """
        durations: dict[str, list[int]] = {stage.name: [] for stage in SeedStage}
        current_stage: SeedStage | None = None
        stage_start = 0

        for snap in self.store.epoch_snapshots:
            if slot_id not in snap.slots:
                continue

            slot = snap.slots[slot_id]

            if current_stage is None:
                current_stage = slot.stage
                stage_start = snap.epoch
            elif slot.stage != current_stage:
                # Stage changed - record duration
                duration = snap.epoch - stage_start
                durations[current_stage.name].append(duration)
                current_stage = slot.stage
                stage_start = snap.epoch

        # Record final stage duration
        if current_stage is not None and self.store.epoch_snapshots:
            final_epoch = self.store.epoch_snapshots[-1].epoch
            duration = final_epoch - stage_start + 1
            durations[current_stage.name].append(duration)

        return durations

    def mean_stage_duration(self, stage: SeedStage) -> float:
        """Get mean duration across all slots for a stage."""
        all_durations = []

        slot_ids = set()
        for snap in self.store.epoch_snapshots:
            slot_ids.update(snap.slots.keys())

        for slot_id in slot_ids:
            durations = self.stage_durations(slot_id)
            all_durations.extend(durations.get(stage.name, []))

        return statistics.mean(all_durations) if all_durations else 0.0

    # =========================================================================
    # Episode Summary
    # =========================================================================

    def episode_summary(self) -> EpisodeSummary:
        """Generate comprehensive episode summary.

        Returns:
            EpisodeSummary with all key metrics
        """
        if not self.store.context:
            return EpisodeSummary(episode_id="unknown")

        summary = EpisodeSummary(
            episode_id=self.store.context.episode_id,
            total_epochs=len(self.store.epoch_snapshots),
        )

        if self.store.epoch_snapshots:
            summary.final_accuracy = self.store.epoch_snapshots[-1].host.val_accuracy
            best_epoch, best_acc = self.best_epoch()
            summary.best_accuracy = best_acc
            summary.best_epoch = best_epoch

        # Get convergence info
        summary.convergence = self.detect_convergence()

        # Get slot summaries
        summary.slot_summaries = self.all_slot_summaries()

        # Aggregate seed statistics
        for slot_summary in summary.slot_summaries.values():
            summary.total_seeds_germinated += slot_summary.total_seeds
            summary.total_seeds_fossilized += slot_summary.fossilized
            summary.total_seeds_pruned += slot_summary.pruned

        return summary

    # =========================================================================
    # Comparative Analysis
    # =========================================================================

    def compare_slots(self) -> dict[str, dict[str, float]]:
        """Compare performance metrics across slots.

        Returns:
            Dict mapping slot_id to metrics dict
        """
        summaries = self.all_slot_summaries()
        return {
            slot_id: {
                "mean_contribution": s.mean_contribution,
                "max_contribution": s.max_contribution,
                "fossilization_rate": s.fossilization_rate,
                "epochs_active": s.epochs_active,
            }
            for slot_id, s in summaries.items()
        }

    def accuracy_by_stage(self) -> dict[str, list[float]]:
        """Get accuracy grouped by dominant seed stage.

        Returns:
            Dict mapping stage name to list of accuracies when that stage was dominant
        """
        by_stage: dict[str, list[float]] = {stage.name: [] for stage in SeedStage}

        for snap in self.store.epoch_snapshots:
            if not snap.slots:
                by_stage["DORMANT"].append(snap.host.val_accuracy)
                continue

            # Find dominant stage (most advanced active stage)
            stages = [slot.stage for slot in snap.slots.values()]
            dominant = max(stages, key=lambda s: s.value)
            by_stage[dominant.name].append(snap.host.val_accuracy)

        return by_stage

    def improvement_rate(self, window: int = 5) -> list[float]:
        """Calculate rolling improvement rate over epochs.

        Args:
            window: Number of epochs for rolling average

        Returns:
            List of improvement rates (accuracy delta per epoch)
        """
        snapshots = list(self.store.epoch_snapshots)  # Convert deque to list for slicing
        if len(snapshots) < window + 1:
            return []

        rates = []
        for i in range(window, len(snapshots)):
            start_acc = snapshots[i - window].host.val_accuracy
            end_acc = snapshots[i].host.val_accuracy
            rate = (end_acc - start_acc) / window
            rates.append(rate)

        return rates
