"""Task presets and factories for Tolaria/Simic wiring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from esper.kasmina.host import CNNHost, MorphogeneticModel, TransformerHost
from esper.leyline.actions import build_action_enum
from esper.leyline import DEFAULT_HOST_LR, DEFAULT_SEED_LR, DEFAULT_BATCH_SIZE_TRAINING, DEFAULT_DROPOUT
from esper.tamiyo.policy.features import TaskConfig
from esper.simic.rewards import LossRewardConfig
from torch.utils.data import Dataset

from esper.utils.data import load_cifar10, load_tinystories, get_cifar10_datasets


@dataclass(slots=True)
class TaskSpec:
    """Task-specific wiring for hosts, dataloaders, and action enums.

    Learning rate fields allow task-specific tuning without code changes:
    - host_lr: Learning rate for host model parameters (backbone)
    - seed_lr: Learning rate for seed module parameters (new capacity)

    Common patterns:
    - Same LR for both (default): Simple baseline
    - Lower seed_lr: More conservative seed training for stability
    - Lower host_lr for transformers: Typical for attention models
    """

    name: str
    topology: str
    task_type: str  # "classification" | "lm"
    model_factory: Callable[[str], MorphogeneticModel]
    dataloader_factory: Callable[..., tuple]
    dataloader_defaults: dict[str, Any] = field(default_factory=dict)
    task_config: TaskConfig = field(default_factory=TaskConfig.for_cifar10)
    loss_reward_config: LossRewardConfig = field(default_factory=LossRewardConfig.default)
    vocab_size: int | None = None
    num_classes: int | None = None
    block_size: int | None = None

    # Learning rates (from leyline - single source of truth)
    host_lr: float = DEFAULT_HOST_LR
    seed_lr: float = DEFAULT_SEED_LR

    action_enum: Any = field(init=False)

    def __post_init__(self) -> None:
        self.action_enum = build_action_enum(self.topology)

    def create_model(
        self,
        device: str = "cuda:0",
        slots: list[str] | None = None,
        permissive_gates: bool = True,
    ) -> MorphogeneticModel:
        """Instantiate model for this task on the target device.

        Args:
            device: Target device for the model.
            slots: Seed slots to enable. Required - cannot be None or empty.
            permissive_gates: If True, quality gates only check structural requirements
                and let Tamiyo learn quality thresholds through reward signals.
        """
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        return self.model_factory(device, slots=slots, permissive_gates=permissive_gates)

    def create_dataloaders(self, **overrides):
        """Instantiate dataloaders with defaults merged with overrides."""
        params = {**self.dataloader_defaults, **overrides}
        return self.dataloader_factory(**params)

    def get_datasets(self) -> tuple[Dataset, Dataset]:
        """Get raw train/test datasets (for SharedBatchIterator).

        Returns:
            Tuple of (trainset, testset) Dataset objects.
        """
        if self.name in ("cifar10", "cifar10_deep", "cifar10_blind"):
            data_root = self.dataloader_defaults.get("data_root", "./data")
            mock = self.dataloader_defaults.get("mock", False)
            return get_cifar10_datasets(data_root=data_root, mock=mock)
        elif self.name == "tinystories":
            # TinyStories uses TinyStoriesDataset class directly
            from esper.utils.data import TinyStoriesDataset
            block_size = self.dataloader_defaults.get("block_size", 256)
            max_train = self.dataloader_defaults.get("max_train_samples")
            max_val = self.dataloader_defaults.get("max_val_samples")
            mock = self.dataloader_defaults.get("mock", False)
            trainset = TinyStoriesDataset(
                split="train", block_size=block_size, max_samples=max_train, mock=mock
            )
            testset = TinyStoriesDataset(
                split="validation", block_size=block_size, max_samples=max_val, mock=mock
            )
            return trainset, testset
        else:
            raise NotImplementedError(f"get_datasets not implemented for task: {self.name}")


def _cifar10_spec() -> TaskSpec:
    """CIFAR-10 classification with CNN host."""
    cifar_config = TaskConfig.for_cifar10()
    loss_cfg = LossRewardConfig.for_cifar10()

    def _make_model(
        device: str, slots: list[str] | None = None, permissive_gates: bool = True
    ) -> MorphogeneticModel:
        # Deliberately weak host (8 base channels vs default 32) to leave
        # headroom for seeds to demonstrate value. Expected accuracy ~40-50%.
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        host = CNNHost(num_classes=10, base_channels=8)
        return MorphogeneticModel(
            host, device=device, slots=slots, task_config=cifar_config,
            permissive_gates=permissive_gates,
        )

    return TaskSpec(
        name="cifar10",
        topology="cnn",
        task_type="classification",
        model_factory=_make_model,
        dataloader_factory=load_cifar10,
        dataloader_defaults={
            "batch_size": DEFAULT_BATCH_SIZE_TRAINING,
            "data_root": "./data",
            "num_workers": 4,
            "mock": False,
            "generator": None,
        },
        task_config=cifar_config,
        loss_reward_config=loss_cfg,
        num_classes=10,
    )


def _cifar10_deep_spec() -> TaskSpec:
    """CIFAR-10 with deeper CNN for better GPU utilization.

    Same narrow channels as cifar10 (base_channels=8) to keep the model
    weak enough that seeds matter, but 6 blocks instead of 3 to increase
    compute per sample and better saturate GPU.

    Use this variant when profiling shows data loading is not the bottleneck
    but GPU utilization is still low.
    """
    cifar_config = TaskConfig.for_cifar10()
    loss_cfg = LossRewardConfig.for_cifar10()

    def _make_model(
        device: str, slots: list[str] | None = None, permissive_gates: bool = True
    ) -> MorphogeneticModel:
        # Deep but narrow: 5 blocks with 8 base channels (8→16→32→64→128)
        # 5 pools → spatial: 32→16→8→4→2→1
        # 4 injection points at 4 distinct spatial resolutions (8×8, 4×4, 2×2, 1×1).
        # Despite 100K params, baseline accuracy is only ~42% due to aggressive
        # downsampling - seeds have plenty of room to contribute.
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        host = CNNHost(num_classes=10, base_channels=8, n_blocks=5, pool_layers=5)
        return MorphogeneticModel(
            host, device=device, slots=slots, task_config=cifar_config,
            permissive_gates=permissive_gates,
        )

    return TaskSpec(
        name="cifar10_deep",
        topology="cnn",
        task_type="classification",
        model_factory=_make_model,
        dataloader_factory=load_cifar10,
        dataloader_defaults={
            "batch_size": DEFAULT_BATCH_SIZE_TRAINING,
            "data_root": "./data",
            "num_workers": 4,
            "mock": False,
            "generator": None,
        },
        task_config=cifar_config,
        loss_reward_config=loss_cfg,
        num_classes=10,
    )


def _tinystories_spec() -> TaskSpec:
    """TinyStories language modeling with transformer host."""
    ts_config = TaskConfig.for_tinystories()
    loss_cfg = LossRewardConfig.for_tinystories()
    block_size = 256
    vocab_size = 50257

    def _make_model(
        device: str, slots: list[str] | None = None, permissive_gates: bool = True
    ) -> MorphogeneticModel:
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        host = TransformerHost(
            vocab_size=vocab_size,
            n_embd=384,
            n_head=6,
            n_layer=6,
            block_size=block_size,
            dropout=DEFAULT_DROPOUT,
        )
        return MorphogeneticModel(
            host, device=device, slots=slots, task_config=ts_config,
            permissive_gates=permissive_gates,
        )

    return TaskSpec(
        name="tinystories",
        topology="transformer",
        task_type="lm",
        model_factory=_make_model,
        dataloader_factory=load_tinystories,
        dataloader_defaults={
            "block_size": block_size,
            "batch_size": 32,
            "max_train_samples": None,
            "max_val_samples": None,
            "num_workers": 0,
            "mock": False,
        },
        task_config=ts_config,
        loss_reward_config=loss_cfg,
        vocab_size=vocab_size,
        block_size=block_size,
    )


def _cifar10_blind_spec() -> TaskSpec:
    """CIFAR-10 with 1x1 convolutions only ("Spatial Blindness").

    This host cannot see spatial features (shapes, edges) and relies purely on
    per-pixel channel statistics. Baseline accuracy caps at ~25-30%.
    Seeds (which have 3x3 kernels or attention) must be used to restore
    spatial awareness.
    """
    cifar_config = TaskConfig.for_cifar10()
    loss_cfg = LossRewardConfig.for_cifar10()

    def _make_model(
        device: str, slots: list[str] | None = None, permissive_gates: bool = True
    ) -> MorphogeneticModel:
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        # Medium-Weak blind host: 32 channels (for bandwidth), 2 blocks, no pooling.
        # This provides a wide feature space for seeds but zero spatial context.
        host = CNNHost(num_classes=10, base_channels=32, n_blocks=2, pool_layers=0, kernel_size=1)
        return MorphogeneticModel(
            host, device=device, slots=slots, task_config=cifar_config,
            permissive_gates=permissive_gates,
        )

    return TaskSpec(
        name="cifar10_blind",
        topology="cnn",
        task_type="classification",
        model_factory=_make_model,
        dataloader_factory=load_cifar10,
        dataloader_defaults={
            "batch_size": DEFAULT_BATCH_SIZE_TRAINING,
            "data_root": "./data",
            "num_workers": 4,
            "mock": False,
            "generator": None,
        },
        task_config=cifar_config,
        loss_reward_config=loss_cfg,
        num_classes=10,
    )


def get_task_spec(name: str) -> TaskSpec:
    """Return TaskSpec preset by name."""
    key = name.lower()
    if key == "cifar10":
        return _cifar10_spec()
    if key == "cifar10_deep":
        return _cifar10_deep_spec()
    if key == "cifar10_blind":
        return _cifar10_blind_spec()
    if key == "tinystories":
        return _tinystories_spec()
    raise ValueError(f"Unknown task '{name}'. Available: cifar10, cifar10_deep, cifar10_blind, tinystories")


__all__ = ["TaskSpec", "get_task_spec"]
