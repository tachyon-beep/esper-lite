"""Task presets and factories for Tolaria/Simic wiring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from esper.kasmina.host import CNNHost, MorphogeneticModel, TransformerHost
from esper.leyline.actions import build_action_enum
from esper.simic.features import TaskConfig
from esper.simic.rewards import LossRewardConfig
from esper.utils.data import load_cifar10, load_tinystories


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

    # Learning rates (previously hardcoded as 0.01 throughout training code)
    host_lr: float = 0.01
    seed_lr: float = 0.01

    action_enum: Any = field(init=False)

    def __post_init__(self) -> None:
        self.action_enum = build_action_enum(self.topology)

    def create_model(self, device: str = "cuda:0") -> MorphogeneticModel:
        """Instantiate model for this task on the target device."""
        return self.model_factory(device)

    def create_dataloaders(self, **overrides):
        """Instantiate dataloaders with defaults merged with overrides."""
        params = {**self.dataloader_defaults, **overrides}
        return self.dataloader_factory(**params)


def _cifar10_spec() -> TaskSpec:
    """CIFAR-10 classification with CNN host."""
    cifar_config = TaskConfig.for_cifar10()
    loss_cfg = LossRewardConfig.for_cifar10()

    def _make_model(device: str) -> MorphogeneticModel:
        # Deliberately weak host (8 base channels vs default 32) to leave
        # headroom for seeds to demonstrate value. Expected accuracy ~40-50%.
        host = CNNHost(num_classes=10, base_channels=8)
        return MorphogeneticModel(host, device=device, task_config=cifar_config)

    return TaskSpec(
        name="cifar10",
        topology="cnn",
        task_type="classification",
        model_factory=_make_model,
        dataloader_factory=load_cifar10,
        dataloader_defaults={
            "batch_size": 128,
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

    def _make_model(device: str) -> MorphogeneticModel:
        host = TransformerHost(
            vocab_size=vocab_size,
            n_embd=384,
            n_head=6,
            n_layer=6,
            block_size=block_size,
            dropout=0.1,
        )
        return MorphogeneticModel(host, device=device, task_config=ts_config)

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


def get_task_spec(name: str) -> TaskSpec:
    """Return TaskSpec preset by name."""
    key = name.lower()
    if key == "cifar10":
        return _cifar10_spec()
    if key == "tinystories":
        return _tinystories_spec()
    raise ValueError(f"Unknown task '{name}'. Available: cifar10, tinystories")


__all__ = ["TaskSpec", "get_task_spec"]
