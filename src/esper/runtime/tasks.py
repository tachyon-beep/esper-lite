"""Task presets and factories for Tolaria/Simic wiring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from esper.kasmina.host import CNNHost, MorphogeneticModel, TransformerHost
from esper.tamiyo.action_enums import build_action_enum
from esper.leyline import DEFAULT_HOST_LR, DEFAULT_SEED_LR, DEFAULT_BATCH_SIZE_TRAINING, DEFAULT_DROPOUT, LossRewardConfig
from esper.tamiyo.policy.features import TaskConfig
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
    topology: Literal["cnn", "transformer"]
    task_type: Literal["classification", "lm"]
    model_factory: Callable[[str], MorphogeneticModel]
    dataloader_factory: Callable[..., tuple[Any, Any]]
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
        return self.model_factory(device, slots=slots, permissive_gates=permissive_gates)  # type: ignore[call-arg]

    def create_dataloaders(self, **overrides: Any) -> tuple[Any, Any]:
        """Instantiate dataloaders with defaults merged with overrides."""
        params = {**self.dataloader_defaults, **overrides}
        result: tuple[Any, Any] = self.dataloader_factory(**params)
        return result

    def get_datasets(self) -> tuple[Dataset[Any], Dataset[Any]]:
        """Get raw train/test datasets (for SharedBatchIterator).

        Returns:
            Tuple of (trainset, testset) Dataset objects.
        """
        # All CIFAR-10 based tasks share the same dataset
        cifar_tasks = (
            "cifar_baseline", "cifar_scale", "cifar_impaired", "cifar_minimal",
            "cifar_multichannel", "cifar_postpool", "cifar_highres", "cifar_deep_multichannel",
        )
        if self.name in cifar_tasks:
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


def _cifar_baseline_spec() -> TaskSpec:
    """CIFAR-10 baseline: capable host with headroom for seed augmentation.

    Testing objective: Can seeds AUGMENT an already-capable host?

    Architecture:
    - Standard 3×3 kernels (spatial features enabled)
    - Moderate channels: 8→16→32
    - 3 pooling layers
    - ~6K parameters

    Expected behavior:
    - Host alone: ~45-50% accuracy
    - With seeds: ~65-75% accuracy
    - Seeds add refinement rather than rescue
    """
    cifar_config = TaskConfig.for_cifar10()
    loss_cfg = LossRewardConfig.for_cifar10()

    def _make_model(
        device: str, slots: list[str] | None = None, permissive_gates: bool = True
    ) -> MorphogeneticModel:
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        host = CNNHost(num_classes=10, base_channels=8)
        return MorphogeneticModel(
            host, device=device, slots=slots, task_config=cifar_config,
            permissive_gates=permissive_gates,
        )

    return TaskSpec(
        name="cifar_baseline",
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


def _cifar_scale_spec() -> TaskSpec:
    """CIFAR-10 scale: deeper network for multi-slot scaling tests.

    Testing objective: Does the system SCALE to deeper networks with more slots?

    Architecture:
    - Standard 3×3 kernels
    - Deep: 5 blocks (8→16→32→64→128)
    - 5 pooling layers (32→16→8→4→2→1)
    - ~100K parameters
    - 5 injection points at different spatial resolutions

    Expected behavior:
    - Host alone: ~42% accuracy (aggressive downsampling limits it)
    - Tests GPU utilization and multi-slot coordination
    - Use when profiling shows GPU isn't saturated
    """
    cifar_config = TaskConfig.for_cifar10()
    loss_cfg = LossRewardConfig.for_cifar10()

    def _make_model(
        device: str, slots: list[str] | None = None, permissive_gates: bool = True
    ) -> MorphogeneticModel:
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        host = CNNHost(num_classes=10, base_channels=8, n_blocks=5, pool_layers=5)
        return MorphogeneticModel(
            host, device=device, slots=slots, task_config=cifar_config,
            permissive_gates=permissive_gates,
        )

    return TaskSpec(
        name="cifar_scale",
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


def _cifar_impaired_spec() -> TaskSpec:
    """CIFAR-10 impaired: severely constrained host requiring seed rescue.

    Testing objective: Can seeds RESCUE a host with severe information bottleneck?

    Architecture:
    - 1×1 kernels only (no spatial receptive field)
    - 1-channel bottleneck (3→1→2→4 progression)
    - 77 parameters total
    - MaxPool provides some coarse spatial awareness

    Expected behavior:
    - Host alone: ~18-22% accuracy (barely above random)
    - The 1-channel bottleneck crushes RGB to luminance-only
    - Seeds must provide nearly all useful capability
    - Good for testing Tamiyo's rescue decision-making
    """
    cifar_config = TaskConfig.for_cifar10()
    loss_cfg = LossRewardConfig.for_cifar10()

    def _make_model(
        device: str, slots: list[str] | None = None, permissive_gates: bool = True
    ) -> MorphogeneticModel:
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        # Severely impaired: 1-channel bottleneck, 1×1 kernels
        # Channel progression: 3→1→2→4 (77 total parameters)
        host = CNNHost(num_classes=10, base_channels=1, n_blocks=3, kernel_size=1)
        return MorphogeneticModel(
            host, device=device, slots=slots, task_config=cifar_config,
            permissive_gates=permissive_gates,
        )

    return TaskSpec(
        name="cifar_impaired",
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


def _cifar_minimal_spec() -> TaskSpec:
    """CIFAR-10 minimal: extreme constraint where seeds provide ALL capability.

    Testing objective: Can seeds work as SOLO load-bearers with a near-useless host?

    Architecture:
    - 1×1 kernels only (no spatial receptive field)
    - NO pooling (seeds must add spatial aggregation)
    - Ultra-narrow channels: 2→4→8 progression
    - 3→2 input bottleneck crushes RGB immediately
    - ~150 parameters total

    Expected behavior:
    - Host alone: ~12-18% accuracy (barely above 10% random chance)
    - No spatial awareness (no MaxPool competition)
    - Seeds at r0c0 have full 32×32 spatial resolution
    - This is the "nuclear option" for rescue testing
    """
    cifar_config = TaskConfig.for_cifar10()
    loss_cfg = LossRewardConfig.for_cifar10()

    def _make_model(
        device: str, slots: list[str] | None = None, permissive_gates: bool = True
    ) -> MorphogeneticModel:
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        # Minimal host: 2→4→8 channels, no pooling, 1×1 kernels
        # The 3→2 input bottleneck crushes RGB to 2 dimensions
        host = CNNHost(
            num_classes=10,
            base_channels=2,   # Ultra-narrow: 2→4→8
            n_blocks=3,        # 3 injection points for seeds
            kernel_size=1,     # No spatial features
            pool_layers=0,     # No pooling - seeds must add spatial aggregation
        )
        return MorphogeneticModel(
            host, device=device, slots=slots, task_config=cifar_config,
            permissive_gates=permissive_gates,
        )

    return TaskSpec(
        name="cifar_minimal",
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


# =============================================================================
# Multichannel Task Specs (PRE_POOL + POST_POOL surfaces)
# =============================================================================


def _cifar_multichannel_spec() -> TaskSpec:
    """CIFAR-10 multichannel: seeds at both PRE_POOL and POST_POOL surfaces.

    Testing objective: Can seeds benefit from SURFACE DIVERSITY (resolution vs semantics)?

    Architecture:
    - Standard 3-block CNN (8→16→32 channels)
    - 6 injection points in 2×3 grid:
        Row 0 (PRE_POOL):  r0c0(32×32), r0c1(16×16), r0c2(8×8)  - high resolution
        Row 1 (POST_POOL): r1c0(16×16), r1c1(8×8), r1c2(4×4)    - after pooling

    Surface semantics:
    - PRE_POOL: Higher spatial resolution, lower-level features
    - POST_POOL: Lower spatial resolution, higher-level (more semantic) features

    Expected behavior:
    - Seeds at PRE_POOL can add fine-grained spatial processing
    - Seeds at POST_POOL can add abstract/semantic processing
    - Tamiyo can learn which surfaces benefit most from augmentation
    """
    cifar_config = TaskConfig.for_cifar10()
    loss_cfg = LossRewardConfig.for_cifar10()

    def _make_model(
        device: str, slots: list[str] | None = None, permissive_gates: bool = True
    ) -> MorphogeneticModel:
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        host = CNNHost(num_classes=10, base_channels=8, n_blocks=3)
        return MorphogeneticModel(
            host, device=device, slots=slots, task_config=cifar_config,
            permissive_gates=permissive_gates,
        )

    return TaskSpec(
        name="cifar_multichannel",
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


def _cifar_postpool_spec() -> TaskSpec:
    """CIFAR-10 post-pool only: seeds at POST_POOL surfaces for semantic processing.

    Testing objective: Are POST_POOL surfaces better for SEMANTIC augmentation?

    Architecture:
    - Standard 3-block CNN (16→32→64 channels)
    - Slots ONLY at POST_POOL surfaces (row 1): r1c0, r1c1, r1c2
    - These inject AFTER pooling at each block

    Surface semantics:
    - POST_POOL surfaces have lower spatial resolution
    - But features are more pooled/aggregated (arguably more semantic)
    - Seeds add processing to already-pooled representations

    Expected behavior:
    - Seeds operate on spatially-reduced feature maps
    - May be more parameter-efficient (smaller spatial dims)
    - Good for testing if semantic-level intervention is sufficient
    """
    cifar_config = TaskConfig.for_cifar10()
    loss_cfg = LossRewardConfig.for_cifar10()

    def _make_model(
        device: str, slots: list[str] | None = None, permissive_gates: bool = True
    ) -> MorphogeneticModel:
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        # Slightly larger channels for more capacity at POST_POOL
        host = CNNHost(num_classes=10, base_channels=16, n_blocks=3)
        return MorphogeneticModel(
            host, device=device, slots=slots, task_config=cifar_config,
            permissive_gates=permissive_gates,
        )

    return TaskSpec(
        name="cifar_postpool",
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


def _cifar_highres_spec() -> TaskSpec:
    """CIFAR-10 high-res: seeds at PRE_POOL surfaces for fine-grained processing.

    Testing objective: Do PRE_POOL surfaces enable better SPATIAL augmentation?

    Architecture:
    - Standard 3-block CNN (16→32→64 channels)
    - Slots ONLY at PRE_POOL surfaces (row 0): r0c0, r0c1, r0c2
    - These inject BEFORE pooling at each block

    Surface semantics:
    - PRE_POOL surfaces have full spatial resolution for their block
    - r0c0 sees 32×32, r0c1 sees 16×16, r0c2 sees 8×8
    - Seeds can add fine-grained spatial processing before downsampling

    Expected behavior:
    - Seeds operate on higher-resolution feature maps
    - More parameters needed (larger spatial dims) but finer control
    - Good for testing if spatial-level intervention helps most
    """
    cifar_config = TaskConfig.for_cifar10()
    loss_cfg = LossRewardConfig.for_cifar10()

    def _make_model(
        device: str, slots: list[str] | None = None, permissive_gates: bool = True
    ) -> MorphogeneticModel:
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        host = CNNHost(num_classes=10, base_channels=16, n_blocks=3)
        return MorphogeneticModel(
            host, device=device, slots=slots, task_config=cifar_config,
            permissive_gates=permissive_gates,
        )

    return TaskSpec(
        name="cifar_highres",
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


def _cifar_deep_multichannel_spec() -> TaskSpec:
    """CIFAR-10 deep multichannel: 5-block network with 10 injection points.

    Testing objective: Does multichannel SCALE to deeper networks?

    Architecture:
    - Deep 5-block CNN (8→16→32→64→128 channels)
    - 10 injection points in 2×5 grid:
        Row 0 (PRE_POOL):  r0c0, r0c1, r0c2, r0c3, r0c4
        Row 1 (POST_POOL): r1c0, r1c1, r1c2, r1c3, r1c4
    - Full pooling on all blocks (32→16→8→4→2→1)

    Expected behavior:
    - 10 slots allows testing multi-slot coordination at scale
    - Early slots (c0, c1) have larger spatial dims
    - Late slots (c3, c4) have more channels but smaller spatial dims
    - Tamiyo must learn to coordinate across many surfaces
    """
    cifar_config = TaskConfig.for_cifar10()
    loss_cfg = LossRewardConfig.for_cifar10()

    def _make_model(
        device: str, slots: list[str] | None = None, permissive_gates: bool = True
    ) -> MorphogeneticModel:
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")
        host = CNNHost(num_classes=10, base_channels=8, n_blocks=5, pool_layers=5)
        return MorphogeneticModel(
            host, device=device, slots=slots, task_config=cifar_config,
            permissive_gates=permissive_gates,
        )

    return TaskSpec(
        name="cifar_deep_multichannel",
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


# Registry of valid task names (single source of truth)
#
# Naming schema (test-objective focused):
#   cifar_baseline        - Can seeds AUGMENT a capable host?
#   cifar_scale           - Does the system SCALE to deeper networks?
#   cifar_impaired        - Can seeds RESCUE a severely constrained host?
#   cifar_minimal         - Can seeds work as SOLO load-bearers?
#   cifar_multichannel    - Can seeds benefit from SURFACE DIVERSITY?
#   cifar_postpool        - Are POST_POOL surfaces better for SEMANTIC augmentation?
#   cifar_highres         - Do PRE_POOL surfaces enable better SPATIAL augmentation?
#   cifar_deep_multichannel - Does multichannel SCALE to deeper networks?
#
VALID_TASKS: frozenset[str] = frozenset({
    "cifar_baseline",
    "cifar_scale",
    "cifar_impaired",
    "cifar_minimal",
    "cifar_multichannel",
    "cifar_postpool",
    "cifar_highres",
    "cifar_deep_multichannel",
    "tinystories",
})


def get_task_spec(name: str) -> TaskSpec:
    """Return TaskSpec preset by name."""
    key = name.lower()
    if key == "cifar_baseline":
        return _cifar_baseline_spec()
    if key == "cifar_scale":
        return _cifar_scale_spec()
    if key == "cifar_impaired":
        return _cifar_impaired_spec()
    if key == "cifar_minimal":
        return _cifar_minimal_spec()
    if key == "cifar_multichannel":
        return _cifar_multichannel_spec()
    if key == "cifar_postpool":
        return _cifar_postpool_spec()
    if key == "cifar_highres":
        return _cifar_highres_spec()
    if key == "cifar_deep_multichannel":
        return _cifar_deep_multichannel_spec()
    if key == "tinystories":
        return _tinystories_spec()
    raise ValueError(
        f"Unknown task '{name}'. Available: {', '.join(sorted(VALID_TASKS))}"
    )


__all__ = ["TaskSpec", "get_task_spec", "VALID_TASKS"]
