"""Strict, JSON-loadable hyperparameter configuration for Simic training.

`TrainingConfig` is the single source of truth for PPO hyperparameters.
Every field maps directly to `train_ppo_vectorized` so the surface cannot
silently drift. Configs can be built from presets or loaded from JSON, with
unknown keys rejected to avoid “paper surfaces.”

Usage:
    from esper.simic.training import TrainingConfig

    # Default config (CIFAR-10 optimized)
    config = TrainingConfig()

    # Task-specific presets
    config = TrainingConfig.for_cifar_baseline()
    config = TrainingConfig.for_cifar_baseline_stable()
    config = TrainingConfig.for_cifar_scale()
    config = TrainingConfig.for_cifar_impaired()
    config = TrainingConfig.for_cifar_minimal()
    config = TrainingConfig.for_tinystories()

    # Load from JSON (fails on unknown keys)
    config = TrainingConfig.from_json_path("config.json")

    # Use with train_ppo_vectorized
    agent, history = train_ppo_vectorized(**config.to_train_kwargs())
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from esper.leyline import (
    DEFAULT_GAMMA,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_LSTM_HIDDEN_DIM,
    DEFAULT_N_ENVS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_CLIP_RATIO,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_ENTROPY_COEF_MIN,
)
from esper.simic.rewards import RewardFamily, RewardMode


@dataclass(slots=True)
class TrainingConfig:
    """Strict training configuration for vectorized PPO.

    All fields map directly to `train_ppo_vectorized` arguments. Unknown
    fields are rejected on load to prevent silent drift between the config
    surface and the runtime.
    """

    # === Training scale ===
    n_episodes: int = 100  # PPO update rounds (batches)
    n_envs: int = DEFAULT_N_ENVS
    max_epochs: int = DEFAULT_EPISODE_LENGTH  # Epochs per env per round (LSTM horizon)
    batch_size_per_env: int | None = None

    # === PPO core (from leyline defaults) ===
    lr: float = DEFAULT_LEARNING_RATE
    clip_ratio: float = DEFAULT_CLIP_RATIO
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    ppo_updates_per_batch: int = 1

    # === Entropy (exploration) ===
    entropy_coef: float = DEFAULT_ENTROPY_COEF
    entropy_coef_start: float | None = None
    entropy_coef_end: float | None = None
    entropy_coef_min: float = DEFAULT_ENTROPY_COEF_MIN
    # Total env-episodes over which to anneal entropy from start to end.
    # With n_envs=K, this produces ceil(entropy_anneal_episodes / K) PPO batches
    # of annealing. Note: this is env-episode-based, not PPO-update-based.
    # Changing n_envs changes the number of PPO updates but not the total env experience.
    entropy_anneal_episodes: int = 0

    # === Telemetry and runtime flags ===
    use_telemetry: bool = True
    amp: bool = False
    # AMP dtype: "auto" (detect BF16 support), "float16", "bfloat16", or "off"
    # BF16 eliminates GradScaler overhead on Ampere+ GPUs (compute capability >= 8.0)
    amp_dtype: str = "auto"
    # torch.compile mode for policy network: "default", "max-autotune", or "off"
    # max-autotune: 2-5x longer compile, potentially 10-20% faster runtime
    # default: fast compile, good balance for development
    # off: disable torch.compile entirely
    compile_mode: str = "default"

    # === LSTM configuration ===
    lstm_hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM
    chunk_length: int | None = None  # None = auto-match max_epochs; set explicitly if different

    # === Slot/reward selection ===
    slots: list[str] = field(default_factory=lambda: ["r0c1"])  # canonical ID (formerly "mid")
    max_seeds: int | None = None
    reward_mode: RewardMode = RewardMode.SHAPED
    reward_family: RewardFamily = RewardFamily.CONTRIBUTION
    param_budget: int = 500_000
    param_penalty_weight: float = 0.1
    sparse_reward_scale: float = 1.0
    # Floor for host-param normalization in rent/shock calculations.
    # Prevents tiny hosts (e.g., ~17 trainable params) being crushed by any seed growth.
    rent_host_params_floor: int = 200

    # === Diagnostics thresholds ===
    plateau_threshold: float = 0.5
    improvement_threshold: float = 2.0
    gradient_telemetry_stride: int = 10

    # === Reproducibility ===
    seed: int = 42

    # === Task selection ===
    # If set, overrides CLI --task. Valid: "cifar_baseline", "cifar_scale", "cifar_impaired", "cifar_minimal", "tinystories"
    task: str | None = None

    # === Gate mode ===
    # If True, quality gates only check structural requirements (epoch counts, alpha completion)
    # and let Tamiyo learn quality thresholds through reward signals. If False, gates enforce
    # hard-coded thresholds for gradient ratios, contribution levels, and stability metrics.
    permissive_gates: bool = True
    # Auto-forward gates: when enabled, Kasmina will advance stages automatically
    # when the corresponding gate passes (removes ADVANCE as a learned decision).
    auto_forward_g1: bool = False  # GERMINATED → TRAINING
    auto_forward_g2: bool = False  # TRAINING → BLENDING
    auto_forward_g3: bool = False  # BLENDING → HOLDING

    # === Ablation Flags ===
    # Used for systematic reward function experiments.
    # These disable specific reward components to measure their contribution.
    disable_pbrs: bool = False  # Disable PBRS stage advancement shaping
    disable_terminal_reward: bool = False  # Disable terminal accuracy bonus
    disable_anti_gaming: bool = False  # Disable ratio_penalty and alpha_shock

    # === Gradient Clipping ===
    # Maximum gradient norm for host/seed training.
    # Default 1.0 is standard for supervised training (PPO policy uses 0.5).
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        """Validate and set defaults.

        Note: String-to-enum coercion for reward_family/reward_mode is handled
        by from_dict(), not here. Direct construction requires proper enum values.
        Use TrainingConfig.from_dict() for external/JSON data.
        """
        # Auto-match chunk_length to max_epochs if not set
        if self.chunk_length is None:
            self.chunk_length = self.max_epochs

        # Reject string slots with helpful error (list("r0c1") would split to ['r','0','c','1'])
        if isinstance(self.slots, str):
            raise TypeError(
                f"slots must be a list of slot IDs, got string '{self.slots}'. "
                f"Did you mean slots=['{self.slots}']?"
            )
        # Normalize tuple to list for convenience
        if not isinstance(self.slots, list):
            self.slots = list(self.slots)

        self._validate()

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    @staticmethod
    def for_cifar_baseline() -> "TrainingConfig":
        """Optimized configuration for cifar_baseline task."""
        return TrainingConfig(entropy_coef=0.1, plateau_threshold=0.4)

    @staticmethod
    def for_cifar_baseline_stable() -> "TrainingConfig":
        """Conservative configuration for CIFAR tasks (slower, more stable PPO).

        Note: lr=1e-4 was removed after max_grad_norm increased from 1.0 to 5.0.
        The low LR was compensating for aggressive clipping; now uses default 3e-4.
        """
        return TrainingConfig(
            n_episodes=200,
            clip_ratio=0.1,
            entropy_coef=0.06,
            entropy_coef_start=0.06,
            entropy_coef_end=0.03,
            entropy_anneal_episodes=200,
            plateau_threshold=0.4,
        )

    @staticmethod
    def for_cifar_scale() -> "TrainingConfig":
        """Preset tuned for cifar_scale task (deeper host)."""
        return TrainingConfig(entropy_coef=0.08, plateau_threshold=0.35)

    @staticmethod
    def for_cifar_impaired() -> "TrainingConfig":
        """Preset for cifar_impaired task (severely constrained host)."""
        return TrainingConfig.for_cifar_baseline()

    @staticmethod
    def for_cifar_minimal() -> "TrainingConfig":
        """Preset for cifar_minimal task (near-random host)."""
        return TrainingConfig.for_cifar_baseline()

    @staticmethod
    def for_tinystories() -> "TrainingConfig":
        """Preset tuned for TinyStories language modeling."""
        return TrainingConfig(
            entropy_coef=0.05,
            entropy_anneal_episodes=10,
            plateau_threshold=0.3,
            improvement_threshold=1.5,
            sparse_reward_scale=0.5,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    @classmethod
    def _validate_known_keys(cls, data: dict[str, Any]) -> None:
        known_fields = {f.name for f in fields(cls)}
        unknown = set(data) - known_fields
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary, rejecting unknown keys.

        Supports aliases:
        - entropy_anneal_rounds -> entropy_anneal_episodes (Tamiyo-centric naming)
        """
        # Handle aliases before validation (detect conflicts)
        if "entropy_anneal_rounds" in data:
            if "entropy_anneal_episodes" in data:
                raise ValueError(
                    "Cannot specify both 'entropy_anneal_rounds' and 'entropy_anneal_episodes' "
                    "(they are aliases for the same field)"
                )
            data = dict(data)  # Don't mutate input
            data["entropy_anneal_episodes"] = data.pop("entropy_anneal_rounds")

        cls._validate_known_keys(data)
        parsed = dict(data)
        if "reward_family" in parsed:
            parsed["reward_family"] = RewardFamily(parsed["reward_family"])
        if "reward_mode" in parsed:
            parsed["reward_mode"] = RewardMode(parsed["reward_mode"])
        if "slots" in parsed and parsed["slots"] is not None:
            parsed["slots"] = list(parsed["slots"])
        return cls(**parsed)

    @classmethod
    def from_json_path(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file, failing on unknown keys."""
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise ValueError("Config JSON must contain an object at the top level")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        raw = asdict(self)
        raw["reward_family"] = self.reward_family.value
        raw["reward_mode"] = self.reward_mode.value
        return raw

    # ------------------------------------------------------------------
    # Runtime mappings
    # ------------------------------------------------------------------
    def to_ppo_kwargs(self) -> dict[str, Any]:
        """Extract PPOAgent constructor kwargs aligned with runtime usage."""
        entropy_steps = 0
        if self.entropy_anneal_episodes > 0:
            entropy_steps = math.ceil(self.entropy_anneal_episodes / self.n_envs) * self.ppo_updates_per_batch

        return {
            "lr": self.lr,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_ratio": self.clip_ratio,
            "entropy_coef": self.entropy_coef,
            "entropy_coef_start": self.entropy_coef_start,
            "entropy_coef_end": self.entropy_coef_end,
            "entropy_coef_min": self.entropy_coef_min,
            "entropy_anneal_steps": entropy_steps,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "chunk_length": self.chunk_length,
            "num_envs": self.n_envs,
            "max_steps_per_env": self.max_epochs,
        }

    def to_train_kwargs(self) -> dict[str, Any]:
        """Extract train_ppo_vectorized kwargs.

        Note: task is only included when explicitly set (not None),
        allowing the function's default to apply when unspecified.
        """
        kwargs: dict[str, Any] = {
            "n_episodes": self.n_episodes,
            "n_envs": self.n_envs,
            "max_epochs": self.max_epochs,
            "batch_size_per_env": self.batch_size_per_env,
            "use_telemetry": self.use_telemetry,
            "lr": self.lr,
            "clip_ratio": self.clip_ratio,
            "entropy_coef": self.entropy_coef,
            "entropy_coef_start": self.entropy_coef_start,
            "entropy_coef_end": self.entropy_coef_end,
            "entropy_coef_min": self.entropy_coef_min,
            "entropy_anneal_episodes": self.entropy_anneal_episodes,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ppo_updates_per_batch": self.ppo_updates_per_batch,
            "amp": self.amp,
            "amp_dtype": self.amp_dtype,
            "compile_mode": self.compile_mode,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "chunk_length": self.chunk_length,
            "slots": list(self.slots),
            "max_seeds": self.max_seeds,
            "reward_mode": self.reward_mode.value,
            "reward_family": self.reward_family.value,
            "param_budget": self.param_budget,
            "param_penalty_weight": self.param_penalty_weight,
            "sparse_reward_scale": self.sparse_reward_scale,
            "rent_host_params_floor": self.rent_host_params_floor,
            "plateau_threshold": self.plateau_threshold,
            "improvement_threshold": self.improvement_threshold,
            "gradient_telemetry_stride": self.gradient_telemetry_stride,
            "seed": self.seed,
            "permissive_gates": self.permissive_gates,
            "auto_forward_g1": self.auto_forward_g1,
            "auto_forward_g2": self.auto_forward_g2,
            "auto_forward_g3": self.auto_forward_g3,
            "disable_pbrs": self.disable_pbrs,
            "disable_terminal_reward": self.disable_terminal_reward,
            "disable_anti_gaming": self.disable_anti_gaming,
            "max_grad_norm": self.max_grad_norm,
        }
        # Only include task when explicitly set (not None)
        if self.task is not None:
            kwargs["task"] = self.task
        return kwargs

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_positive(self, value: int, name: str) -> None:
        if value < 1:
            raise ValueError(f"{name} must be >= 1 (got {value})")

    def _validate_range(
        self, value: float, name: str, min_val: float, max_val: float,
        min_inclusive: bool = True, max_inclusive: bool = True
    ) -> None:
        """Validate value is within range."""
        min_ok = value >= min_val if min_inclusive else value > min_val
        max_ok = value <= max_val if max_inclusive else value < max_val
        if not (min_ok and max_ok):
            min_bracket = "[" if min_inclusive else "("
            max_bracket = "]" if max_inclusive else ")"
            raise ValueError(
                f"{name} must be in {min_bracket}{min_val}, {max_val}{max_bracket} (got {value})"
            )

    def _validate(self) -> None:
        self._validate_positive(self.n_episodes, "n_episodes")
        self._validate_positive(self.n_envs, "n_envs")
        self._validate_positive(self.max_epochs, "max_epochs")
        self._validate_positive(self.ppo_updates_per_batch, "ppo_updates_per_batch")
        self._validate_positive(self.gradient_telemetry_stride, "gradient_telemetry_stride")
        if self.batch_size_per_env is not None:
            self._validate_positive(self.batch_size_per_env, "batch_size_per_env")

        if self.chunk_length != self.max_epochs:
            raise ValueError(
                "chunk_length must match max_epochs for the current training loop"
            )

        if self.entropy_anneal_episodes < 0:
            raise ValueError("entropy_anneal_episodes cannot be negative")

        if self.max_seeds is not None and self.max_seeds < 1:
            raise ValueError("max_seeds must be >= 1 when provided")

        if not self.slots:
            raise ValueError("slots cannot be empty")
        # Validate slot IDs using canonical format validation
        from esper.leyline.slot_id import validate_slot_ids, SlotIdError
        try:
            validate_slot_ids(list(self.slots))
        except SlotIdError as e:
            raise ValueError(f"Invalid slot configuration: {e}") from e

        # Validate task if specified (dynamically lookup valid tasks)
        if self.task is not None:
            # Import here to avoid circular dependency at module load time
            from esper.runtime.tasks import VALID_TASKS
            if self.task not in VALID_TASKS:
                raise ValueError(
                    f"Invalid task '{self.task}'. Valid options: {sorted(VALID_TASKS)}"
                )

        if self.param_budget <= 0:
            raise ValueError("param_budget must be positive")
        self._validate_positive(self.rent_host_params_floor, "rent_host_params_floor")

        # Validate amp_dtype
        valid_amp_dtypes = {"auto", "float16", "bfloat16", "off"}
        if self.amp_dtype not in valid_amp_dtypes:
            raise ValueError(
                f"amp_dtype must be one of {sorted(valid_amp_dtypes)} (got '{self.amp_dtype}')"
            )

        # Validate compile_mode
        valid_compile_modes = {"default", "max-autotune", "reduce-overhead", "off"}
        if self.compile_mode not in valid_compile_modes:
            raise ValueError(
                f"compile_mode must be one of {sorted(valid_compile_modes)} (got '{self.compile_mode}')"
            )
        if self.param_penalty_weight < 0:
            raise ValueError("param_penalty_weight must be non-negative")
        if self.sparse_reward_scale <= 0:
            raise ValueError("sparse_reward_scale must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

        if (
            self.reward_family == RewardFamily.LOSS
            and self.reward_mode != RewardMode.SHAPED
        ):
            raise ValueError("reward_mode applies only to contribution rewards")

        # PPO hyperparameter ranges
        self._validate_range(self.gamma, "gamma", 0.0, 1.0, min_inclusive=False, max_inclusive=True)
        self._validate_range(self.gae_lambda, "gae_lambda", 0.0, 1.0, min_inclusive=False, max_inclusive=True)
        self._validate_range(self.clip_ratio, "clip_ratio", 0.0, 1.0, min_inclusive=False, max_inclusive=True)
        self._validate_range(self.lr, "lr", 0.0, float("inf"), min_inclusive=False, max_inclusive=False)
        self._validate_range(self.entropy_coef, "entropy_coef", 0.0, float("inf"), min_inclusive=True, max_inclusive=False)

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable summary of configuration."""
        auto_forward = []
        if self.auto_forward_g1:
            auto_forward.append("G1")
        if self.auto_forward_g2:
            auto_forward.append("G2")
        if self.auto_forward_g3:
            auto_forward.append("G3")

        lines = [
            "TrainingConfig:",
            f"  PPO: lr={self.lr}, gamma={self.gamma}, gae_lambda={self.gae_lambda}, clip={self.clip_ratio}",
            f"  Rounds: {self.n_episodes} (env episodes={self.n_episodes * self.n_envs}), "
            f"envs={self.n_envs}, max_epochs={self.max_epochs}",
            f"  Entropy: {self.entropy_coef}" + (f" -> {self.entropy_coef_end}" if self.entropy_coef_end else ""),
            f"  Updates/batch: {self.ppo_updates_per_batch}, amp={'on' if self.amp else 'off'}, amp_dtype={self.amp_dtype}, compile={self.compile_mode}",
            f"  LSTM: hidden={self.lstm_hidden_dim}, chunk={self.chunk_length}",
            f"  Slots: {','.join(self.slots)} | reward_family={self.reward_family.value} mode={self.reward_mode.value}",
            f"  Telemetry: {'enabled' if self.use_telemetry else 'disabled'}, gates={'permissive' if self.permissive_gates else 'strict'}",
            f"  Auto-forward: {','.join(auto_forward) if auto_forward else 'off'}",
        ]
        return "\n".join(lines)


__all__ = ["TrainingConfig"]
