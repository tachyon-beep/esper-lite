"""Environment configuration utilities.

Derived from Sprint 0 enablement requirements described in
`docs/project/implementation_plan.md`. Values are loaded via environment
variables (see `.env.example`).
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EsperSettings(BaseSettings):
    """Top-level configuration container for Esper-Lite services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        frozen=True,
        extra="ignore",
    )

    redis_url: str = Field(alias="REDIS_URL", default="redis://localhost:6379/0")
    oona_normal_stream: str = Field(alias="OONA_NORMAL_STREAM", default="oona.normal")
    oona_emergency_stream: str = Field(
        alias="OONA_EMERGENCY_STREAM", default="oona.emergency"
    )
    oona_telemetry_stream: str = Field(
        alias="OONA_TELEMETRY_STREAM", default="oona.telemetry"
    )
    oona_policy_stream: str = Field(alias="OONA_POLICY_STREAM", default="oona.policy")
    oona_message_ttl_ms: int | None = Field(alias="OONA_MESSAGE_TTL_MS", default=900_000)
    kernel_freshness_window_ms: int = Field(
        alias="KERNEL_FRESHNESS_WINDOW_MS", default=60_000
    )
    kernel_nonce_cache_size: int = Field(alias="KERNEL_NONCE_CACHE_SIZE", default=4096)
    tezzeret_inductor_cache_dir: str | None = Field(
        alias="TEZZERET_INDUCTOR_CACHE_DIR",
        default=None,
    )

    prometheus_pushgateway: str = Field(
        alias="PROMETHEUS_PUSHGATEWAY", default="http://localhost:9091"
    )
    elasticsearch_url: str = Field(
        alias="ELASTICSEARCH_URL", default="http://localhost:9200"
    )

    leyline_schema_dir: str = Field(alias="LEYLINE_SCHEMA_DIR", default="./contracts/leyline")
    leyline_namespace: str = Field(
        alias="LEYLINE_DEFAULT_NAMESPACE", default="esper.leyline.v1"
    )

    tamiyo_policy_dir: str = Field(alias="TAMIYO_POLICY_DIR", default="./var/tamiyo/policies")
    tamiyo_conservative_mode: bool = Field(
        alias="TAMIYO_CONSERVATIVE_MODE", default=False
    )
    tamiyo_field_report_retention_hours: int = Field(
        alias="TAMIYO_FIELD_REPORT_RETENTION_HOURS", default=24
    )
    tamiyo_field_report_max_retries: int = Field(
        alias="TAMIYO_FIELD_REPORT_MAX_RETRIES", default=3
    )
    # Optional blend-mode annotations for Kasmina (P8)
    tamiyo_enable_blend_mode_ann: bool = Field(
        alias="TAMIYO_ENABLE_BLEND_MODE_ANN", default=False
    )
    tamiyo_blend_mode_default: str = Field(
        alias="TAMIYO_BLEND_MODE_DEFAULT", default="CONVEX"
    )
    tamiyo_blend_conf_gate_k: float = Field(
        alias="TAMIYO_BLEND_CONF_GATE_K", default=1.0
    )
    tamiyo_blend_conf_gate_tau: float = Field(
        alias="TAMIYO_BLEND_CONF_GATE_TAU", default=1.0
    )
    tamiyo_blend_alpha_lo: float = Field(
        alias="TAMIYO_BLEND_ALPHA_LO", default=0.0
    )
    tamiyo_blend_alpha_hi: float = Field(
        alias="TAMIYO_BLEND_ALPHA_HI", default=1.0
    )
    tamiyo_blend_alpha_vec_max: int = Field(
        alias="TAMIYO_BLEND_ALPHA_VEC_MAX", default=64
    )
    # P9 — Field Report Lifecycle (observation windows + retry backoff)
    tamiyo_fr_obs_window_epochs: int = Field(
        alias="TAMIYO_FR_OBS_WINDOW_EPOCHS", default=3
    )
    tamiyo_fr_retry_backoff_ms: int = Field(
        alias="TAMIYO_FR_RETRY_BACKOFF_MS", default=1000
    )
    tamiyo_fr_retry_backoff_mult: float = Field(
        alias="TAMIYO_FR_RETRY_BACKOFF_MULT", default=2.0
    )
    # Policy update verification knobs
    tamiyo_verify_updates: bool = Field(
        alias="TAMIYO_VERIFY_UPDATES", default=True
    )
    tamiyo_update_freshness_sec: int = Field(
        alias="TAMIYO_UPDATE_FRESHNESS_SEC", default=0
    )
    # Step-level evaluation timeout for Tamiyo (ms). Used as the default when
    # TamiyoService is constructed without an explicit override.
    tamiyo_step_timeout_ms: float = Field(
        alias="TAMIYO_STEP_TIMEOUT_MS", default=5.0
    )
    # Optional device preference for Tamiyo policy: "cpu", "cuda", "cuda:0", etc.
    # When unset, TamiyoService will auto-detect CUDA and prefer it if available.
    tamiyo_device: str | None = Field(alias="TAMIYO_DEVICE", default=None)
    # Tamiyo compile toggle (optional override)
    # None → use TamiyoPolicy defaults (CUDA + device="cuda" → enabled)
    tamiyo_enable_compile: bool | None = Field(
        alias="TAMIYO_ENABLE_COMPILE", default=None
    )

    urza_database_url: str = Field(alias="URZA_DATABASE_URL", default="sqlite:///./var/urza/catalog.db")
    urza_artifact_dir: str = Field(alias="URZA_ARTIFACT_DIR", default="./var/urza/artifacts")
    urza_cache_ttl_seconds: int | None = Field(alias="URZA_CACHE_TTL_SECONDS", default=None)

    log_level: str = Field(alias="ESP_LOG_LEVEL", default="INFO")

    # ----------------------
    # Tolaria (Training) CFG
    # ----------------------
    # LR + Optimizer governance
    tolaria_lr_policy: str | None = Field(alias="TOLARIA_LR_POLICY", default=None)
    tolaria_lr_warmup_steps: int = Field(alias="TOLARIA_LR_WARMUP_STEPS", default=0)
    tolaria_opt_rebuild_enabled: bool = Field(
        alias="TOLARIA_OPT_REBUILD_ENABLED", default=False
    )
    tolaria_opt_rebuild_fence: str = Field(
        alias="TOLARIA_OPT_REBUILD_FENCE", default="epoch"
    )  # epoch|n_steps
    tolaria_opt_rebuild_backoff_ms: int = Field(
        alias="TOLARIA_OPT_REBUILD_BACKOFF_MS", default=10_000
    )

    # Rollback (two-tier)
    tolaria_rollback_enabled: bool = Field(
        alias="TOLARIA_ROLLBACK_ENABLED", default=False
    )
    tolaria_rollback_fast_cap_mb: int = Field(
        alias="TOLARIA_ROLLBACK_FAST_CAP_MB", default=32
    )
    tolaria_rollback_deadline_ms: int = Field(
        alias="TOLARIA_ROLLBACK_DEADLINE_MS", default=250
    )
    tolaria_rollback_snapshot_steps: int = Field(
        alias="TOLARIA_ROLLBACK_SNAPSHOT_STEPS", default=1
    )

    # Emergency protocol
    tolaria_emergency_enabled: bool = Field(
        alias="TOLARIA_EMERGENCY_ENABLED", default=False
    )
    tolaria_emergency_bypass_max_per_min: int = Field(
        alias="TOLARIA_EMERGENCY_BYPASS_MAX_PER_MIN", default=60
    )
    tolaria_emergency_l4_on_rollback_deadline: bool = Field(
        alias="TOLARIA_EMERGENCY_L4_ON_ROLLBACK_DEADLINE", default=True
    )
    tolaria_emergency_l4_failed_epochs_threshold: int = Field(
        alias="TOLARIA_EMERGENCY_L4_FAILED_EPOCHS", default=3
    )
    tolaria_rollback_signal_name: str | None = Field(
        alias="TOLARIA_ROLLBACK_SIGNAL_NAME", default=None
    )

    # Multi-seed aggregation
    tolaria_aggregation_scheme: str = Field(
        alias="TOLARIA_AGGREGATION_SCHEME", default="mean"
    )  # mean|sum|state_weighted
    tolaria_pcgrad_enabled: bool = Field(
        alias="TOLARIA_PCGRAD_ENABLED", default=False
    )
    tolaria_aggregation_mode: str = Field(
        alias="TOLARIA_AGGREGATION_MODE", default="seed"
    )  # seed|microbatch
    tolaria_aggregation_attribution: str = Field(
        alias="TOLARIA_AGGREGATION_ATTRIBUTION", default="approx"
    )  # approx|probe|dataloader
    tolaria_aggregation_conflict_warn: float = Field(
        alias="TOLARIA_AGGREGATION_CONFLICT_WARN", default=0.75
    )
    tolaria_agg_per_layer_enabled: bool = Field(
        alias="TOLARIA_AGG_PER_LAYER_ENABLED", default=False
    )
    tolaria_agg_per_layer_topk: int = Field(
        alias="TOLARIA_AGG_PER_LAYER_TOPK", default=5
    )
    # Per-seed telemetry thresholds
    tolaria_seed_share_jump_warn: float = Field(
        alias="TOLARIA_SEED_SHARE_JUMP_WARN", default=0.3
    )
    tolaria_seed_conflict_ratio_warn: float = Field(
        alias="TOLARIA_SEED_CONFLICT_RATIO_WARN", default=0.5
    )
    # Compact per-seed telemetry (single event per seed)
    tolaria_seed_health_compact: bool = Field(
        alias="TOLARIA_SEED_HEALTH_COMPACT", default=False
    )

    # Optimizer rebuild storm guard
    tolaria_opt_rebuild_min_interval_steps: int = Field(
        alias="TOLARIA_OPT_REBUILD_MIN_INTERVAL_STEPS", default=0
    )

    # Profiler
    tolaria_profiler_enabled: bool = Field(
        alias="TOLARIA_PROFILER_ENABLED", default=False
    )
    tolaria_profiler_dir: str = Field(
        alias="TOLARIA_PROFILER_DIR", default="./var/profiler"
    )
    tolaria_profiler_active_steps: int = Field(
        alias="TOLARIA_PROFILER_ACTIVE_STEPS", default=50
    )

    # ----------------------
    # Urabrask (Producer) CFG
    # ----------------------
    urabrask_enabled: bool = Field(alias="URABRASK_ENABLED", default=False)
    urabrask_producer_interval_s: int = Field(alias="URABRASK_PRODUCER_INTERVAL_S", default=300)
    urabrask_topn_per_cycle: int = Field(alias="URABRASK_TOPN_PER_CYCLE", default=5)
    urabrask_only_safe_tier: bool = Field(alias="URABRASK_ONLY_SAFE_TIER", default=True)
    urabrask_oona_publish_enabled: bool = Field(alias="URABRASK_OONA_PUBLISH_ENABLED", default=False)
    urabrask_timeout_ms: int = Field(alias="URABRASK_TIMEOUT_MS", default=200)

    # -----------------------------
    # Urabrask (Benchmarks) CFG v1
    # -----------------------------
    urabrask_bench_enabled: bool = Field(alias="URABRASK_BENCH_ENABLED", default=False)
    urabrask_bench_interval_s: int = Field(alias="URABRASK_BENCH_INTERVAL_S", default=1800)
    urabrask_bench_topn: int = Field(alias="URABRASK_BENCH_TOPN", default=3)
    urabrask_bench_timeout_ms: int = Field(alias="URABRASK_BENCH_TIMEOUT_MS", default=500)
    urabrask_bench_min_interval_s: int = Field(alias="URABRASK_BENCH_MIN_INTERVAL_S", default=3600)

    # -----------------------------
    # Urabrask (Signing + WAL)
    # -----------------------------
    urabrask_signing_enabled: bool = Field(alias="URABRASK_SIGNING_ENABLED", default=False)
    urabrask_wal_path: str = Field(alias="URABRASK_WAL_PATH", default="./var/urza/urabrask_wal.jsonl")

    # -----------------------------
    # Urabrask (Crucible artifacts)
    # -----------------------------
    urabrask_crucible_artifacts_dir: str = Field(
        alias="URABRASK_CRUCIBLE_ARTIFACTS_DIR", default="./var/urabrask/crucible"
    )
    urabrask_crucible_artifacts_keep: int = Field(
        alias="URABRASK_CRUCIBLE_ARTIFACTS_KEEP", default=5
    )

    # -----------------------------
    # Urabrask (Crucible v1 probes)
    # -----------------------------
    urabrask_crucible_allow_oom: bool = Field(
        alias="URABRASK_CRUCIBLE_ALLOW_OOM", default=False
    )
    urabrask_crucible_memory_watermark_mb: float = Field(
        alias="URABRASK_CRUCIBLE_MEMORY_WATERMARK_MB", default=64.0
    )
    # Test/dev helper to avoid real OOM allocations
    urabrask_crucible_simulate_oom: bool = Field(
        alias="URABRASK_CRUCIBLE_SIMULATE_OOM", default=False
    )
    urabrask_crucible_gpu_timing_enabled: bool = Field(
        alias="URABRASK_CRUCIBLE_GPU_TIMING_ENABLED", default=False
    )

__all__ = ["EsperSettings"]
