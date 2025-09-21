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

    urza_database_url: str = Field(alias="URZA_DATABASE_URL", default="sqlite:///./var/urza/catalog.db")
    urza_artifact_dir: str = Field(alias="URZA_ARTIFACT_DIR", default="./var/urza/artifacts")

    log_level: str = Field(alias="ESP_LOG_LEVEL", default="INFO")

__all__ = ["EsperSettings"]
