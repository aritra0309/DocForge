"""Configuration system — loads from multiple sources with Pydantic validation.

Priority order (highest wins):
1. Explicit Python API arguments
2. Environment variables (DOCFORGE_*)
3. Project-level docforge.toml
4. User-level ~/.config/docforge/config.toml
5. Built-in defaults
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    Field,
    PlainValidator,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

_ERR_MAX_CHUNK_SIZE = "max_chunk_size must be >= target_chunk_size"

# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------


def _expand_path(v: str) -> Path:
    return Path(v).expanduser().resolve()


ExpandPath = Annotated[Path, PlainValidator(lambda v: _expand_path(str(v)))]


class CrawlerConfig(BaseModel):
    max_pages_per_version: int = Field(default=5000, ge=1)
    rate_limit_rps: int = Field(default=5, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)
    retry_attempts: int = Field(default=3, ge=0)
    retry_backoff: Literal["exponential", "linear", "fixed"] = "exponential"
    respect_robots_txt: bool = True
    enable_js_rendering: bool = False
    cache_ttl_hours: int = Field(default=168, ge=0)


class ChunkerConfig(BaseModel):
    target_chunk_size: int = Field(default=512, ge=64)
    max_chunk_size: int = Field(default=1024, ge=128)
    overlap_tokens: int = Field(default=64, ge=0)
    strategy: Literal["auto", "heading", "api_ref", "tutorial", "code", "table"] = "auto"

    @field_validator("max_chunk_size")
    @classmethod
    def max_gte_target(cls, v: int, info: Any) -> int:
        if "target_chunk_size" in info.data and v < info.data["target_chunk_size"]:
            raise ValueError(_ERR_MAX_CHUNK_SIZE)
        return v


class EmbeddingsConfig(BaseModel):
    provider: str = "sentence-transformers"
    model: str = "BAAI/bge-base-en-v1.5"
    batch_size: int = Field(default=64, ge=1, le=1024)
    cache_embeddings: bool = True


class StorageConfig(BaseModel):
    backend: str = "chromadb"
    path: ExpandPath = Field(default_factory=lambda: _expand_path("~/.docforge/vectordb"))


class GeneralConfig(BaseModel):
    data_dir: ExpandPath = Field(default_factory=lambda: _expand_path("~/.docforge"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    parallelism: int = Field(default=8, ge=1, le=128)


# ---------------------------------------------------------------------------
# Full config model — used as the canonical runtime config
# ---------------------------------------------------------------------------


class DocForgeConfig(BaseModel):
    """Complete DocForge configuration with validated defaults."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @model_validator(mode="after")
    def ensure_data_dir_exists(self) -> DocForgeConfig:
        self.general.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage.path.parent.mkdir(parents=True, exist_ok=True)
        return self


# ---------------------------------------------------------------------------
# Settings loader (pydantic-settings) with environment variable support
# ---------------------------------------------------------------------------


class _DocForgeSettings(BaseSettings):
    """Low-level settings loader that reads env vars + TOML files.

    This class exists solely to leverage pydantic-settings' env var parsing.
    Normal code should use ``load_config()`` instead.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCFORGE_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    general__data_dir: str | None = None
    general__log_level: str | None = None
    general__parallelism: int | None = None

    crawler__max_pages_per_version: int | None = None
    crawler__rate_limit_rps: int | None = None
    crawler__timeout_seconds: int | None = None
    crawler__retry_attempts: int | None = None
    crawler__retry_backoff: str | None = None
    crawler__respect_robots_txt: bool | None = None
    crawler__enable_js_rendering: bool | None = None
    crawler__cache_ttl_hours: int | None = None

    chunker__target_chunk_size: int | None = None
    chunker__max_chunk_size: int | None = None
    chunker__overlap_tokens: int | None = None
    chunker__strategy: str | None = None

    embeddings__provider: str | None = None
    embeddings__model: str | None = None
    embeddings__batch_size: int | None = None
    embeddings__cache_embeddings: bool | None = None

    storage__backend: str | None = None
    storage__path: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_TOML = """\
[general]
data_dir = "~/.docforge"
log_level = "INFO"
parallelism = 8

[crawler]
max_pages_per_version = 5000
rate_limit_rps = 5
timeout_seconds = 30
retry_attempts = 3
retry_backoff = "exponential"
respect_robots_txt = true
enable_js_rendering = false
cache_ttl_hours = 168

[chunker]
target_chunk_size = 512
max_chunk_size = 1024
overlap_tokens = 64
strategy = "auto"

[embeddings]
provider = "sentence-transformers"
model = "BAAI/bge-base-en-v1.5"
batch_size = 64
cache_embeddings = true

[storage]
backend = "chromadb"
path = "~/.docforge/vectordb"
"""


def _find_project_config() -> Path | None:
    """Walk up from CWD looking for ``docforge.toml``."""
    cwd = Path.cwd()
    for parent in [cwd, *list(cwd.parents)]:
        candidate = parent / "docforge.toml"
        if candidate.is_file():
            return candidate
    return None


def _load_toml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` into ``base`` (mutates ``base``)."""
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    *,
    overrides: dict[str, Any] | None = None,
    user_config_path: Path | None = None,
    project_config_path: Path | None = None,
) -> DocForgeConfig:
    """Load and merge configuration from all sources.

    Priority (highest wins):
    1. ``overrides`` dict (explicit Python API arguments)
    2. ``DOCFORGE_*`` environment variables
    3. Project-level ``docforge.toml`` (walked up from CWD)
    4. User-level ``~/.config/docforge/config.toml``
    5. Built-in defaults

    Args:
        overrides: Optional dict of config overrides (same shape as the TOML).
        user_config_path: Explicit path to user config. Defaults to
            ``~/.config/docforge/config.toml``.
        project_config_path: Explicit path to project config. If not set,
            walks up from CWD looking for ``docforge.toml``.

    Returns:
        A validated ``DocForgeConfig`` instance.

    Raises:
        pydantic.ValidationError: If any config values are invalid.
    """
    # Layer 5: built-in defaults
    raw: dict[str, Any] = tomllib.loads(_DEFAULT_TOML)

    # Layer 4: user-level config
    user_path = user_config_path or Path.home() / ".config" / "docforge" / "config.toml"
    user_cfg = _load_toml(user_path)
    _merge_dict(raw, user_cfg)

    # Layer 3: project-level config
    proj_path = project_config_path or _find_project_config()
    proj_cfg = _load_toml(proj_path)
    _merge_dict(raw, proj_cfg)

    # Layer 2: environment variables
    env_settings = _DocForgeSettings()
    env_raw: dict[str, Any] = {}
    _apply_env(env_raw, env_settings)
    _merge_dict(raw, env_raw)

    # Layer 1: explicit overrides
    if overrides:
        _merge_dict(raw, overrides)

    return DocForgeConfig.model_validate(raw)


def _apply_env(target: dict[str, Any], env: _DocForgeSettings) -> None:
    """Map flat env var fields into nested dict structure."""
    _set_if(target, ("general", "data_dir"), env.general__data_dir)
    _set_if(target, ("general", "log_level"), env.general__log_level)
    _set_if(target, ("general", "parallelism"), env.general__parallelism)
    _set_if(target, ("crawler", "max_pages_per_version"), env.crawler__max_pages_per_version)
    _set_if(target, ("crawler", "rate_limit_rps"), env.crawler__rate_limit_rps)
    _set_if(target, ("crawler", "timeout_seconds"), env.crawler__timeout_seconds)
    _set_if(target, ("crawler", "retry_attempts"), env.crawler__retry_attempts)
    _set_if(target, ("crawler", "retry_backoff"), env.crawler__retry_backoff)
    _set_if(target, ("crawler", "respect_robots_txt"), env.crawler__respect_robots_txt)
    _set_if(target, ("crawler", "enable_js_rendering"), env.crawler__enable_js_rendering)
    _set_if(target, ("crawler", "cache_ttl_hours"), env.crawler__cache_ttl_hours)
    _set_if(target, ("chunker", "target_chunk_size"), env.chunker__target_chunk_size)
    _set_if(target, ("chunker", "max_chunk_size"), env.chunker__max_chunk_size)
    _set_if(target, ("chunker", "overlap_tokens"), env.chunker__overlap_tokens)
    _set_if(target, ("chunker", "strategy"), env.chunker__strategy)
    _set_if(target, ("embeddings", "provider"), env.embeddings__provider)
    _set_if(target, ("embeddings", "model"), env.embeddings__model)
    _set_if(target, ("embeddings", "batch_size"), env.embeddings__batch_size)
    _set_if(target, ("embeddings", "cache_embeddings"), env.embeddings__cache_embeddings)
    _set_if(target, ("storage", "backend"), env.storage__backend)
    _set_if(target, ("storage", "path"), env.storage__path)


def _set_if(target: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    if value is not None:
        d = target
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value


__all__ = [
    "ChunkerConfig",
    "CrawlerConfig",
    "DocForgeConfig",
    "EmbeddingsConfig",
    "GeneralConfig",
    "StorageConfig",
    "load_config",
]
