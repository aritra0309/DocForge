"""Unit tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from docforge.core.config import (
    DocForgeConfig,
    load_config,
)


class TestDefaults:
    def test_loads_with_defaults(self) -> None:
        config = load_config()
        assert config.general.log_level == "INFO"
        assert config.general.parallelism == 8
        assert config.crawler.max_pages_per_version == 5000
        assert config.crawler.rate_limit_rps == 5
        assert config.chunker.target_chunk_size == 512
        assert config.chunker.max_chunk_size == 1024
        assert config.embeddings.provider == "sentence-transformers"
        assert config.embeddings.model == "BAAI/bge-base-en-v1.5"
        assert config.storage.backend == "chromadb"

    def test_data_dir_created(self) -> None:
        config = load_config()
        assert config.general.data_dir.exists()


class TestEnvOverride:
    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCFORGE_GENERAL__LOG_LEVEL", "DEBUG")
        config = load_config()
        assert config.general.log_level == "DEBUG"

    def test_env_var_overrides_general(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCFORGE_GENERAL__PARALLELISM", "16")
        config = load_config()
        assert config.general.parallelism == 16

    def test_env_var_overrides_crawler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCFORGE_CRAWLER__RATE_LIMIT_RPS", "10")
        config = load_config()
        assert config.crawler.rate_limit_rps == 10

    def test_env_var_overrides_chunker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCFORGE_CHUNKER__STRATEGY", "heading")
        config = load_config()
        assert config.chunker.strategy == "heading"

    def test_env_var_overrides_embeddings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCFORGE_EMBEDDINGS__PROVIDER", "openai")
        monkeypatch.setenv("DOCFORGE_EMBEDDINGS__MODEL", "text-embedding-3-small")
        config = load_config()
        assert config.embeddings.provider == "openai"
        assert config.embeddings.model == "text-embedding-3-small"

    def test_env_var_overrides_storage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCFORGE_STORAGE__BACKEND", "qdrant")
        config = load_config()
        assert config.storage.backend == "qdrant"


class TestOverrideDict:
    def test_overrides_dict_wins_over_defaults(self) -> None:
        config = load_config(overrides={"general": {"log_level": "ERROR"}})
        assert config.general.log_level == "ERROR"

    def test_overrides_dict_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCFORGE_GENERAL__LOG_LEVEL", "WARNING")
        config = load_config(overrides={"general": {"log_level": "ERROR"}})
        assert config.general.log_level == "ERROR"

    def test_deep_nested_override(self) -> None:
        config = load_config(
            overrides={
                "crawler": {
                    "max_pages_per_version": 1000,
                    "rate_limit_rps": 2,
                }
            }
        )
        assert config.crawler.max_pages_per_version == 1000
        assert config.crawler.rate_limit_rps == 2
        # other crawler fields should remain at defaults
        assert config.crawler.timeout_seconds == 30


class TestProjectConfig:
    def test_project_toml_overrides_defaults(self, tmp_path: Path) -> None:
        proj_toml = tmp_path / "docforge.toml"
        proj_toml.write_text('[general]\nlog_level = "WARNING"\n')
        config = load_config(project_config_path=proj_toml)
        assert config.general.log_level == "WARNING"
        assert config.general.parallelism == 8  # remains default


class TestUserConfig:
    def test_user_config_overrides_defaults(self, tmp_path: Path) -> None:
        user_dir = tmp_path / ".config" / "docforge"
        user_dir.mkdir(parents=True)
        user_cfg = user_dir / "config.toml"
        user_cfg.write_text("[general]\nparallelism = 4\n")
        config = load_config(user_config_path=user_cfg)
        assert config.general.parallelism == 4


class TestPrecedence:
    def test_env_beats_user_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        user_dir = tmp_path / ".config" / "docforge"
        user_dir.mkdir(parents=True)
        user_cfg = user_dir / "config.toml"
        user_cfg.write_text("[crawler]\nrate_limit_rps = 2\n")
        monkeypatch.setenv("DOCFORGE_CRAWLER__RATE_LIMIT_RPS", "10")
        config = load_config(user_config_path=user_cfg)
        assert config.crawler.rate_limit_rps == 10  # env wins over user file

    def test_override_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCFORGE_GENERAL__LOG_LEVEL", "WARNING")
        config = load_config(overrides={"general": {"log_level": "CRITICAL"}})
        assert config.general.log_level == "CRITICAL"  # explicit wins over env

    def test_project_beats_user(self, tmp_path: Path) -> None:
        user_dir = tmp_path / ".config" / "docforge"
        user_dir.mkdir(parents=True)
        user_cfg = user_dir / "config.toml"
        user_cfg.write_text("[general]\nparallelism = 4\n")
        proj_toml = tmp_path / "docforge.toml"
        proj_toml.write_text("[general]\nparallelism = 16\n")
        config = load_config(user_config_path=user_cfg, project_config_path=proj_toml)
        assert config.general.parallelism == 16  # project beats user


class TestValidation:
    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(Exception):
            load_config(overrides={"general": {"log_level": "INVALID"}})

    def test_invalid_max_chunk_size_raises(self) -> None:
        with pytest.raises(Exception):
            load_config(overrides={"chunker": {"max_chunk_size": 50}})

    def test_parallelism_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            load_config(overrides={"general": {"parallelism": 0}})

    def test_batch_size_cap(self) -> None:
        with pytest.raises(Exception):
            load_config(overrides={"embeddings": {"batch_size": 9999}})

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(Exception):
            load_config(overrides={"chunker": {"strategy": "nonexistent"}})

    def test_retry_backoff_invalid_value(self) -> None:
        with pytest.raises(Exception):
            load_config(overrides={"crawler": {"retry_backoff": "polynomial"}})


class TestPathExpansion:
    def test_data_dir_expands_tilde(self) -> None:
        config = DocForgeConfig()
        assert "~" not in str(config.general.data_dir)
        assert config.general.data_dir.is_absolute()

    def test_storage_path_expands_tilde(self) -> None:
        config = DocForgeConfig()
        assert "~" not in str(config.storage.path)
        assert config.storage.path.is_absolute()
