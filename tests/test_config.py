import pytest
import os
import yaml
from src.utils.config import Config, load_config


class TestConfig:
    """Tests for the Config class."""

    def test_init_empty(self):
        """Test creating an empty config."""
        config = Config()
        assert config._data == {}

    def test_init_with_data(self):
        """Test creating a config with data."""
        data = {"key": "value", "nested": {"inner": 123}}
        config = Config(data)
        assert config._data == data

    def test_get_nested_value(self):
        """Test getting a nested value using dot notation."""
        config = Config({"training": {"lr": 0.001, "epochs": 50}})
        assert config.get("training.lr") == 0.001
        assert config.get("training.epochs") == 50

    def test_get_default_value(self):
        """Test getting a non-existent key returns default."""
        config = Config({"key": "value"})
        assert config.get("nonexistent") is None
        assert config.get("nonexistent", "default") == "default"

    def test_contains(self):
        """Test checking if key exists."""
        config = Config({"key": "value", "nested": {"inner": 1}})
        assert "key" in config
        assert "nested.inner" in config
        assert "nonexistent" not in config

    def test_merge(self):
        """Test merging configs."""
        base = Config({"a": 1, "b": {"c": 2}})
        overrides = {"b": {"d": 3}, "e": 4}
        merged = base.merge(overrides)

        assert merged._data["a"] == 1
        assert merged._data["b"]["c"] == 2
        assert merged._data["b"]["d"] == 3
        assert merged._data["e"] == 4

    def test_to_dict(self):
        """Test converting config to dictionary."""
        data = {"key": "value"}
        config = Config(data)
        assert config.to_dict() == data
        assert config.to_dict() is not config._data

    def test_repr(self):
        """Test string representation."""
        config = Config({"key": "value"})
        assert "Config" in repr(config)


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_defaults(self):
        """Test loading default config."""
        config = load_config()
        assert config.get("training.epochs") == 50
        assert config.get("training.batch_size") == 8

    def test_load_from_file(self, temp_dir):
        """Test loading config from YAML file."""
        yaml_content = {"training": {"epochs": 100, "batch_size": 16}}
        yaml_path = os.path.join(temp_dir, "test_config.yaml")

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_config(yaml_path)
        assert config.get("training.epochs") == 100
        assert config.get("training.batch_size") == 16

    def test_load_with_overrides(self):
        """Test loading config with overrides."""
        overrides = {"training": {"epochs": 200}}
        config = load_config(overrides=overrides)
        assert config.get("training.epochs") == 200

    def test_load_nonexistent_file_returns_defaults(self):
        """Test loading a nonexistent file returns defaults."""
        config = load_config("/nonexistent/path.yaml")
        assert config.get("training.epochs") == 50
