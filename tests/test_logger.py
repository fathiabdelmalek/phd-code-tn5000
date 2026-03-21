import pytest
import os
from src.utils.logger import Logger, get_logger


class TestLogger:
    """Tests for the Logger class."""

    def test_init(self, temp_dir):
        """Test logger initialization."""
        logger = Logger(temp_dir, "test")
        assert str(logger.exp_dir) == str(temp_dir)
        assert logger.name == "test"
        assert logger.console is True
        assert logger.file is True

    def test_info_logging(self, temp_dir):
        """Test info logging."""
        logger = Logger(temp_dir, "test")
        logger.info("Test message")

        assert os.path.exists(logger.log_file)
        with open(logger.log_file, "r") as f:
            content = f.read()
            assert "INFO: Test message" in content

    def test_warning_logging(self, temp_dir):
        """Test warning logging."""
        logger = Logger(temp_dir, "test")
        logger.warning("Warning message")

        with open(logger.log_file, "r") as f:
            content = f.read()
            assert "WARNING: Warning message" in content

    def test_error_logging(self, temp_dir):
        """Test error logging."""
        logger = Logger(temp_dir, "test")
        logger.error("Error message")

        with open(logger.log_file, "r") as f:
            content = f.read()
            assert "ERROR: Error message" in content

    def test_console_only(self, temp_dir, capsys):
        """Test logging to console only."""
        logger = Logger(temp_dir, "test", file=False)
        logger.info("Console only message")

        captured = capsys.readouterr()
        assert "Console only message" in captured.out
        assert not os.path.exists(logger.log_file)

    def test_log_metrics(self, temp_dir):
        """Test logging metrics."""
        logger = Logger(temp_dir, "test")
        metrics = {"loss": 0.5, "accuracy": 0.95}
        logger.log_metrics(metrics, step=1)

        assert len(logger.get_history()) == 1
        assert logger.get_history()[0]["loss"] == 0.5
        assert logger.get_history()[0]["step"] == 1

    def test_log_scalar(self, temp_dir):
        """Test logging a single scalar."""
        logger = Logger(temp_dir, "test")
        logger.log_scalar("lr", 0.001)

        history = logger.get_history()
        assert len(history) == 1
        assert "lr" in history[0]

    def test_save_metrics(self, temp_dir):
        """Test saving metrics to file."""
        logger = Logger(temp_dir, "test")
        logger.log_metrics({"loss": 0.5})
        logger.save_metrics()

        save_path = os.path.join(temp_dir, "logs", "metrics.json")
        assert os.path.exists(save_path)


class TestGetLogger:
    """Tests for the get_logger factory function."""

    def test_get_logger(self, temp_dir):
        """Test creating a logger via factory."""
        logger = get_logger(temp_dir, "factory_test")
        assert isinstance(logger, Logger)
        assert logger.name == "factory_test"
