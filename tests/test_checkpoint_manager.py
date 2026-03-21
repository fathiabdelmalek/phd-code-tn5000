import pytest
import torch
import os
from src.core.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """Tests for the CheckpointManager class."""

    def test_init(self, temp_dir):
        """Test checkpoint manager initialization."""
        cm = CheckpointManager(temp_dir)

        assert str(cm.exp_dir) == str(temp_dir)
        assert cm.best_path.name == "best.pt"
        assert cm.last_path.name == "last.pt"
        assert os.path.exists(cm.weights_dir)

    def test_save_checkpoint(self, temp_dir, mock_model):
        """Test saving a checkpoint."""
        cm = CheckpointManager(temp_dir)
        cm.save(mock_model, epoch=0, metrics={"map": 0.5})

        assert os.path.exists(cm.last_path)

    def test_save_best_checkpoint(self, temp_dir, mock_model):
        """Test saving a best checkpoint."""
        cm = CheckpointManager(temp_dir)
        cm.save(mock_model, is_best=True, metrics={"map": 0.9})

        assert os.path.exists(cm.best_path)

    def test_save_with_optimizer(self, temp_dir, mock_model):
        """Test saving checkpoint with optimizer."""
        cm = CheckpointManager(temp_dir)
        optimizer = torch.optim.Adam(mock_model.parameters())

        cm.save(mock_model, optimizer=optimizer, epoch=5)

        assert os.path.exists(cm.last_path)

    def test_load_checkpoint(self, temp_dir, mock_model):
        """Test loading a checkpoint."""
        cm = CheckpointManager(temp_dir)
        optimizer = torch.optim.Adam(mock_model.parameters())

        cm.save(mock_model, optimizer=optimizer, epoch=10, metrics={"map": 0.8})

        new_model = type(mock_model)()
        loaded_info = cm.load(new_model, filepath=cm.last_path)

        assert loaded_info["epoch"] == 10
        assert loaded_info["metrics"]["map"] == 0.8

    def test_load_nonexistent(self, temp_dir, mock_model):
        """Test loading a nonexistent checkpoint raises error."""
        cm = CheckpointManager(temp_dir)

        with pytest.raises(FileNotFoundError):
            cm.load(mock_model, filepath="nonexistent.pt")

    def test_get_best_weights(self, temp_dir, mock_model):
        """Test getting best weights path."""
        cm = CheckpointManager(temp_dir)

        assert cm.get_best_weights() is None

        cm.save(mock_model, is_best=True)
        assert cm.get_best_weights() is not None
        assert cm.get_best_weights().endswith("best.pt")

    def test_get_last_weights(self, temp_dir, mock_model):
        """Test getting last weights path."""
        cm = CheckpointManager(temp_dir)

        assert cm.get_last_weights() is None

        cm.save(mock_model)
        assert cm.get_last_weights() is not None
        assert cm.get_last_weights().endswith("last.pt")

    def test_save_metrics(self, temp_dir):
        """Test saving metrics to JSON."""
        cm = CheckpointManager(temp_dir)
        metrics = {"map50": 0.85, "precision": 0.9}

        cm.save_metrics(metrics)

        save_path = os.path.join(temp_dir, "results", "metrics.json")
        assert os.path.exists(save_path)
