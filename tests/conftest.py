import pytest
import tempfile
import os


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    import torch.nn as nn

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3)

        def forward(self, x):
            return self.conv(x)

    return MockModel()


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""

    class MockDataLoader:
        def __init__(self):
            self.data = [("img1", "target1"), ("img2", "target2")]

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

    return MockDataLoader()
