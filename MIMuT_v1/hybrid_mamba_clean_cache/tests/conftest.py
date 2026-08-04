"""Shared fixtures for muscriptor tests."""

import os

import pytest
from pathlib import Path

WEIGHTS_PATH = (
    Path(__file__).parent.parent / "muscriptor_weights_01684fbb_350.safetensors"
)
SONG_PATH = Path(os.environ.get("MUSCRIPTOR_TEST_AUDIO", "/path/to/test_audio.wav"))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests that require model weights")


@pytest.fixture(scope="session")
def transcription_model():
    """Load the TranscriptionModel once for the whole test session."""
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"Weights not found at {WEIGHTS_PATH}")
    from muscriptor.transcription_model import TranscriptionModel

    return TranscriptionModel.load_model(weights_path=WEIGHTS_PATH, device="cpu")
