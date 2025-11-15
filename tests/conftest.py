"""Pytest configuration and fixtures."""

import sys
from unittest.mock import MagicMock


def pytest_configure(config):
    """Configure pytest before tests run."""
    # Mock heavy dependencies that are not needed for unit tests

    # Mock wyoming if not available
    if "wyoming" not in sys.modules:
        wyoming_mock = MagicMock()
        sys.modules["wyoming"] = wyoming_mock
        sys.modules["wyoming.audio"] = MagicMock()
        sys.modules["wyoming.error"] = MagicMock()
        sys.modules["wyoming.event"] = MagicMock()
        sys.modules["wyoming.info"] = MagicMock()
        sys.modules["wyoming.tts"] = MagicMock()
        sys.modules["wyoming.server"] = MagicMock()

    # Mock gladostts - this requires CUDA and heavy ML libraries
    if "gladostts" not in sys.modules:
        sys.modules["gladostts"] = MagicMock()
        sys.modules["gladostts.glados"] = MagicMock()
