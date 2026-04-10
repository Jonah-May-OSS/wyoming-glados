"""Pytest configuration and fixtures."""

import importlib.util
import sys
from unittest.mock import MagicMock


def pytest_configure(config):
    """Configure pytest before tests run."""
    _ = config
    # Mock heavy dependencies that are not needed for unit tests

    # If Wyoming is available, don't mock it.
    if "wyoming" not in sys.modules:
        try:
            wyoming_available = importlib.util.find_spec("wyoming") is not None
        except ValueError:
            # Some tests install placeholder modules with __spec__=None.
            wyoming_available = False
    else:
        wyoming_available = True

    if not wyoming_available:
        # Mock wyoming if not available
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
