"""Tests for the GLaDOS process management module.

These are unit tests that test the logic without requiring heavy dependencies.
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.process import GladosProcess, GladosProcessManager  # noqa: E402

# ---------------------------------------------------------------------------
# GladosProcess Logic Tests
# ---------------------------------------------------------------------------


class TestGladosProcessLogic:
    """Test the GladosProcess logic with mocked dependencies."""

    def test_process_initialization(self):
        """Test basic process initialization logic."""
        voice_name = "default"
        mock_runner = MagicMock()

        process = GladosProcess(voice_name, mock_runner)

        assert process.voice_name == "default"
        assert process.runner == mock_runner
        assert isinstance(process.last_used, int)

    @pytest.mark.asyncio
    async def test_tts_processing_flow(self):
        """Test normal TTS flow."""
        mock_runner = MagicMock()

        mock_audio = MagicMock()
        mock_audio.raw_data = b"audio_data"
        mock_audio.frame_rate = 22050
        mock_audio.sample_width = 2
        mock_audio.channels = 1

        mock_runner.run_tts.return_value = mock_audio

        p = GladosProcess("default", mock_runner)

        results = []
        async for chunk in p.run_tts("Test text"):
            results.append(chunk)

        assert len(results) == 1
        raw, rate, width, ch = results[0]
        assert raw == b"audio_data"
        assert rate == 22050
        mock_runner.run_tts.assert_called_once_with("Test text", 1.0)

    @pytest.mark.asyncio
    async def test_async_generator_pattern(self):
        """Test async generator pattern used in run_tts."""

        async def mock_gen():
            yield b"audio1", 22050, 2, 1
            yield b"audio2", 22050, 2, 1

        results = []
        async for data, rate, width, channels in mock_gen():
            results.append((data, rate, width, channels))

        assert len(results) == 2
        assert results[0][0] == b"audio1"
        assert results[1][0] == b"audio2"

    @pytest.mark.asyncio
    async def test_run_tts_exception_path(self):
        """Ensure exception inside run_tts is yielded and re-raised."""
        mock_runner = MagicMock()
        mock_runner.run_tts.side_effect = RuntimeError("boom")

        p = GladosProcess("voice", mock_runner)

        with pytest.raises(RuntimeError):
            async for _ in p.run_tts("text"):
                pass


# ---------------------------------------------------------------------------
# GladosProcessManager Logic Tests
# ---------------------------------------------------------------------------


class TestGladosProcessManagerLogic:
    """Test the GladosProcessManager logic."""

    @pytest.mark.asyncio
    async def test_process_cache_behavior(self):
        """Test caching behavior."""
        mock_runner = MagicMock()
        mgr = GladosProcessManager(mock_runner)

        p1 = await mgr.get_process()
        p2 = await mgr.get_process()

        assert p1 is p2
        assert "default" in mgr.processes

    @pytest.mark.asyncio
    async def test_multiple_voices(self):
        mock_runner = MagicMock()
        mgr = GladosProcessManager(mock_runner)

        p1 = await mgr.get_process("one")
        p2 = await mgr.get_process("two")
        p3 = await mgr.get_process("one")

        assert p1 is p3
        assert p1 is not p2
        assert len(mgr.processes) == 2

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self):
        mock_runner = MagicMock()
        mgr = GladosProcessManager(mock_runner)

        # Hammer the manager with concurrent calls
        procs = await asyncio.gather(*[mgr.get_process("voice") for _ in range(20)])

        # All processes should be identical (lock prevents duplicates)
        assert len({id(p) for p in procs}) == 1
        assert len(mgr.processes) == 1

    @pytest.mark.asyncio
    async def test_timestamp_update(self):
        mock_runner = MagicMock()
        mgr = GladosProcessManager(mock_runner)

        p = await mgr.get_process("voice")
        first = p.last_used

        await asyncio.sleep(0.001)  # ensure clock tick
        p = await mgr.get_process("voice")
        second = p.last_used

        assert second > first

    @pytest.mark.asyncio
    async def test_get_process_updates_timestamp_on_creation(self):
        """Covers missing coverage: timestamp updated on first creation."""
        mock_runner = MagicMock()
        mgr = GladosProcessManager(mock_runner)

        before = time.monotonic_ns()
        p = await mgr.get_process("newvoice")

        assert p.last_used >= before
        assert p.voice_name == "newvoice"

    def test_is_multispeaker_default(self):
        p = GladosProcess("v", MagicMock())
        assert p.is_multispeaker() is False
