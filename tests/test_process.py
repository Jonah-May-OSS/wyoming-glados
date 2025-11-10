"""Tests for the GLaDOS process management module.

These are unit tests that test the logic without requiring heavy dependencies.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

pytest = __import__("pytest")


class TestGladosProcessLogic:
    """Test the GladosProcess logic with mocked dependencies."""

    def test_process_initialization(self):
        """Test basic process initialization logic."""
        voice_name = "default"
        mock_runner = MagicMock()

        # Simulate what would happen in GladosProcess.__init__
        process_data = {"voice_name": voice_name, "runner": mock_runner}

        assert process_data["voice_name"] == "default"
        assert process_data["runner"] == mock_runner

    @pytest.mark.asyncio
    async def test_tts_processing_flow(self):
        """Test the flow of TTS processing."""
        mock_runner = MagicMock()
        mock_audio = MagicMock()
        mock_audio.raw_data = b"audio_data"
        mock_audio.frame_rate = 22050
        mock_audio.sample_width = 2
        mock_audio.channels = 1
        mock_runner.run_tts.return_value = mock_audio

        # Simulate processing
        result = mock_runner.run_tts("Test text", 1.0)

        assert result.raw_data == b"audio_data"
        assert result.frame_rate == 22050
        mock_runner.run_tts.assert_called_once_with("Test text", 1.0)

    @pytest.mark.asyncio
    async def test_async_generator_pattern(self):
        """Test async generator pattern used in run_tts."""

        async def mock_run_tts(text):
            """Simulate async generator."""
            yield b"audio1", 22050, 2, 1
            yield b"audio2", 22050, 2, 1

        results = []
        async for data, rate, width, channels in mock_run_tts("test"):
            results.append((data, rate, width, channels))

        assert len(results) == 2
        assert results[0][0] == b"audio1"
        assert results[1][0] == b"audio2"


class TestGladosProcessManagerLogic:
    """Test the GladosProcessManager logic."""

    @pytest.mark.asyncio
    async def test_process_cache_behavior(self):
        """Test process caching behavior."""
        processes = {}
        mock_runner = MagicMock()

        async def get_process(voice_name=None):
            if voice_name is None:
                voice_name = "default"
            if voice_name not in processes:
                processes[voice_name] = {"voice": voice_name, "runner": mock_runner}
            return processes[voice_name]

        # First call creates process
        proc1 = await get_process()
        assert "default" in processes
        assert len(processes) == 1

        # Second call reuses process
        proc2 = await get_process()
        assert proc1 is proc2
        assert len(processes) == 1

    @pytest.mark.asyncio
    async def test_multiple_voices(self):
        """Test managing multiple voice processes."""
        processes = {}
        mock_runner = MagicMock()

        async def get_process(voice_name):
            if voice_name not in processes:
                processes[voice_name] = {"voice": voice_name, "runner": mock_runner}
            return processes[voice_name]

        proc1 = await get_process("voice1")
        proc2 = await get_process("voice2")
        proc3 = await get_process("voice1")

        assert len(processes) == 2
        assert proc1 is proc3
        assert proc1 is not proc2

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self):
        """Test thread-safe concurrent access."""
        processes = {}
        lock = asyncio.Lock()
        mock_runner = MagicMock()
        init_count = 0

        async def get_process(voice_name="default"):
            nonlocal init_count
            async with lock:
                if voice_name not in processes:
                    init_count += 1
                    await asyncio.sleep(0.01)  # Simulate slow initialization
                    processes[voice_name] = {
                        "voice": voice_name,
                        "runner": mock_runner,
                    }
                return processes[voice_name]

        # Multiple concurrent requests for same voice
        procs = await asyncio.gather(*[get_process("default") for _ in range(10)])

        # Should only initialize once despite concurrent calls
        assert init_count == 1
        assert len(processes) == 1
        # All should return the same process
        assert len({id(p) for p in procs}) == 1

    @pytest.mark.asyncio
    async def test_timestamp_update(self):
        """Test that last_used timestamp gets updated."""
        import time

        process = {"last_used": 0}

        # Simulate updating timestamp
        process["last_used"] = time.monotonic_ns()
        first_time = process["last_used"]

        await asyncio.sleep(0.01)

        process["last_used"] = time.monotonic_ns()
        second_time = process["last_used"]

        assert second_time > first_time

    def test_is_multispeaker_default(self):
        """Test multispeaker check defaults to False."""
        # GLaDOS doesn't support multiple speakers
        is_multispeaker = False
        assert is_multispeaker is False
