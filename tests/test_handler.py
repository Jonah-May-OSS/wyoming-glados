"""Tests for the GLaDOS event handler module.

These are unit tests that test the handler logic with mocked dependencies.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from wyoming.event import Event
from wyoming.info import Describe, Info, TtsProgram, TtsVoice
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.handler import GladosEventHandler  # noqa: E402
from server.process import GladosProcess, GladosProcessManager  # noqa: E402

# ---------------------------------------------------------------------------
# Helper Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cli_args():
    """Create mock CLI arguments."""
    args = argparse.Namespace()
    args.streaming = True
    return args


@pytest.fixture
def mock_wyoming_info():
    """Create mock Wyoming Info object."""
    from wyoming.info import Attribution

    return Info(
        tts=[
            TtsProgram(
                name="GLaDOS",
                attribution=Attribution(
                    name="GLaDOS",
                    url="https://github.com/R2D2FISH/glados-tts",
                ),
                installed=True,
                description="GLaDOS TTS",
                version="1.0",
                voices=[
                    TtsVoice(
                        name="default",
                        attribution=Attribution(
                            name="GLaDOS",
                            url="https://github.com/R2D2FISH/glados-tts",
                        ),
                        installed=True,
                        description="Default voice",
                        version="1.0",
                        languages=["en"],
                    )
                ],
            )
        ]
    )


@pytest.fixture
def mock_process_manager():
    """Create a mock process manager."""
    manager = MagicMock(spec=GladosProcessManager)

    # Create a mock process that returns async generator
    async def mock_run_tts(text, alpha=1.0):
        yield b"audio_data", 22050, 2, 1

    mock_process = MagicMock(spec=GladosProcess)
    mock_process.run_tts = mock_run_tts

    # Make get_process return the mock process
    manager.get_process = AsyncMock(return_value=mock_process)

    return manager


@pytest.fixture
def handler(mock_cli_args, mock_wyoming_info, mock_process_manager):
    """Create a GladosEventHandler instance for testing."""
    # Create mock reader and writer for AsyncEventHandler
    mock_reader = MagicMock(spec=asyncio.StreamReader)
    mock_writer = MagicMock(spec=asyncio.StreamWriter)

    handler = GladosEventHandler(
        wyoming_info=mock_wyoming_info,
        cli_args=mock_cli_args,
        process_manager=mock_process_manager,
        reader=mock_reader,
        writer=mock_writer,
    )
    # Mock write_event method
    handler.write_event = AsyncMock()
    return handler


# ---------------------------------------------------------------------------
# GladosEventHandler Initialization Tests
# ---------------------------------------------------------------------------


class TestGladosEventHandlerInit:
    """Test the GladosEventHandler initialization."""

    def test_handler_initialization(
        self, mock_cli_args, mock_wyoming_info, mock_process_manager
    ):
        """Test basic handler initialization."""
        mock_reader = MagicMock(spec=asyncio.StreamReader)
        mock_writer = MagicMock(spec=asyncio.StreamWriter)

        handler = GladosEventHandler(
            wyoming_info=mock_wyoming_info,
            cli_args=mock_cli_args,
            process_manager=mock_process_manager,
            reader=mock_reader,
            writer=mock_writer,
        )

        assert handler.cli_args == mock_cli_args
        assert handler.process_manager == mock_process_manager
        assert handler.is_streaming is None
        assert handler._synthesize is None
        assert handler.audio_started is False
        assert handler.sbd is not None


# ---------------------------------------------------------------------------
# Event Handling Tests
# ---------------------------------------------------------------------------


class TestDescribeEvent:
    """Test Describe event handling."""

    @pytest.mark.asyncio
    async def test_describe_event_sends_info(self, handler, mock_wyoming_info):
        """Test that Describe event triggers info response."""
        event = Describe().event()

        result = await handler.handle_event(event)

        assert result is True
        handler.write_event.assert_called_once()
        # Verify the info event was sent
        sent_event = handler.write_event.call_args[0][0]
        assert sent_event.type == "info"


class TestSynthesizeEvent:
    """Test Synthesize event handling (non-streaming mode)."""

    @pytest.mark.asyncio
    async def test_synthesize_event_basic(self, handler):
        """Test basic Synthesize event handling."""
        synthesize = Synthesize(text="Hello world", voice=None)
        event = synthesize.event()

        result = await handler.handle_event(event)

        assert result is True
        # Should call write_event multiple times: AudioStart, AudioChunk, AudioStop
        assert handler.write_event.call_count >= 3

    @pytest.mark.asyncio
    async def test_synthesize_removes_asterisks(self, handler):
        """Test that asterisks are removed from text."""
        synthesize = Synthesize(text="*Hello* world*", voice=None)
        event = synthesize.event()

        result = await handler.handle_event(event)

        assert result is True
        # Verify process_manager.get_process was called
        handler.process_manager.get_process.assert_called()

    @pytest.mark.asyncio
    async def test_synthesize_during_streaming_ignored(self, handler):
        """Test that Synthesize events during streaming are ignored."""
        handler.is_streaming = True

        synthesize = Synthesize(text="Hello", voice=None)
        event = synthesize.event()

        result = await handler.handle_event(event)

        assert result is True
        # Should not call process_manager when streaming
        handler.process_manager.get_process.assert_not_called()


class TestStreamingEvents:
    """Test streaming mode event handling."""

    @pytest.mark.asyncio
    async def test_synthesize_start_event(self, handler):
        """Test SynthesizeStart event."""
        stream_start = SynthesizeStart(voice=None)
        event = stream_start.event()

        result = await handler.handle_event(event)

        assert result is True
        assert handler.is_streaming is True
        assert handler._synthesize is not None
        assert handler._synthesize.text == ""

    @pytest.mark.asyncio
    async def test_synthesize_chunk_event(self, handler):
        """Test SynthesizeChunk event with complete sentence."""
        # First start streaming
        stream_start = SynthesizeStart(voice=None)
        await handler.handle_event(stream_start.event())

        # Reset write_event mock to clear previous calls
        handler.write_event.reset_mock()

        # Send chunk with complete sentence (needs lookahead to next sentence)
        stream_chunk = SynthesizeChunk(text="Hello world. Another")
        event = stream_chunk.event()

        result = await handler.handle_event(event)

        assert result is True
        # Should have sent audio events for the complete sentence
        assert handler.write_event.call_count >= 3  # AudioStart, AudioChunk, AudioStop

    @pytest.mark.asyncio
    async def test_synthesize_chunk_incomplete_sentence(self, handler):
        """Test SynthesizeChunk with incomplete sentence."""
        # First start streaming
        stream_start = SynthesizeStart(voice=None)
        await handler.handle_event(stream_start.event())

        # Reset write_event mock
        handler.write_event.reset_mock()

        # Send chunk without sentence boundary
        stream_chunk = SynthesizeChunk(text="Hello")
        event = stream_chunk.event()

        result = await handler.handle_event(event)

        assert result is True
        # Should not have sent audio (no complete sentence)
        handler.write_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_stop_event_with_text(self, handler):
        """Test SynthesizeStop event with remaining text."""
        # First start streaming
        stream_start = SynthesizeStart(voice=None)
        await handler.handle_event(stream_start.event())

        # Add incomplete text
        stream_chunk = SynthesizeChunk(text="Hello")
        await handler.handle_event(stream_chunk.event())

        # Reset write_event mock
        handler.write_event.reset_mock()

        # Stop streaming
        stream_stop = SynthesizeStop()
        event = stream_stop.event()

        result = await handler.handle_event(event)

        assert result is True
        # Should have sent audio events + SynthesizeStopped
        assert handler.write_event.call_count >= 4

    @pytest.mark.asyncio
    async def test_synthesize_stop_event_without_text(self, handler):
        """Test SynthesizeStop event with no remaining text."""
        # First start streaming
        stream_start = SynthesizeStart(voice=None)
        await handler.handle_event(stream_start.event())

        # Reset write_event mock
        handler.write_event.reset_mock()

        # Stop streaming immediately
        stream_stop = SynthesizeStop()
        event = stream_stop.event()

        result = await handler.handle_event(event)

        assert result is True
        # Should only send SynthesizeStopped
        handler.write_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_disabled_in_args(self, handler, mock_cli_args):
        """Test that streaming events are ignored when streaming is disabled."""
        handler.cli_args.streaming = False

        stream_start = SynthesizeStart(voice=None)
        event = stream_start.event()

        result = await handler.handle_event(event)

        assert result is True
        assert handler.is_streaming is None  # Should not have changed


class TestHandleSynthesize:
    """Test the _handle_synthesize method."""

    @pytest.mark.asyncio
    async def test_handle_synthesize_sends_audio_events(self, handler):
        """Test that _handle_synthesize sends proper audio events."""
        synthesize = Synthesize(text="Test", voice=None)

        result = await handler._handle_synthesize(synthesize)

        assert result is True
        assert handler.write_event.call_count >= 3

        # Check that AudioStart, AudioChunk, and AudioStop were sent
        calls = handler.write_event.call_args_list
        event_types = [call[0][0].type for call in calls]

        assert "audio-start" in event_types
        assert "audio-chunk" in event_types
        assert "audio-stop" in event_types

    @pytest.mark.asyncio
    async def test_handle_synthesize_audio_start_sent_once(self, handler):
        """Test that AudioStart is only sent once per handler lifetime."""
        synthesize = Synthesize(text="Test", voice=None)

        # First call
        await handler._handle_synthesize(synthesize)
        first_count = handler.write_event.call_count

        handler.write_event.reset_mock()

        # Second call
        await handler._handle_synthesize(synthesize)
        second_count = handler.write_event.call_count

        # Second call should have one fewer event (no AudioStart)
        assert second_count == first_count - 1

    @pytest.mark.asyncio
    async def test_handle_synthesize_exception_handling(self, handler):
        """Test exception handling in _handle_synthesize."""
        # Make get_process raise an exception
        handler.process_manager.get_process.side_effect = RuntimeError("Test error")

        synthesize = Synthesize(text="Test", voice=None)

        with pytest.raises(RuntimeError):
            await handler._handle_synthesize(synthesize)

    @pytest.mark.asyncio
    async def test_handle_synthesize_tts_error_sends_error_event(self, handler):
        """Test that TTS errors are sent as Error events."""

        # Create a process that raises an exception
        async def mock_run_tts_error(text, alpha=1.0):
            raise RuntimeError("TTS failed")
            yield  # Make it a generator

        mock_process = MagicMock(spec=GladosProcess)
        mock_process.run_tts = mock_run_tts_error
        handler.process_manager.get_process = AsyncMock(return_value=mock_process)

        synthesize = Synthesize(text="Test", voice=None)

        result = await handler._handle_synthesize(synthesize)

        assert result is True
        # Should have sent Error event and AudioStop
        assert handler.write_event.call_count >= 2

        # Find the error event
        calls = handler.write_event.call_args_list
        error_found = False
        for call in calls:
            event = call[0][0]
            if event.type == "error":
                error_found = True
                assert "TTS" in event.data["code"]

        assert error_found


class TestErrorHandling:
    """Test error handling in handle_event."""

    @pytest.mark.asyncio
    async def test_exception_in_event_handling_sends_error(self, handler):
        """Test that exceptions are caught and Error events are sent."""
        # Create an event that will trigger an error
        synthesize = Synthesize(text="Test", voice=None)
        event = synthesize.event()

        # Make process_manager raise an exception
        handler.process_manager.get_process.side_effect = RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            await handler.handle_event(event)

        # Should have sent an Error event before re-raising
        error_sent = False
        for call in handler.write_event.call_args_list:
            if call[0][0].type == "error":
                error_sent = True
                break

        assert error_sent


class TestUnknownEvents:
    """Test handling of unknown or unsupported events."""

    @pytest.mark.asyncio
    async def test_unknown_event_type_returns_true(self, handler):
        """Test that unknown event types return True without errors."""
        # Create a generic event that's not a recognized type
        event = Event(type="unknown", data={})

        result = await handler.handle_event(event)

        assert result is True
        # Should not have called write_event
        handler.write_event.assert_not_called()
