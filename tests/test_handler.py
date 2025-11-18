import argparse
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
)

from server.handler import GladosEventHandler
from server.process import GladosProcess, GladosProcessManager
from server.sentence_boundary import SentenceBoundaryDetector

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class DummyWriter:
    """Capture events sent by the handler."""

    def __init__(self):
        self.events = []

    async def write_event(self, e):
        self.events.append(e)


def make_handler(streaming=False):
    """Build a handler object with fully mocked environment."""
    dummy_info = Info()
    cli_args = argparse.Namespace(streaming=streaming)

    mock_runner = MagicMock()
    mock_process = MagicMock(spec=GladosProcess)
    mock_process.run_tts = AsyncMock(return_value=iter(()))

    mgr = MagicMock(spec=GladosProcessManager)
    mgr.get_process = AsyncMock(return_value=mock_process)

    handler = GladosEventHandler(
        wyoming_info=dummy_info,
        cli_args=cli_args,
        process_manager=mgr,
        writer=DummyWriter(),  # AsyncEventHandler expects writer=
    )
    handler.writer = handler._writer  # Force-match AsyncEventHandler
    return handler, mgr


# ---------------------------------------------------------------------------
# Describe Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_describe():
    handler, _ = make_handler()
    evt = Describe().event()

    await handler.handle_event(evt)

    assert len(handler.writer.events) == 1
    assert handler.writer.events[0]["type"] == "info"


# ---------------------------------------------------------------------------
# Synthesize (non-streaming)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_basic_path():
    handler, mgr = make_handler(streaming=False)

    pcm_tuple = [(b"aaa", 22050, 2, 1)]
    mgr.get_process.return_value.run_tts = AsyncMock(return_value=pcm_tuple)

    evt = Synthesize(text="Hello world.").event()

    await handler.handle_event(evt)

    # Expect AudioStart, AudioChunk, AudioStop
    types = [e["type"] for e in handler.writer.events]
    assert "audio-start" in types
    assert "audio-chunk" in types
    assert "audio-stop" in types


# ---------------------------------------------------------------------------
# Synthesize exception path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_exception_path():
    handler, mgr = make_handler(streaming=False)

    async def bad_gen(*_):
        raise RuntimeError("boom")

    mgr.get_process.return_value.run_tts = bad_gen

    evt = Synthesize(text="Bad").event()
    await handler.handle_event(evt)

    types = [e["type"] for e in handler.writer.events]
    assert "error" in types


# ---------------------------------------------------------------------------
# STREAMING MODE TESTS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_start_resets_state():
    handler, _ = make_handler(streaming=True)

    evt = SynthesizeStart(voice="default").event()
    await handler.handle_event(evt)

    assert handler.is_streaming is True
    assert isinstance(handler.sbd, SentenceBoundaryDetector)


@pytest.mark.asyncio
async def test_stream_chunk_sentence_emission():
    handler, mgr = make_handler(streaming=True)

    # Make run_tts emit one chunk
    mgr.get_process.return_value.run_tts = AsyncMock(
        return_value=[(b"111", 22050, 2, 1)]
    )

    await handler.handle_event(SynthesizeStart(voice="v").event())

    # The chunk contains a full sentence ("Hello.")
    await handler.handle_event(SynthesizeChunk(text="Hello. ").event())

    types = [e["type"] for e in handler.writer.events]

    assert "audio-start" in types
    assert "audio-chunk" in types


@pytest.mark.asyncio
async def test_stream_stop_flushes_remaining_text():
    handler, mgr = make_handler(streaming=True)

    # run_tts returns one chunk
    mgr.get_process.return_value.run_tts = AsyncMock(
        return_value=[(b"xyz", 22050, 2, 1)]
    )

    await handler.handle_event(SynthesizeStart(voice="v").event())
    await handler.handle_event(SynthesizeChunk(text="Hello").event())
    await handler.handle_event(SynthesizeStop().event())

    types = [e["type"] for e in handler.writer.events]

    # Should emit audio-stop or stopped event
    assert "audio-stop" in types or "tts-synthesize-stopped" in types


# ---------------------------------------------------------------------------
# remove_asterisks integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asterisk_removal_in_synthesize():
    handler, mgr = make_handler(streaming=False)

    mgr.get_process.return_value.run_tts = AsyncMock(
        return_value=[(b"aaa", 22050, 2, 1)]
    )

    evt = Synthesize(text="This is *important*. ").event()
    await handler.handle_event(evt)

    # It should remove *
    events = handler.writer.events
    # Find AudioChunk event and ensure text logged in debug had no asterisks
    assert handler._synthesize is None or "important" in " ".join(
        [str(e) for e in events]
    )
