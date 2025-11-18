import sys
import pytest
import argparse
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.handler import GladosEventHandler
from server.process import GladosProcessManager, GladosProcess
from server.sentence_boundary import SentenceBoundaryDetector
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import (
    Synthesize,
    SynthesizeStart,
    SynthesizeChunk,
    SynthesizeStop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyAsyncWriter:
    """Valid async writer to inject into the handler."""
    def __init__(self):
        self.events = []

    async def write_event(self, event):
        self.events.append(event)


def async_gen(*items):
    """Turn any list of tuples into an async generator."""
    async def generator():
        for i in items:
            yield i
    return generator()


def make_handler(streaming=False):
    """Construct a handler with a mocked process manager."""
    dummy_info = Info()
    cli_args = argparse.Namespace(streaming=streaming)

    mock_runner = MagicMock()
    mock_process = MagicMock(spec=GladosProcess)
    mock_process.run_tts = AsyncMock(return_value=async_gen())  # default: no output

    mgr = MagicMock(spec=GladosProcessManager)
    mgr.get_process = AsyncMock(return_value=mock_process)

    writer = DummyAsyncWriter()

    handler = GladosEventHandler(
        wyoming_info=dummy_info,
        cli_args=cli_args,
        process_manager=mgr,
        writer=writer,
    )
    return handler, mgr, writer


# ---------------------------------------------------------------------------
# Describe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_describe():
    handler, mgr, writer = make_handler()

    await handler.handle_event(Describe().event())

    assert len(writer.events) == 1
    assert writer.events[0]["type"] == "info"


# ---------------------------------------------------------------------------
# Non-streaming Synthesize
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesize_basic_path():
    handler, mgr, writer = make_handler(streaming=False)

    mgr.get_process.return_value.run_tts = AsyncMock(
        return_value=async_gen((b"aaa", 22050, 2, 1))
    )

    await handler.handle_event(Synthesize(text="Hello world.").event())

    types = [e["type"] for e in writer.events]
    assert "audio-start" in types
    assert "audio-chunk" in types
    assert "audio-stop" in types


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesize_exception_path():
    handler, mgr, writer = make_handler(streaming=False)

    async def bad_gen(*_):
        raise RuntimeError("boom")
        yield  # avoids StopIteration signature errors

    mgr.get_process.return_value.run_tts = bad_gen

    await handler.handle_event(Synthesize(text="oops").event())

    types = [e["type"] for e in writer.events]
    assert "error" in types


# ---------------------------------------------------------------------------
# Streaming mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_start_resets_state():
    handler, mgr, writer = make_handler(streaming=True)

    await handler.handle_event(SynthesizeStart(voice="v1").event())

    assert handler.is_streaming is True
    assert isinstance(handler.sbd, SentenceBoundaryDetector)


@pytest.mark.asyncio
async def test_stream_chunk_sentence_emission():
    handler, mgr, writer = make_handler(streaming=True)

    mgr.get_process.return_value.run_tts = AsyncMock(
        return_value=async_gen((b"abc", 22050, 2, 1))
    )

    await handler.handle_event(SynthesizeStart(voice="v").event())
    await handler.handle_event(SynthesizeChunk(text="Hello. ").event())

    types = [e["type"] for e in writer.events]
    assert "audio-start" in types
    assert "audio-chunk" in types


@pytest.mark.asyncio
async def test_stream_stop_flushes_remaining_text():
    handler, mgr, writer = make_handler(streaming=True)

    mgr.get_process.return_value.run_tts = AsyncMock(
        return_value=async_gen((b"xyz", 22050, 2, 1))
    )

    await handler.handle_event(SynthesizeStart(voice="v").event())
    await handler.handle_event(SynthesizeChunk(text="Hi").event())
    await handler.handle_event(SynthesizeStop().event())

    types = [e["type"] for e in writer.events]
    assert "audio-stop" in types


# ---------------------------------------------------------------------------
# Asterisk removal test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_asterisk_removal_in_synthesize():
    handler, mgr, writer = make_handler(streaming=False)

    mgr.get_process.return_value.run_tts = AsyncMock(
        return_value=async_gen((b"aaa", 22050, 2, 1))
    )

    await handler.handle_event(Synthesize(text="This is *important*.").event())

    # verify any event text logged does not contain the asterisk
    text_dump = " ".join(str(e) for e in writer.events)
    assert "important" in text_dump
