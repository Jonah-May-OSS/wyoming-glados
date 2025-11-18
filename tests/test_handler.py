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
# Proper async writer
# ---------------------------------------------------------------------------

class DummyAsyncWriter:
    """A real async writer as required by AsyncEventHandler."""
    def __init__(self):
        self.events = []

    async def write_event(self, event):
        self.events.append(event)


# Correct async generator
def make_async_gen(*items):
    async def gen():
        for item in items:
            yield item
        await asyncio.sleep(0)  # prevents StopIteration blowing up
    return gen()


# ---------------------------------------------------------------------------
# Factory to build a *correct* GladosEventHandler
# ---------------------------------------------------------------------------

def build_handler(streaming=False):
    dummy_info = Info()
    cli_args = argparse.Namespace(streaming=streaming)
    writer = DummyAsyncWriter()

    # Mock the GladosProcess
    mock_runner = MagicMock()
    mock_process = MagicMock(spec=GladosProcess)
    mock_process.run_tts = AsyncMock(return_value=make_async_gen())

    # Mock the manager
    mgr = MagicMock(spec=GladosProcessManager)
    mgr.get_process = AsyncMock(return_value=mock_process)

    # KEY FIX: AsyncEventHandler expects keyword argument 'writer'
    handler = GladosEventHandler(
        wyoming_info=dummy_info,
        cli_args=cli_args,
        process_manager=mgr,
        writer=writer,     # important!
        reader=None,
    )

    return handler, mgr, writer


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_describe():
    handler, mgr, writer = build_handler()

    await handler.handle_event(Describe().event())

    assert len(writer.events) == 1
    assert writer.events[0]["type"] == "info"


@pytest.mark.asyncio
async def test_synthesize_basic_path():
    handler, mgr, writer = build_handler(streaming=False)

    mgr.get_process.return_value.run_tts = AsyncMock(
        return_value=make_async_gen((b"aaa", 22050, 2, 1))
    )

    await handler.handle_event(Synthesize(text="Hello world.").event())

    types = [e["type"] for e in writer.events]
    assert "audio-start" in types
    assert "audio-chunk" in types
    assert "audio-stop" in types


@pytest.mark.asyncio
async def test_synthesize_exception_path():
    handler, mgr, writer = build_handler(streaming=False)

    async def err_gen(*_):
        raise RuntimeError("boom")
        yield None  # prevent StopIteration

    mgr.get_process.return_value.run_tts = err_gen

    await handler.handle_event(Synthesize(text="oops").event())

    types = [e["type"] for e in writer.events]
    assert "error" in types


@pytest.mark.asyncio
async def test_stream_start_resets_state():
    handler, mgr, writer = build_handler(streaming=True)

    await handler.handle_event(SynthesizeStart(voice="v1").event())

    assert handler.is_streaming is True
    assert isinstance(handler.sbd, SentenceBoundaryDetector)


@pytest.mark.asyncio
async def test_stream_chunk_sentence_emission():
    handler, mgr, writer = build_handler(streaming=True)

    mgr.get_process.return_value.run_tts = AsyncMock(
        return_value=make_async_gen((b"abc", 22050, 2, 1))
    )

    await handler.handle_event(SynthesizeStart(voice="v").event())
    await handler.handle_event(SynthesizeChunk(text="Hello. ").event())

    types = [e["type"] for e in writer.events]
    assert "audio-start" in types
    assert "audio-chunk" in types


@pytest.mark.asyncio
async def test_stream_stop_flushes_remaining_text():
    handler, mgr, writer = build_handler(streaming=True)

    mgr.get_process.return_value.run_tts = AsyncMock(
        return_value=make_async_gen((b"xyz", 22050, 2, 1))
    )

    await handler.handle_event(SynthesizeStart(voice="v").event())
    await handler.handle_event(SynthesizeChunk(text="Hi").event())
    await handler.handle_event(SynthesizeStop().event())

    assert "audio-stop" in [e["type"] for e in writer.events]


@pytest.mark.asyncio
async def test_asterisk_removal_in_synthesize():
    handler, mgr, writer = build_handler(streaming=False)

    mgr.get_process.return_value.run_tts = AsyncMock(
        return_value=make_async_gen((b"aaa", 22050, 2, 1))
    )

    await handler.handle_event(Synthesize(text="This is *important*.").event())

    dumped = " ".join(str(e) for e in writer.events)
    assert "important" in dumped
