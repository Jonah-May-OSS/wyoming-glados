"""Process and process-manager helpers for the GLaDOS TTS runner."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from piper_runtime import PiperTTSRunner

    Runner = PiperTTSRunner

_LOGGER = logging.getLogger(__name__)

# Audio format produced by the VITS model (22050 Hz, 16-bit mono).

SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2
CHANNELS = 1

# Sentinel marking the end of a PCM stream.

_stream_end = object()


class GladosProcess:
    """Info for a running GLaDOS process (one TTS instance)."""

    def __init__(self, voice_name: str, runner: Runner) -> None:
        """Wrap a loaded runner, stamping it as used now.

        `last_used` starts at construction rather than zero so a process is
        never eligible for idle eviction before it has served anything.
        """
        self.voice_name = voice_name
        self.runner = runner
        self.last_used = time.monotonic_ns()

    def is_multispeaker(self) -> bool:
        """Return whether this process supports multiple speakers."""
        return False  # Assuming GLaDOS doesn't support multiple speakers in this case

    async def run_tts(
        self, text: str, alpha: float = 1.0
    ) -> AsyncGenerator[tuple[bytes | None, int, int, int], None]:
        """Process the text, yielding PCM chunks as the model produces them.

        Inference runs in a worker thread so the event loop stays responsive;
        a one-chunk queue bounds audio buffered ahead of a slow client.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        cancelled = threading.Event()

        def offer(item: Any) -> bool:
            """Send one item with backpressure, stopping promptly on cancel."""
            put = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while not cancelled.is_set():
                try:
                    put.result(timeout=0.1)
                    return True
                except TimeoutError:
                    continue
            put.cancel()
            return False

        def produce() -> None:
            stream = self.runner.run_tts_stream(text, alpha)
            try:
                for pcm in stream:
                    if not offer(pcm):
                        return
            finally:
                close = getattr(stream, "close", None)
                try:
                    if close is not None:
                        close()
                finally:
                    if not cancelled.is_set():
                        offer(_stream_end)

        future = loop.run_in_executor(None, produce)
        try:
            while True:
                item = await queue.get()
                if item is _stream_end:
                    break
                # The voice config's rate, not the module constant: a
                # 16 kHz voice announced as 22050 plays ~1.38x too fast.
                rate = getattr(self.runner, "sample_rate", SAMPLE_RATE)
                yield (item, rate, SAMPLE_WIDTH, CHANNELS)
            # Surface any inference error raised in the worker thread.
            await future
        except Exception as e:
            _LOGGER.error(
                "TTS processing failed for text: %s... Error: %s", text[:50], e
            )
            raise
        finally:
            # Cancelling the async consumer cannot interrupt an ONNX call
            # already running in the executor. Stop it from starting another
            # sentence, unblock a pending queue put, and wait for that call to
            # finish before releasing this stream.
            cancelled.set()
            if not future.done():
                with contextlib.suppress(Exception):
                    await asyncio.shield(future)


class GladosProcessManager:
    """Manages the GLaDOS TTS process and its runner."""

    def __init__(self, runner: Runner) -> None:
        """Initialize the TTS process manager with an existing runner."""
        self.runner = runner  # Use the passed-in runner, don't initialize a new one
        self.processes: dict[str, GladosProcess] = {}
        self.processes_lock = asyncio.Lock()  # Lock for thread safety
        _LOGGER.debug("Glados TTS process manager initialized.")

    async def get_process(self, voice_name: str | None = None) -> GladosProcess:
        """Get the TTS process for the given voice or initialize a new one."""
        if voice_name is None:
            voice_name = "default"  # Assuming default voice if none provided
        async with self.processes_lock:  # Lock access to the process dictionary
            if voice_name not in self.processes:
                # Initialize a new process if it doesn't exist

                _LOGGER.debug("Initializing new process for voice: %s", voice_name)
                self.processes[voice_name] = GladosProcess(voice_name, self.runner)
            # Update last used timestamp

            self.processes[voice_name].last_used = time.monotonic_ns()
        return self.processes[voice_name]
