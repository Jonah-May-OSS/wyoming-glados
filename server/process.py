"""Process and process-manager helpers for the GLaDOS TTS runner."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from gladostts.glados import TTSRunner  # Import your TTSRunner from glados.py

_LOGGER = logging.getLogger(__name__)

# Audio format produced by the GLaDOS vocoder (22050 Hz, 16-bit mono).

SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2
CHANNELS = 1

# Sentinel marking the end of a PCM stream.

_stream_end = object()


class GladosProcess:
    """Info for a running GLaDOS process (one TTS instance)."""

    def __init__(self, voice_name: str, runner: TTSRunner) -> None:
        self.voice_name = voice_name
        self.runner = runner
        self.last_used = time.monotonic_ns()

    def is_multispeaker(self) -> bool:
        """Return whether this process supports multiple speakers."""
        return False  # Assuming GLaDOS doesn't support multiple speakers in this case

    async def run_tts(
        self, text: str, alpha: float = 1.0, lang: str | None = None
    ) -> AsyncGenerator[tuple[bytes | None, int, int, int], None]:
        """Process the text, yielding PCM chunks as the model produces them.

        Inference runs in a worker thread so the event loop stays responsive;
        each PCM chunk is forwarded here as soon as the vocoder emits it, so
        playback can start before the utterance is fully synthesized.

        ``lang`` selects the phonemizer language (None = English default).
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()

        def produce() -> None:
            stream = self.runner.run_tts_stream(text, alpha, lang=lang)
            try:
                for pcm in stream:
                    loop.call_soon_threadsafe(queue.put_nowait, pcm)
            finally:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()

        try:
            future = loop.run_in_executor(None, produce)
            future.add_done_callback(lambda _: queue.put_nowait(_stream_end))
            while True:
                item = await queue.get()
                if item is _stream_end:
                    break
                yield (item, SAMPLE_RATE, SAMPLE_WIDTH, CHANNELS)
            # Surface any inference error raised in the worker thread.
            await future
        except Exception as e:
            _LOGGER.error(
                "TTS processing failed for text: %s... Error: %s", text[:50], e
            )
            raise


class GladosProcessManager:
    """Manages GLaDOS TTS process, initializes and interacts with TTSRunner."""

    def __init__(self, runner: TTSRunner) -> None:
        """Initialize the TTS process manager with an existing TTSRunner."""
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
