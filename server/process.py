import logging
import time
import asyncio
from typing import Optional
from gladostts.glados import TTSRunner  # Import your TTSRunner from glados.py

_LOGGER = logging.getLogger(__name__)


class GladosProcess:
    """Info for a running GLaDOS process (one TTS instance)."""

    def __init__(self, voice_name: str, runner: TTSRunner) -> None:
        self.voice_name = voice_name
        self.runner = runner
        self.last_used = time.monotonic_ns()

    def is_multispeaker(self) -> bool:
        """Check if the model has multiple speakers (example method, modify as needed)."""
        return False  # Assuming GLaDOS doesn't support multiple speakers in this case

    async def run_tts(self, text: str, alpha: float = 1.0):
        """Process the text and handle TTS output."""
        try:
            audio_segment = self.runner.run_tts(text, alpha)
            yield audio_segment.raw_data, audio_segment.frame_rate, audio_segment.sample_width, audio_segment.channels
        except Exception as e:
            _LOGGER.error(f"TTS processing failed for text: {text[:50]}... Error: {e}")
            raise


class GladosProcessManager:
    """Manages GLaDOS TTS process, initializes and interacts with TTSRunner."""

    def __init__(self, runner: TTSRunner) -> None:
        """Initialize the TTS process manager with an existing TTSRunner."""
        self.runner = runner  # Use the passed-in runner, don't initialize a new one
        self.processes = {}
        self.processes_lock = asyncio.Lock()  # Lock for thread safety
        _LOGGER.debug("Glados TTS process manager initialized.")

    async def get_process(self, voice_name: Optional[str] = None) -> GladosProcess:
        """Get the TTS process for the given voice or initialize a new one."""
        if voice_name is None:
            voice_name = "default"  # Assuming default voice if none provided
        async with self.processes_lock:  # Lock access to the process dictionary
            if (
                voice_name not in self.processes
                or self.processes[voice_name].runner != self.runner
            ):
                # Initialize a new process if it doesn't exist

                _LOGGER.debug(f"Initializing new process for voice: {voice_name}")
                self.processes[voice_name] = GladosProcess(voice_name, self.runner)
            # Update last used timestamp

            self.processes[voice_name].last_used = time.monotonic_ns()
        return self.processes[voice_name]
