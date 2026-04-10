"""Event handler implementation for Wyoming GLaDOS TTS events."""

import argparse
import logging
from typing import Any

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .process import GladosProcessManager
from .sentence_boundary import SentenceBoundaryDetector, remove_asterisks

_LOGGER = logging.getLogger(__name__)


class GladosEventHandler(AsyncEventHandler):
    """Handle Wyoming TTS events for one client connection."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        process_manager: GladosProcessManager,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.process_manager = process_manager
        self.sbd = (
            SentenceBoundaryDetector()
        )  # Initialize the sentence boundary detector
        self.is_streaming: bool | None = None
        self._synthesize: Synthesize | None = None

        self.audio_started = False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True
        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    # Ignore since this is only sent for compatibility reasons.

                    return True
                # Sent outside a stream, so we must process it

                synthesize = Synthesize.from_event(event)
                synthesize.text = remove_asterisks(synthesize.text)
                self.audio_started = False
                return await self._handle_synthesize(synthesize, send_stop=True)
            if not self.cli_args.streaming:
                return True
            if SynthesizeStart.is_type(event.type):
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                self.audio_started = False
                _LOGGER.debug("Text stream started: voice=%s", stream_start.voice)
                return True
            if SynthesizeChunk.is_type(event.type):
                assert self._synthesize is not None
                stream_chunk = SynthesizeChunk.from_event(event)

                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    _LOGGER.debug("Synthesizing stream sentence: %s", sentence)

                    self._synthesize.text = sentence

                    # Handle the synthesis (e.g., convert text to PCM, send AudioChunks)

                    await self._handle_synthesize(self._synthesize, send_stop=False)
                return True
            if SynthesizeStop.is_type(event.type):
                assert self._synthesize is not None
                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    # Final audio chunk(s)

                    await self._handle_synthesize(self._synthesize, send_stop=False)

                if self.audio_started:
                    await self.write_event(AudioStop().event())
                    self.audio_started = False

                self.is_streaming = False
                # End of audio

                await self.write_event(SynthesizeStopped().event())

                _LOGGER.debug("Text stream stopped")
                return True
            # Unknown event type, ignore
            return True
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_synthesize(
        self, synthesize: Synthesize, send_stop: bool = True
    ) -> bool:
        _LOGGER.debug(synthesize)

        glados_proc = await self.process_manager.get_process()

        try:
            # Start processing with run_tts

            async for pcm, rate, width, channels in glados_proc.run_tts(
                synthesize.text
            ):
                if pcm is None:
                    continue

                # Send AudioStart event if it's not already sent

                if not self.audio_started:
                    await self.write_event(
                        AudioStart(rate=rate, width=width, channels=channels).event()
                    )
                    self.audio_started = True
                # Send audio chunk as soon as it's generated

                await self.write_event(
                    AudioChunk(
                        audio=pcm, rate=rate, width=width, channels=channels
                    ).event()
                )
            _LOGGER.debug("Sent AudioChunk event for chunk: %s", synthesize.text)
        except Exception as e:
            _LOGGER.error("Error during TTS processing: %s", e)
            await self.write_event(
                Error(text=str(e), code="TTSProcessingError").event()
            )

        if send_stop:
            await self.write_event(AudioStop().event())
            self.audio_started = False
        # Stop the audio stream when done

        _LOGGER.debug("Completed request")
        return True
