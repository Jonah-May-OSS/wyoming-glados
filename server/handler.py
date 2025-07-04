import argparse
import json
import logging
import os
import math
from typing import Any, Dict, Optional

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeStart, SynthesizeStop, SynthesizeStopped

from .process import GladosProcessManager
from .sentence_boundary import SentenceBoundaryDetector, remove_asterisks

_LOGGER = logging.getLogger(__name__)

class GladosEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        process_manager: GladosProcessManager,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.process_manager = process_manager
        self.sbd = SentenceBoundaryDetector()
        self.is_streaming: Optional[bool] = None
        self._synthesize: Optional[Synthesize] = None

        # Add the audio_started flag here
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

                synthesize = Synthesize.from_event(event)
                synthesize.text = remove_asterisks(synthesize.text)
                return await self._handle_synthesize(synthesize)

            if not self.cli_args.streaming:
                return True

            if SynthesizeStart.is_type(event.type):
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                _LOGGER.debug("Text stream started: voice=%s", stream_start.voice)
                return True

            if SynthesizeStop.is_type(event.type):
                assert self._synthesize is not None
                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    await self._handle_synthesize(self._synthesize)

                await self.write_event(SynthesizeStopped().event())
                _LOGGER.debug("Text stream stopped")
                return True

            if not Synthesize.is_type(event.type):
                return True

            synthesize = Synthesize.from_event(event)
            return await self._handle_synthesize(synthesize)
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_synthesize(self, synthesize: Synthesize) -> bool:
        _LOGGER.debug(synthesize)

        glados_proc = await self.process_manager.get_process()

        try:
            # Start processing with run_tts
            async for pcm, rate, width, channels in glados_proc.run_tts(synthesize.text):
                # Check if PCM data is coming through
                _LOGGER.debug(f"PCM data size: {len(pcm)} bytes")
                
                # Send AudioStart event if it's not already sent
                if not self.audio_started:
                    await self.write_event(
                        AudioStart(rate=rate, width=width, channels=channels).event()
                    )
                    self.audio_started = True

                # Send audio chunk
                await self.write_event(
                    AudioChunk(audio=pcm, rate=rate, width=width, channels=channels).event()
                )

            _LOGGER.debug(f"Sent AudioChunk event for chunk: {synthesize.text}")

        except Exception as e:
            _LOGGER.error(f"Error during TTS processing: {e}")
            await self.write_event(
                Error(text=str(e), code="TTSProcessingError").event()
            )

        # Stop the audio stream when done
        await self.write_event(AudioStop().event())
        _LOGGER.debug("Completed request")

        return True
