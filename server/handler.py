"""Event handler for clients of the server, now with Wyoming TTS streaming support."""

import argparse
import logging
from typing import Optional

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeStart,
    SynthesizeChunk,
    SynthesizeStop,
    SynthesizeStopped,
)
import nltk
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
from gladostts.glados import TTSRunner

_LOGGER = logging.getLogger(__name__)

# Ensure NLTK 'punkt' data is downloaded
try:
    nltk.data.find("tokenizers/punkt_tab")
    _LOGGER.debug("NLTK 'punkt_tab' tokenizer data is already available.")
except LookupError:
    _LOGGER.info("Downloading NLTK 'punkt_tab' tokenizer data...")
    nltk.download("punkt_tab")


class GladosEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        glados_tts: TTSRunner,
        samples_per_chunk: int,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the GLaDOS event handler."""
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.glados_tts = glados_tts
        self.samples_per_chunk = samples_per_chunk

        # For streaming
        self._stream_buffer: list[str] = []
        self._streaming = False

    def handle_tts_request(self, text: str, delay: float = 250) -> AudioSegment:
        """Generate an AudioSegment for the given text with optional delay between sentences."""
        sentences = sent_tokenize(text)
        if not sentences:
            return AudioSegment.silent(duration=0)
        audio = self.glados_tts.run_tts(sentences[0])
        pause = AudioSegment.silent(duration=delay)
        for sentence in sentences[1:]:
            audio += pause + self.glados_tts.run_tts(sentence)
        return audio

    async def handle_event(self, event: Event) -> bool:
        """Handle incoming events from the client, including TTS streaming."""
        # --- Service description ---
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        # --- TTS streaming events ---
        if SynthesizeStart.is_type(event.type):
            _LOGGER.debug("Received synthesize-start")
            self._stream_buffer.clear()
            self._streaming = True
            return True

        if SynthesizeChunk.is_type(event.type):
            chunk = SynthesizeChunk.from_event(event)
            _LOGGER.debug("Received synthesize-chunk: %r", chunk.text)
            self._stream_buffer.append(chunk.text)
            return True

        if SynthesizeStop.is_type(event.type):
            _LOGGER.debug(
                "Received synthesize-stop, performing TTS on accumulated text"
            )
            full_text = "".join(self._stream_buffer).strip()
            self._streaming = False

            if full_text:
                audio = self.handle_tts_request(full_text)
            else:
                audio = AudioSegment.silent(duration=0)

            # stream audio out
            rate = audio.frame_rate
            width = audio.sample_width
            channels = audio.channels

            await self.write_event(
                AudioStart(rate=rate, width=width, channels=channels).event()
            )

            raw = audio.raw_data
            bps = width * channels
            size = bps * self.samples_per_chunk
            for i in range(0, len(raw), size):
                await self.write_event(
                    AudioChunk(
                        audio=raw[i : i + size],
                        rate=rate,
                        width=width,
                        channels=channels,
                    ).event()
                )

            await self.write_event(AudioStop().event())
            await self.write_event(SynthesizeStopped().event())
            _LOGGER.debug("Completed streaming response")
            return True

        # --- Legacy single‐shot TTS (only if not in streaming flow) ---
        if Synthesize.is_type(event.type):
            if self._streaming:
                _LOGGER.debug("Ignoring legacy synthesize during streaming")
                return True

            synth = Synthesize.from_event(event)
            raw_text = synth.text
            _LOGGER.debug("Received legacy synthesize: %r", raw_text)

            # cleanup & auto-punctuate
            text = " ".join(raw_text.strip().splitlines())
            if self.cli_args.auto_punctuation and text:
                if not any(text.endswith(p) for p in self.cli_args.auto_punctuation):
                    text += self.cli_args.auto_punctuation[0]
            _LOGGER.debug("Synthesize: raw_text=%r, text=%r", raw_text, text)

            if text:
                audio = self.handle_tts_request(text)
            else:
                audio = AudioSegment.silent(duration=0)

            rate = audio.frame_rate
            width = audio.sample_width
            channels = audio.channels

            await self.write_event(
                AudioStart(rate=rate, width=width, channels=channels).event()
            )

            raw = audio.raw_data
            bps = width * channels
            size = bps * self.samples_per_chunk
            for i in range(0, len(raw), size):
                await self.write_event(
                    AudioChunk(
                        audio=raw[i : i + size],
                        rate=rate,
                        width=width,
                        channels=channels,
                    ).event()
                )

            await self.write_event(AudioStop().event())
            _LOGGER.debug("Completed legacy request")
            return True

        # Anything else, ignore
        _LOGGER.warning("Unexpected event: %s", event)
        return True
