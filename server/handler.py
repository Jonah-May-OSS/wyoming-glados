#!/usr/bin/env python3


"""Event handler for clients of the server, now with true pipelined TTS streaming support."""

import argparse
import logging
from typing import List

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
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.glados_tts = glados_tts
        self.samples_per_chunk = samples_per_chunk
        self._stream_buffer: List[str] = []
        self._streaming = False

    async def handle_event(self, event: Event) -> bool:
        """Dispatch incoming events to their dedicated handlers."""
        etype = event.type
        if Describe.is_type(etype):
            return await self._handle_describe()
        if SynthesizeStart.is_type(etype):
            return await self._handle_synthesize_start()
        if SynthesizeChunk.is_type(etype):
            return await self._handle_synthesize_chunk(event)
        if SynthesizeStop.is_type(etype):
            return await self._handle_synthesize_stop()
        if Synthesize.is_type(etype):
            return await self._handle_legacy_synthesize(event)
        return await self._handle_unexpected_event(event)

    async def _handle_describe(self) -> bool:
        await self.write_event(self.wyoming_info_event)
        _LOGGER.debug("Sent info")
        return True

    async def _handle_synthesize_start(self) -> bool:
        _LOGGER.debug("Received synthesize-start")
        self._stream_buffer.clear()
        self._streaming = True
        return True

    async def _handle_synthesize_chunk(self, event: Event) -> bool:
        chunk = SynthesizeChunk.from_event(event)
        _LOGGER.debug("Received synthesize-chunk: %r", chunk.text)
        self._stream_buffer.append(chunk.text)
        return True

    async def _handle_synthesize_stop(self) -> bool:
        _LOGGER.debug(
            "Received synthesize-stop, performing pipelined TTS on accumulated text"
        )
        self._streaming = False
        full_text = "".join(self._stream_buffer).strip()
        if not full_text:
            await self.write_event(AudioStart(rate=22050, width=2, channels=1).event())
            await self.write_event(AudioStop().event())
            await self.write_event(SynthesizeStopped().event())
            return True
        try:
            for pcm, rate, width, channels in self.glados_tts.stream_tts(
                full_text, alpha=1.0, samples_per_chunk=self.samples_per_chunk
            ):
                await self._emit_audio_event(pcm, rate, width, channels)
        except Exception as e:
            _LOGGER.error("Streaming TTS failed: %s", e)
            await self.write_event(SynthesizeStopped().event())
            return True
        _LOGGER.debug("Completed streaming response")
        return True

    async def _emit_audio_event(
        self, pcm: bytes, rate: int, width: int, channels: int
    ) -> None:
        if pcm == b"__AUDIO_START__":
            await self.write_event(
                AudioStart(rate=rate, width=width, channels=channels).event()
            )
        elif pcm == b"__AUDIO_STOP__":
            await self.write_event(AudioStop().event())
        elif pcm == b"__SYNTH_STOPPED__":
            await self.write_event(SynthesizeStopped().event())
        else:
            await self.write_event(
                AudioChunk(audio=pcm, rate=rate, width=width, channels=channels).event()
            )

    async def _handle_legacy_synthesize(self, event: Event) -> bool:
        if self._streaming:
            _LOGGER.debug("Ignoring legacy synthesize during streaming")
            return True
        synth = Synthesize.from_event(event)
        raw_text = synth.text
        _LOGGER.debug("Received legacy synthesize: %r", raw_text)

        text = " ".join(raw_text.strip().splitlines())
        if (
            self.cli_args.auto_punctuation
            and text
            and not any(text.endswith(p) for p in self.cli_args.auto_punctuation)
        ):
            text += self.cli_args.auto_punctuation[0]
        _LOGGER.debug("Synthesize: raw_text=%r, text=%r", raw_text, text)

        audio = (
            self.handle_tts_request(text) if text else AudioSegment.silent(duration=0)
        )
        await self._output_audio_stream(audio)
        _LOGGER.debug("Completed legacy request")
        return True

    async def _output_audio_stream(self, audio: AudioSegment) -> None:
        rate = audio.frame_rate
        width = audio.sample_width
        channels = audio.channels
        await self.write_event(
            AudioStart(rate=rate, width=width, channels=channels).event()
        )
        raw = audio.raw_data
        frame_size = width * channels
        chunk_size = frame_size * self.samples_per_chunk
        for i in range(0, len(raw), chunk_size):
            await self.write_event(
                AudioChunk(
                    audio=raw[i : i + chunk_size],
                    rate=rate,
                    width=width,
                    channels=channels,
                ).event()
            )
        await self.write_event(AudioStop().event())

    async def _handle_unexpected_event(self, event: Event) -> bool:
        _LOGGER.warning("Unexpected event: %s", event)
        return True

    def handle_tts_request(self, text: str, delay: float = 250) -> AudioSegment:
        sentences = sent_tokenize(text)
        if not sentences:
            return AudioSegment.silent(duration=0)
        audio = self.glados_tts.run_tts(sentences[0])
        pause = AudioSegment.silent(duration=delay)
        for sentence in sentences[1:]:
            audio += pause + self.glados_tts.run_tts(sentence)
        return audio
