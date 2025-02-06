"""Event handler for clients of the server."""

import argparse
import logging
from typing import Optional

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize

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
        *args,
        **kwargs,
    ) -> None:
        """Initialize the GLaDOS event handler."""
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.glados_tts = glados_tts

    def handle_tts_request(self, text: str, delay: float = 250) -> AudioSegment:
        """Generate an AudioSegment for the given text with optional delay between sentences.

        Args:
            text: The text to synthesize.
            delay: The delay between sentences in milliseconds.

        Returns:
            An AudioSegment containing the synthesized speech.
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return AudioSegment.silent(duration=0)
        audio = self.glados_tts.run_tts(sentences[0])
        pause = AudioSegment.silent(duration=delay)

        for sentence in sentences[1:]:
            new_line = self.glados_tts.run_tts(sentence)
            audio += pause + new_line
        return audio

    async def handle_event(self, event: Event) -> bool:
        """Handle incoming events from the client.

        Args:
            event: The event to handle.

        Returns:
            True if the connection should remain open, False otherwise.
        """
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True
        if not Synthesize.is_type(event.type):
            _LOGGER.warning("Unexpected event: %s", event)
            return True
        synthesize = Synthesize.from_event(event)
        _LOGGER.debug("Received synthesis request: %s", synthesize)

        raw_text = synthesize.text

        # Join multiple lines

        text = " ".join(raw_text.strip().splitlines())

        if self.cli_args.auto_punctuation and text:
            # Add automatic punctuation (important for some voices)

            if not any(
                text.endswith(punc_char) for punc_char in self.cli_args.auto_punctuation
            ):
                text += self.cli_args.auto_punctuation[0]
        # Actual TTS synthesis

        _LOGGER.debug("Synthesize: raw_text='%s', text='%s'", raw_text, text)

        if text:
            try:
                audio = self.handle_tts_request(text)
            except Exception as e:
                _LOGGER.exception("Error during TTS synthesis: %s", e)
                # Optionally, send an error message to the client

                return True
        else:
            audio = AudioSegment.silent(duration=0)
        rate = audio.frame_rate
        width = audio.sample_width
        channels = audio.channels

        await self.write_event(
            AudioStart(
                rate=rate,
                width=width,
                channels=channels,
            ).event(),
        )

        # Audio data

        audio_bytes = audio.raw_data
        bytes_per_sample = width * channels
        bytes_per_chunk = bytes_per_sample * self.cli_args.samples_per_chunk
        num_chunks = (
            len(audio_bytes) + bytes_per_chunk - 1
        ) // bytes_per_chunk  # Ceiling division

        # Split into chunks and send

        for i in range(num_chunks):
            offset = i * bytes_per_chunk
            chunk = audio_bytes[offset : offset + bytes_per_chunk]
            await self.write_event(
                AudioChunk(
                    audio=chunk,
                    rate=rate,
                    width=width,
                    channels=channels,
                ).event(),
            )
        await self.write_event(AudioStop().event())
        _LOGGER.debug("Completed request")

        return True
