"""Event handler implementation for Wyoming GLaDOS TTS events."""

import argparse
import asyncio
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

        # Streaming pipeline state: sentences are synthesized ahead of time
        # while the previous sentence's audio is still being written to the
        # client. _sentence_queue holds (pcm_queue, pump_task) pairs in
        # sentence order; _drain_task writes their audio to the client.
        self._sentence_queue: asyncio.Queue[Any] | None = None
        self._drain_task: asyncio.Task[None] | None = None

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
                await self._start_pipeline()
                _LOGGER.debug("Text stream started: voice=%s", stream_start.voice)
                return True
            if SynthesizeChunk.is_type(event.type):
                assert self._synthesize is not None
                stream_chunk = SynthesizeChunk.from_event(event)

                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    _LOGGER.debug("Synthesizing stream sentence: %s", sentence)
                    await self._enqueue_sentence(sentence)
                return True
            if SynthesizeStop.is_type(event.type):
                assert self._synthesize is not None
                final_text = self.sbd.finish()
                if final_text:
                    # Final audio chunk(s)

                    await self._enqueue_sentence(final_text)

                await self._finish_pipeline()

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

    async def _write_audio(
        self, pcm: bytes, rate: int, width: int, channels: int
    ) -> int:
        """Write one PCM buffer to the client in transport-sized chunks."""
        if not self.audio_started:
            await self.write_event(
                AudioStart(rate=rate, width=width, channels=channels).event()
            )
            self.audio_started = True

        samples_per_chunk = max(
            1,
            int(getattr(self.cli_args, "samples_per_chunk", 1024)),
        )

        # Keep sentence-level synthesis quality, but stream PCM in transport-sized chunks.
        chunks_sent = 0
        bytes_per_chunk = max(1, width * channels * samples_per_chunk)
        for offset in range(0, len(pcm), bytes_per_chunk):
            await self.write_event(
                AudioChunk(
                    audio=pcm[offset : offset + bytes_per_chunk],
                    rate=rate,
                    width=width,
                    channels=channels,
                ).event()
            )
            chunks_sent += 1
        return chunks_sent

    async def _report_tts_error(self, err: Exception) -> None:
        """Send an Error event and reset the audio stream state."""
        _LOGGER.error("Error during TTS processing: %s", err)
        await self.write_event(Error(text=str(err), code="TTSProcessingError").event())
        if self.audio_started:
            await self.write_event(AudioStop().event())
            self.audio_started = False
            _LOGGER.debug("Audio stream reset after synthesis error.")

    async def _handle_synthesize(
        self, synthesize: Synthesize, send_stop: bool = True
    ) -> bool:
        _LOGGER.debug(synthesize)

        glados_proc = await self.process_manager.get_process()
        total_chunks_sent = 0

        try:
            # Start processing with run_tts

            async for pcm, rate, width, channels in glados_proc.run_tts(
                synthesize.text
            ):
                if pcm is None:
                    continue

                total_chunks_sent += await self._write_audio(pcm, rate, width, channels)

            _LOGGER.debug(
                "Sent %s AudioChunk events for chunk: %s",
                total_chunks_sent,
                synthesize.text,
            )
        except Exception as e:
            await self._report_tts_error(e)

        if send_stop:
            await self.write_event(AudioStop().event())
            self.audio_started = False
        # Stop the audio stream when done

        _LOGGER.debug("Completed request")
        return True

    # ------------------------------------------------------------------
    # Streaming pipeline: synthesize the next sentence while the previous
    # sentence's audio is still being written to the client.
    # ------------------------------------------------------------------

    async def _start_pipeline(self) -> None:
        """(Re)start the sentence pipeline for a new text stream."""
        await self._cancel_pipeline()
        # maxsize=1 bounds prefetch: one sentence draining to the client plus
        # one synthesizing ahead, so a flood of input text can't pile up
        # unbounded synthesis tasks.
        self._sentence_queue = asyncio.Queue(maxsize=1)
        self._drain_task = asyncio.create_task(self._drain_audio())

    async def _enqueue_sentence(self, text: str) -> None:
        """Start synthesis of one sentence and queue its audio for draining."""
        assert self._sentence_queue is not None
        pcm_queue: asyncio.Queue[Any] = asyncio.Queue()
        pump_task = asyncio.create_task(self._pump_sentence(text, pcm_queue))
        await self._sentence_queue.put((pcm_queue, pump_task))

    async def _pump_sentence(self, text: str, pcm_queue: asyncio.Queue[Any]) -> None:
        """Synthesize one sentence, pushing PCM chunks into pcm_queue.

        The queue is terminated with None on success or the exception itself
        on failure, so the drain task never blocks forever.
        """
        try:
            glados_proc = await self.process_manager.get_process()
            async for chunk in glados_proc.run_tts(text):
                await pcm_queue.put(chunk)
            await pcm_queue.put(None)
        except Exception as err:
            await pcm_queue.put(err)

    async def _drain_audio(self) -> None:
        """Write synthesized audio to the client in sentence order."""
        assert self._sentence_queue is not None
        while True:
            item = await self._sentence_queue.get()
            if item is None:
                # End of stream
                return
            pcm_queue, pump_task = item
            try:
                while True:
                    chunk = await pcm_queue.get()
                    if chunk is None:
                        break
                    if isinstance(chunk, Exception):
                        raise chunk
                    pcm, rate, width, channels = chunk
                    if pcm is None:
                        continue
                    await self._write_audio(pcm, rate, width, channels)
                await pump_task
            except Exception as err:
                # Report the failed sentence but keep the stream alive for
                # the sentences that follow.
                await self._report_tts_error(err)

    async def _finish_pipeline(self) -> None:
        """Signal end-of-stream and wait for all queued audio to be written."""
        if self._sentence_queue is None or self._drain_task is None:
            return
        await self._sentence_queue.put(None)
        try:
            await self._drain_task
        finally:
            self._drain_task = None
            self._sentence_queue = None

    async def _cancel_pipeline(self) -> None:
        """Drop any in-flight pipeline work without writing further audio."""
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                # Expected when awaiting a task that we explicitly cancelled.
                pass
            except Exception:
                pass
            self._drain_task = None
        self._sentence_queue = None

    async def disconnect(self) -> None:
        """Cancel in-flight synthesis when the client goes away."""
        await self._cancel_pipeline()
