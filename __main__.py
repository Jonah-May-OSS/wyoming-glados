#!/usr/bin/env python3

"""Utility for running the GLaDOS TTS server."""

import argparse
import asyncio
import contextlib
import logging
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from piper_runtime import PiperTTSRunner
from server.handler import GladosEventHandler
from server.process import GladosProcessManager

SCRIPT_DIR = Path(__file__).resolve().parent

# logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NanosecondFormatter(logging.Formatter):
    """Custom formatter to include nanoseconds in log timestamps."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Format the record's timestamp, appending nanoseconds.

        `datefmt` is honoured rather than ignored. It used to be accepted and
        dropped, so a caller passing one got the hardcoded layout back with no
        indication its argument had gone nowhere.
        """
        ct = record.created
        t = time.localtime(ct)
        s = time.strftime(datefmt or "%Y-%m-%d %H:%M:%S", t)
        return f"{s}.{int(ct * 1e9) % 1_000_000_000:09d}"


def setup_logging(debug: bool, log_format: str) -> None:
    """Configure root logging handlers and verbosity."""
    formatter = NanosecondFormatter(log_format)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    rootlogger = logging.getLogger()
    rootlogger.setLevel(logging.DEBUG if debug else logging.INFO)
    rootlogger.handlers = [handler]

    logger.debug("Logging has been configured.")


def _warmup(runner: PiperTTSRunner) -> None:
    """Run the warmup synthesis, logging rather than killing the server.

    Engines are built during session creation now that TensorRT is given
    explicit shape profiles, so this is a smoke test that inference works
    rather than the seven-bucket, three-pass, budgeted walk it replaced.

    Guarded because an exception here would otherwise vanish: a failed smoke
    test should be visible in the log, not fatal to a server that can still
    fall back to another provider.
    """
    started = time.monotonic()
    try:
        runner.warmup()
    except Exception:
        logger.exception("TensorRT warmup failed; serving without warm engines")
    else:
        logger.info("Warmup complete in %.1fs", time.monotonic() - started)


async def main() -> None:
    """Run the GLaDOS TTS server until the process is stopped."""
    parser = argparse.ArgumentParser(description="GLaDOS TTS Server")
    parser.add_argument(
        "--uri",
        default="stdio://",
        help="Server URI (e.g., 'unix://', 'tcp://')",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=os.environ.get("MODELS_DIR", "/usr/src/models"),
        help="Directory containing the model files",
    )
    parser.add_argument(
        "--auto-punctuation",
        default=".?!",
        help="Characters to use for automatic punctuation",
    )
    parser.add_argument(
        "--samples-per-chunk",
        type=int,
        default=1024,
        help="Number of samples per audio chunk",
    )
    parser.add_argument(
        "--streaming",  # Add the streaming argument
        action="store_true",
        help="Enable streaming mode",
    )
    parser.add_argument(
        "--log-format",
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        help="Format for log messages",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--voice-name",
        default=os.environ.get("VOICE_NAME", "glados"),
        help="Basename of <name>.onnx in --models-dir",
    )
    parser.add_argument(
        "--speaker",
        default=os.environ.get("SPEAKER") or None,
        help=(
            "Speaker to synthesize as, for multi-speaker voices "
            "(p1, p2, dota2, potato). Names resolve through the voice "
            "config's speaker_id_map. Ignored by single-speaker voices. "
            "Unset, the voice's own default_speaker is used (p2 for GLaDOS); "
            "a voice declaring none falls back to id 0, which is merely "
            "whichever speaker came first in the corpus."
        ),
    )
    args = parser.parse_args()

    # Setup logging

    setup_logging(args.debug, args.log_format)

    # Fetch the voice before startup. download.py verifies what is already
    # on disk and only pulls what is missing, so this is cheap on restart.

    try:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "download.py"),
                "--model-dir",
                str(args.models_dir),
                "--voice-name",
                args.voice_name,
                *(["--debug"] if args.debug else []),
            ],
            timeout=300,
            check=True,
        )
        logger.info("Voice model downloaded (or already up-to-date).")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        error_msg = (
            "timeout after 300s"
            if isinstance(e, subprocess.TimeoutExpired)
            else f"exit {e.returncode}"
        )
        logger.error("Voice model download failed (%s).", error_msg)

        # Whether that is fatal depends on what is already on disk. Continuing
        # with no voice at all means PiperTTSRunner below raises
        # FileNotFoundError from _load, which surfaces as a traceback about a
        # missing file rather than the actual problem, which is that the
        # download failed. Say so and exit.
        #
        # But a failed refresh with a usable voice present is survivable, and
        # exiting there would take a working offline deployment down over a
        # transient network problem. Serve what we have.
        # Both files, not just the graph: PiperTTSRunner reads the .json for
        # the phoneme_id_map and speaker table, and does so without a guard, so
        # a half-downloaded voice would still die with a bare traceback about a
        # missing .json - the exact failure this branch exists to prevent.
        voice_path = args.models_dir / f"{args.voice_name}.onnx"
        config_path = args.models_dir / f"{args.voice_name}.onnx.json"
        missing = [str(p) for p in (voice_path, config_path) if not p.exists()]
        if missing:
            logger.error(
                "Voice incomplete after a failed download (missing %s); cannot serve.",
                ", ".join(missing),
            )
            sys.exit(1)
        logger.warning(
            "Continuing with the existing voice at %s; it may be out of date.",
            voice_path,
        )

    # Define voice attribution and voices
    #
    # This credits the VITS voice trained in this repository, not R2D2FISH's
    # ForwardTacotron models, which the piper backend no longer uses.

    voice_attribution = Attribution(
        name="Jonah-May-OSS",
        url="https://github.com/Jonah-May-OSS/wyoming-glados",
    )
    # The advertised name is the one that was loaded, so it cannot drift from
    # --voice-name. It used to be hardcoded "default" while the server loaded
    # "glados", which would have become ambiguous as soon as a second voice
    # existed.
    voices = [
        TtsVoice(
            name=args.voice_name,
            description=f"GLaDOS VITS voice ({args.voice_name})",
            attribution=voice_attribution,
            installed=True,
            languages=["en"],
            version="2",
        )
    ]

    # Define TTS program information (streaming support enabled)

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="glados-tts",
                description="A GLaDOS TTS using VITS via ONNX Runtime.",
                attribution=voice_attribution,
                installed=True,
                voices=voices,
                version="2",
                supports_synthesize_streaming=True,  # ← ADDED
            )
        ],
    )

    # Initialize GLaDOS TTS

    logger.debug("Initializing GLaDOS TTS engine...")
    glados_tts = PiperTTSRunner(
        models_dir=args.models_dir,
        voice_name=args.voice_name,
        speaker=args.speaker,
    )
    # NOTE: startup blocks until the TensorRT engines exist, and that happens
    # in PiperTTSRunner() above - _load calls _trt_profiles, which creates real
    # sessions to discover the profile inputs. The server binds afterwards.
    #
    # This used to run on a background thread, with a comment claiming the
    # server accepted connections immediately and early requests fell back to
    # the CUDA provider. That stopped being true when engine construction moved
    # into session creation: by the time the thread started there was nothing
    # left to defer, and _warmup measures 0.03s against a cold build of ~60s.
    # The thread deferred nothing and hid where the time actually went.
    #
    # So this is synchronous and honest. The real mitigation is the on-disk
    # engine cache: mount --models-dir on a volume and only the first start on
    # a given voice and GPU pays the build. Home Assistant will mark the entity
    # unavailable for that first cold start.
    #
    # Backgrounding the whole runner construction would let the server bind
    # first, but then the first synthesize blocks for the same minute with no
    # way to say why - a timeout mid-request rather than an entity that is
    # briefly unavailable. That trade is worth making deliberately, not as a
    # side effect.
    _warmup(glados_tts)
    logger.debug("GLaDOS TTS engine initialized successfully.")

    # Create the GladosProcessManager instance

    process_manager = GladosProcessManager(glados_tts)

    # Make sure default voice is loaded.

    await process_manager.get_process()

    # Start the server with the updated handler

    server = AsyncServer.from_uri(args.uri)
    logger.info("Server started and listening on %s", args.uri)

    handler_factory = partial(
        GladosEventHandler,
        wyoming_info,
        args,
        process_manager,
    )

    # Run the server

    try:
        await server.run(handler_factory)
    except (RuntimeError, OSError) as e:
        logger.exception("An error occurred while running the server: %s", e)
        sys.exit(1)


def run():
    """Run the async main entrypoint."""
    asyncio.run(main())


if __name__ == "__main__":  # pragma: no cover
    with contextlib.suppress(KeyboardInterrupt):
        run()
