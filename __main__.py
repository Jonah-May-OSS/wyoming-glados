#!/usr/bin/env python3

"""Utility for running the GLaDOS TTS server."""

import argparse
import asyncio
import logging
import sys
import time
from typing import Optional
from functools import partial
from pathlib import Path

import warnings

# 1) hide that nested-tensor warning so it never pollutes your logs
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False",
    module=r"torch\.nn\.modules\.transformer",
)


# 2) actually turn it off under the hood
import torch.nn.modules.transformer as _tfm

_tfm.enable_nested_tensor = False

import nltk
from nltk import data as nltk_data
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

# Configure logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Ensure 'gladostts' module is importable

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gladostts.glados import TTSRunner
from server.handler import GladosEventHandler


class NanosecondFormatter(logging.Formatter):
    """Custom formatter to include nanoseconds in log timestamps."""

    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        """Formats the time with nanosecond precision."""
        ct = record.created
        t = time.localtime(ct)
        s = time.strftime("%Y-%m-%d %H:%M:%S", t)
        return f"{s}.{int(ct * 1e9) % 1_000_000_000:09d}"


def setup_logging(debug: bool, log_format: str) -> None:
    """
    Sets up logging with the specified level and format.

    Args:
        debug (bool): Whether to enable DEBUG level logging.
        log_format (str): Format string for log messages.
    """
    formatter = NanosecondFormatter(log_format)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    rootlogger = logging.getLogger()
    rootlogger.setLevel(logging.DEBUG if debug else logging.INFO)
    rootlogger.handlers = [handler]

    logger.debug("Logging has been configured.")


async def main() -> None:
    """Main entry point for the GLaDOS TTS server."""
    parser = argparse.ArgumentParser(description="GLaDOS TTS Server")
    parser.add_argument(
        "--uri",
        default="stdio://",
        help="Server URI (e.g., 'unix://', 'tcp://')",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=SCRIPT_DIR / "gladostts" / "models",
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
        "--log-format",
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        help="Format for log messages",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug, args.log_format)

    # Validate models directory

    models_dir = args.models_dir.resolve()
    if not models_dir.exists():
        logger.error("Models directory does not exist: %s", models_dir)
        sys.exit(1)
    # Define TTS voices

    voice_attribution = Attribution(
        name="R2D2FISH", url="https://github.com/R2D2FISH/glados-tts"
    )
    voices = [
        TtsVoice(
            name="default",
            description="Default GLaDOS voice",
            attribution=voice_attribution,
            installed=True,
            languages=["en"],
            version=2,
        )
    ]

    # Define TTS program information

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="glados-tts",
                description="A GLaDOS TTS using Forward Tacotron and HiFiGAN.",
                attribution=voice_attribution,
                installed=True,
                voices=voices,
                version=2,
            )
        ],
    )

    # Initialize GLaDOS TTS
    logger.debug("Initializing GLaDOS TTS engine...")
    glados_tts = TTSRunner(
        use_p1=False,
        log=args.debug,
        models_dir=models_dir,
    )

    logger.debug("GLaDOS TTS engine initialized successfully.")

    # sanity-check that our RNN weights are contiguous
    try:
        # if the ScriptModule exposes .rnn, flatten now and log
        glados_tts.glados.rnn.flatten_parameters()
        logger.debug("Flattened RNN weights for best cuDNN performance.")
    except Exception:
        logger.debug("No .rnn to flatten (or already contiguous).")

    # Ensure NLTK 'punkt' data is downloaded
    try:
        nltk_data.find("tokenizers/punkt_tab")
        logger.debug("NLTK 'punkt' tokenizer data is already available.")
    except LookupError:
        logger.debug("Downloading NLTK 'punkt' tokenizer data...")
        nltk_download("punkt_tab", quiet=not args.debug)

    # Start the server

    logger.info("Starting the GLaDOS TTS server on %s", args.uri)
    server = AsyncServer.from_uri(args.uri)
    logger.info("Server started and listening on %s", args.uri)
    try:
        await server.run(partial(GladosEventHandler, wyoming_info, args, glados_tts))
    except Exception as e:
        logger.exception("An error occurred while running the server: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested. Exiting...")
    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
        sys.exit(1)
