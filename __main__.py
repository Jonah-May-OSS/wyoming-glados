#!/usr/bin/env python3

"""Utility for running the GLaDOS TTS server."""

import argparse
import asyncio
import logging
import sys
from functools import partial
from pathlib import Path

import nltk
from nltk import data as nltk_data
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

# Configure logging

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

# Ensure 'gladostts' module is importable

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gladostts.glados import TTSRunner
from server.handler import GladosEventHandler


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
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Set logging level based on debug flag

    if args.debug:
        _LOGGER.setLevel(logging.DEBUG)
    _LOGGER.debug("Starting GLaDOS TTS server with arguments: %s", args)

    # Validate models directory

    models_dir = args.models_dir.resolve()
    if not models_dir.exists():
        _LOGGER.error("Models directory does not exist: %s", models_dir)
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

    _LOGGER.debug("Initializing GLaDOS TTS engine...")
    glados_tts = TTSRunner(
        use_p1=False,
        log=args.debug,
        models_dir=models_dir,
    )

    # Ensure NLTK 'punkt' data is downloaded

    try:
        nltk_data.find("tokenizers/punkt_tab")
        _LOGGER.debug("NLTK 'punkt' tokenizer data is already available.")
    except LookupError:
        _LOGGER.debug("Downloading NLTK 'punkt' tokenizer data...")
        nltk_download("punkt_tab", quiet=not args.debug)
    # Start the server

    _LOGGER.info("Starting the GLaDOS TTS server...")
    server = AsyncServer.from_uri(args.uri)
    try:
        await server.run(partial(GladosEventHandler, wyoming_info, args, glados_tts))
    except Exception as e:
        _LOGGER.exception("An error occurred while running the server: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _LOGGER.info("Server shutdown requested. Exiting...")
    except Exception as e:
        _LOGGER.exception("An unexpected error occurred: %s", e)
        sys.exit(1)
