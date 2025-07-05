#!/usr/bin/env python3


"""Utility for downloading GLaDOS TTS models."""

import argparse
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Union
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import urlopen

DEFAULT_URL = (
    "https://github.com/Jonah-May-OSS/glados-tts/releases/download/1.0.0/{file}"
)
DEFAULT_MODEL_DIR = "./gladostts/models"

_LOGGER = logging.getLogger(__name__)


def _quote_url(url: str) -> str:
    """Quote the file part of the URL in case it contains UTF-8 characters."""
    parts = list(urlsplit(url))
    parts[2] = quote(parts[2])
    return urlunsplit(parts)


def get_file_hash(path: Path, bytes_per_chunk: int = 8192) -> str:
    """Calculate the MD5 hash of a file in chunks."""
    md5_hash = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(bytes_per_chunk), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def is_valid_file(file_path: Path, expected_md5: str) -> bool:
    """Check if the file exists, is of sufficient size, and matches the MD5 hash."""
    if not file_path.exists():
        return False
    if file_path.stat().st_size < 1024:
        _LOGGER.warning("File %s is too small.", file_path)
        return False
    md5_hash = get_file_hash(file_path)
    if md5_hash != expected_md5:
        _LOGGER.warning(
            "MD5 hash mismatch for %s. Expected %s, got %s.",
            file_path,
            expected_md5,
            md5_hash,
        )
        return False
    return True


def ensure_model_exists(download_dir: Path, base_url: str):
    """Ensure that all required model files are present and valid."""
    # List of model files and their expected MD5 checksums

    model_files = [
        {
            "filename": "glados-new.pt",
            "md5": "d6945ffd96ee0619d0d49a581b5b83ad",
        },
        {
            "filename": "tacotron-trt.ts",
            "md5": "b0d3d95a30bfecb2904be498767636ff",
        },
        {
            "filename": "en_us_cmudict_ipa_forward.pt",
            "md5": "33887f7f579f010ce4463534306120b0",
        },
        {
            "filename": "emb/glados_p2.pt",
            "md5": "ff2ad1438e9acb1f8e8607864c239ffc",
        },
        {
            "filename": "vocoder-gpu.pt",
            "md5": "d35c13c01d2cacd348aa216649bbfac3",
        },
        {
            "filename": "vocoder-trt.ts",
            "md5": "57990678b466251884a43ccc2ca7dda4",
        },
    ]

    for model in model_files:
        model_file = model["filename"]
        model_file_path = download_dir / model_file
        model_file_path.parent.mkdir(parents=True, exist_ok=True)

        if is_valid_file(model_file_path, model["md5"]):
            _LOGGER.info("File %s is valid.", model_file_path)
            continue  # No need to download
        # Remove invalid or incomplete file

        if model_file_path.exists():
            model_file_path.unlink()
        # Download the file

        try:
            filename = model_file.split("/")[-1]
            model_url = base_url.format(file=filename)
            _LOGGER.info("Downloading %s to %s", model_url, model_file_path)
            with urlopen(_quote_url(model_url)) as response, open(
                model_file_path, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)
            _LOGGER.info("Downloaded %s", model_file_path)

            # Verify MD5 hash after download

            if is_valid_file(model_file_path, model["md5"]):
                _LOGGER.info("Verified MD5 hash for %s.", model_file_path)
            else:
                _LOGGER.error(
                    "MD5 hash mismatch after download for %s.", model_file_path
                )
                if model_file_path.exists():
                    model_file_path.unlink()
        except Exception:
            _LOGGER.exception(
                "Failed to download %s from %s",
                model_file_path,
                _quote_url(model_url),
            )
            if model_file_path.exists():
                model_file_path.unlink()  # Remove incomplete file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLaDOS TTS Model Downloader")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(DEFAULT_MODEL_DIR),
        help="Directory for the models",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="URL for downloading models",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    ensure_model_exists(args.model_dir, args.url)
