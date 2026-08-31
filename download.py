#!/usr/bin/env python3


"""Utility for downloading GLaDOS TTS models."""

import argparse
import hashlib
import logging
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


class ModelFile(TypedDict):
    """A model file to download and its expected MD5 checksum (None to skip)."""

    filename: str
    md5: str | None


# `latest` rather than a pinned tag so publishing a retrained voice does not
# also require a code change here. Override with --url to pin a release.
DEFAULT_URL = (
    "https://github.com/Jonah-May-OSS/wyoming-glados/releases/latest/download/{file}"
)
DEFAULT_MODEL_DIR = "./models"

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


def remote_size(url: str, opener: Callable[..., Any] | None = None) -> int | None:
    """Content-Length the server reports for `url`, or None if unavailable.

    Used to decide whether an unversioned file on disk is still current. A HEAD
    keeps this cheap: no body is transferred, so the common case (nothing has
    changed) costs one round trip.
    """
    try:
        # Resolved here, not as a default argument: a default binds urlopen at
        # def time, which silently escapes patch("download.urlopen") and lets
        # tests make real network calls while asserting they made none.
        open_url = urlopen if opener is None else opener
        request = Request(_quote_url(url), method="HEAD")
        with open_url(request) as response:
            length = response.headers.get("Content-Length")
        return int(length) if length is not None else None
    except Exception:
        # Offline, or the host does not answer HEAD. Callers treat None as
        # "cannot tell", and keep whatever is on disk: a missing network must
        # not delete a working voice.
        _LOGGER.debug("Could not determine remote size for %s", url, exc_info=True)
        return None


def is_valid_file(file_path: Path, expected_md5: str | None) -> bool:
    """Check if the file exists, is of sufficient size, and matches the MD5 hash."""
    if not file_path.exists():
        return False
    if file_path.stat().st_size < 1024:
        _LOGGER.warning("File %s is too small.", file_path)
        return False

    # Some artifacts (e.g., locally rebuilt TensorRT engines) are expected
    # to differ from the release hash across machines/runtime versions.
    if expected_md5 is None:
        return True

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


def ensure_model_exists(
    download_dir: Path, base_url: str, voice_name: str = "glados"
) -> bool:
    """Ensure that all required model files are present and valid.

    Returns False if any file is still missing or invalid afterwards. __main__
    runs this as a subprocess with check=True, so swallowing failures here made
    that error branch unreachable: a download that failed outright still exited
    0 and was logged as "downloaded (or already up-to-date)", with the real
    problem surfacing later as a confusing model-not-found at session load.
    """
    # List of model files and their expected MD5 checksums

    # The VITS voice is two files: the ONNX graph and the config carrying the
    # phoneme_id_map the runtime needs to turn phonemes into model inputs.
    # Checksums are left unset until the voice is published to a release; a
    # None checksum only skips verification, it still requires the file.
    # Derived from voice_name, not hardcoded: __main__.py exposes --voice-name
    # and VOICE_NAME, and used to invoke this script without passing either. A
    # non-default voice therefore downloaded glados.onnx, exited 0, and left
    # PiperTTSRunner to raise FileNotFoundError for a file nothing had tried to
    # fetch - a traceback that blamed the wrong thing entirely.
    model_files: list[ModelFile] = [
        {
            "filename": f"{voice_name}.onnx",
            "md5": None,
        },
        {
            "filename": f"{voice_name}.onnx.json",
            "md5": None,
        },
    ]

    all_present = True
    for model in model_files:
        model_file = model["filename"]
        model_file_path = download_dir / model_file
        model_file_path.parent.mkdir(parents=True, exist_ok=True)

        model_url = base_url.format(file=model_file.rsplit("/", maxsplit=1)[-1])
        stale_but_usable = False

        if is_valid_file(model_file_path, model["md5"]):
            # A checksum settles it. Without one, "exists and over 1024 bytes"
            # is not the same as "current": DEFAULT_URL points at
            # releases/latest, so republishing a retrained voice changes what
            # that URL serves while the stale file on disk keeps passing. Every
            # existing deployment would then stay on the old voice forever,
            # which is the opposite of what pointing at `latest` is for.
            #
            # Compare sizes to catch that. It is weaker than a checksum - a
            # retrained voice that happens to be byte-identical in length is
            # not detected - but it needs no published hash, and it fails safe:
            # remote_size returns None when offline, and the file is kept.
            if model["md5"] is not None:
                _LOGGER.info("File %s is valid.", model_file_path)
                continue

            try:
                local_size = model_file_path.stat().st_size
            except OSError:
                # is_valid_file vouched for this file; if it cannot be stat'd
                # there is nothing to compare a remote size against, so take
                # that verdict rather than re-downloading on a stat error.
                _LOGGER.info("File %s is valid.", model_file_path)
                continue

            expected_size = remote_size(model_url)
            if expected_size is None or expected_size == local_size:
                _LOGGER.info("File %s is valid.", model_file_path)
                continue

            _LOGGER.info(
                "File %s is %d bytes but %s now serves %d; re-downloading.",
                model_file_path,
                local_size,
                model_url,
                expected_size,
            )
            stale_but_usable = True

        # An invalid file must go: leaving it would let __main__'s "continue
        # with the existing voice" fallback serve something corrupt.
        #
        # A valid-but-stale one is deliberately kept. The sidecar rename below
        # replaces it atomically, so deleting it first buys nothing and costs
        # everything if the re-download then fails - a transient network error
        # while refreshing a working voice would leave the deployment with no
        # voice at all.
        if not stale_but_usable and model_file_path.exists():
            model_file_path.unlink()
        # Download the file

        part_path = model_file_path.with_name(model_file_path.name + ".part")
        try:
            _LOGGER.info("Downloading %s to %s", model_url, model_file_path)
            # Download to a sidecar and rename only once complete. Both
            # files carry md5=None, so is_valid_file's only real gate is
            # "larger than 1024 bytes" - a download killed midway would leave a
            # truncated model that passes that check on every later run and is
            # never re-fetched. fetch.py guards the same failure for wiki audio
            # with is_wav_complete; this is the same idea, done atomically.
            with (
                urlopen(_quote_url(model_url)) as response,
                open(part_path, "wb") as out_file,
            ):
                headers = getattr(response, "headers", None)
                declared = headers.get("Content-Length") if headers else None
                shutil.copyfileobj(response, out_file)

            # With md5=None the only gate downstream is "larger than 1024
            # bytes", which a truncated transfer or an error page served with
            # HTTP 200 can both satisfy. The response already said how many
            # bytes to expect, so check them before the rename makes the file
            # look authoritative.
            try:
                expected_bytes = None if declared is None else int(declared)
            except (TypeError, ValueError):
                # A server that omits or malforms Content-Length gives us
                # nothing to check against; that is not itself a failure.
                expected_bytes = None

            written = part_path.stat().st_size
            if expected_bytes is not None and written != expected_bytes:
                raise OSError(
                    f"{model_url}: expected {expected_bytes} bytes, received {written}"
                )

            part_path.replace(model_file_path)
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
                all_present = False
        except Exception:
            _LOGGER.exception(
                "Failed to download %s from %s",
                model_file_path,
                _quote_url(model_url),
            )
            # Only the sidecar is ever partial; model_file_path is written
            # solely by the atomic rename. Deleting it here would throw away
            # the very file the stale-but-usable path is preserving.
            part_path.unlink(missing_ok=True)
            all_present = False

    return all_present


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="GLaDOS TTS Model Downloader")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(DEFAULT_MODEL_DIR),
        help="Directory for the models",
    )
    parser.add_argument(
        "--voice-name",
        type=str,
        default="glados",
        help="Voice to fetch; selects <voice-name>.onnx and .onnx.json",
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

    if not ensure_model_exists(args.model_dir, args.url, args.voice_name):
        _LOGGER.error("One or more voice files are missing or invalid.")
        sys.exit(1)
