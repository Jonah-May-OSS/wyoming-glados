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


# The voice lives on its own tag, not on `latest`.
#
# `latest` was the obvious choice - a retrained voice reaches deployments with
# no code change - but it only works if every release carries the voice, and
# release.yaml cannot. That workflow zips the repo and attaches release.zip;
# the voice is a 73 MB artifact exported from a training checkpoint that is not
# in the repo, so CI has nothing to build it from. The first ordinary code
# release after a voice release would therefore become `latest` with no
# glados.onnx attached, and every fresh deployment would 404.
#
# Pinning decouples the two: code releases cut as often as they like, and
# shipping a retrained voice is an explicit one-line bump here, reviewable in
# the PR that ships it. This is what the pre-Piper code did (it pinned
# glados-tts 1.0.0), arrived at the same way.
VOICE_RELEASE = "voice-2026.09.01"

# md5 of each asset on VOICE_RELEASE, keyed by filename. Bump with the tag.
#
# Only valid for the voice these describe, so they are keyed by the default
# voice name and applied only to it. A --voice-name pointing at some other
# voice hosted at the same URL would otherwise fail verification against
# glados' hashes and re-download forever.
DEFAULT_VOICE_NAME = "glados"
VOICE_CHECKSUMS: dict[str, str] = {
    "glados.onnx": "fbe7ad25c02eff554e69f3a8128f71a7",
    "glados.onnx.json": "bcd03937e5bdaeac0ef5c7fafee91d06",
}
DEFAULT_URL = (
    "https://github.com/Jonah-May-OSS/wyoming-glados/releases/download/"
    f"{VOICE_RELEASE}/{{file}}"
)
DEFAULT_MODEL_DIR = "./models"

# Below this a "download" is an error page or a truncated transfer, not a
# voice. Deliberately crude: it only has to reject obvious garbage, since
# the checksum settles the rest.
MIN_PLAUSIBLE_FILE_BYTES = 1024

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
    if file_path.stat().st_size < MIN_PLAUSIBLE_FILE_BYTES:
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


def _classify(model: ModelFile, path: Path, url: str) -> tuple[bool, bool]:
    """Decide what to do with one file already on disk.

    Returns (leave_alone, still_usable):

    * leave_alone - the file is current; skip it entirely.
    * still_usable - it works, but is out of date. Callers keep it until a
      replacement is safely in hand, so a failed refresh never costs a working
      voice.

    A checksum settles currency outright. Without one, "exists and over 1024
    bytes" is not the same as "current": bumping VOICE_RELEASE changes what
    DEFAULT_URL serves while the stale file on disk keeps passing that test, so
    an upgraded deployment would stay on the old voice forever. The same holds
    for a voice tag republished in place.

    Sizes catch that. Weaker than a checksum (a retrained voice that happens to
    match byte-for-byte in length is missed), but it needs no published hash
    and fails safe: remote_size returns None when offline, and the file stays.
    """
    if not is_valid_file(path, model["md5"]):
        return False, False
    if model["md5"] is not None:
        return True, True

    try:
        local_size = path.stat().st_size
    except OSError:
        # is_valid_file vouched for it; with nothing to compare against, take
        # that verdict rather than re-downloading on a stat error.
        return True, True

    expected_size = remote_size(url)
    if expected_size is None or expected_size == local_size:
        return True, True

    _LOGGER.info(
        "File %s is %d bytes but %s now serves %d; re-downloading.",
        path,
        local_size,
        url,
        expected_size,
    )
    return False, True


def _voice_checksums(voice_name: str, base_url: str) -> dict[str, str]:
    """Return the published checksums, but only where they actually apply.

    VOICE_CHECKSUMS describes the assets on VOICE_RELEASE for the default
    voice. Applied to anything else they are not a weaker check but a wrong
    one: verification would fail on a perfectly good file, and since a
    mismatch means "re-download", the result is an unbreakable loop rather
    than a clear error.

    Two ways to end up somewhere else, both supported on purpose: --voice-name
    for a different voice, and --url for a different host (an air-gapped
    mirror, or a locally exported voice served over HTTP).
    """
    if voice_name != DEFAULT_VOICE_NAME:
        return {}
    if base_url != DEFAULT_URL:
        return {}
    return dict(VOICE_CHECKSUMS)


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
    # Derived from voice_name, not hardcoded: __main__.py exposes --voice-name
    # and VOICE_NAME, and used to invoke this script without passing either. A
    # non-default voice therefore downloaded glados.onnx, exited 0, and left
    # PiperTTSRunner to raise FileNotFoundError for a file nothing had tried to
    # fetch - a traceback that blamed the wrong thing entirely.
    #
    # The checksums are what make a VOICE_RELEASE bump actually reach an
    # existing deployment, and without them the bump is silently inert. The
    # size fallback in _classify() cannot help here: the ONNX graph is the same
    # shape at every epoch, so only the weight VALUES differ. Exports of four
    # different checkpoints all came to exactly 76771148 bytes, so a size
    # comparison can never see a retrained voice for this architecture.
    #
    # Publishing a hash is safe because the export is deterministic - the same
    # checkpoint exported four times produced one md5, matching the artifact on
    # the release. Bump these together with VOICE_RELEASE; that they move as a
    # set is the point, and _voice_checksums() keeps them from being applied to
    # a voice they do not describe.
    md5_by_file = _voice_checksums(voice_name, base_url)
    model_files: list[ModelFile] = [
        {
            "filename": f"{voice_name}.onnx",
            "md5": md5_by_file.get(f"{voice_name}.onnx"),
        },
        {
            "filename": f"{voice_name}.onnx.json",
            "md5": md5_by_file.get(f"{voice_name}.onnx.json"),
        },
    ]

    # Two phases: fetch every file that needs replacing into a sidecar, then
    # rename them into place only once ALL of them arrived.
    #
    # Per-file commits could leave a mismatched pair. The .onnx and .onnx.json
    # are one artefact - the config carries the phoneme_id_map and speaker
    # table for that exact graph - so refreshing the config while the graph
    # failed produces a voice that loads and then speaks garbage. __main__'s
    # fallback only checks that both files exist, so it would serve it.
    all_present = True
    pending: list[tuple[ModelFile, Path, Path, bool]] = []

    for model in model_files:
        model_file = model["filename"]
        model_file_path = download_dir / model_file
        model_file_path.parent.mkdir(parents=True, exist_ok=True)

        model_url = base_url.format(file=model_file.rsplit("/", maxsplit=1)[-1])

        leave_alone, usable = _classify(model, model_file_path, model_url)
        if leave_alone:
            _LOGGER.info("File %s is valid.", model_file_path)
            continue

        part_path = model_file_path.with_name(model_file_path.name + ".part")
        pending.append((model, model_file_path, part_path, usable))

    # Phase 1: fetch into sidecars. Nothing on disk changes yet.
    fetched: list[tuple[ModelFile, Path, Path]] = []
    for model, model_file_path, part_path, _usable in pending:
        model_file = model["filename"]
        model_url = base_url.format(file=model_file.rsplit("/", maxsplit=1)[-1])
        try:
            _LOGGER.info("Downloading %s to %s", model_url, model_file_path)
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
            fetched.append((model, model_file_path, part_path))
        except Exception:
            _LOGGER.exception(
                "Failed to download %s from %s",
                model_file_path,
                _quote_url(model_url),
            )
            all_present = False

    # Phase 2: commit, but only as a set.
    if pending and len(fetched) != len(pending):
        _LOGGER.error(
            "Only %d of %d voice files downloaded; keeping the existing voice "
            "rather than committing a mismatched set.",
            len(fetched),
            len(pending),
        )
        for _model, model_file_path, part_path, usable in pending:
            part_path.unlink(missing_ok=True)
            # A file that was already unusable stays unusable, and leaving it
            # would let __main__'s existence check serve something corrupt.
            # One that was merely stale still works, so it is kept.
            if not usable and model_file_path.exists():
                model_file_path.unlink()
        return False

    for model, model_file_path, part_path in fetched:
        part_path.replace(model_file_path)
        _LOGGER.info("Downloaded %s", model_file_path)
        if is_valid_file(model_file_path, model["md5"]):
            _LOGGER.info("Verified MD5 hash for %s.", model_file_path)
        else:
            _LOGGER.error("MD5 hash mismatch after download for %s.", model_file_path)
            if model_file_path.exists():
                model_file_path.unlink()
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
