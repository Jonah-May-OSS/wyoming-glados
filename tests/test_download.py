"""Unit tests for model download and validation helpers."""

import contextlib
import hashlib
import io
import logging
from unittest.mock import MagicMock, patch

import download
from download import (
    _quote_url,
    ensure_model_exists,
    get_file_hash,
    is_valid_file,
)


# -----------------------------
# _quote_url
# -----------------------------
def test_quote_url_encodes_path():
    url = "https://example.com/models/ümlaut.pt"
    quoted = _quote_url(url)
    assert "ümlaut" not in quoted
    assert "%C3%BCmlaut.pt" in quoted


# -----------------------------
# get_file_hash
# -----------------------------
def test_get_file_hash(tmp_path):
    file = tmp_path / "test.bin"
    file.write_bytes(b"hello world")
    expected = hashlib.md5(b"hello world").hexdigest()
    assert get_file_hash(file) == expected


# -----------------------------
# is_valid_file
# -----------------------------
def test_is_valid_file_valid(tmp_path):
    good = tmp_path / "good.bin"
    content = b"0" * 2048
    good.write_bytes(content)
    md5 = hashlib.md5(content).hexdigest()
    assert is_valid_file(good, md5) is True


def test_is_valid_file_too_small(tmp_path, caplog):
    small = tmp_path / "small.bin"
    small.write_bytes(b"x" * 10)
    md5 = hashlib.md5(b"x" * 10).hexdigest()

    with caplog.at_level(logging.WARNING):
        assert is_valid_file(small, md5) is False
        assert "too small" in caplog.text


def test_is_valid_file_bad_md5(tmp_path, caplog):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"x" * 2048)

    with caplog.at_level(logging.WARNING):
        assert is_valid_file(bad, "ffff") is False
        assert "MD5 hash mismatch" in caplog.text


# ================================================================
# ensure_model_exists — download missing files
# ================================================================
def test_ensure_model_exists_downloads_missing_files(tmp_path):
    base_url = "https://example.com/{file}"

    with (
        patch("download.urlopen") as urlopen_mock,
        patch("shutil.copyfileobj") as copy_mock,
        patch("download.get_file_hash") as hash_mock,
    ):
        hash_mock.return_value = "ffffffffffffffffffffffffffffffff"

        urlopen_mock.return_value = MagicMock(
            __enter__=lambda s: io.BytesIO(b"x" * 4096),
            __exit__=lambda *exc: False,
        )

        ensure_model_exists(tmp_path, base_url)

        assert urlopen_mock.call_count == 2
        assert copy_mock.call_count == 2


# ================================================================
# ensure_model_exists — skip valid files
# ================================================================
def test_ensure_model_exists_skips_valid_files(tmp_path):
    base_url = "https://example.com/{file}"

    model_paths = [
        tmp_path / "glados.onnx",
        tmp_path / "glados.onnx.json",
    ]

    for p in model_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"0" * 2048)

    # A HEAD is expected: the voice files carry md5=None, so "present and over
    # 1024 bytes" cannot distinguish a current file from a stale one after a
    # VOICE_RELEASE bump. The remote size is what settles it.
    head = MagicMock()
    head.headers = {"Content-Length": "2048"}
    head.__enter__ = lambda self: self
    head.__exit__ = lambda self, *exc: False

    with (
        patch("download.is_valid_file", return_value=True),
        patch("download.urlopen", return_value=head) as urlopen_mock,
    ):
        ensure_model_exists(tmp_path, base_url)

    # Probed, but nothing re-fetched: every call is a HEAD.
    assert urlopen_mock.call_count == len(model_paths)
    assert all(
        call.args[0].get_method() == "HEAD" for call in urlopen_mock.call_args_list
    )
    for path in model_paths:
        assert path.read_bytes() == b"0" * 2048


def test_a_partial_refresh_commits_nothing(tmp_path):
    """The graph and its config are one artefact; never commit half of them.

    The .onnx.json carries the phoneme_id_map and speaker table for that exact
    graph, so a refreshed config beside a stale graph loads fine and then
    speaks garbage - and __main__ only checks that both files exist.
    """
    base_url = "https://example.com/{file}"
    old_bytes = b"0" * 2048
    graph = tmp_path / "glados.onnx"
    config = tmp_path / "glados.onnx.json"
    for path in (graph, config):
        path.write_bytes(old_bytes)

    def fake_urlopen(request_or_url, *_args, **_kwargs):
        response = MagicMock()
        response.__enter__ = lambda self: self
        response.__exit__ = lambda self, *exc: False
        response.headers = {"Content-Length": "4096"}
        if getattr(request_or_url, "get_method", lambda: "GET")() == "HEAD":
            return response
        # The config downloads; the graph does not.
        url = getattr(request_or_url, "full_url", request_or_url)
        if str(url).endswith(".json"):
            response.read = MagicMock(side_effect=[b"1" * 4096, b""])
            return response
        raise OSError("connection reset")

    with (
        patch("download.is_valid_file", return_value=True),
        patch("download.urlopen", side_effect=fake_urlopen),
    ):
        assert ensure_model_exists(tmp_path, base_url) is False

    # Neither file moved: an old-but-matched pair beats a new mismatched one.
    assert graph.read_bytes() == old_bytes
    assert config.read_bytes() == old_bytes
    assert not list(tmp_path.glob("*.part"))


def test_a_failed_refresh_keeps_the_working_voice(tmp_path):
    """A transient network error must not leave the deployment with no voice.

    The size check can route a perfectly good file into the re-download path.
    If that download then fails, deleting the original first would take a
    working offline deployment down - and __main__ would exit rather than
    serve, because its fallback looks for exactly this file.
    """
    base_url = "https://example.com/{file}"
    good = b"0" * 2048
    model_paths = [tmp_path / "glados.onnx", tmp_path / "glados.onnx.json"]
    for path in model_paths:
        path.write_bytes(good)

    def fake_urlopen(request_or_url, *_args, **_kwargs):
        if getattr(request_or_url, "get_method", lambda: "GET")() == "HEAD":
            response = MagicMock()
            response.__enter__ = lambda self: self
            response.__exit__ = lambda self, *exc: False
            response.headers = {"Content-Length": "4096"}  # remote changed
            return response
        raise OSError("connection reset")  # the refresh fails

    with (
        patch("download.is_valid_file", return_value=True),
        patch("download.urlopen", side_effect=fake_urlopen),
    ):
        assert ensure_model_exists(tmp_path, base_url) is False

    for path in model_paths:
        assert path.exists(), "the working voice must survive a failed refresh"
        assert path.read_bytes() == good
        assert not path.with_suffix(path.suffix + ".part").exists()


def test_ensure_model_exists_refetches_when_the_remote_size_changed(tmp_path):
    """A retrained voice at a bumped VOICE_RELEASE must actually arrive."""
    base_url = "https://example.com/{file}"
    model_paths = [tmp_path / "glados.onnx", tmp_path / "glados.onnx.json"]
    for path in model_paths:
        path.write_bytes(b"0" * 2048)

    body = b"1" * 4096

    def fake_urlopen(request_or_url, *_args, **_kwargs):
        response = MagicMock()
        response.__enter__ = lambda self: self
        response.__exit__ = lambda self, *exc: False
        response.headers = {"Content-Length": str(len(body))}
        if getattr(request_or_url, "get_method", lambda: "GET")() == "HEAD":
            return response
        response.read = MagicMock(side_effect=[body, b""])
        return response

    with (
        patch("download.is_valid_file", return_value=True),
        patch("download.urlopen", side_effect=fake_urlopen),
    ):
        ensure_model_exists(tmp_path, base_url)

    for path in model_paths:
        assert path.read_bytes() == body


# ================================================================
# ensure_model_exists — invalid then valid (download occurs)
# ================================================================
def test_ensure_model_exists_removes_invalid_and_downloads(tmp_path):
    base_url = "https://example.com/{file}"

    bad_file = tmp_path / "glados.onnx"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_bytes(b"bad")

    def fake_is_valid_file(path, _expected):
        if path.name == "glados.onnx":
            fake_is_valid_file.count += 1
            return fake_is_valid_file.count > 1
        return True

    fake_is_valid_file.count = 0

    def fake_copy(src, dst):
        dst.write(src.read())

    with (
        patch("download.is_valid_file", side_effect=fake_is_valid_file),
        patch("download.urlopen") as urlopen_mock,
        patch("shutil.copyfileobj", side_effect=fake_copy),
    ):
        urlopen_mock.return_value = MagicMock(
            __enter__=lambda s: io.BytesIO(b"x" * 4096),
            __exit__=lambda *exc: False,
        )

        ensure_model_exists(tmp_path, base_url)

        assert urlopen_mock.call_count == 1
        assert bad_file.exists()
        assert bad_file.stat().st_size >= 1024


# ================================================================
# ensure_model_exists — EXCEPTION branch (line 130)
# ================================================================
def test_ensure_model_exists_download_exception_hits_except(
    tmp_path, monkeypatch, caplog
):
    """
    DIRECTLY exercises the `except Exception:` block.
    This covers line 130 and its cleanup behavior.
    """

    file_path = tmp_path / "glados.onnx"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("partial")

    # Only fail for glados-new.pt
    def fake_is_valid(path, _md5):
        return path.name != "glados.onnx"

    monkeypatch.setattr(download, "is_valid_file", fake_is_valid)

    # Force exception inside urlopen → triggers except block
    monkeypatch.setattr(
        download, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(Exception("boom"))
    )

    monkeypatch.setattr(download, "get_file_hash", lambda *a: "ignored")

    with caplog.at_level(logging.ERROR):
        ensure_model_exists(tmp_path, download.DEFAULT_URL)

    assert "Failed to download" in caplog.text
    assert "boom" in caplog.text
    assert not file_path.exists()


# ================================================================
# ensure_model_exists — MD5 mismatch AFTER download (lines 134–156)
# ================================================================
def test_ensure_model_exists_md5_mismatch_after_download(tmp_path, monkeypatch, caplog):
    """
    Exercises the post-download MD5 mismatch error branch.
    This covers lines 134–156.
    """

    file_path = tmp_path / "glados.onnx"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # First model invalid → triggers download
    def fake_is_valid(path, _md5):
        return path.name != "glados.onnx"

    monkeypatch.setattr(download, "is_valid_file", fake_is_valid)

    # Download succeeds
    monkeypatch.setattr(
        download,
        "urlopen",
        lambda *a, **kw: MagicMock(
            __enter__=lambda s: io.BytesIO(b"x" * 4096),
            __exit__=lambda *exc: False,
        ),
    )

    # Force MD5 mismatch after download
    monkeypatch.setattr(download, "get_file_hash", lambda *a: "WRONGHASH")

    monkeypatch.setattr(
        download.shutil, "copyfileobj", lambda src, dst: dst.write(src.read())
    )

    with caplog.at_level(logging.ERROR):
        ensure_model_exists(tmp_path, download.DEFAULT_URL)

    assert "MD5 hash mismatch after download" in caplog.text
    assert not file_path.exists()


# ================================================================
# Failure must be reportable: __main__ runs this with check=True
# ================================================================
def test_returns_false_when_a_download_fails(tmp_path, monkeypatch):
    """A failed fetch must be visible to the caller, not swallowed.

    __main__ runs download.py as a subprocess with check=True and logs the
    failure. That branch was unreachable while every error was caught here and
    the process still exited 0.
    """
    monkeypatch.setattr(download, "is_valid_file", lambda *_a: False)
    monkeypatch.setattr(
        download, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(OSError("no net"))
    )
    assert ensure_model_exists(tmp_path, download.DEFAULT_URL) is False


def test_returns_true_when_every_file_is_present(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "is_valid_file", lambda *_a: True)
    assert ensure_model_exists(tmp_path, download.DEFAULT_URL) is True


def test_download_is_atomic_via_a_sidecar(tmp_path, monkeypatch):
    """Nothing may appear at the final path until the transfer completes.

    Both model files carry md5=None, so is_valid_file's only real gate is
    "bigger than 1024 bytes". Writing in place meant a process killed midway
    left a truncated model that satisfied that gate on every later run and was
    never re-fetched. Asserting mid-transfer state is what distinguishes the
    sidecar from an in-place write; an exception-based test does not, because
    the except handler unlinks either way.
    """
    seen = []

    class Observing(io.RawIOBase):
        def readinto(self, buffer):
            seen.append(
                {
                    "finals": sorted(
                        q.name
                        for q in tmp_path.glob("glados.onnx*")
                        if not q.name.endswith(".part")
                    ),
                    "parts": sorted(q.name for q in tmp_path.glob("*.part")),
                }
            )
            return 0  # EOF, ending the copy

    monkeypatch.setattr(download, "is_valid_file", lambda *_a: False)
    monkeypatch.setattr(
        download, "urlopen", lambda *a, **kw: contextlib.closing(Observing())
    )

    ensure_model_exists(tmp_path, download.DEFAULT_URL)

    # Both files are fetched, and no final path is written during EITHER
    # transfer. The commit is now a set operation: the sidecars are renamed
    # only once every file has arrived, so both .part files coexist during the
    # second transfer rather than the first having already been committed.
    # That is what stops a failed .onnx and a successful .onnx.json from
    # leaving a graph and a config that do not belong to each other.
    assert len(seen) == 2
    assert seen[0]["finals"] == [], "final path was written in place"
    assert seen[1]["finals"] == [], "a file was committed before the set was complete"
    assert seen[0]["parts"] == ["glados.onnx.part"]
    assert seen[1]["parts"] == ["glados.onnx.json.part", "glados.onnx.part"]


def test_default_url_is_pinned_to_a_voice_tag_not_latest():
    """The voice must not be fetched from releases/latest.

    release.yaml attaches only release.zip, and cannot build the 73 MB voice
    because it is exported from a training checkpoint that is not in the repo.
    Pointing at `latest` therefore means the first ordinary code release after
    a voice release becomes latest with no glados.onnx attached, and every
    fresh deployment 404s on the voice.
    """
    assert "/releases/latest/" not in download.DEFAULT_URL
    assert f"/releases/download/{download.VOICE_RELEASE}/" in download.DEFAULT_URL
    assert download.DEFAULT_URL.format(file="glados.onnx").endswith("/glados.onnx")


class TestVoiceChecksums:
    """The published hashes must apply to the published voice, and only it."""

    def test_default_voice_on_the_default_url_gets_checksums(self):
        # Without these a VOICE_RELEASE bump is silently inert: the ONNX graph
        # is the same shape at every epoch, so exports of different
        # checkpoints are byte-identical in LENGTH and the size fallback in
        # _classify() can never see a retrained voice.
        sums = download._voice_checksums(
            download.DEFAULT_VOICE_NAME, download.DEFAULT_URL
        )
        assert sums["glados.onnx"] == download.VOICE_CHECKSUMS["glados.onnx"]
        assert sums["glados.onnx.json"]

    def test_a_different_voice_gets_none(self):
        assert download._voice_checksums("othervoice", download.DEFAULT_URL) == {}

    def test_a_different_url_gets_none(self):
        # A mirror or a locally served export is a supported setup; checking it
        # against glados' hashes would fail a good file, and since a mismatch
        # means re-download, that is an unbreakable loop rather than an error.
        assert (
            download._voice_checksums(
                download.DEFAULT_VOICE_NAME, "https://example.com/{file}"
            )
            == {}
        )

    def test_returns_a_copy_so_callers_cannot_mutate_the_table(self):
        sums = download._voice_checksums(
            download.DEFAULT_VOICE_NAME, download.DEFAULT_URL
        )
        sums["glados.onnx"] = "tampered"
        assert download.VOICE_CHECKSUMS["glados.onnx"] != "tampered"
