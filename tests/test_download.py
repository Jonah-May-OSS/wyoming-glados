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

        assert urlopen_mock.call_count == 6
        assert copy_mock.call_count == 6


# ================================================================
# ensure_model_exists — skip valid files
# ================================================================
def test_ensure_model_exists_skips_valid_files(tmp_path):
    base_url = "https://example.com/{file}"

    model_paths = [
        tmp_path / "glados-new.pt",
        tmp_path / "tacotron-trt.ts",
        tmp_path / "en_us_cmudict_ipa_forward.pt",
        tmp_path / "emb/glados_p2.pt",
        tmp_path / "vocoder-gpu.pt",
        tmp_path / "vocoder-trt.ts",
    ]

    for p in model_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"0" * 2048)

    with (
        patch("download.is_valid_file", return_value=True),
        patch("download.urlopen") as urlopen_mock,
    ):
        ensure_model_exists(tmp_path, base_url)
        urlopen_mock.assert_not_called()


# ================================================================
# ensure_model_exists — invalid then valid (download occurs)
# ================================================================
def test_ensure_model_exists_removes_invalid_and_downloads(tmp_path):
    base_url = "https://example.com/{file}"

    bad_file = tmp_path / "glados-new.pt"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_bytes(b"bad")

    def fake_is_valid_file(path, expected):
        if path.name == "glados-new.pt":
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

    file_path = tmp_path / "glados-new.pt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("partial")

    # Only fail for glados-new.pt
    def fake_is_valid(path, md5):
        return path.name != "glados-new.pt"

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

    file_path = tmp_path / "glados-new.pt"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # First model invalid → triggers download
    def fake_is_valid(path, md5):
        return path.name != "glados-new.pt"

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
