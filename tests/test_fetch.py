"""Tests for the polite/resumable dataset fetcher."""

import urllib.error

import pytest

from dataset_tools.fetch import (
    ALLOWED_HOSTS,
    MAX_DOWNLOAD_BYTES,
    FetchError,
    fetch_audio,
    fetch_pages,
    fetch_with_retries,
    find_filename_collisions,
    http_opener,
    is_wav_complete,
    safe_filename,
)
from dataset_tools.portalwiki import PAGES, VoiceLine

WAV = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 64

_PAGE_HTML = '<ul><li>"<i>Line.</i>" <a href="https://x/a.wav">Download</a></li></ul>'


def _line(url: str = "https://x/a.wav") -> VoiceLine:
    return VoiceLine(url, "Line.", "portal1", "", "")


def _no_sleep(_seconds: float) -> None:
    return None


class TestFetchWithRetries:
    def test_returns_payload(self):
        assert fetch_with_retries("u", lambda _u: b"ok", sleep=_no_sleep) == b"ok"

    def test_retries_then_succeeds(self):
        calls = []

        def opener(url):
            calls.append(url)
            if len(calls) < 3:
                raise urllib.error.URLError("flaky")
            return b"ok"

        assert fetch_with_retries("u", opener, sleep=_no_sleep) == b"ok"
        assert len(calls) == 3

    def test_raises_after_exhausting_retries(self):
        def opener(_url):
            raise urllib.error.URLError("down")

        with pytest.raises(FetchError, match="down"):
            fetch_with_retries("u", opener, retries=2, sleep=_no_sleep)

    def test_backoff_is_exponential(self):
        waits = []

        def opener(_url):
            raise TimeoutError("slow")

        with pytest.raises(FetchError):
            fetch_with_retries("u", opener, retries=4, backoff=1.0, sleep=waits.append)
        assert waits == [1.0, 2.0, 4.0]


class TestSafeFilename:
    def test_takes_basename(self):
        assert safe_filename("https://x/y/GLaDOS_01.wav") == "GLaDOS_01.wav"

    def test_percent_decodes(self):
        assert safe_filename("https://x/It%27s.wav") == "It's.wav"

    def test_replaces_path_unsafe_characters(self):
        assert safe_filename("https://x/a%3Ab%2Ac.wav") == "a_b_c.wav"


class TestIsWavComplete:
    def test_accepts_riff_wave(self, tmp_path):
        path = tmp_path / "a.wav"
        path.write_bytes(WAV)
        assert is_wav_complete(path)

    def test_rejects_missing_file(self, tmp_path):
        assert not is_wav_complete(tmp_path / "nope.wav")

    def test_rejects_header_only_file(self, tmp_path):
        path = tmp_path / "a.wav"
        path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 20)
        assert not is_wav_complete(path)

    def test_rejects_non_riff_payload(self, tmp_path):
        path = tmp_path / "a.wav"
        path.write_bytes(b"<html>error page</html>" + b"\x00" * 64)
        assert not is_wav_complete(path)


class TestFetchPages:
    def test_fetches_and_caches_every_page(self, tmp_path):
        seen = []

        def opener(url):
            seen.append(url)
            return _PAGE_HTML.encode()

        lines = fetch_pages(tmp_path, opener=opener, sleep=_no_sleep)
        assert len(seen) == len(PAGES)
        assert len(lines) == len(PAGES)
        for page in PAGES:
            assert (tmp_path / f"{page}.html").exists()

    def test_second_run_uses_cache(self, tmp_path):
        calls = []

        def opener(url):
            calls.append(url)
            return _PAGE_HTML.encode()

        fetch_pages(tmp_path, opener=opener, sleep=_no_sleep)
        fetch_pages(tmp_path, opener=opener, sleep=_no_sleep)
        assert len(calls) == len(PAGES)


class TestFetchAudio:
    def test_downloads_missing_file(self, tmp_path):
        report = fetch_audio(
            [_line()], tmp_path, opener=lambda _u: WAV, sleep=_no_sleep
        )
        assert report.downloaded == 1
        assert report.ok
        assert (tmp_path / "a.wav").read_bytes() == WAV

    def test_skips_already_complete_file(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(WAV)
        calls = []

        def opener(url):
            calls.append(url)
            return WAV

        report = fetch_audio([_line()], tmp_path, opener=opener, sleep=_no_sleep)
        assert (report.skipped, report.downloaded) == (1, 0)
        assert calls == []

    def test_redownloads_truncated_file(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        report = fetch_audio(
            [_line()], tmp_path, opener=lambda _u: WAV, sleep=_no_sleep
        )
        assert report.downloaded == 1

    def test_records_failures_and_continues(self, tmp_path):
        def opener(url):
            if "bad" in url:
                raise urllib.error.URLError("410")
            return WAV

        lines = [_line("https://x/bad.wav"), _line("https://x/good.wav")]
        report = fetch_audio(lines, tmp_path, opener=opener, retries=1, sleep=_no_sleep)
        assert report.downloaded == 1
        assert not report.ok
        assert len(report.failures) == 1
        assert (tmp_path / "good.wav").exists()

    def test_leaves_no_part_files_behind(self, tmp_path):
        fetch_audio([_line()], tmp_path, opener=lambda _u: WAV, sleep=_no_sleep)
        assert list(tmp_path.glob("*.part")) == []

    def test_reports_progress(self, tmp_path):
        seen = []
        lines = [_line("https://x/a.wav"), _line("https://x/b.wav")]
        fetch_audio(
            lines,
            tmp_path,
            opener=lambda _u: WAV,
            sleep=_no_sleep,
            on_progress=lambda done, total: seen.append((done, total)),
        )
        assert seen == [(1, 2), (2, 2)]


class TestFilenameCollisions:
    def test_none_for_distinct_names(self):
        lines = [_line("https://x/a.wav"), _line("https://y/b.wav")]
        assert find_filename_collisions(lines) == {}

    def test_detects_same_basename_from_different_urls(self):
        lines = [_line("https://x/a.wav"), _line("https://y/a.wav")]
        assert list(find_filename_collisions(lines)) == ["a.wav"]


class TestMalformedUrls:
    def test_value_error_becomes_fetch_error_without_retrying(self):
        calls = []

        def opener(url):
            calls.append(url)
            raise ValueError(f"unknown url type: {url!r}")

        with pytest.raises(FetchError, match="unknown url type"):
            fetch_with_retries("/w/index.php", opener, retries=3, sleep=_no_sleep)
        assert len(calls) == 1

    def test_one_bad_url_does_not_abort_the_batch(self, tmp_path):
        def opener(url):
            if url.startswith("/"):
                raise ValueError("unknown url type")
            return WAV

        lines = [_line("/relative.wav"), _line("https://x/good.wav")]
        report = fetch_audio(lines, tmp_path, opener=opener, sleep=_no_sleep)
        assert report.downloaded == 1
        assert len(report.failures) == 1
        assert (tmp_path / "good.wav").exists()


class TestDownloadDestinationIsRestricted:
    """Every URL fetched comes from wiki HTML, so targets are untrusted.

    Anyone able to edit a page can point a link at localhost, at a cloud
    metadata endpoint, or at anything else reachable from the machine running
    the crawl.
    """

    def test_a_host_outside_the_allowlist_is_refused(self):
        opener = http_opener()
        with pytest.raises(FetchError, match="not in ALLOWED_HOSTS"):
            opener("https://evil.example.com/a.wav")

    def test_loopback_is_refused(self):
        opener = http_opener()
        with pytest.raises(FetchError, match="not in ALLOWED_HOSTS"):
            opener("https://127.0.0.1/a.wav")

    def test_cloud_metadata_endpoint_is_refused(self):
        opener = http_opener()
        with pytest.raises(FetchError, match="not in ALLOWED_HOSTS"):
            opener("https://169.254.169.254/latest/meta-data/")

    def test_plain_http_is_refused_even_on_an_allowed_host(self):
        # Downgrading to http would expose the crawl to tampering in transit,
        # so the scheme is checked before the host.
        opener = http_opener()
        allowed = next(iter(ALLOWED_HOSTS))
        with pytest.raises(FetchError, match="non-HTTPS"):
            opener(f"http://{allowed}/a.wav")

    def test_the_wiki_itself_is_allowed(self):
        assert "theportalwiki.com" in ALLOWED_HOSTS

    def test_the_size_cap_is_smaller_than_available_memory(self):
        # A guard that is effectively unbounded is not a guard.
        assert 0 < MAX_DOWNLOAD_BYTES <= 64 * 1024 * 1024
