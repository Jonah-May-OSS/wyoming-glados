"""Polite, resumable fetching of wiki pages and voice-line audio.

Fetching the corpus means ~1,800 requests to a community-run wiki, so every
call is rate limited, retried with backoff, and cached to disk. Re-running
after an interruption re-downloads only what is missing.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .portalwiki import PAGES, WIKI_BASE, VoiceLine, parse_page

USER_AGENT = (
    "wyoming-glados-dataset/0.1 (+https://github.com/Jonah-May-OSS/wyoming-glados)"
)

# Seconds between requests. The wiki is community hosted and this crawl is a
# one-off, so it is deliberately unhurried.
DEFAULT_DELAY = 0.25
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 30.0

# A RIFF/WAVE header is 44 bytes; anything at or below that has no audio.
_WAV_HEADER_BYTES = 44

Opener = Callable[[str], bytes]


class FetchError(RuntimeError):
    """A URL could not be retrieved after exhausting retries."""


@dataclass
class FetchReport:
    """Outcome of a batch fetch."""

    downloaded: int = 0
    skipped: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing failed."""
        return not self.failures


# Hosts this crawler is allowed to talk to.
#
# Every URL it fetches comes from parsing wiki HTML, so the target set is
# attacker-influenced: anyone who can edit a page can point a link at
# 127.0.0.1, at a cloud metadata endpoint, or at any host reachable from
# whoever runs the crawl. Restricting the destination is what stops an edit
# from turning this into a request forgery.
#
# The media hosts are not the wiki host. Every .wav on the four GLaDOS pages
# is served from i1.theportalwiki.net (i2 appears alongside it for other
# assets), while only the pages themselves come from theportalwiki.com. The
# first version of this list held theportalwiki.com plus two hosts that appear
# nowhere in the corpus, so it would have refused all 1001 audio URLs on the
# Portal 2 page alone - the allowlist was written from what the crawler seemed
# like it should fetch rather than from what the pages actually link to.
ALLOWED_HOSTS = frozenset(
    {
        "theportalwiki.com",
        "i1.theportalwiki.net",
        "i2.theportalwiki.net",
    }
)

# Ceiling on a single download. The corpus is ~1,800 short voice lines; the
# largest is a few hundred KB. Without a cap, response.read() sizes the buffer
# from whatever the server sends, so one hostile or broken URL can exhaust
# memory before anything gets a chance to reject it.
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024


def _check_url(url: str) -> str:
    """Return `url` if it targets an allowed HTTPS host, else raise.

    Applied to redirects as well as the initial request: an allowed host that
    answers with a 302 to somewhere else would otherwise walk straight past a
    check done only at the start.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        raise FetchError(f"{url}: refusing a non-HTTPS URL")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise FetchError(f"{url}: host {host!r} is not in ALLOWED_HOSTS")
    return url


class _AllowlistRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-check every redirect target against the allowlist."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Reject a redirect that leaves the allowlist, then defer upstream."""
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http_opener(
    *, user_agent: str = USER_AGENT, timeout: float = DEFAULT_TIMEOUT
) -> Opener:
    """Build an opener that issues real HTTP requests."""
    opener = urllib.request.build_opener(_AllowlistRedirectHandler)

    def _open(url: str) -> bytes:
        request = urllib.request.Request(
            _check_url(url), headers={"User-Agent": user_agent}
        )
        with opener.open(request, timeout=timeout) as response:
            # One byte past the cap, so a body sitting exactly on the limit is
            # accepted and anything larger is detectable without reading it
            # all.
            body = bytes(response.read(MAX_DOWNLOAD_BYTES + 1))
        if len(body) > MAX_DOWNLOAD_BYTES:
            raise FetchError(f"{url}: response exceeds {MAX_DOWNLOAD_BYTES} bytes")
        return body

    return _open


def _sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def fetch_with_retries(
    url: str,
    opener: Opener,
    *,
    retries: int = DEFAULT_RETRIES,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = _sleep,
) -> bytes:
    """Fetch a URL, retrying transient failures with exponential backoff."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return opener(url)
        except ValueError as exc:
            # A malformed URL will never succeed; do not burn retries on it.
            raise FetchError(f"{url}: {exc}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            if attempt < retries - 1:
                sleep(backoff * (2**attempt))
    raise FetchError(f"{url}: {last}") from last


def safe_filename(url: str) -> str:
    """Local filename for an audio URL, percent-decoded and path-safe."""
    name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    return "".join("_" if ch in r'\/:*?"<>|' else ch for ch in name)


def is_wav_complete(path: Path) -> bool:
    """Report whether `path` holds a plausible, non-empty RIFF/WAVE file.

    Guards against resuming onto a truncated file from an interrupted run.
    """
    try:
        if path.stat().st_size <= _WAV_HEADER_BYTES:
            return False
        with path.open("rb") as handle:
            header = handle.read(12)
    except OSError:
        return False
    return header[:4] == b"RIFF" and header[8:12] == b"WAVE"


def fetch_pages(
    cache_dir: Path,
    *,
    opener: Opener,
    delay: float = DEFAULT_DELAY,
    sleep: Callable[[float], None] = _sleep,
) -> list[VoiceLine]:
    """Fetch and parse every source page, caching the HTML on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lines: list[VoiceLine] = []
    for page, slug in PAGES.items():
        path = cache_dir / f"{page}.html"
        if not path.exists():
            raw = fetch_with_retries(WIKI_BASE + slug, opener)
            # Write through a .part file, as fetch_audio does. path.exists() is
            # the resume check, so a run interrupted mid-write would otherwise
            # leave a truncated page that every later run treats as complete -
            # silently dropping whatever voice lines were cut off. Renaming
            # into place makes the cached page either absent or whole.
            temp = path.with_suffix(path.suffix + ".part")
            temp.write_bytes(raw)
            temp.replace(path)
            sleep(delay)
        html = path.read_text(encoding="utf-8", errors="replace")
        lines.extend(parse_page(html, page))
    return lines


def find_filename_collisions(
    lines: Sequence[VoiceLine],
) -> dict[str, list[str]]:
    """Map local filenames to URLs where more than one URL claims the name."""
    by_name: dict[str, list[str]] = {}
    for line in lines:
        by_name.setdefault(safe_filename(line.url), []).append(line.url)
    return {name: urls for name, urls in by_name.items() if len(urls) > 1}


def fetch_audio(
    lines: Iterable[VoiceLine],
    audio_dir: Path,
    *,
    opener: Opener,
    delay: float = DEFAULT_DELAY,
    retries: int = DEFAULT_RETRIES,
    sleep: Callable[[float], None] = _sleep,
    on_progress: Callable[[int, int], None] | None = None,
) -> FetchReport:
    """Download each line's audio into `audio_dir`, skipping complete files."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    todo = list(lines)
    report = FetchReport()
    for index, line in enumerate(todo, start=1):
        path = audio_dir / safe_filename(line.url)
        if is_wav_complete(path):
            report.skipped += 1
        else:
            try:
                payload = fetch_with_retries(
                    line.url, opener, retries=retries, sleep=sleep
                )
            except FetchError as exc:
                report.failures.append((line.url, str(exc)))
            else:
                # Write via a temp file so an interrupted run never leaves a
                # truncated .wav that a later run would treat as complete.
                temp = path.with_suffix(path.suffix + ".part")
                temp.write_bytes(payload)
                temp.replace(path)
                report.downloaded += 1
            sleep(delay)
        if on_progress is not None:
            on_progress(index, len(todo))
    return report
