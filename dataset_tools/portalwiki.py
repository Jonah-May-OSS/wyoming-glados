"""Parse GLaDOS voice-line listings from theportalwiki.com.

Each wiki page renders one voice line per ``<li>``, carrying both a
hand-written transcript and a link to the source ``.wav``::

    <li>"<i>Transcript text.</i>" | ... <a href=".../Name.wav">Download</a></li>

That pairing is why this project needs no ASR pass. The wiki transcripts are
written by hand and also cover unused/cut lines, which have no closed-caption
entry in the shipped games and so cannot be recovered from the game files.

The wiki serves 16-bit PCM at 44.1 kHz (mono for the Portal 1 era, stereo for
much of Portal 2), i.e. original quality rather than lossy re-encodes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

WIKI_BASE = "https://theportalwiki.com/wiki/"

# Wiki page slugs, keyed by the short page name used throughout this package.
PAGES: dict[str, str] = {
    "portal1": "GLaDOS_voice_lines_(Portal)",
    "portal2": "GLaDOS_voice_lines_(Portal_2)",
    "coop": "GLaDOS_voice_lines_(Cooperative_Testing_Initiative)",
    "other": "GLaDOS_voice_lines_(Other)",
}

# Speaker IDs for the multi-speaker VITS model.
#
# Co-op lines merge into `p2`: they are the same game, cast and recording
# session as single-player Portal 2, and the corpus is small enough that
# splitting them would starve the main voice of data.
#
# Dota 2 gets its own ID rather than merging into `p2`. It is a much later
# session with different post-processing, so folding it into a Portal speaker
# would blur the voice; as a separate ID it stays available without cost.
SPEAKER_P1 = "p1"
SPEAKER_P2 = "p2"
SPEAKER_DOTA2 = "dota2"

# `Other` page sections whose lines belong to a speaker other than the page
# default. Matched case-insensitively against the enclosing <h2>.
_OTHER_SECTION_SPEAKERS: dict[str, str] = {
    "portal 1 unused/alternate lines": SPEAKER_P1,
    "dota 2": SPEAKER_DOTA2,
}

_PAGE_SPEAKERS: dict[str, str] = {
    "portal1": SPEAKER_P1,
    "portal2": SPEAKER_P2,
    "coop": SPEAKER_P2,
    "other": SPEAKER_P2,
}


def is_audio_url(href: str) -> bool:
    """Report whether this is an absolute URL that points at a .wav.

    The wiki renders missing files as redlinks to Special:Upload, whose query
    string carries the intended filename::

        /w/index.php?title=Special:Upload&wpDestFile=GLaDOS_foo.wav

    A naive endswith(".wav") matches that, yielding a relative URL that no
    fetcher can resolve and a line whose audio does not exist at all.
    """
    if not href.lower().startswith(("http://", "https://")):
        return False
    path = href.split("?", 1)[0].split("#", 1)[0]
    return path.lower().endswith(".wav")


_WS_RE = re.compile(r"\s+")

# <h2> is the section heading; <h3> the subsection beneath it.
_H2_LEVEL = 2
# Wiki annotations such as "[sic]" or "[laughs]" describe the audio rather
# than appearing in it, so they must not reach the phonemizer.
_ANNOTATION_RE = re.compile(r"\[[^\]]*\]")
# Parentheses are used the same way, for editorial notes and sound effects:
# "(page flip)", "(beep beep beep)", "(The last part is cut off by the level
# transition.)", and one transcript that is nothing but "(subtitled as ...)".
_PAREN_RE = re.compile(r"\([^)]*\)")

# Annotations that comment on the TEXT rather than describe a sound. These are
# safe to strip and keep the clip; everything else means the audio contains
# something the transcript no longer accounts for.
_TEXT_ONLY_ANNOTATIONS = frozenset({"sic", "citation needed", "unused"})


def audio_annotations(raw: str) -> list[str]:
    """Annotations in `raw` that describe SOUND, not text.

    A stripped "[train horn]" or "(phone ringing)" leaves the transcript clean
    while the audio still contains the effect, so the model is taught that a
    burst of non-speech belongs to the surrounding words. The same goes for
    "[gentle laughter]" and "[hums 'For He's A Jolly Good Fellow']". Callers
    use this to drop those clips rather than silently mistrain on them.
    """
    found = _ANNOTATION_RE.findall(raw) + _PAREN_RE.findall(raw)
    return [
        note
        for note in found
        if note.strip("[]() ").strip().lower() not in _TEXT_ONLY_ANNOTATIONS
    ]


# Stripping variants that also swallow the dashes framing an annotation. The
# wiki writes "...loud noises--[train horn]--", and removing only the brackets
# strands "-- --" in the middle of the line. Dashes NOT adjacent to an
# annotation are left alone: they mark GLaDOS interrupting herself ("finally
# be ba--"), which is real speech the model should learn.
# Quote characters that can delimit a spoken line on the wiki. Straight
# quotes are the convention; the curly pair appears occasionally, and
# treating one as ordinary text would silently drop a real line.
# A frozenset, deliberately, not a string. `"" in "\"“”"` is True - every
# string contains the empty string - so a run with no character before or
# after it passed the quote test vacuously, and stage directions that open
# a <li> were kept as transcripts.
_QUOTE_CHARS = frozenset('"“”')

# Inner quotation marks, skipped when looking for the delimiter above. GLaDOS
# quotes things constantly, and the wiki nests them:
#
#     <li>"'<i>Shall not be mourned.' That's exactly what it says.</i>" ...
#
# The character right before the run is then the inner ‘, not the outer ",
# so testing only the adjacent character discarded three real lines.
_INNER_QUOTE_CHARS = frozenset("'‘’")


def _delimiter_before(text: str) -> str:
    """Last meaningful character before an italic run, inner quotes skipped."""
    stripped = text.rstrip()
    while stripped and stripped[-1] in _INNER_QUOTE_CHARS:
        stripped = stripped[:-1].rstrip()
    return stripped[-1] if stripped else ""


def _delimiter_after(text: str) -> str:
    """First meaningful character after an italic run, inner quotes skipped."""
    stripped = text.lstrip()
    while stripped and stripped[0] in _INNER_QUOTE_CHARS:
        stripped = stripped[1:].lstrip()
    return stripped[0] if stripped else ""


_ANNOTATION_STRIP_RE = re.compile(r"-*\[[^\]]*\]-*")
_PAREN_STRIP_RE = re.compile(r"-*\([^)]*\)-*")


def clean_transcript(raw: str) -> str:
    """Normalize a wiki transcript into a training-ready line."""
    text = _PAREN_STRIP_RE.sub(" ", _ANNOTATION_STRIP_RE.sub(" ", raw))
    text = _WS_RE.sub(" ", text).strip()
    return text.strip('"').strip()


@dataclass(frozen=True)
class VoiceLine:
    """One voice line: its audio URL, transcript and wiki provenance."""

    url: str
    transcript: str
    page: str
    section: str
    subsection: str
    # True when the raw wiki transcript described a sound - see
    # audio_annotations(). The transcript below has been cleaned, but the AUDIO
    # still contains whatever it described, so build drops these.
    has_audio_annotation: bool = False

    @property
    def filename(self) -> str:
        """Basename of the source .wav, unique across the whole wiki."""
        return self.url.rsplit("/", 1)[-1]

    @property
    def speaker(self) -> str:
        """Speaker ID this line trains under."""
        override = _OTHER_SECTION_SPEAKERS.get(self.section.strip().lower())
        if self.page == "other" and override is not None:
            return override
        return _PAGE_SPEAKERS[self.page]


@dataclass
class _Frame:
    """Capture state for one <li>, which may nest inside another."""

    transcript: list[str]
    url: str
    i_depth: int
    # Text seen OUTSIDE any <i>, used to decide whether an italic run is a
    # spoken line or an editorial aside - see _VoiceLineParser.
    outside: list[str] = field(default_factory=list)
    # Italic runs closed so far, each with whether it was quote-delimited.
    runs: list[tuple[str, bool]] = field(default_factory=list)
    # The character immediately before the italic run currently open.
    opened_after: str = ""
    # Index of a run whose closing quote has not been looked for yet.
    awaiting_close: int = -1


class _VoiceLineParser(HTMLParser):
    """Collect (transcript, wav URL) pairs, tracking the enclosing headings.

    Headings are tracked at two levels because the `Other` page nests chapter
    subsections under a game-level <h2>; the speaker mapping keys off that
    <h2>, so flattening the two would lose it.

    List items are tracked as a stack rather than a depth counter: the wiki
    groups alternate takes of a line into a nested <ul>, and each nested <li>
    is a real voice line that must be emitted with its own transcript and
    audio, not folded into its parent.
    """

    def __init__(self, page: str) -> None:
        super().__init__(convert_charrefs=True)
        self.page = page
        self.lines: list[VoiceLine] = []
        self._h2 = ""
        self._h3 = ""
        self._heading_level = 0
        self._heading_buf: list[str] = []
        self._frames: list[_Frame] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("h2", "h3"):
            self._heading_level = int(tag[1])
            self._heading_buf = []
        elif tag == "li":
            self._frames.append(_Frame(transcript=[], url="", i_depth=0))
        elif self._frames:
            frame = self._frames[-1]
            if tag == "i":
                if frame.i_depth == 0:
                    # Remember the character this run opens after, so a
                    # closing </i> can decide whether it was quoted.
                    frame.opened_after = _delimiter_before("".join(frame.outside))
                frame.i_depth += 1
            elif tag == "a" and not frame.url:
                href = dict(attrs).get("href") or ""
                if is_audio_url(href):
                    frame.url = href

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h2", "h3") and self._heading_level:
            text = _WS_RE.sub(" ", "".join(self._heading_buf)).strip()
            if self._heading_level == _H2_LEVEL:
                self._h2, self._h3 = text, ""
            else:
                self._h3 = text
            self._heading_level = 0
        elif tag == "i" and self._frames and self._frames[-1].i_depth:
            frame = self._frames[-1]
            frame.i_depth -= 1
            if frame.i_depth == 0:
                frame.runs.append(("".join(frame.transcript), False))
                frame.transcript = []
                # The closing quote sits in the text after </i>, which has not
                # been seen yet.
                frame.awaiting_close = len(frame.runs) - 1
        elif tag == "li" and self._frames:
            self._emit(self._frames.pop())

    def handle_data(self, data: str) -> None:
        if self._heading_level:
            self._heading_buf.append(data)
        elif self._frames and self._frames[-1].i_depth:
            self._frames[-1].transcript.append(data)
        elif self._frames:
            frame = self._frames[-1]
            if frame.awaiting_close >= 0:
                closed_by = _delimiter_after(data)
                text, _ = frame.runs[frame.awaiting_close]
                quoted = (
                    frame.opened_after in _QUOTE_CHARS and closed_by in _QUOTE_CHARS
                )
                frame.runs[frame.awaiting_close] = (text, quoted)
                frame.awaiting_close = -1
            frame.outside.append(data)

    def _emit(self, frame: _Frame) -> None:
        """Emit the spoken line from one <li>, if it has one.

        A <li> can hold more than one italic run, and only the first is the
        line. The wiki writes editorial asides in italics too, immediately
        after the transcript and inside the same <li>:

            <li>"<i>...you'll be dead.</i>" ... <i>"with the sphere, cycle
            through these:"</i>)</li>

        Concatenating every run appended that instruction to the transcript,
        so the model was trained to read stage directions aloud. Some entries
        are an aside and nothing else, with the audio being a sound effect:

            <li><i>Upon destroying of the last three cores, this will
            sound:</i> <a ...>"*scream*"</a></li>

        That one paired a death scream with a sentence GLaDOS never says.

        The discriminator is the wiki's own convention: a spoken line is
        wrapped in quotes that sit OUTSIDE the <i>, an aside is not. Note the
        aside above contains quotes of its own, so the test has to be on the
        delimiters around the run rather than on its content.
        """
        quoted = [text for text, is_quoted in frame.runs if is_quoted]
        if not quoted:
            return
        raw = quoted[0]
        transcript = clean_transcript(raw)
        if transcript and frame.url:
            self.lines.append(
                VoiceLine(
                    url=frame.url,
                    transcript=transcript,
                    page=self.page,
                    section=self._h2,
                    subsection=self._h3,
                    has_audio_annotation=bool(audio_annotations(raw)),
                )
            )


def parse_page(html: str, page: str) -> list[VoiceLine]:
    """Extract every transcribed voice line from one wiki page.

    Lines without a transcript (pure sound effects) or without audio are
    skipped, and repeated links to the same .wav collapse to one entry.
    """
    if page not in PAGES:
        raise ValueError(f"Unknown page {page!r}; expected one of {sorted(PAGES)}")
    parser = _VoiceLineParser(page)
    parser.feed(html)
    parser.close()

    seen: set[str] = set()
    unique: list[VoiceLine] = []
    for line in parser.lines:
        if line.url in seen:
            continue
        seen.add(line.url)
        unique.append(line)
    return unique
