"""Guess sentence and clause boundaries in streamed text."""

from collections.abc import Iterable

import regex as re

# Kept as one non-capturing group so the `+` in BOUNDARY_RE applies to the whole
# alternation. Without it, `+` bound only to the last branch and a run like
# "..." matched one period at a time, emitting bare "." segments.
SENTENCE_END = r"(?:[.!?…]|[。！？]|[؟]|[।॥])"
CLAUSE_BREAK = r"[,;:，、؛：]"
BOUNDARY_RE = re.compile(
    rf"(?P<strong>{SENTENCE_END}+)|(?P<clause>{CLAUSE_BREAK})(?=\s)"
)
ABBREVIATION_RE = re.compile(
    r"(?:(?:\b\p{L}\.){1,4}|\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.)$",
    re.IGNORECASE | re.UNICODE,
)
# A run of dots or the ellipsis character ("...", "…"), as opposed to a single
# sentence-ending period.
ELLIPSIS_RE = re.compile(r"[.…]{2,}|…")
# What the text after an ellipsis looks like when it starts a new sentence:
# a capital, possibly behind an opening quote or bracket.
SENTENCE_START_RE = re.compile(r"[\"'“‘«(\[]*[\p{Lu}\p{Lt}]", re.UNICODE)
WORD_RE = re.compile(r"\p{L}[\p{L}\p{M}'’-]*|\d+", re.UNICODE)
WORD_ASTERISKS = re.compile(r"\*+([^\*]+)\*+")
LINE_ASTERISKS = re.compile(r"(?<=^|\n)\s*\*+")
MIN_CLAUSE_WORDS = 6


class SentenceBoundaryDetector:
    """Detect sentence or long-clause boundaries in streamed text."""

    def __init__(self, min_clause_words: int = MIN_CLAUSE_WORDS) -> None:
        self.remaining_text = ""
        self.min_clause_words = min_clause_words

    def add_chunk(self, chunk: str) -> Iterable[str]:
        """Add a new chunk of text and yield stable sentence/clause segments."""
        self.remaining_text += chunk

        while self.remaining_text:
            segment = self._get_next_segment()
            if segment is None:
                break

            yield remove_asterisks(segment)

    def finish(self) -> str:
        """Finalize and return the last sentence, clearing state."""
        text = self.remaining_text.strip()
        self.remaining_text = ""
        return remove_asterisks(text)

    def _get_next_segment(self) -> str | None:
        for match in BOUNDARY_RE.finditer(self.remaining_text):
            boundary_index = match.end()
            candidate = self.remaining_text[:boundary_index]
            trailing = self.remaining_text[boundary_index:]

            if match.lastgroup == "strong":
                if (
                    self._is_abbreviation(candidate)
                    or self._is_decimal(candidate, trailing)
                    or self._is_pause_ellipsis(match.group("strong"), trailing)
                ):
                    continue

                segment = candidate.strip()
                if not segment:
                    continue

                self.remaining_text = trailing.lstrip()
                return segment

            if self._count_words(candidate) < self.min_clause_words:
                continue

            if not self._has_trailing_words(trailing):
                continue

            segment = candidate.strip()
            if not segment:
                continue

            self.remaining_text = trailing.lstrip()
            return segment

        return None

    @staticmethod
    def _count_words(text: str) -> int:
        return len(WORD_RE.findall(text))

    @staticmethod
    def _has_trailing_words(text: str) -> bool:
        return bool(WORD_RE.search(text))

    @staticmethod
    def _is_abbreviation(candidate: str) -> bool:
        return bool(ABBREVIATION_RE.search(candidate.strip()))

    @staticmethod
    def _is_decimal(candidate: str, trailing: str) -> bool:
        """A digit before the period means a decimal, or that we cannot tell yet.

        While streaming, the fractional digits may not have arrived; treat a
        trailing "3." as a decimal so the segment stays buffered until the next
        chunk decides it.
        """
        stripped_candidate = candidate.rstrip()
        if not re.search(r"\d\.$", stripped_candidate):
            return False
        rest = trailing.lstrip()
        return not rest or bool(re.match(r"\d", rest))

    @staticmethod
    def _is_pause_ellipsis(boundary: str, trailing: str) -> bool:
        """An ellipsis is a dramatic pause unless a new sentence clearly follows.

        GLaDOS leans on "..." mid-sentence ("something more... educational.").
        Splitting there hands the vocoder the tail as its own stub utterance,
        which comes out clipped, so keep the pause inside the sentence. A
        following capital does start a new sentence and still splits. While
        streaming the tail may not have arrived yet; as with a trailing decimal
        point, treat the undecidable case as a pause and let the next chunk --
        or finish() -- settle it.
        """
        if not ELLIPSIS_RE.fullmatch(boundary):
            return False
        rest = trailing.lstrip()
        if not rest:
            return True
        return not SENTENCE_START_RE.match(rest)


def remove_asterisks(text: str) -> str:
    """Remove *asterisks* surrounding **words**"""
    text = WORD_ASTERISKS.sub(r"\1", text)
    text = LINE_ASTERISKS.sub("", text)
    return text
