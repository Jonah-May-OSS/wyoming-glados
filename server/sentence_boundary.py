"""Guess sentence and clause boundaries in streamed text."""

from collections.abc import Iterable

import regex as re

SENTENCE_END = r"[.!?…]|[。！？]|[؟]|[।॥]"
CLAUSE_BREAK = r"[,;:，、؛：]"
BOUNDARY_RE = re.compile(
    rf"(?P<strong>{SENTENCE_END}+)|(?P<clause>{CLAUSE_BREAK})(?=\s)"
)
ABBREVIATION_RE = re.compile(
    r"(?:(?:\b\p{L}\.){1,4}|\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.)$",
    re.IGNORECASE | re.UNICODE,
)
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
                if self._is_abbreviation(candidate) or self._is_decimal(
                    candidate, trailing
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
        stripped_candidate = candidate.rstrip()
        return bool(
            re.search(r"\d\.$", stripped_candidate)
            and re.match(r"\d", trailing.lstrip())
        )


def remove_asterisks(text: str) -> str:
    """Remove *asterisks* surrounding **words**"""
    text = WORD_ASTERISKS.sub(r"\1", text)
    text = LINE_ASTERISKS.sub("", text)
    return text
