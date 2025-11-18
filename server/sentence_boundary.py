"""Guess the sentence boundaries in text."""

from collections.abc import Iterable

import regex as re

SENTENCE_END = r"[.!?…]|[。！？]|[؟]|[।॥]"
ABBREVIATION_RE = re.compile(r"\b\p{L}{1,3}\.$", re.UNICODE)

SENTENCE_BOUNDARY_RE = re.compile(
    rf"(.*?(?:{SENTENCE_END}+))(?=\s+[\p{{Lu}}\p{{Lt}}\p{{Lo}}]|(?:\s+\d+\.\s+))",
    re.DOTALL,
)

WORD_ASTERISKS = re.compile(r"\*+([^\*]+)\*+")
LINE_ASTERISKS = re.compile(r"(?<=^|\n)\s*\*+")


class SentenceBoundaryDetector:
    def __init__(self) -> None:
        self.remaining_text = ""
        self.current_sentence = ""

    def add_chunk(self, chunk: str) -> Iterable[str]:
        """Add a new chunk of text and yield sentences."""
        self.remaining_text += chunk

        while self.remaining_text:
            match = SENTENCE_BOUNDARY_RE.search(self.remaining_text)
            if not match:
                break

            # Extract full matched sentence portion (text up to match end)
            sentence_portion = self.remaining_text[: match.end()]

            # Append to current sentence buffer
            self.current_sentence += sentence_portion

            # Emit full sentence unless it ends in an abbreviation
            if not ABBREVIATION_RE.search(self.current_sentence[-5:]):
                yield remove_asterisks(self.current_sentence.strip())
                self.current_sentence = ""

            # Remove used portion from buffer
            self.remaining_text = self.remaining_text[match.end():].lstrip()

    def finish(self) -> str:
        """Finalize and return the last sentence, clearing state."""
        combined = (self.current_sentence + self.remaining_text)

        # reset state
        self.remaining_text = ""
        self.current_sentence = ""

        # If leftover text contains meaningful characters, return it
        cleaned = combined.strip()
        if cleaned:
            return remove_asterisks(cleaned)

        return ""


def remove_asterisks(text: str) -> str:
    """Remove *asterisks* surrounding **words**"""
    text = WORD_ASTERISKS.sub(r"\1", text)
    text = LINE_ASTERISKS.sub("", text)
    return text
