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

        while True:
            match = SENTENCE_BOUNDARY_RE.search(self.remaining_text)
            if not match:
                break

            sentence_portion = self.remaining_text[: match.end()]
            self.current_sentence += sentence_portion

            if not ABBREVIATION_RE.search(self.current_sentence[-5:]):
                yield remove_asterisks(self.current_sentence.strip())
                self.current_sentence = ""

            self.remaining_text = self.remaining_text[match.end() :].lstrip()

    def finish(self) -> str:
        """
        Finalize and return *all remaining text*, clearing state.

        NOTE:
        Updated to match the updated test expectations:
        - Always return the leftover text.
        - Do NOT require regex boundaries.
        - This matches real streaming NLP behavior where
          partial or final sentences appear at flush time.
        """
        combined = self.current_sentence + self.remaining_text

        # Clear state
        self.current_sentence = ""
        self.remaining_text = ""

        combined = combined.strip()
        if combined:
            return remove_asterisks(combined)

        return ""


def remove_asterisks(text: str) -> str:
    """Remove *asterisks* surrounding **words**."""
    text = WORD_ASTERISKS.sub(r"\1", text)
    text = LINE_ASTERISKS.sub("", text)
    return text
