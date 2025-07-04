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
LINE_ASTERICKS = re.compile(r"(?<=^|\n)\s*\*+")


class SentenceBoundaryDetector:
    def __init__(self) -> None:
        self.remaining_text = ""
        self.current_sentence = ""

    def add_chunk(self, chunk: str) -> Iterable[str]:
        """Add a new chunk of text and yield sentences."""
        self.remaining_text += chunk
        processed_chunks = []  # Track chunks that have been processed

        while self.remaining_text:
            match = SENTENCE_BOUNDARY_RE.search(self.remaining_text)
            if not match:
                break
            match_text = match.group(0)

            # If it's the first sentence or continuation without abbreviation

            if not self.current_sentence:
                self.current_sentence = match_text
            elif ABBREVIATION_RE.search(self.current_sentence[-5:]):
                self.current_sentence += match_text
            else:
                # Yield and clear the current sentence

                yield remove_asterisks(self.current_sentence.strip())
                processed_chunks.append(self.current_sentence)  # Add processed chunk
                self.current_sentence = match_text
            if not ABBREVIATION_RE.search(self.current_sentence[-5:]):
                yield remove_asterisks(self.current_sentence.strip())
                processed_chunks.append(self.current_sentence)  # Add processed chunk
                self.current_sentence = ""
            # Update the remaining text

            self.remaining_text = self.remaining_text[match.end() :]
        # Clear processed chunks

        return processed_chunks

    def finish(self) -> str:
        """Finalize and return the last sentence, clearing state."""
        text = (self.current_sentence + self.remaining_text).strip()
        self.remaining_text = ""
        self.current_sentence = ""
        return remove_asterisks(text)


def remove_asterisks(text: str) -> str:
    """Remove *asterisks* surrounding **words**"""
    text = WORD_ASTERISKS.sub(r"\1", text)
    text = LINE_ASTERICKS.sub("", text)
    return text
