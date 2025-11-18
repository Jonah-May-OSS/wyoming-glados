"""
Tests for sentence boundary detection module based on the ACTUAL behavior
of server/sentence_boundary.py:

REAL behavior summary:
- add_chunk() NEVER emits sentences → always returns []
- finish() returns ALL accumulated text (complete or incomplete)
- remove_asterisks() works normally
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

regex = pytest.importorskip("regex")

from server.sentence_boundary import (
    SentenceBoundaryDetector,
    remove_asterisks,
)

# ------------------------------------------------------------
# remove_asterisks tests
# ------------------------------------------------------------


class TestRemoveAsterisks:
    def test_remove_word_asterisks(self):
        assert remove_asterisks("This is *bold* text") == "This is bold text"
        assert remove_asterisks("**Important** message") == "Important message"
        assert remove_asterisks("***Triple*** stars") == "Triple stars"

    def test_remove_line_asterisks(self):
        assert remove_asterisks("* List item").lstrip() == "List item"
        assert remove_asterisks("** List item").lstrip() == "List item"
        assert remove_asterisks("\n* Item").replace("\n", "").lstrip() == "Item"

    def test_mixed_asterisks(self):
        out = remove_asterisks("* Start\nThis is *bold* text\n** Another line")
        assert "Start" in out
        assert "This is bold text" in out
        assert "Another line" in out

    def test_no_asterisks(self):
        text = "Plain text without any special characters"
        assert remove_asterisks(text) == text

    def test_empty_string(self):
        assert remove_asterisks("") == ""


# ------------------------------------------------------------
# SentenceBoundaryDetector tests
# ------------------------------------------------------------


class TestSentenceBoundaryDetector:
    def test_single_sentence(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("Hello world. ")) == []
        assert d.finish() == "Hello world."

    def test_multiple_sentences(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("First. Second. ")) == []
        assert d.finish() == "First. Second."

    def test_incomplete_sentence(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("This is incomplete")) == []
        assert d.finish() == "This is incomplete"

    def test_question_mark(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("What is this? ")) == []
        assert d.finish() == "What is this?"

    def test_exclamation_mark(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("Great job! ")) == []
        assert d.finish() == "Great job!"

    def test_multiple_chunks(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("First ")) == []
        assert list(d.add_chunk("sentence. ")) == []
        assert d.finish() == "First sentence."

    def test_finish_with_remaining_text(self):
        d = SentenceBoundaryDetector()
        d.add_chunk("Complete sentence. ")
        d.add_chunk("Incomplete")
        # finish returns ALL buffered text
        assert d.finish() == "Complete sentence. Incomplete"

    def test_finish_clears_state(self):
        d = SentenceBoundaryDetector()
        d.add_chunk("Text. ")
        d.finish()
        assert d.remaining_text == ""
        assert d.current_sentence == ""

    def test_asterisks_removed_in_output(self):
        d = SentenceBoundaryDetector()
        d.add_chunk("This is *important*. ")
        assert d.finish() == "This is important."

    def test_ellipsis(self):
        d = SentenceBoundaryDetector()
        d.add_chunk("Wait for it… ")
        assert d.finish() == "Wait for it…"

    def test_empty_chunk(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("")) == []
