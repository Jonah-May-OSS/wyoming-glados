"""
Tests for sentence boundary detection module.

These tests match the behavior of the actual production
SentenceBoundaryDetector implementation found in server/sentence_boundary.py.

Key expectations:

- add_chunk() emits complete sentences immediately when regex detects them.
- finish() only returns leftover partial sentences, not completed ones.
- Trailing punctuation alone does NOT force a flush.
- remove_asterisks() removes inline and leading asterisks correctly.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip all tests if regex is not available
regex = pytest.importorskip("regex")

from server.sentence_boundary import (  # noqa: E402
    SentenceBoundaryDetector,
    remove_asterisks,
)


# -------------------------------------------------------------------
# remove_asterisks tests
# -------------------------------------------------------------------

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
        text = "* Start\nThis is *bold* text\n** Another line"
        result = remove_asterisks(text)
        assert "Start" in result
        assert "This is bold text" in result
        assert "Another line" in result

    def test_no_asterisks(self):
        text = "Plain text without any special characters"
        assert remove_asterisisks(text) == text

    def test_empty_string(self):
        assert remove_asterisks("") == ""


# -------------------------------------------------------------------
# SentenceBoundaryDetector Tests
# -------------------------------------------------------------------

class TestSentenceBoundaryDetector:

    def test_single_sentence(self):
        detector = SentenceBoundaryDetector()
        out = list(detector.add_chunk("Hello world. "))
        # production emits immediately
        assert out == ["Hello world."]
        assert detector.finish() == ""

    def test_multiple_sentences(self):
        detector = SentenceBoundaryDetector()
        out = list(detector.add_chunk("First one. Second one. Third. "))
        # production emits each sentence as detected
        assert "First one." in out
        assert "Second one." in out
        assert "Third." in out
        assert detector.finish() == ""

    def test_incomplete_sentence(self):
        detector = SentenceBoundaryDetector()
        out = list(detector.add_chunk("This is incomplete"))
        # no emission because no boundary
        assert out == []
        # finish returns leftover
        assert detector.finish() == "This is incomplete"

    def test_abbreviation_handling(self):
        detector = SentenceBoundaryDetector()
        out = list(detector.add_chunk("The U.S. is a country. "))
        # Abbreviation handling is conservative; may emit one or zero initially
        # but full sentence is eventually emitted before finish.
        assert out in (["The U.S. is a country."], [])
        if not out:
            assert detector.finish() == "The U.S. is a country."
        else:
            assert detector.finish() == ""

    def test_question_mark_emits_in_add_chunk(self):
        detector = SentenceBoundaryDetector()
        out = list(detector.add_chunk("What is this? "))
        # production emits immediately
        assert out == ["What is this?"]
        assert detector.finish() == ""

    def test_exclamation_mark_emits_in_add_chunk(self):
        detector = SentenceBoundaryDetector()
        out = list(detector.add_chunk("Great job! "))
        assert out == ["Great job!"]
        assert detector.finish() == ""

    def test_multiple_chunks_sentence_completion(self):
        detector = SentenceBoundaryDetector()
        out1 = list(detector.add_chunk("First "))
        assert out1 == []
        out2 = list(detector.add_chunk("sentence. "))
        assert out2 == ["First sentence."]
        assert detector.finish() == ""

    def test_finish_with_remaining_text(self):
        detector = SentenceBoundaryDetector()
        detector.add_chunk("Complete sentence. ")
        detector.add_chunk("Incomplete")
        # the complete sentence was emitted
        # finish should return only leftover incomplete part
        assert detector.finish() == "Incomplete"

    def test_finish_clears_state(self):
        detector = SentenceBoundaryDetector()
        detector.add_chunk("Text. ")
        detector.finish()
        assert detector.remaining_text == ""
        assert detector.current_sentence == ""

    def test_asterisks_removed_in_output(self):
        detector = SentenceBoundaryDetector()
        out = list(detector.add_chunk("This is *important*. "))
        assert out == ["This is important."]
        assert detector.finish() == ""

    def test_ellipsis(self):
        detector = SentenceBoundaryDetector()
        out = list(detector.add_chunk("Wait for it… "))
        assert out == ["Wait for it…"]
        assert detector.finish() == ""

    def test_empty_chunk(self):
        detector = SentenceBoundaryDetector()
        out = list(detector.add_chunk(""))
        assert out == []
