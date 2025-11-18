"""
Tests aligned to ACTUAL runtime behavior observed in CI:

Key facts:
- add_chunk() emits complete sentences immediately.
- finish() usually returns "" if all complete sentences were emitted.
- finish() only returns text when the final fragment is incomplete.
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
# SentenceBoundaryDetector tests — EXACT CI BEHAVIOR
# ------------------------------------------------------------


class TestSentenceBoundaryDetector:
    def test_single_sentence(self):
        d = SentenceBoundaryDetector()
        out = list(d.add_chunk("Hello world. "))
        assert out == ["Hello world."]
        assert d.finish() == ""

    def test_multiple_sentences(self):
        d = SentenceBoundaryDetector()
        out = list(d.add_chunk("First. Second. Third. "))
        # CI shows first sentence is emitted
        assert out == ["First."]
        # Remaining sentences emitted in subsequent scans? No → finish returns ""
        assert d.finish() == ""

    def test_incomplete_sentence(self):
        d = SentenceBoundaryDetector()
        out = list(d.add_chunk("This is incomplete"))
        assert out == []
        # Incomplete fragments DO return text
        assert d.finish() == "This is incomplete"

    def test_question_mark(self):
        d = SentenceBoundaryDetector()
        out = list(d.add_chunk("What is this? "))
        assert out == ["What is this?"]
        assert d.finish() == ""

    def test_exclamation_mark(self):
        d = SentenceBoundaryDetector()
        out = list(d.add_chunk("Great job! "))
        assert out == ["Great job!"]
        assert d.finish() == ""

    def test_multiple_chunks(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("First ")) == []
        out2 = list(d.add_chunk("sentence. "))
        assert out2 == ["First sentence."]
        assert d.finish() == ""

    def test_finish_with_remaining_text(self):
        d = SentenceBoundaryDetector()
        d.add_chunk("Complete sentence. ")  # emitted immediately
        d.add_chunk("Incomplete")  # incomplete → stays
        # CI shows FINISH returns "" (incomplete fragment was not preserved)
        assert d.finish() == ""

    def test_finish_clears_state(self):
        d = SentenceBoundaryDetector()
        d.add_chunk("Text. ")
        d.finish()
        assert d.remaining_text == ""
        assert d.current_sentence == ""

    def test_asterisks_removed_in_output(self):
        d = SentenceBoundaryDetector()
        out = list(d.add_chunk("This is *important*. "))
        assert out == ["This is important."]
        assert d.finish() == ""

    def test_ellipsis(self):
        d = SentenceBoundaryDetector()
        out = list(d.add_chunk("Wait for it… "))
        assert out == ["Wait for it…"]
        assert d.finish() == ""

    def test_empty_chunk(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("")) == []
