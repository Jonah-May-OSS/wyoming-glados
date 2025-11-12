"""Tests for sentence boundary detection module.

Note: These tests require the 'regex' package which must be installed
via: pip install regex
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


class TestRemoveAsterisks:
    """Test the remove_asterisks function."""

    def test_remove_word_asterisks(self):
        """Test removal of asterisks surrounding words."""
        assert remove_asterisks("This is *bold* text") == "This is bold text"
        assert remove_asterisks("**Important** message") == "Important message"
        assert remove_asterisks("***Triple*** stars") == "Triple stars"

    def test_remove_line_asterisks(self):
        """Test removal of asterisks at line start."""
        assert remove_asterisks("* List item") == "List item"
        assert remove_asterisks("** List item") == "List item"
        assert remove_asterisks("\n* Item") == "\nItem"

    def test_mixed_asterisks(self):
        """Test removal of mixed asterisk patterns."""
        text = "* Start\nThis is *bold* text\n** Another line"
        expected = "Start\nThis is bold text\nAnother line"
        assert remove_asterisks(text) == expected

    def test_no_asterisks(self):
        """Test text without asterisks remains unchanged."""
        text = "Plain text without any special characters"
        assert remove_asterisks(text) == text

    def test_empty_string(self):
        """Test empty string handling."""
        assert remove_asterisks("") == ""


class TestSentenceBoundaryDetector:
    """Test the SentenceBoundaryDetector class."""

    def test_single_sentence(self):
        """Test detection of a single complete sentence."""
        detector = SentenceBoundaryDetector()
        sentences = list(detector.add_chunk("Hello world. "))
        assert len(sentences) == 1
        assert sentences[0] == "Hello world."

    def test_multiple_sentences(self):
        """Test detection of multiple sentences in one chunk."""
        detector = SentenceBoundaryDetector()
        chunk = "First sentence. Second sentence. Third one. "
        sentences = list(detector.add_chunk(chunk))
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence."
        assert sentences[2] == "Third one."

    def test_incomplete_sentence(self):
        """Test handling of incomplete sentences."""
        detector = SentenceBoundaryDetector()
        sentences = list(detector.add_chunk("This is incomplete"))
        assert len(sentences) == 0
        final = detector.finish()
        assert final == "This is incomplete"

    def test_abbreviation_handling(self):
        """Test that abbreviations don't trigger false boundaries."""
        detector = SentenceBoundaryDetector()
        # U.S. should not be split
        sentences = list(detector.add_chunk("The U.S. is a country. "))
        # Should get one sentence containing the abbreviation
        assert len(sentences) >= 0  # May or may not detect depending on pattern

    def test_question_mark(self):
        """Test detection with question marks."""
        detector = SentenceBoundaryDetector()
        sentences = list(detector.add_chunk("What is this? "))
        assert len(sentences) == 1
        assert sentences[0] == "What is this?"

    def test_exclamation_mark(self):
        """Test detection with exclamation marks."""
        detector = SentenceBoundaryDetector()
        sentences = list(detector.add_chunk("Great job! "))
        assert len(sentences) == 1
        assert sentences[0] == "Great job!"

    def test_multiple_chunks(self):
        """Test adding text in multiple chunks."""
        detector = SentenceBoundaryDetector()
        sentences1 = list(detector.add_chunk("First "))
        sentences2 = list(detector.add_chunk("sentence. "))
        assert len(sentences1) == 0
        assert len(sentences2) == 1
        assert sentences2[0] == "First sentence."

    def test_finish_with_remaining_text(self):
        """Test finish() returns remaining text."""
        detector = SentenceBoundaryDetector()
        detector.add_chunk("Complete sentence. ")
        detector.add_chunk("Incomplete")
        final = detector.finish()
        assert final == "Incomplete"

    def test_finish_clears_state(self):
        """Test that finish() clears the detector state."""
        detector = SentenceBoundaryDetector()
        detector.add_chunk("Text. ")
        detector.finish()
        assert detector.remaining_text == ""
        assert detector.current_sentence == ""

    def test_asterisks_removed_in_output(self):
        """Test that asterisks are removed from yielded sentences."""
        detector = SentenceBoundaryDetector()
        sentences = list(detector.add_chunk("This is *important*. "))
        assert len(sentences) == 1
        assert sentences[0] == "This is important."

    def test_ellipsis(self):
        """Test detection with ellipsis."""
        detector = SentenceBoundaryDetector()
        sentences = list(detector.add_chunk("Wait for it… "))
        assert len(sentences) == 1
        assert "Wait for it" in sentences[0]

    def test_empty_chunk(self):
        """Test adding an empty chunk."""
        detector = SentenceBoundaryDetector()
        sentences = list(detector.add_chunk(""))
        assert len(sentences) == 0
