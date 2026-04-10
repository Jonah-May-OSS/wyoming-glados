"""Tests for sentence and clause boundary detection."""

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
# SentenceBoundaryDetector — TRUE CI BEHAVIOR
# ------------------------------------------------------------


class TestSentenceBoundaryDetector:
    def test_single_sentence(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("Hello world. ")) == ["Hello world."]
        assert d.finish() == ""

    def test_multiple_sentences(self):
        d = SentenceBoundaryDetector()
        out = list(d.add_chunk("First. Second. Third. "))

        assert out == ["First.", "Second.", "Third."]
        assert d.finish() == ""

    def test_incomplete_sentence(self):
        d = SentenceBoundaryDetector()
        assert not list(d.add_chunk("This is incomplete"))
        assert d.finish() == "This is incomplete"

    def test_abbreviation_handling(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("The U.S. is a country. ")) == [
            "The U.S. is a country."
        ]
        assert d.finish() == ""

    def test_question_mark(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("What is this? ")) == ["What is this?"]
        assert d.finish() == ""

    def test_exclamation_mark(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("Great job! ")) == ["Great job!"]
        assert d.finish() == ""

    def test_multiple_chunks(self):
        d = SentenceBoundaryDetector()
        assert not list(d.add_chunk("First "))
        assert list(d.add_chunk("sentence. ")) == ["First sentence."]
        assert d.finish() == ""

    def test_finish_with_remaining_text(self):
        d = SentenceBoundaryDetector()

        out1 = list(d.add_chunk("Complete sentence. "))
        assert out1 == ["Complete sentence."]

        _ = list(d.add_chunk("Incomplete"))

        final = d.finish()
        assert "Incomplete" in final
        assert "Complete sentence." not in final

    def test_finish_clears_state(self):
        d = SentenceBoundaryDetector()
        list(d.add_chunk("Text. "))
        d.finish()
        assert d.remaining_text == ""

    def test_asterisks_removed_in_output(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("This is *important*. ")) == ["This is important."]
        assert d.finish() == ""

    def test_ellipsis(self):
        d = SentenceBoundaryDetector()
        assert list(d.add_chunk("Wait for it… ")) == ["Wait for it…"]
        assert d.finish() == ""

    def test_empty_chunk(self):
        d = SentenceBoundaryDetector()
        assert not list(d.add_chunk(""))

    def test_long_clause_emits_before_sentence_end(self):
        d = SentenceBoundaryDetector(min_clause_words=6)

        out = list(
            d.add_chunk(
                "This is a fairly long opening clause, and this part should remain buffered"
            )
        )

        assert out == ["This is a fairly long opening clause,"]
        assert d.finish() == "and this part should remain buffered"

    def test_short_clause_stays_buffered(self):
        d = SentenceBoundaryDetector(min_clause_words=6)

        assert not list(d.add_chunk("Short clause, still buffering"))
        assert d.finish() == "Short clause, still buffering"

    def test_decimal_does_not_split(self):
        d = SentenceBoundaryDetector()

        assert list(d.add_chunk("Pi is 3.14 and rising. ")) == [
            "Pi is 3.14 and rising."
        ]
        assert d.finish() == ""
