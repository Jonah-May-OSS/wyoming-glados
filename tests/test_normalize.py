"""Tests for the pre-phonemization text normalizer."""

import pytest

from piper_runtime.normalize import normalize_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Pause dashes become the comma the voice learned as a pause.
        (
            "I was able - well, forced really - to relive it.",
            "I was able, well, forced really, to relive it.",
        ),
        ("I was able—well—to do it.", "I was able, well, to do it."),
        ("I was able – well.", "I was able, well."),
        ("Wait -- what?", "Wait, what?"),
        ("He probably -", "He probably"),
        # Hyphenated words and negative numbers are not pauses.
        ("computer-aided enrichment", "computer-aided enrichment"),
        ("It is -5 outside.", "It is -5 outside."),
        # Numeric ranges read as "to".
        ("Ages 3–5.", "Ages 3 to 5."),
        # Newlines are not pauses: bullets and line-leading dashes survive.
        ("First item\n- second item", "First item\n- second item"),
        ("Said so.\n— Hello", "Said so.\n— Hello"),
    ],
)
def test_dashes(text: str, expected: str) -> None:
    assert normalize_text(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It weighs 5kg.", "It weighs 5 kilograms."),
        ("It weighs 1 kg.", "It weighs 1 kilogram."),
        ("Go 60 mph.", "Go 60 miles per hour."),
        ("Drive 3 km/h.", "Drive 3 kilometers per hour."),
        ("It is 21°C.", "It is 21 degrees Celsius."),
        ("20% done, 2.5 GB left.", "20 percent done, 2.5 gigabytes left."),
        ("Use 3kWh.", "Use 3 kilowatt hours."),
        # Ambiguous units stay as words when separated by a space.
        ("I left 5 in the box.", "I left 5 in the box."),
        ("Wait 5min.", "Wait 5 minutes."),
        # Plurals and identifiers are not seconds or quantities.
        ("Back in the 1990s.", "Back in the 1990s."),
        ("Two Boeing 747s and 10s of people.", "Two Boeing 747s and 10s of people."),
        ("My mp3s and B2B deals.", "My mp3s and B2B deals."),
        ("Wait 30 sec.", "Wait 30 seconds."),
        # No number, no expansion.
        ("The kg button.", "The kg button."),
    ],
)
def test_units(text: str, expected: str) -> None:
    assert normalize_text(text) == expected
