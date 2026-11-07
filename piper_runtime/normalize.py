"""Text normalization applied before espeak phonemization.

espeak drops dashes outright ("able - well", "able—well", "able--well") or
turns spaced em/en dashes into ";", which the voice saw only 5 times in
training. A comma is the pause token the model actually learned (872 uses),
so every pause-dash becomes one. Hyphenated words and numeric ranges are left
alone.

espeak also reads unit abbreviations letter by letter ("kg" -> "k g"), so
known units after a number are expanded to words.
"""

from __future__ import annotations

import re

# Pause dashes. Order matters: handle runs/spaced forms before single chars.
# Whitespace is horizontal only ([ \t]): a newline before a dash is a list
# bullet or a line break, not a pause, and must survive.
_DIGIT_RANGE = re.compile(r"(?<=\d)[ \t]*[–—][ \t]*(?=\d)")  # 3–5 -> 3 to 5
_PAUSE_DASH = re.compile(
    r"(?<=\S)[ \t]*(?:—|–|―|--+)[ \t]*"  # em, en, horizontal bar, double hyphen
    r"|(?<=\S)[ \t]+-+[ \t]+(?=\S)"  # spaced hyphen between words
    r"|(?<=\S)[ \t]+-+[ \t]*$"  # trailing interrupted hyphen: "He probably -"
)
_COMMA_RUN = re.compile(r"\s*,(?:\s*,)+")
_COMMA_BEFORE_END = re.compile(r",\s*(?=[.!?…]|$)")

# unit -> (singular, plural). Matched only directly after a number.
_UNITS: dict[str, tuple[str, str]] = {
    "mg": ("milligram", "milligrams"),
    "g": ("gram", "grams"),
    "kg": ("kilogram", "kilograms"),
    "t": ("tonne", "tonnes"),
    "lb": ("pound", "pounds"),
    "lbs": ("pound", "pounds"),
    "oz": ("ounce", "ounces"),
    "mm": ("millimeter", "millimeters"),
    "cm": ("centimeter", "centimeters"),
    "m": ("meter", "meters"),
    "km": ("kilometer", "kilometers"),
    "in": ("inch", "inches"),
    "ft": ("foot", "feet"),
    "yd": ("yard", "yards"),
    "mi": ("mile", "miles"),
    "ml": ("milliliter", "milliliters"),
    "mL": ("milliliter", "milliliters"),
    "l": ("liter", "liters"),
    "L": ("liter", "liters"),
    "gal": ("gallon", "gallons"),
    "km/h": ("kilometer per hour", "kilometers per hour"),
    "kph": ("kilometer per hour", "kilometers per hour"),
    "mph": ("mile per hour", "miles per hour"),
    "m/s": ("meter per second", "meters per second"),
    "°C": ("degree Celsius", "degrees Celsius"),
    "°F": ("degree Fahrenheit", "degrees Fahrenheit"),
    "°": ("degree", "degrees"),
    "%": ("percent", "percent"),
    "ms": ("millisecond", "milliseconds"),
    # No bare "s": "1990s", "747s" and "10s of people" are plurals, not seconds.
    "sec": ("second", "seconds"),
    "min": ("minute", "minutes"),
    "h": ("hour", "hours"),
    "hr": ("hour", "hours"),
    "hrs": ("hour", "hours"),
    "Hz": ("hertz", "hertz"),
    "kHz": ("kilohertz", "kilohertz"),
    "MHz": ("megahertz", "megahertz"),
    "GHz": ("gigahertz", "gigahertz"),
    "W": ("watt", "watts"),
    "kW": ("kilowatt", "kilowatts"),
    "kWh": ("kilowatt hour", "kilowatt hours"),
    "V": ("volt", "volts"),
    "A": ("amp", "amps"),
    "mAh": ("milliamp hour", "milliamp hours"),
    "KB": ("kilobyte", "kilobytes"),
    "MB": ("megabyte", "megabytes"),
    "GB": ("gigabyte", "gigabytes"),
    "TB": ("terabyte", "terabytes"),
}
# Longest first so "km/h" wins over "km", "kWh" over "kW".
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
# Units that are also common words/letters need a space-free attachment or
# must not be followed by another letter: "5 in the box" must stay "in".
_AMBIGUOUS = {"in", "A", "t", "h", "m", "g", "l", "L", "V", "W", "min"}
# (?<!\w): digits inside an identifier ("mp3s", "B2B") are not a quantity.
_UNIT_RE = re.compile(
    rf"(?<!\w)(?P<num>-?\d+(?:[.,]\d+)*)(?P<sp>\s?)(?P<unit>{_UNIT_ALT})(?![\w/])"
)


def _unit_sub(m: re.Match[str]) -> str:
    num, sp, unit = m.group("num"), m.group("sp"), m.group("unit")
    if unit in _AMBIGUOUS and sp:
        # "5 in", "3 A", "2 min" with a space are too often plain words.
        return m.group(0)
    singular, plural = _UNITS[unit]
    word = singular if num.lstrip("-") in ("1", "1.0") else plural
    return f"{num} {word}"


def normalize_text(text: str) -> str:
    """Rewrite dashes as comma pauses and expand units after numbers."""
    text = _UNIT_RE.sub(_unit_sub, text)
    text = _DIGIT_RANGE.sub(" to ", text)
    text = _PAUSE_DASH.sub(", ", text)
    text = _COMMA_RUN.sub(",", text)
    text = _COMMA_BEFORE_END.sub("", text)
    return text.strip()
