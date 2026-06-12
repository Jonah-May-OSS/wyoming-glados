"""Language/voice mapping for multilingual GLaDOS synthesis (Path A).

The GLaDOS acoustic model was trained on English speech, so non-English output
is the model "reading" foreign-language IPA with an English accent. We expose
one Wyoming voice per supported language and map the selected voice back to the
phonemizer language code understood by the multilingual "Latin IPA" checkpoint.

English voices map to ``None`` so synthesis keeps the original cmudict
phonemizer + english_cleaners path unchanged.
"""

from typing import NamedTuple


class VoiceSpec(NamedTuple):
    """A user-selectable voice and the phonemizer language it maps to."""

    name: str
    description: str
    # Wyoming language tag advertised to clients (e.g. Home Assistant).
    wyoming_lang: str
    # Phonemizer language passed to the multilingual model; None = English
    # (original cmudict phonemizer + english_cleaners, output unchanged).
    phonemizer_lang: str | None


# Order matters only for display; "default" stays first for back-compat.
# wyoming_lang uses underscore locale codes (en_US, de_DE, ...) to match the
# Piper convention Home Assistant is built against: the HA Wyoming integration
# does no language-code normalization and exposes voice.languages verbatim as
# the TTS entity's supported_languages, so these must match HA's pipeline codes.
VOICES: list[VoiceSpec] = [
    VoiceSpec("default", "Default GLaDOS voice (English)", "en_US", None),
    VoiceSpec("glados-de", "GLaDOS voice reading German", "de_DE", "de"),
    VoiceSpec("glados-fr", "GLaDOS voice reading French", "fr_FR", "fr"),
    VoiceSpec("glados-es", "GLaDOS voice reading Spanish", "es_ES", "es"),
]

# Lookup by exact voice name (lower-cased).
_voice_to_phon: dict[str, str | None] = {v.name.lower(): v.phonemizer_lang for v in VOICES}

# Fallback lookup by bare language code, for clients that send a language
# instead of a voice name. en -> None keeps the English default path.
_lang_to_phon: dict[str, str | None] = {
    "en": None,
    "de": "de",
    "fr": "fr",
    "es": "es",
}


def resolve_phonemizer_lang(voice: object | None) -> str | None:
    """Map a Wyoming SynthesizeVoice to a phonemizer language code.

    Returns ``None`` for English / unknown requests, which selects the original
    English synthesis path. Note: this Wyoming version's SynthesizeVoice.from_dict
    places a language-only request into ``.name``, so we check both fields.
    """
    if voice is None:
        return None

    key = (getattr(voice, "name", None) or getattr(voice, "language", None) or "")
    key = key.strip().lower()
    if not key:
        return None

    if key in _voice_to_phon:
        return _voice_to_phon[key]

    # Accept bare/region-tagged language codes too (e.g. "de", "de-DE").
    short = key.replace("_", "-").split("-")[0]
    return _lang_to_phon.get(short)
