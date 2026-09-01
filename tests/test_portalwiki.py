"""Tests for the theportalwiki.com voice-line parser."""

from dataset_tools.portalwiki import (
    PAGES,
    VoiceLine,
    audio_annotations,
    clean_transcript,
    is_audio_url,
    parse_page,
)

# Mirrors the real markup: transcript in <i> inside quotes, then several
# links to the same .wav (download icon, download text, play icon, play text).
_LI = (
    '<li>"<i>{text}</i>" | '
    '<span><a href="{url}" class="internal" title="x.wav">Download</a></span> '
    '<span><a href="{url}" rel="nofollow"><img src="play.png" /></a></span></li>'
)


def _li(text: str, url: str) -> str:
    return _LI.format(text=text, url=url)


def test_parses_transcript_and_url():
    html = f"<ul>{_li('Hello and, again, welcome.', 'https://x/a.wav')}</ul>"
    (line,) = parse_page(html, "portal1")
    assert line.transcript == "Hello and, again, welcome."
    assert line.url == "https://x/a.wav"
    assert line.filename == "a.wav"


def test_repeated_links_to_same_wav_collapse_to_one_entry():
    html = f"<ul>{_li('One.', 'https://x/a.wav')}</ul>"
    assert len(parse_page(html, "portal1")) == 1


def test_duplicate_wav_across_entries_is_deduplicated():
    html = f"<ul>{_li('One.', 'https://x/a.wav')}{_li('One.', 'https://x/a.wav')}</ul>"
    assert len(parse_page(html, "portal1")) == 1


def test_entries_without_transcript_or_audio_are_skipped():
    html = (
        "<ul>"
        '<li><a href="https://x/sfx.wav">Download</a></li>'
        '<li>"<i>No audio here.</i>"</li>'
        f"{_li('Kept.', 'https://x/a.wav')}"
        "</ul>"
    )
    (line,) = parse_page(html, "portal1")
    assert line.transcript == "Kept."


def test_headings_are_tracked_at_two_levels():
    html = (
        "<h2>Portal 2 Unused/alternate lines</h2>"
        "<h3>Chapter 1: GLaDOS Awakening</h3>"
        f"<ul>{_li('Line.', 'https://x/a.wav')}</ul>"
    )
    (line,) = parse_page(html, "other")
    assert line.section == "Portal 2 Unused/alternate lines"
    assert line.subsection == "Chapter 1: GLaDOS Awakening"


def test_h2_resets_stale_h3():
    html = (
        "<h2>First</h2><h3>Sub</h3>"
        "<h2>Second</h2>"
        f"<ul>{_li('Line.', 'https://x/a.wav')}</ul>"
    )
    (line,) = parse_page(html, "other")
    assert line.section == "Second"
    assert line.subsection == ""


def test_nested_markup_inside_transcript_is_flattened():
    html = '<ul><li>"<i>Well <b>done</b>, android.</i>" '
    html += '<a href="https://x/a.wav">Download</a></li></ul>'
    (line,) = parse_page(html, "portal2")
    assert line.transcript == "Well done, android."


def test_entities_are_decoded():
    html = f"<ul>{_li('It&#39;s your &quot;cake&quot;.', 'https://x/a.wav')}</ul>"
    (line,) = parse_page(html, "portal2")
    assert line.transcript == 'It\'s your "cake".'


def test_unknown_page_rejected():
    try:
        parse_page("<ul></ul>", "nope")
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected ValueError")


class TestCleanTranscript:
    def test_strips_wiki_annotations(self):
        assert clean_transcript("Oh. [sic] Hello.") == "Oh. Hello."

    def test_collapses_whitespace(self):
        assert clean_transcript("Too\n  many   spaces.") == "Too many spaces."

    def test_strips_wrapping_quotes(self):
        assert clean_transcript('"Quoted."') == "Quoted."


class TestSpeakerAssignment:
    def _line(self, page: str, section: str = "") -> VoiceLine:
        return VoiceLine(
            url="https://x/a.wav",
            transcript="t",
            page=page,
            section=section,
            subsection="",
        )

    def test_portal1_page(self):
        assert self._line("portal1").speaker == "p1"

    def test_portal2_and_coop_share_a_speaker(self):
        assert self._line("portal2").speaker == "p2"
        assert self._line("coop").speaker == "p2"

    def test_other_page_defaults_to_p2(self):
        assert self._line("other", "Leaderboard responses").speaker == "p2"

    def test_portal1_unused_section_maps_to_p1(self):
        line = self._line("other", "Portal 1 Unused/alternate lines")
        assert line.speaker == "p1"

    def test_dota2_section_gets_its_own_speaker(self):
        assert self._line("other", "Dota 2").speaker == "dota2"

    def test_section_match_is_case_insensitive(self):
        assert self._line("other", "  DOTA 2 ").speaker == "dota2"

    def test_section_overrides_apply_only_to_other_page(self):
        assert self._line("portal2", "Dota 2").speaker == "p2"


def test_all_pages_have_a_default_speaker():
    for page in PAGES:
        line = VoiceLine("https://x/a.wav", "t", page, "", "")
        assert line.speaker


def test_nested_list_items_are_each_emitted():
    """The wiki groups alternate takes as a nested <ul> inside a parent <li>."""
    inner = _li("Ow.", "https://x/take1.wav") + _li(
        "It's eating me.", "https://x/take2.wav"
    )
    html = (
        "<ul><li>"
        '"<i>Parent line.</i>" '
        '<a href="https://x/parent.wav">Download</a>'
        f"<ul>{inner}</ul>"
        "</li></ul>"
    )
    lines = parse_page(html, "portal2")
    assert [line.transcript for line in lines] == [
        "Ow.",
        "It's eating me.",
        "Parent line.",
    ]
    assert {line.filename for line in lines} == {
        "take1.wav",
        "take2.wav",
        "parent.wav",
    }


def test_nested_item_does_not_steal_parent_audio():
    html = (
        "<ul><li>"
        '"<i>Parent.</i>" <a href="https://x/parent.wav">Download</a>'
        f"<ul>{_li('Child.', 'https://x/child.wav')}</ul>"
        "</li></ul>"
    )
    by_text = {line.transcript: line.filename for line in parse_page(html, "portal2")}
    assert by_text == {"Parent.": "parent.wav", "Child.": "child.wav"}


def test_annotation_only_transcript_is_dropped():
    """e.g. "[hums 'For He's A Jolly Good Fellow']" is humming, not speech."""
    html = f"<ul>{_li('[hums a tune]', 'https://x/hum.wav')}</ul>"
    assert parse_page(html, "portal2") == []


class TestIsAudioUrl:
    def test_accepts_absolute_wav(self):
        assert is_audio_url("https://i1.theportalwiki.net/img/e/e5/GLaDOS.wav")

    def test_rejects_relative_url(self):
        assert not is_audio_url("/img/e/e5/GLaDOS.wav")

    def test_rejects_special_upload_redlink(self):
        """Missing files render as upload redlinks carrying .wav in the query."""
        assert not is_audio_url(
            "/w/index.php?title=Special:Upload&wpDestFile=GLaDOS_acid03.wav"
        )

    def test_rejects_wav_only_in_query_string(self):
        assert not is_audio_url("https://x/index.php?file=a.wav")

    def test_ignores_query_and_fragment_after_a_wav_path(self):
        assert is_audio_url("https://x/a.wav?v=2")


def test_redlink_entry_yields_no_line():
    """A line whose audio was never uploaded has nothing to train on."""
    html = (
        '<ul><li>"<i>Missing audio.</i>" '
        '<a href="/w/index.php?title=Special:Upload&wpDestFile=GLaDOS_x.wav">'
        "Upload</a></li></ul>"
    )
    assert parse_page(html, "portal2") == []


def test_redlink_does_not_shadow_a_real_link_in_the_same_item():
    html = (
        '<ul><li>"<i>Has audio.</i>" '
        '<a href="/w/index.php?title=Special:Upload&wpDestFile=GLaDOS_x.wav">Up</a> '
        '<a href="https://x/real.wav">Download</a></li></ul>'
    )
    (line,) = parse_page(html, "portal2")
    assert line.filename == "real.wav"


def test_parenthesised_editorial_notes_are_stripped():
    # "(page flip)", "(The last part is cut off...)" and a transcript that is
    # nothing but "(subtitled as ...)" all reached the phonemizer before.
    assert clean_transcript("(page flip) 'No guts. No glory.'") == (
        "'No guts. No glory.'"
    )
    assert clean_transcript("Everything's fine.(unused)") == "Everything's fine."


def test_dangling_dashes_from_a_stripped_annotation_are_removed():
    # The train-horn line: "...loud noises--[train horn]--" left "-- --".
    raw = "startled by loud noises--[train horn]--"
    assert clean_transcript(raw) == "startled by loud noises"


def test_intra_word_dashes_survive():
    assert clean_transcript("Ba-- I mean, fine.") == "Ba-- I mean, fine."


def test_audio_annotations_ignores_text_only_notes():
    assert audio_annotations("She said teh[sic] thing") == []
    assert audio_annotations("Plain text") == []


def test_audio_annotations_reports_sounds():
    assert audio_annotations("noises--[train horn]--") == ["[train horn]"]
    assert audio_annotations("(phone ringing) Hello") == ["(phone ringing)"]
    assert audio_annotations("[gentle laughter] It's been fun.") == [
        "[gentle laughter]"
    ]


def test_parse_page_flags_lines_whose_audio_holds_a_sound():
    html = (
        "<ul>"
        + _li("Hello and, again, welcome.", "https://x/clean.wav")
        + _li("startled by loud noises--[train horn]--", "https://x/horn.wav")
        + "</ul>"
    )
    lines = {line.filename: line for line in parse_page(html, next(iter(PAGES)))}
    assert lines["clean.wav"].has_audio_annotation is False
    assert lines["horn.wav"].has_audio_annotation is True


class TestOnlyQuotedItalicsAreTranscripts:
    """A <li> can hold editorial italics beside the spoken line.

    The wiki's convention is that a spoken line is wrapped in quotes sitting
    OUTSIDE the <i>, while an aside is not. Every case below is taken from the
    live pages, and each one reached metadata.csv before this was fixed.
    """

    def test_an_aside_after_the_line_is_not_appended(self):
        # GLaDOS_escape_02_spheredrop1-03: the aside is italic too, and
        # concatenating every run trained the model to read the stage
        # direction aloud. Note the aside contains quotes of its own, so the
        # test has to be on the delimiters, not on the content.
        html = (
            '<ul><li>"<i>Because you\'ll be dead.</i>" '
            '<a href="https://x/drop.wav">Download</a> '
            '(add <i>"with the sphere, cycle through these:"</i>)</li></ul>'
        )
        lines = parse_page(html, next(iter(PAGES)))
        assert [line.transcript for line in lines] == ["Because you'll be dead."]

    def test_a_wiki_note_run_together_with_the_line_is_dropped(self):
        # GLaDOS_sp_incinerator_01_15 ended "...Food for thought.Note: This
        # line was used in the Portal 2 trailer." - no space, so it did not
        # even read as a separate sentence.
        html = (
            '<ul><li>"<i>Food for thought.</i>" '
            '<a href="https://x/inc.wav">Download</a>'
            "<i>Note: This line was used in the Portal 2 trailer.</i></li></ul>"
        )
        lines = parse_page(html, next(iter(PAGES)))
        assert [line.transcript for line in lines] == ["Food for thought."]

    def test_an_unquoted_stage_direction_yields_no_line_at_all(self):
        # GLaDOS_escape_02_sphere_Death_Scream: the audio is a scream, and the
        # italic text is the wiki describing it. That paired a death scream
        # with a sentence GLaDOS never says.
        html = (
            "<ul><li><i>Upon destroying of the last three cores, this will "
            'sound:</i> <a href="https://x/scream.wav">"*scream*"</a></li></ul>'
        )
        assert parse_page(html, next(iter(PAGES))) == []

    def test_a_line_opening_with_an_inner_quotation_is_kept(self):
        # GLaDOS_escape_02_miscbabble-26. The line opens `"'` - the outer
        # double quote then the inner quotation's apostrophe - so testing only
        # the character adjacent to the run rejected three real lines.
        html = (
            "<ul><li>\"'<i>Shall not be mourned.' That's exactly what it "
            'says.</i>" <a href="https://x/mourn.wav">Download</a></li></ul>'
        )
        lines = parse_page(html, next(iter(PAGES)))
        assert [line.transcript for line in lines] == [
            "Shall not be mourned.' That's exactly what it says."
        ]

    def test_a_run_with_nothing_around_it_is_not_treated_as_quoted(self):
        # `"" in '"'` is True in Python - every string contains the empty
        # string - so an italic run at the very start of a <li> passed the
        # quote test vacuously while the check was written against a string.
        html = (
            "<ul><li><i>Alternate version of</i> "
            '<a href="https://x/alt.wav">Download</a></li></ul>'
        )
        assert parse_page(html, next(iter(PAGES))) == []
