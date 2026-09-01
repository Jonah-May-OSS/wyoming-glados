"""Tests for assembling the Piper training dataset."""

import numpy as np
import pytest

from dataset_tools.audio import write_wav
from dataset_tools.build import (
    POTATO_SPEAKER,
    _prune_orphans,
    build_dataset,
    sanitize_transcript,
)
from dataset_tools.portalwiki import VoiceLine

RATE = 22050


def _speech(seconds: float) -> np.ndarray:
    """Make a tone loud enough to survive silence trimming."""
    t = np.arange(int(RATE * seconds)) / RATE
    return (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


@pytest.fixture
def corpus(tmp_path):
    """Build a source directory plus a factory for lines pointing into it."""
    source = tmp_path / "audio"
    source.mkdir()

    def add(
        name: str,
        seconds: float = 3.0,
        text: str = "A perfectly normal line.",
        page: str = "portal2",
    ) -> VoiceLine:
        write_wav(source / name, _speech(seconds), RATE)
        return VoiceLine(f"https://x/{name}", text, page, "", "")

    return source, add


class TestSanitizeTranscript:
    def test_replaces_delimiter(self):
        assert sanitize_transcript("a | b") == "a , b"

    def test_collapses_whitespace(self):
        assert sanitize_transcript("a\n  b") == "a b"


class TestBuildDataset:
    def test_writes_metadata_and_wavs(self, corpus, tmp_path):
        source, add = corpus
        lines = [add("a.wav"), add("b.wav", page="portal1")]
        out = tmp_path / "ds"

        report = build_dataset(lines, source, out)

        rows = (out / "metadata.csv").read_text(encoding="utf-8").splitlines()
        assert rows == [
            "a.wav|A perfectly normal line.",
            "b.wav|A perfectly normal line.",
        ]
        assert (out / "wavs" / "a.wav").exists()
        assert report.total_clips == 2
        assert set(report.clips) == {"p1", "p2"}

    def test_counts_duration_per_speaker(self, corpus, tmp_path):
        source, add = corpus
        report = build_dataset([add("a.wav", 4.0)], source, tmp_path / "ds")
        assert report.seconds["p2"] == pytest.approx(4.0, abs=0.2)

    def test_skips_missing_audio(self, corpus, tmp_path):
        source, _add = corpus
        line = VoiceLine("https://x/gone.wav", "Text.", "portal2", "", "")
        report = build_dataset([line], source, tmp_path / "ds")
        assert report.excluded["missing audio"] == 1
        assert report.total_clips == 0

    def test_skips_undecodable_audio(self, corpus, tmp_path):
        source, _add = corpus
        (source / "bad.wav").write_bytes(b"not a wav at all")
        line = VoiceLine("https://x/bad.wav", "Text.", "portal2", "", "")
        report = build_dataset([line], source, tmp_path / "ds")
        assert report.excluded["decode failed"] == 1

    def test_drops_clips_below_minimum(self, corpus, tmp_path):
        source, add = corpus
        out = tmp_path / "ds"
        report = build_dataset([add("tiny.wav", 0.2)], source, out, min_seconds=0.7)
        assert report.total_clips == 0
        assert not (out / "wavs" / "tiny.wav").exists()
        assert (out / "metadata.csv").read_text(encoding="utf-8") == ""

    def test_drops_clips_above_maximum(self, corpus, tmp_path):
        source, add = corpus
        out = tmp_path / "ds"
        report = build_dataset([add("long.wav", 6.0)], source, out, max_seconds=2.0)
        assert report.total_clips == 0
        assert not (out / "wavs" / "long.wav").exists()

    def test_sanitizes_transcripts_and_counts_them(self, corpus, tmp_path):
        source, add = corpus
        line = add("a.wav", text="Well | done, android.")
        report = build_dataset([line], source, tmp_path / "ds")
        assert report.sanitized_transcripts == 1
        row = (tmp_path / "ds" / "metadata.csv").read_text(encoding="utf-8")
        assert row.strip() == "a.wav|Well , done, android."

    def test_flags_implausible_speaking_rate(self, corpus, tmp_path):
        source, add = corpus
        # Three characters spread over four seconds is far too slow.
        line = add("a.wav", 4.0, text="No!")
        out = tmp_path / "ds"
        report = build_dataset([line], source, out)
        assert len(report.suspicious) == 1
        assert "chars_per_second" in (out / "review_transcripts.csv").read_text(
            encoding="utf-8"
        )

    def test_no_review_file_when_nothing_flagged(self, corpus, tmp_path):
        source, add = corpus
        out = tmp_path / "ds"
        build_dataset([add("a.wav")], source, out)
        assert not (out / "review_transcripts.csv").exists()

    def test_stale_review_file_is_removed(self, corpus, tmp_path):
        source, add = corpus
        out = tmp_path / "ds"
        out.mkdir()
        (out / "review_transcripts.csv").write_text("stale", encoding="utf-8")
        build_dataset([add("a.wav")], source, out)
        assert not (out / "review_transcripts.csv").exists()


class TestSpeakerModes:
    def test_single_speaker_omits_the_speaker_column(self, corpus, tmp_path):
        """A single-speaker checkpoint has no speaker embedding to match."""
        source, add = corpus
        out = tmp_path / "ds"
        build_dataset([add("a.wav")], source, out)
        row = (out / "metadata.csv").read_text(encoding="utf-8").strip()
        assert row.count("|") == 1

    def test_multi_speaker_adds_the_speaker_column(self, corpus, tmp_path):
        source, add = corpus
        out = tmp_path / "ds"
        build_dataset([add("a.wav")], source, out, multi_speaker=True)
        row = (out / "metadata.csv").read_text(encoding="utf-8").strip()
        assert row == "a.wav|p2|A perfectly normal line."

    def test_audit_breaks_down_by_source_even_when_pooled(self, corpus, tmp_path):
        """Pooling must not cost the per-source numbers the decision needs."""
        source, add = corpus
        lines = [add("a.wav"), add("b.wav", page="portal1")]
        report = build_dataset(lines, source, tmp_path / "ds")
        assert set(report.clips) == {"p1", "p2"}


class TestSummary:
    def test_reports_speakers_and_exclusions(self, corpus, tmp_path):
        source, add = corpus
        lines = [add("a.wav"), add("tiny.wav", 0.2)]
        report = build_dataset(lines, source, tmp_path / "ds")
        summary = report.summary()
        assert "p2" in summary
        assert "TOTAL" in summary
        assert "Excluded" in summary


class TestPruneOrphans:
    def test_removes_clips_the_metadata_no_longer_references(self, tmp_path):
        """A rebuild with tighter filters must not leave excluded audio behind."""
        wavs = tmp_path / "wavs"
        wavs.mkdir()
        (wavs / "keep.wav").write_bytes(b"a")
        (wavs / "drop.wav").write_bytes(b"b")
        assert _prune_orphans(wavs, {"keep.wav"}) == 1
        assert [p.name for p in wavs.glob("*.wav")] == ["keep.wav"]

    def test_returns_zero_when_nothing_is_stale(self, tmp_path):
        wavs = tmp_path / "wavs"
        wavs.mkdir()
        (wavs / "keep.wav").write_bytes(b"a")
        assert _prune_orphans(wavs, {"keep.wav"}) == 0

    def test_leaves_non_wav_files_alone(self, tmp_path):
        wavs = tmp_path / "wavs"
        wavs.mkdir()
        (wavs / "notes.txt").write_text("keep me")
        _prune_orphans(wavs, set())
        assert (wavs / "notes.txt").exists()


class TestSpectralGate:
    def _line(self, name):
        return VoiceLine(
            url=f"https://example.test/{name}.wav",
            transcript="A test line with enough words to be plausible.",
            page="portal2",
            section="Chapter 1",
            subsection="Test",
        )

    def _write(self, path, freq, rate=RATE, seconds=2.0):
        t = np.arange(int(rate * seconds)) / rate
        write_wav(path, (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32), rate)

    def test_band_limited_clips_are_excluded(self, tmp_path):
        """The potato-battery lines are recognised acoustically, not by chapter."""
        source, out = tmp_path / "src", tmp_path / "out"
        source.mkdir()
        self._write(source / "filtered.wav", 2000)
        report = build_dataset([self._line("filtered")], source, out)
        assert report.total_clips == 0
        assert report.excluded["lo-fi filtered (potato battery)"] == 1

    def test_full_band_clips_are_kept(self, tmp_path):
        source, out = tmp_path / "src", tmp_path / "out"
        source.mkdir()
        self._write(source / "clean.wav", 200)
        report = build_dataset([self._line("clean")], source, out)
        assert report.total_clips == 1

    def test_threshold_is_configurable(self, tmp_path):
        source, out = tmp_path / "src", tmp_path / "out"
        source.mkdir()
        self._write(source / "filtered.wav", 2000)
        report = build_dataset(
            [self._line("filtered")], source, out, min_low_mid_db=-1000.0
        )
        assert report.total_clips == 1

    def test_excluded_clips_are_recorded_for_audit(self, tmp_path):
        source, out = tmp_path / "src", tmp_path / "out"
        source.mkdir()
        self._write(source / "filtered.wav", 2000)
        build_dataset([self._line("filtered")], source, out)
        audit = (out / "filtered_clips.csv").read_text(encoding="utf-8")
        assert "filtered.wav" in audit

    def test_excluded_audio_is_not_left_on_disk(self, tmp_path):
        source, out = tmp_path / "src", tmp_path / "out"
        source.mkdir()
        self._write(source / "filtered.wav", 2000)
        build_dataset([self._line("filtered")], source, out)
        assert not list((out / "wavs").glob("*.wav"))


class TestDuplicateFilenames:
    def test_second_line_sharing_a_filename_is_dropped(self, tmp_path):
        """One clip with two transcripts is silent training-data corruption."""
        source, out = tmp_path / "src", tmp_path / "out"
        source.mkdir()
        rate = RATE
        t = np.arange(int(rate * 2.0)) / rate
        write_wav(
            source / "clip.wav",
            (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32),
            rate,
        )

        def line(text):
            return VoiceLine(
                url="https://example.test/clip.wav",
                transcript=text,
                page="portal2",
                section="Chapter 1",
                subsection="Test",
            )

        report = build_dataset(
            [line("First transcript here."), line("Second one.")], source, out
        )
        assert report.total_clips == 1
        assert report.excluded["duplicate filename"] == 1

    def test_metadata_has_no_repeated_filenames(self, tmp_path):
        source, out = tmp_path / "src", tmp_path / "out"
        source.mkdir()
        rate = RATE
        t = np.arange(int(rate * 2.0)) / rate
        write_wav(
            source / "clip.wav",
            (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32),
            rate,
        )
        line = VoiceLine(
            url="https://example.test/clip.wav",
            transcript="Only one row should survive.",
            page="portal2",
            section="Chapter 1",
            subsection="Test",
        )
        build_dataset([line, line], source, out)
        names = [
            r.split("|")[0]
            for r in (out / "metadata.csv").read_text().splitlines()
            if r
        ]
        assert len(names) == len(set(names)) == 1


class TestSoundEffectExclusion:
    def test_clip_whose_audio_holds_a_sound_is_dropped(self, corpus, tmp_path):
        source, add = corpus
        keep = add("keep.wav")
        drop = add("horn.wav")
        drop = VoiceLine(
            drop.url, drop.transcript, drop.page, "", "", has_audio_annotation=True
        )
        report = build_dataset([keep, drop], source, tmp_path / "out")

        names = (tmp_path / "out" / "metadata.csv").read_text(encoding="utf-8")
        assert "keep.wav" in names
        assert "horn.wav" not in names
        assert report.excluded["sound effect in audio"] == 1

    def test_its_wav_is_not_written(self, corpus, tmp_path):
        source, add = corpus
        drop = add("horn.wav")
        drop = VoiceLine(
            drop.url, drop.transcript, drop.page, "", "", has_audio_annotation=True
        )
        build_dataset([drop], source, tmp_path / "out")
        # Dropped before process_file runs, so nothing is written or deleted.
        assert not (tmp_path / "out" / "wavs" / "horn.wav").exists()


def _lofi(seconds: float) -> np.ndarray:
    """Make a mid-band tone with no energy below 300 Hz, tripping the potato gate."""
    t = np.arange(int(RATE * seconds)) / RATE
    return (0.5 * np.sin(2 * np.pi * 2000.0 * t)).astype(np.float32)


class TestPotatoSpeaker:
    @pytest.fixture
    def lofi_corpus(self, tmp_path):
        source = tmp_path / "audio"
        source.mkdir()
        write_wav(source / "clean.wav", _speech(3.0), RATE)
        write_wav(source / "potato.wav", _lofi(3.0), RATE)
        return source, [
            VoiceLine("https://x/clean.wav", "A normal line.", "portal2", "", ""),
            VoiceLine("https://x/potato.wav", "A filtered line.", "portal2", "", ""),
        ]

    def test_single_speaker_still_drops_them(self, lofi_corpus, tmp_path):
        source, lines = lofi_corpus
        report = build_dataset(lines, source, tmp_path / "out")
        meta = (tmp_path / "out" / "metadata.csv").read_text(encoding="utf-8")
        assert "potato.wav" not in meta
        assert report.excluded["lo-fi filtered (potato battery)"] == 1

    def test_multi_speaker_without_the_flag_still_drops_them(
        self, lofi_corpus, tmp_path
    ):
        source, lines = lofi_corpus
        build_dataset(lines, source, tmp_path / "out", multi_speaker=True)
        meta = (tmp_path / "out" / "metadata.csv").read_text(encoding="utf-8")
        assert "potato.wav" not in meta

    def test_flag_keeps_them_under_their_own_speaker(self, lofi_corpus, tmp_path):
        source, lines = lofi_corpus
        build_dataset(
            lines,
            source,
            tmp_path / "out",
            multi_speaker=True,
            potato_speaker=True,
        )
        meta = (tmp_path / "out" / "metadata.csv").read_text(encoding="utf-8")
        rows = {
            line.split("|")[0]: line.split("|")[1]
            for line in meta.splitlines()
            if line.strip()
        }
        assert rows["potato.wav"] == POTATO_SPEAKER
        # The clean clip must keep its own source ID, not be relabelled.
        assert rows["clean.wav"] == "p2"

    def test_audit_counts_potato_under_its_own_speaker(self, lofi_corpus, tmp_path):
        """The duration gate must match metadata.csv, not the source era.

        These clips used to be counted under p2 while being written as
        `potato`, so the audit could not answer the one question it exists for:
        does this speaker have enough audio to deserve an embedding?
        """
        source, lines = lofi_corpus
        report = build_dataset(
            lines,
            source,
            tmp_path / "out",
            multi_speaker=True,
            potato_speaker=True,
        )
        assert report.clips[POTATO_SPEAKER] == 1
        assert report.seconds[POTATO_SPEAKER] > 0
        assert report.clips["p2"] == 1
