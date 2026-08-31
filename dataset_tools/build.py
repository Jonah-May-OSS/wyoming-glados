"""Turn fetched audio and transcripts into a Piper training dataset.

Emits the metadata format Piper's trainer expects. Single speaker pools every
source into one voice and uses two columns::

    utt1.wav|Text for utterance 1.

Multi speaker keeps one ID per recording era and adds a column::

    utt1.wav|p2|Text for utterance 1.

The column count is not cosmetic: a single-speaker checkpoint has no speaker
embedding, so emitting three columns with one speaker name would build an
embedding the fine-tune checkpoint does not have.

The per-source duration audit is reported either way, since it is what decides
whether multi-speaker training is viable at all.
"""

from __future__ import annotations

import collections
from collections.abc import Container, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .audio import AudioError, process_file
from .fetch import safe_filename
from .portalwiki import VoiceLine

# Clips shorter than this are usually single barks with too little context to
# align well; longer ones inflate VRAM during training.
MIN_SECONDS = 0.7
MAX_SECONDS = 15.0

# Plausible speaking rate. Outside this band the transcript probably does not
# match the audio, which is the failure mode most damaging to a small
# fine-tune. Flagged for review rather than dropped, since GLaDOS delivers
# some lines very slowly on purpose.
MIN_CHARS_PER_SECOND = 3.0
MAX_CHARS_PER_SECOND = 30.0

# Clips whose low-band/mid-band energy ratio falls below this are band-pass
# filtered rather than plainly recorded - in this corpus, the potato-battery
# scenes. Clean lines measure around +7 dB and filtered ones below -30 dB, so
# the cut sits in the empty space between the two populations.
#
# This is checked acoustically, not by wiki chapter: the filtered scenes also
# appear on the co-op page and in Chapter 5, so provenance misses them. Pooled
# into a single voice the model learns the timbre as a variation it can apply
# to any word, and it leaks into unrelated output.
# Raised from -12.0: at -12 the closest surviving clip was
# fgbgladostransfer15 ("GET YOUR HANDS OFF ME! NO! STOP! No!") at -11.0 dB,
# and the next survivor above it sat at -7.0 dB. That 4 dB gap is a natural
# break, so -10.0 drops exactly that one clip and nothing else.
MIN_LOW_MID_DB = -10.0

# Speaker ID for the band-pass filtered potato-battery scenes, used only when
# multi-speaker training keeps them instead of dropping them.
POTATO_SPEAKER = "potato"

_DELIMITER = "|"


@dataclass
class BuildReport:
    """Outcome of a dataset build."""

    clips: collections.Counter[str] = field(default_factory=collections.Counter)
    # Durations are floats, so this cannot be a Counter: Counter's values are
    # typed int, and the annotation would misreport total_seconds.
    seconds: dict[str, float] = field(
        default_factory=lambda: collections.defaultdict(float)
    )
    excluded: collections.Counter[str] = field(default_factory=collections.Counter)
    suspicious: list[tuple[str, float]] = field(default_factory=list)
    filtered_out: list[tuple[str, float]] = field(default_factory=list)
    sanitized_transcripts: int = 0
    pruned: int = 0

    @property
    def total_clips(self) -> int:
        """Clips written across all speakers."""
        return sum(self.clips.values())

    @property
    def total_seconds(self) -> float:
        """Speech seconds written across all speakers."""
        return sum(self.seconds.values())

    def summary(self) -> str:
        """Human-readable audit, including the per-speaker duration gate."""
        lines = ["Speaker      clips   duration"]
        for speaker in sorted(self.clips):
            secs = self.seconds[speaker]
            lines.append(
                f"  {speaker:<10} {self.clips[speaker]:>5}   {secs / 60:>6.1f} min"
            )
        lines.append(
            f"  {'TOTAL':<10} {self.total_clips:>5}   "
            f"{self.total_seconds / 60:>6.1f} min"
        )
        if self.excluded:
            lines.append("Excluded:")
            for reason, count in self.excluded.most_common():
                lines.append(f"  {reason}: {count}")
        if self.sanitized_transcripts:
            lines.append(
                f"Transcripts with delimiters removed: {self.sanitized_transcripts}"
            )
        if self.pruned:
            lines.append(f"Stale clips removed from wavs/: {self.pruned}")
        if self.suspicious:
            lines.append(
                f"Suspicious transcript/duration ratio: {len(self.suspicious)} "
                "(review these)"
            )
        return "\n".join(lines)


def sanitize_transcript(text: str) -> str:
    """Strip characters that would corrupt the pipe-delimited metadata file."""
    return " ".join(text.replace(_DELIMITER, ",").split())


def build_dataset(
    lines: Iterable[VoiceLine],
    source_dir: Path,
    out_dir: Path,
    *,
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
    trim: bool = True,
    normalize: bool = True,
    multi_speaker: bool = False,
    min_low_mid_db: float = MIN_LOW_MID_DB,
    potato_speaker: bool = False,
) -> BuildReport:
    """Normalize audio and write `metadata.csv` alongside a `wavs/` directory.

    With `multi_speaker` false every source is pooled into one voice and the
    speaker column is omitted. The audit still breaks down by source either
    way, so the pooling decision can be revisited without a re-fetch.
    """
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    report = BuildReport()
    rows: list[str] = []
    kept_names: set[str] = set()

    for line in lines:
        source = source_dir / safe_filename(line.url)
        if not source.exists():
            report.excluded["missing audio"] += 1
            continue

        if line.has_audio_annotation:
            # The wiki marked a sound in this clip - a train horn, a phone
            # ringing, humming, laughter. clean_transcript() removes the note
            # from the TEXT, but the audio still contains the sound, so the
            # transcript no longer describes what is heard. That mismatch
            # corrupts alignment and the duration predictor, which shows up as
            # artifacts on unrelated words. The text half is fixable; the audio
            # half is not, so drop the clip.
            report.excluded["sound effect in audio"] += 1
            continue

        name = safe_filename(line.url)
        if name in kept_names:
            # Two wiki URLs reduced to the same local filename, so they share
            # one .wav on disk. Emitting both would pair a single clip with two
            # different transcripts, which is silent training-data corruption.
            # `fetch` warns about collisions; this is the backstop.
            report.excluded["duplicate filename"] += 1
            continue
        try:
            result = process_file(
                source, wav_dir / name, trim=trim, normalize=normalize
            )
        except AudioError:
            report.excluded["decode failed"] += 1
            continue

        # The potato-battery scenes carry a heavy band-pass filter. Pooled into
        # one voice the model learns that timbre as a variation it can apply to
        # any word, so single-speaker builds drop them. A speaker embedding
        # binds a timbre to an identity instead, which is exactly the problem
        # here - so in multi-speaker builds they become their own speaker and
        # PotatOS becomes selectable rather than discarded.
        speaker = line.speaker
        if result.low_mid_db < min_low_mid_db:
            if not (multi_speaker and potato_speaker):
                report.excluded["lo-fi filtered (potato battery)"] += 1
                report.filtered_out.append((name, result.low_mid_db))
                (wav_dir / name).unlink(missing_ok=True)
                continue
            speaker = POTATO_SPEAKER

        duration = result.output_seconds
        if duration < min_seconds:
            report.excluded[f"shorter than {min_seconds}s"] += 1
            (wav_dir / name).unlink(missing_ok=True)
            continue
        if duration > max_seconds:
            report.excluded[f"longer than {max_seconds}s"] += 1
            (wav_dir / name).unlink(missing_ok=True)
            continue

        transcript = sanitize_transcript(line.transcript)
        if transcript != line.transcript:
            report.sanitized_transcripts += 1

        rate = len(transcript) / duration if duration else 0.0
        if not MIN_CHARS_PER_SECOND <= rate <= MAX_CHARS_PER_SECOND:
            report.suspicious.append((name, rate))

        if multi_speaker:
            rows.append(f"{name}{_DELIMITER}{speaker}{_DELIMITER}{transcript}")
        else:
            rows.append(f"{name}{_DELIMITER}{transcript}")
        kept_names.add(name)
        # Count under the speaker actually written to metadata.csv, not the
        # source era. summary() is the per-speaker duration gate that decides
        # whether a speaker has enough audio to justify an embedding, and
        # potato clips booked against p2 made that undecidable for the one
        # speaker whose viability is genuinely in question.
        report.clips[speaker] += 1
        report.seconds[speaker] += duration

    (out_dir / "metadata.csv").write_text(
        "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
    )
    report.pruned = _prune_orphans(wav_dir, kept_names)
    _write_suspicious(out_dir, report.suspicious)
    _write_filtered(out_dir, report.filtered_out)
    return report


def _write_filtered(out_dir: Path, filtered: Sequence[tuple[str, float]]) -> None:
    """Record which clips the spectral gate removed, so the cut is auditable."""
    path = out_dir / "filtered_clips.csv"
    if not filtered:
        path.unlink(missing_ok=True)
        return
    body = "\n".join(f"{name}{_DELIMITER}{db:.1f}" for name, db in sorted(filtered))
    path.write_text(f"filename{_DELIMITER}low_mid_db\n{body}\n", encoding="utf-8")


def _prune_orphans(wav_dir: Path, kept: Container[str]) -> int:
    """Delete clips in `wav_dir` that the new metadata.csv does not reference.

    Rebuilding with tighter filters leaves the previously written clips behind.
    Nothing reads them - the trainer selects from metadata.csv - but a stray
    `wavs/*.wav` glob then silently trains on material that was deliberately
    excluded, which is the kind of bug that only shows up in the output voice.
    """
    removed = 0
    for path in wav_dir.glob("*.wav"):
        if path.name not in kept:
            path.unlink()
            removed += 1
    return removed


def _write_suspicious(out_dir: Path, suspicious: Sequence[tuple[str, float]]) -> None:
    """Write flagged clips to a review file, or remove a stale one."""
    path = out_dir / "review_transcripts.csv"
    if not suspicious:
        path.unlink(missing_ok=True)
        return
    body = "\n".join(f"{name}{_DELIMITER}{rate:.1f}" for name, rate in suspicious)
    path.write_text(f"filename{_DELIMITER}chars_per_second\n{body}\n", encoding="utf-8")
