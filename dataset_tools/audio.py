"""Normalize wiki audio into the form Piper's VITS trainer expects.

The wiki serves 16-bit PCM at 44.1 kHz, mono for the Portal 1 era and stereo
for much of Portal 2. Training wants mono 22.05 kHz, which is a clean 2:1
decimation - but only after low-pass filtering, so `scipy.signal.resample_poly`
does the work rather than naive sample dropping, which would alias.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

TARGET_RATE = 22050

# Silence trimming. Thresholds are relative to each clip's own peak, so a
# quietly mastered line is not trimmed more aggressively than a loud one.
SILENCE_FLOOR_DB = -40.0
TRIM_WINDOW_MS = 10.0
# Padding kept either side of detected speech. Trimming hard against the first
# sample clips consonant onsets and breaths, which VITS learns as clicks.
TRIM_PAD_MS = 50.0

# Peak level after normalization. The three speakers come from separately
# mastered games, so levels are matched rather than left as-is.
PEAK_TARGET_DB = -1.0

_INT16_MAX = 32767.0


class AudioError(RuntimeError):
    """A source file could not be decoded."""


@dataclass(frozen=True)
class Processed:
    """Result of normalizing one clip."""

    source_seconds: float
    output_seconds: float
    source_rate: int
    source_channels: int
    low_mid_db: float

    @property
    def trimmed_seconds(self) -> float:
        """Seconds removed by silence trimming."""
        return self.source_seconds - self.output_seconds


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM wav as float32 in [-1, 1], shaped (frames, channels)."""
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, OSError) as exc:
        raise AudioError(f"{path}: {exc}") from exc

    if width != 2:
        raise AudioError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / _INT16_MAX
    try:
        # A truncated file can leave a frame count that is not a whole number
        # of channels. Raised as AudioError so build_dataset skips the clip
        # rather than aborting a run that is most of the way through 1,800.
        return samples.reshape(-1, channels), rate
    except ValueError as exc:
        raise AudioError(f"{path}: truncated or malformed frame data: {exc}") from exc


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average channels down to a single mono track."""
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=1)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample with an anti-aliasing polyphase filter."""
    if source_rate == target_rate:
        return samples
    common = np.gcd(source_rate, target_rate)
    return np.asarray(
        resample_poly(samples, target_rate // common, source_rate // common),
        dtype=np.float32,
    )


def trim_silence(
    samples: np.ndarray,
    rate: int,
    *,
    floor_db: float = SILENCE_FLOOR_DB,
    window_ms: float = TRIM_WINDOW_MS,
    pad_ms: float = TRIM_PAD_MS,
) -> np.ndarray:
    """Trim leading and trailing silence, keeping a short pad either side."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0.0:
        return samples

    window = max(1, int(rate * window_ms / 1000.0))
    usable = (samples.size // window) * window
    if usable == 0:
        return samples
    windows = samples[:usable].reshape(-1, window)
    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    threshold = peak * (10.0 ** (floor_db / 20.0))
    loud = np.flatnonzero(rms >= threshold)
    if loud.size == 0:
        return samples

    pad = int(rate * pad_ms / 1000.0)
    start = max(0, loud[0] * window - pad)
    end = min(samples.size, (loud[-1] + 1) * window + pad)
    return samples[start:end]


def normalize_peak(
    samples: np.ndarray, *, target_db: float = PEAK_TARGET_DB
) -> np.ndarray:
    """Scale so the loudest sample sits at `target_db` dBFS."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0.0:
        return samples
    return np.asarray(samples * (10.0 ** (target_db / 20.0)) / peak, dtype=np.float32)


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    """Write mono float samples as 16-bit PCM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * _INT16_MAX).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        # pylint: disable=no-member
        # "wb" makes this a Wave_write; pylint infers Wave_read from the
        # signature alone and flags every setter below.
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


# Bands used to detect band-pass-filtered material. The Portal 2 potato-battery
# lines suppress everything below ~300 Hz by around 32 dB while boosting the
# 800-4000 Hz range, so the ratio between the two separates them cleanly from
# unprocessed speech. Measured on 120 clips of each: clean lines sit near
# +7 dB, filtered ones below -30 dB.
LOW_BAND_HZ = (100.0, 300.0)
MID_BAND_HZ = (800.0, 4000.0)
_FFT_SIZE = 1024


def low_mid_ratio_db(samples: np.ndarray, rate: int) -> float:
    """Ratio of low-band to mid-band energy, in dB.

    Identifies lo-fi filtering acoustically rather than by provenance. Chapter
    or page boundaries do not work for this: the same filtered scenes appear on
    the co-op page and in Chapter 5, so only the audio itself is reliable.

    Returns 0.0 (i.e. "unremarkable") for clips too short to analyse, so a tiny
    clip is never excluded on the strength of a meaningless measurement.
    """
    if samples.size < _FFT_SIZE * 2:
        return 0.0
    window = np.hanning(_FFT_SIZE)
    frames = [
        np.abs(np.fft.rfft(samples[i : i + _FFT_SIZE] * window)) ** 2
        for i in range(0, samples.size - _FFT_SIZE, _FFT_SIZE)
    ]
    if not frames:
        return 0.0
    psd = np.mean(frames, axis=0)
    freqs = np.fft.rfftfreq(_FFT_SIZE, 1.0 / rate)

    def band(bounds: tuple[float, float]) -> float:
        low, high = bounds
        return float(psd[(freqs >= low) & (freqs < high)].sum())

    mid = band(MID_BAND_HZ)
    if mid <= 0.0:
        return 0.0
    return float(10.0 * np.log10((band(LOW_BAND_HZ) + 1e-20) / mid))


def process_file(
    source: Path,
    destination: Path,
    *,
    target_rate: int = TARGET_RATE,
    trim: bool = True,
    normalize: bool = True,
) -> Processed:
    """Convert one wiki .wav into a training-ready mono clip."""
    raw, rate = read_wav(source)
    source_seconds = raw.shape[0] / rate if rate else 0.0
    channels = raw.shape[1] if raw.ndim > 1 else 1

    samples = to_mono(raw)
    samples = resample(samples, rate, target_rate)
    if trim:
        samples = trim_silence(samples, target_rate)
    if normalize:
        samples = normalize_peak(samples)
    write_wav(destination, samples, target_rate)

    return Processed(
        source_seconds=source_seconds,
        output_seconds=samples.size / target_rate,
        source_rate=rate,
        source_channels=channels,
        low_mid_db=low_mid_ratio_db(samples, target_rate),
    )
