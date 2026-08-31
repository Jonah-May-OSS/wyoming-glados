"""Tests for dataset audio normalization."""

import wave

import numpy as np
import pytest

from dataset_tools.audio import (
    TARGET_RATE,
    AudioError,
    low_mid_ratio_db,
    normalize_peak,
    process_file,
    read_wav,
    resample,
    to_mono,
    trim_silence,
    write_wav,
)


def _write_pcm(path, samples, rate=44100, channels=1, width=2):
    pcm = np.round(np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _tone(seconds=1.0, rate=44100, freq=440.0, amp=0.5):
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestReadWav:
    def test_reads_mono(self, tmp_path):
        path = tmp_path / "a.wav"
        _write_pcm(path, _tone(0.1))
        samples, rate = read_wav(path)
        assert rate == 44100
        assert samples.shape == (4410, 1)

    def test_reads_stereo_interleaved(self, tmp_path):
        path = tmp_path / "a.wav"
        left = np.full(100, 0.5, dtype=np.float32)
        right = np.full(100, -0.5, dtype=np.float32)
        _write_pcm(path, np.column_stack([left, right]).ravel(), channels=2)
        samples, _ = read_wav(path)
        assert samples.shape == (100, 2)
        assert samples[0, 0] > 0 > samples[0, 1]

    def test_rejects_non_16_bit(self, tmp_path):
        path = tmp_path / "a.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(1)
            handle.setframerate(44100)
            handle.writeframes(b"\x00" * 100)
        with pytest.raises(AudioError, match="8-bit"):
            read_wav(path)

    def test_rejects_unreadable_file(self, tmp_path):
        path = tmp_path / "a.wav"
        path.write_bytes(b"not a wav")
        with pytest.raises(AudioError):
            read_wav(path)


class TestToMono:
    def test_averages_channels(self):
        stereo = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        assert to_mono(stereo).tolist() == [0.5, 0.5]

    def test_passes_through_1d(self):
        mono = np.array([0.1, 0.2], dtype=np.float32)
        assert to_mono(mono).tolist() == pytest.approx([0.1, 0.2])


class TestResample:
    def test_halves_length_for_2_to_1(self):
        assert resample(_tone(1.0), 44100, 22050).size == 22050

    def test_is_a_no_op_at_matching_rates(self):
        samples = _tone(0.1)
        assert resample(samples, 22050, 22050) is samples

    def test_preserves_a_tone_below_nyquist(self):
        out = resample(_tone(1.0, freq=440.0), 44100, 22050)
        spectrum = np.abs(np.fft.rfft(out))
        peak_hz = np.fft.rfftfreq(out.size, 1 / 22050)[int(np.argmax(spectrum))]
        assert peak_hz == pytest.approx(440.0, abs=2.0)

    def test_attenuates_content_above_new_nyquist(self):
        """Naive decimation would alias a 15 kHz tone down into the audible band.

        Measured away from the edges: the filter's startup transient is a much
        larger excursion than the steady-state stopband response.
        """
        out = resample(_tone(1.0, freq=15000.0), 44100, 22050)
        edge = out.size // 10
        steady_peak = float(np.max(np.abs(out[edge:-edge])))
        assert steady_peak < 0.005


class TestTrimSilence:
    def _padded(self, rate=22050):
        silence = np.zeros(rate, dtype=np.float32)
        return np.concatenate([silence, _tone(1.0, rate=rate), silence])

    def test_removes_leading_and_trailing_silence(self):
        trimmed = trim_silence(self._padded(), 22050, pad_ms=0.0)
        assert trimmed.size == pytest.approx(22050, abs=500)

    def test_keeps_requested_padding(self):
        no_pad = trim_silence(self._padded(), 22050, pad_ms=0.0)
        padded = trim_silence(self._padded(), 22050, pad_ms=50.0)
        assert padded.size > no_pad.size

    def test_returns_all_silence_unchanged(self):
        silent = np.zeros(1000, dtype=np.float32)
        assert trim_silence(silent, 22050).size == 1000

    def test_handles_empty_input(self):
        assert trim_silence(np.zeros(0, dtype=np.float32), 22050).size == 0

    def test_handles_clip_shorter_than_one_window(self):
        tiny = _tone(0.001, rate=22050)
        assert trim_silence(tiny, 22050).size == tiny.size


class TestNormalizePeak:
    def test_scales_to_target(self):
        out = normalize_peak(_tone(0.1, amp=0.1), target_db=-1.0)
        assert float(np.max(np.abs(out))) == pytest.approx(0.891, abs=0.01)

    def test_attenuates_when_too_loud(self):
        out = normalize_peak(np.array([1.0, -1.0], dtype=np.float32))
        assert float(np.max(np.abs(out))) < 1.0

    def test_leaves_silence_alone(self):
        silent = np.zeros(10, dtype=np.float32)
        assert float(np.max(np.abs(normalize_peak(silent)))) == 0.0


class TestWriteWav:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "out.wav"
        write_wav(path, _tone(0.1, rate=22050), 22050)
        samples, rate = read_wav(path)
        assert rate == 22050
        assert samples.shape == (2205, 1)

    def test_clips_out_of_range_samples(self, tmp_path):
        path = tmp_path / "out.wav"
        write_wav(path, np.array([2.0, -2.0], dtype=np.float32), 22050)
        samples, _ = read_wav(path)
        assert float(np.max(np.abs(samples))) <= 1.0

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "out.wav"
        write_wav(path, _tone(0.01, rate=22050), 22050)
        assert path.exists()


class TestProcessFile:
    def test_converts_stereo_44k_to_mono_22k(self, tmp_path):
        source = tmp_path / "in.wav"
        mono = _tone(1.0)
        _write_pcm(source, np.column_stack([mono, mono]).ravel(), channels=2)

        out = tmp_path / "out.wav"
        result = process_file(source, out)

        assert (result.source_rate, result.source_channels) == (44100, 2)
        samples, rate = read_wav(out)
        assert rate == TARGET_RATE
        assert samples.shape[1] == 1

    def test_reports_trimmed_duration(self, tmp_path):
        source = tmp_path / "in.wav"
        silence = np.zeros(44100, dtype=np.float32)
        _write_pcm(source, np.concatenate([silence, _tone(1.0), silence]))
        result = process_file(tmp_path / "in.wav", tmp_path / "out.wav")
        assert result.source_seconds == pytest.approx(3.0, abs=0.01)
        assert result.trimmed_seconds > 1.5

    def test_trim_and_normalize_can_be_disabled(self, tmp_path):
        source = tmp_path / "in.wav"
        _write_pcm(source, _tone(1.0, amp=0.1))
        result = process_file(source, tmp_path / "out.wav", trim=False, normalize=False)
        assert result.output_seconds == pytest.approx(1.0, abs=0.01)
        samples, _ = read_wav(tmp_path / "out.wav")
        assert float(np.max(np.abs(samples))) == pytest.approx(0.1, abs=0.01)


class TestLowMidRatio:
    def _tone(self, freq, rate=22050, seconds=0.5):
        t = np.arange(int(rate * seconds)) / rate
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    def test_low_heavy_audio_scores_positive(self):
        assert low_mid_ratio_db(self._tone(200), 22050) > 0

    def test_band_limited_audio_scores_negative(self):
        """Removing the low band is exactly what the potato filter does."""
        assert low_mid_ratio_db(self._tone(2000), 22050) < -12

    def test_short_clips_are_treated_as_unremarkable(self):
        """Too little data to judge; must not be excluded on a bad measurement."""
        assert low_mid_ratio_db(np.zeros(100, dtype=np.float32), 22050) == 0.0

    def test_silence_does_not_divide_by_zero(self):
        assert low_mid_ratio_db(np.zeros(8192, dtype=np.float32), 22050) == 0.0

    def test_mixed_content_sits_between_the_extremes(self):
        mixed = self._tone(200) + self._tone(2000)
        score = low_mid_ratio_db(mixed, 22050)
        assert -12 < score < 12


RATE = TARGET_RATE


class TestMalformedAudio:
    def test_truncated_frames_raise_audio_error(self, tmp_path, monkeypatch):
        """A corrupt clip must not abort a build most of the way through."""
        path = tmp_path / "broken.wav"
        write_wav(path, np.zeros(1000, dtype=np.float32), RATE)

        import wave as wave_mod

        real_open = wave_mod.open

        class Truncated:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._inner.close()
                return False

            def getnchannels(self):
                return 3  # frame data is not a whole number of channels

            def getsampwidth(self):
                return 2

            def getframerate(self):
                return RATE

            def readframes(self, n):
                return self._inner.readframes(n)

            def getnframes(self):
                return self._inner.getnframes()

        monkeypatch.setattr(
            wave_mod, "open", lambda p, m="rb": Truncated(real_open(p, m))
        )
        with pytest.raises(AudioError, match="truncated or malformed"):
            read_wav(path)
