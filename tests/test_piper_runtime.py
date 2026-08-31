"""Tests for the ONNX Runtime Piper backend."""

import logging

import numpy as np
import pytest

from piper_runtime.runner import (
    _MISSING_PROFILE_MARKER,
    BUILDER_OPTIMIZATION_LEVEL,
    DEFAULT_NOISE_SCALE,
    DEFAULT_NOISE_W,
    PROFILE_PHONEMES,
    WARMUP_PHONEMES,
    PiperTTSRunner,
    build_providers,
    build_scales,
    engine_cache_dir,
    float_to_pcm16,
    missing_profile_inputs,
    model_fingerprint,
    phonemes_to_ids,
    profile_for,
    resolve_speaker_id,
    session_wants_scales,
    session_wants_sid,
)

ID_MAP = {"^": [1], "_": [0], "$": [2], "h": [10], "i": [11], "ˈoʊ": [12]}


class FakeSession:
    """Returns a fixed tone, recording the feeds it was given."""

    def __init__(self, samples=None):
        self.samples = (
            samples if samples is not None else np.full(100, 0.5, dtype=np.float32)
        )
        self.feeds = []

    def run(self, output_names, input_feed):
        self.feeds.append(input_feed)
        return [self.samples]

    def get_providers(self):
        return ["CPUExecutionProvider"]


class FakePhonemizer:
    """Splits on '.' into sentences of single-character phonemes."""

    def phonemize(self, voice, text):
        self.voice = voice
        return [list(part.strip()) for part in text.split(".") if part.strip()]


def _runner(session=None, phonemizer=None):
    return PiperTTSRunner(
        session=session or FakeSession(),
        phonemizer=phonemizer or FakePhonemizer(),
        phoneme_id_map=ID_MAP,
    )


class TestPhonemesToIds:
    def test_wraps_with_bos_pad_and_eos(self):
        assert phonemes_to_ids(["h", "i"], ID_MAP) == [1, 0, 10, 0, 11, 0, 2]

    def test_empty_phonemes_still_produce_markers(self):
        assert phonemes_to_ids([], ID_MAP) == [1, 0, 2]

    def test_unknown_phonemes_are_skipped(self):
        """Matches piper.phoneme_ids, which warns and drops unmapped phonemes."""
        assert phonemes_to_ids(["h", "ZZ", "i"], ID_MAP) == [1, 0, 10, 0, 11, 0, 2]

    def test_multi_id_phonemes_are_expanded(self):
        id_map = dict(ID_MAP, x=[20, 21])
        assert phonemes_to_ids(["x"], id_map) == [1, 0, 20, 21, 0, 2]


class TestBuildProviders:
    def test_tensorrt_is_first_when_enabled(self, tmp_path):
        providers = build_providers(tmp_path, use_trt=True)
        assert providers[0][0] == "TensorrtExecutionProvider"

    def test_falls_back_through_cuda_then_cpu(self, tmp_path):
        providers = build_providers(tmp_path, use_trt=True)
        assert providers[-2:] == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_tensorrt_omitted_when_disabled(self, tmp_path):
        assert build_providers(tmp_path, use_trt=False) == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_tensorrt_omitted_without_a_cache_dir(self):
        assert "TensorrtExecutionProvider" not in build_providers(None, use_trt=True)

    def test_fp16_is_disabled(self, tmp_path):
        """FP32 measured faster than FP16 and avoids the layernorm overflow."""
        options = build_providers(tmp_path, use_trt=True)[0][1]
        assert options["trt_fp16_enable"] is False

    def test_engine_and_timing_caches_are_enabled(self, tmp_path):
        options = build_providers(tmp_path, use_trt=True)[0][1]
        assert options["trt_engine_cache_enable"] is True
        assert options["trt_timing_cache_enable"] is True
        assert options["trt_engine_cache_path"] == str(tmp_path)

    def test_no_profiles_unless_discovered(self, tmp_path):
        """Nothing is invented: with no discovered profiles, none are set.

        This used to assert that profiles are ALWAYS absent, on the belief that
        inferred profiles were fine. They are not -- without them TensorRT
        rebuilds per shape (2939 ms/utterance against 9.9 ms). The absence of
        profile keys is only correct for the no-discovery case below.
        """
        options = build_providers(tmp_path, use_trt=True)[0][1]
        assert not [key for key in options if "profile" in key]

    def test_discovered_profiles_are_passed_through(self, tmp_path):
        """Discovered profiles must reach the EP verbatim, not be dropped."""
        profiles = {
            "trt_profile_min_shapes": "input:1x8",
            "trt_profile_opt_shapes": "input:1x128",
            "trt_profile_max_shapes": "input:1x512",
        }
        options = build_providers(tmp_path, use_trt=True, profiles=profiles)[0][1]
        for key, value in profiles.items():
            assert options[key] == value


class TestFloatToPcm16:
    def test_scales_to_int16(self):
        pcm = float_to_pcm16(np.array([1.0, -1.0, 0.0], dtype=np.float32))
        assert np.frombuffer(pcm, dtype="<i2").tolist() == [32767, -32767, 0]

    def test_clips_out_of_range(self):
        pcm = float_to_pcm16(np.array([2.0, -2.0], dtype=np.float32))
        assert np.frombuffer(pcm, dtype="<i2").tolist() == [32767, -32767]

    def test_flattens_multidimensional_output(self):
        """The model emits (batch, 1, 1, samples)."""
        pcm = float_to_pcm16(np.zeros((1, 1, 1, 64), dtype=np.float32))
        assert len(pcm) == 64 * 2

    def test_two_bytes_per_sample(self):
        assert len(float_to_pcm16(np.zeros(10, dtype=np.float32))) == 20


class TestBuildScales:
    def test_alpha_maps_to_length_scale(self):
        assert build_scales(1.5).tolist() == pytest.approx(
            [DEFAULT_NOISE_SCALE, 1.5, DEFAULT_NOISE_W]
        )

    def test_noise_can_be_overridden(self):
        scales = build_scales(1.0, noise_scale=0.0, noise_w=0.0)
        assert scales.tolist() == pytest.approx([0.0, 1.0, 0.0])

    def test_dtype_is_float32(self):
        assert build_scales(1.0).dtype == np.float32


class TestRunTtsStream:
    def test_yields_one_chunk_per_sentence(self):
        runner = _runner()
        chunks = list(runner.run_tts_stream("hi. ho."))
        assert len(chunks) == 2
        assert all(isinstance(c, bytes) and c for c in chunks)

    def test_empty_text_yields_nothing(self):
        assert list(_runner().run_tts_stream("   ")) == []

    def test_feeds_have_expected_shapes(self):
        session = FakeSession()
        list(_runner(session=session).run_tts_stream("hi."))
        feed = session.feeds[0]
        assert feed["input"].ndim == 2
        assert feed["input"].dtype == np.int64
        # Bucketing pads the tensor, so the true length is <= the padded width.
        assert feed["input_lengths"][0] <= feed["input"].shape[1]
        assert feed["input_lengths"].dtype == np.int64
        assert feed["scales"].shape == (3,)

    def test_alpha_is_passed_through_as_length_scale(self):
        session = FakeSession()
        list(_runner(session=session).run_tts_stream("hi.", alpha=1.4))
        assert session.feeds[0]["scales"][1] == pytest.approx(1.4)

    def test_silent_output_is_still_emitted(self):
        """Zero samples are valid audio; only empty byte strings are skipped."""
        session = FakeSession(samples=np.zeros(50, dtype=np.float32))
        assert len(list(_runner(session=session).run_tts_stream("hi."))) == 1

    def test_sentence_with_no_phonemes_is_skipped(self):
        class EmptySentencePhonemizer:
            def phonemize(self, voice, text):
                return [[], ["h"]]

        session = FakeSession()
        runner = _runner(session=session, phonemizer=EmptySentencePhonemizer())
        assert len(list(runner.run_tts_stream("anything"))) == 1

    def test_phonemizer_is_called_with_en_us(self):
        phonemizer = FakePhonemizer()
        list(_runner(phonemizer=phonemizer).run_tts_stream("hi."))
        assert phonemizer.voice == "en-us"


class TestConstruction:
    def test_injected_session_requires_a_phoneme_id_map(self):
        with pytest.raises(ValueError, match="phoneme_id_map"):
            PiperTTSRunner(session=FakeSession(), phonemizer=FakePhonemizer())

    def test_synthesize_without_session_raises(self):
        runner = _runner()
        runner.session = None
        with pytest.raises(RuntimeError, match="Session"):
            runner.synthesize_ids([1, 0, 2])

    def test_over_long_sentence_warns_about_the_profile_bound(self, caplog):
        """Crossing the profile bound must be diagnosable, not opaque."""
        runner = _runner(session=FakeSession())
        with caplog.at_level(logging.WARNING, logger="piper_runtime.runner"):
            runner.synthesize_ids([1] * (PROFILE_PHONEMES[2] + 1))
        assert "shape profile" in caplog.text

    def test_normal_sentence_does_not_warn(self, caplog):
        runner = _runner(session=FakeSession())
        with caplog.at_level(logging.WARNING, logger="piper_runtime.runner"):
            runner.synthesize_ids([1] * 32)
        assert "shape profile" not in caplog.text


class TestPhonemizeThreadSafety:
    """espeak-ng holds its state in C globals; the pipeline runs three at once."""

    def test_phonemize_is_serialised(self):
        import threading
        import time

        peak = []
        active = 0
        guard = threading.Lock()

        class RacyPhonemizer:
            def phonemize(self, _voice, _text):
                nonlocal active
                with guard:
                    active += 1
                    peak.append(active)
                time.sleep(0.01)  # widen the window a real espeak call has
                with guard:
                    active -= 1
                return [["h"]]

        runner = _runner(phonemizer=RacyPhonemizer())
        threads = [
            threading.Thread(target=runner.phonemize, args=("hello",)) for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert max(peak) == 1, "two threads were inside espeak at once"


class TestShapePassthrough:
    """Bucketing is gone: explicit TensorRT profiles cover the shape range, so
    padding every utterance up to one of seven sizes was pure waste.

    The one exception is below the profile's lower bound, where padding is what
    keeps the shape inside the engine.
    """

    def test_input_is_not_padded(self):
        session = FakeSession()
        runner = _runner(session=session)
        length = PROFILE_PHONEMES[0] + 4
        runner.synthesize_ids([1] * length)
        feed = session.feeds[0]
        assert feed["input"].shape == (1, length)
        assert feed["input_lengths"].tolist() == [length]

    def test_lengths_pass_through_distinctly(self):
        session = FakeSession()
        runner = _runner(session=session)
        lengths = (PROFILE_PHONEMES[0], 20, 31, 64)
        for n in lengths:
            runner.synthesize_ids([1] * n)
        shapes = {tuple(f["input"].shape) for f in session.feeds}
        assert shapes == {(1, n) for n in lengths}

    def test_short_input_is_padded_up_to_the_profile_minimum(self):
        """Under the lower bound TensorRT rebuilds per shape, silently.

        "Yes." is a real utterance for a voice assistant and lands well under
        the bound, so this is the common case, not an edge case.
        """
        session = FakeSession()
        runner = _runner(session=session)
        runner.synthesize_ids([1, 0, 10])
        feed = session.feeds[0]
        assert feed["input"].shape == (1, PROFILE_PHONEMES[0])
        # The true count is preserved, so the model masks the padding and the
        # audio is unchanged.
        assert feed["input_lengths"].tolist() == [3]

    def test_padding_uses_the_pad_id(self):
        session = FakeSession()
        runner = _runner(session=session)
        runner.synthesize_ids([1, 0, 10])
        tail = session.feeds[0]["input"][0][3:].tolist()
        assert tail == [runner._pad_id] * (PROFILE_PHONEMES[0] - 3)


class TestWarmup:
    def test_warmup_runs_exactly_one_synthesis(self):
        """It is a smoke test now, not a shape-profile widening exercise."""
        session = FakeSession()
        _runner(session=session).warmup()
        assert len(session.feeds) == 1

    def test_warmup_uses_a_voiced_phoneme_not_padding(self):
        """Padding predicts near-zero durations, priming unrealistic shapes."""
        session = FakeSession()
        _runner(session=session).warmup()
        ids = set(session.feeds[0]["input"][0].tolist())
        assert ID_MAP["i"][0] in ids

    def test_warmup_prefers_the_earliest_available_phoneme(self):
        session = FakeSession()
        PiperTTSRunner(
            session=session,
            phonemizer=FakePhonemizer(),
            phoneme_id_map=dict(ID_MAP, **{WARMUP_PHONEMES[0]: [77]}),
        ).warmup()
        assert 77 in set(session.feeds[0]["input"][0].tolist())

    def test_warmup_falls_back_to_padding_without_a_known_vowel(self):
        session = FakeSession()
        PiperTTSRunner(
            session=session,
            phonemizer=FakePhonemizer(),
            phoneme_id_map={"^": [1], "_": [0], "$": [2]},
        ).warmup()
        assert set(session.feeds[0]["input"][0].tolist()) == {0}


class FakeInput:
    def __init__(self, name):
        self.name = name


class BakedSession(FakeSession):
    """A model exported with scales frozen in as an initializer."""

    def get_inputs(self):
        return [FakeInput("input"), FakeInput("input_lengths")]


class UnbakedSession(FakeSession):
    """An older voice that still takes scales at runtime."""

    def get_inputs(self):
        return [FakeInput("input"), FakeInput("input_lengths"), FakeInput("scales")]


class TestSessionWantsScales:
    def test_true_when_the_model_declares_a_scales_input(self):
        assert session_wants_scales(UnbakedSession()) is True

    def test_false_when_scales_are_baked_in(self):
        assert session_wants_scales(BakedSession()) is False

    def test_assumes_true_when_inputs_cannot_be_inspected(self):
        """Injected test doubles need not implement get_inputs."""
        assert session_wants_scales(FakeSession()) is True


class TestBakedModels:
    def test_scales_are_omitted_for_a_baked_model(self):
        session = BakedSession()
        list(_runner(session=session).run_tts_stream("hi."))
        assert "scales" not in session.feeds[0]

    def test_scales_are_still_sent_to_an_unbaked_model(self):
        session = UnbakedSession()
        list(_runner(session=session).run_tts_stream("hi."))
        assert session.feeds[0]["scales"].shape == (3,)

    def test_alpha_is_ignored_rather_than_failing_when_baked(self):
        """Speech rate is fixed at export time; requests must not error."""
        session = BakedSession()
        chunks = list(_runner(session=session).run_tts_stream("hi.", alpha=1.4))
        assert len(chunks) == 1
        assert "scales" not in session.feeds[0]


class TestEngineCacheDir:
    def _model(self, tmp_path, name, content):
        path = tmp_path / name
        path.write_bytes(content)
        return path

    def test_same_content_gives_the_same_directory(self, tmp_path):
        a = self._model(tmp_path, "a.onnx", b"weights")
        b = self._model(tmp_path, "b.onnx", b"weights")
        assert engine_cache_dir(tmp_path, a) == engine_cache_dir(tmp_path, b)

    def test_different_weights_give_different_directories(self, tmp_path):
        """ONNX Runtime keys engines on topology, so retrains would collide."""
        old = self._model(tmp_path, "old.onnx", b"epoch-33")
        new = self._model(tmp_path, "new.onnx", b"epoch-64")
        assert engine_cache_dir(tmp_path, old) != engine_cache_dir(tmp_path, new)

    def test_lives_under_the_shared_trt_cache_root(self, tmp_path):
        model = self._model(tmp_path, "v.onnx", b"weights")
        assert engine_cache_dir(tmp_path, model).parent == tmp_path / "trt_cache"

    def test_reverting_reuses_the_earlier_directory(self, tmp_path):
        model = self._model(tmp_path, "v.onnx", b"epoch-33")
        first = engine_cache_dir(tmp_path, model)
        model.write_bytes(b"epoch-64")
        assert engine_cache_dir(tmp_path, model) != first
        model.write_bytes(b"epoch-33")
        assert engine_cache_dir(tmp_path, model) == first


class TestModelFingerprint:
    def test_is_stable_across_chunk_sizes(self, tmp_path):
        path = tmp_path / "m.onnx"
        path.write_bytes(b"x" * 5000)
        assert model_fingerprint(path, chunk_size=16) == model_fingerprint(path)

    def test_is_short_and_hex(self, tmp_path):
        path = tmp_path / "m.onnx"
        path.write_bytes(b"weights")
        digest = model_fingerprint(path)
        assert len(digest) == 16
        assert all(c in "0123456789abcdef" for c in digest)


class TestTimingCache:
    def test_timing_cache_can_be_shared_across_models(self, tmp_path):
        """Kernel timings describe the GPU, so a retrain should reuse them."""
        engines, timings = tmp_path / "abc123", tmp_path
        options = build_providers(engines, use_trt=True, timing_cache_dir=timings)[0][1]
        assert options["trt_engine_cache_path"] == str(engines)
        assert options["trt_timing_cache_path"] == str(timings)

    def test_timing_cache_defaults_to_the_engine_directory(self, tmp_path):
        options = build_providers(tmp_path, use_trt=True)[0][1]
        assert options["trt_timing_cache_path"] == str(tmp_path)

    def test_timing_cache_is_forced(self, tmp_path):
        """Without this ORT discards the cache when GPU clocks drift."""
        options = build_providers(tmp_path, use_trt=True)[0][1]
        assert options["trt_force_timing_cache"] is True

    def test_builder_optimization_level_is_lowered(self, tmp_path):
        """ORT defaults to 3; cold builds are the dominant cost here."""
        options = build_providers(tmp_path, use_trt=True)[0][1]
        assert options["trt_builder_optimization_level"] == BUILDER_OPTIMIZATION_LEVEL
        assert BUILDER_OPTIMIZATION_LEVEL < 3

    def test_builder_level_is_overridable(self, tmp_path):
        options = build_providers(tmp_path, use_trt=True, builder_optimization_level=5)[
            0
        ][1]
        assert options["trt_builder_optimization_level"] == 5


class TestSpeakerSelection:
    def test_unknown_speaker_is_an_error_not_a_silent_default(self):
        # Falling back to id 0 would serve a different character than asked
        # for, with nothing in the logs to say so.
        with pytest.raises(ValueError, match="Unknown speaker"):
            resolve_speaker_id({"speaker_id_map": {"p2": 1}}, "potato")

    def test_name_resolves_through_the_voice_config(self):
        id_map = {"speaker_id_map": {"p1": 0, "p2": 1, "dota2": 2, "potato": 3}}
        assert resolve_speaker_id(id_map, "p2") == 1
        assert resolve_speaker_id(id_map, "potato") == 3

    def test_no_speaker_means_first_id(self):
        assert resolve_speaker_id({"speaker_id_map": {"p2": 1}}, None) == 0

    def test_numeric_speaker_passes_through(self):
        # A voice that declares no speaker count cannot contradict the id.
        assert resolve_speaker_id({}, 2) == 2
        assert resolve_speaker_id({}, "2") == 2

    def test_out_of_range_id_is_rejected_at_startup(self):
        # This used to reach the model and index the speaker embedding out of
        # bounds, failing on the first request instead of at load.
        cfg = {"num_speakers": 4, "speaker_id_map": {"p1": 0, "p2": 1}}
        for bad in (4, "4", -1, 99):
            with pytest.raises(ValueError, match="out of range"):
                resolve_speaker_id(cfg, bad)

    def test_in_range_ids_are_accepted(self):
        cfg = {"num_speakers": 4, "speaker_id_map": {"p1": 0, "p2": 1}}
        assert resolve_speaker_id(cfg, 0) == 0
        assert resolve_speaker_id(cfg, "3") == 3

    def test_single_speaker_voice_ignores_a_bad_speaker(self):
        """--speaker is documented as ignored here, so it must not crash.

        Resolving unconditionally turned any unrecognized SPEAKER into a
        startup failure on a voice whose graph has no sid input at all.
        """

        class SingleSpeaker(FakeSession):
            def get_inputs(self):
                return [FakeInput("input"), FakeInput("input_lengths")]

        runner = PiperTTSRunner(
            session=SingleSpeaker(),
            phonemizer=FakePhonemizer(),
            phoneme_id_map=ID_MAP,
            speaker="nobody-by-that-name",
        )
        assert runner.wants_sid is False
        assert runner.speaker_id == 0

    def test_multi_speaker_voice_still_rejects_a_bad_speaker(self):
        class MultiSpeaker(FakeSession):
            def get_inputs(self):
                return [
                    FakeInput("input"),
                    FakeInput("input_lengths"),
                    FakeInput("sid"),
                ]

        with pytest.raises(ValueError, match="Unknown speaker"):
            PiperTTSRunner(
                session=MultiSpeaker(),
                phonemizer=FakePhonemizer(),
                phoneme_id_map=ID_MAP,
                speaker="nobody-by-that-name",
            )

    def test_range_falls_back_to_the_id_map(self):
        # num_speakers is absent on some configs; the map still bounds it.
        cfg = {"speaker_id_map": {"p1": 0, "p2": 1}}
        assert resolve_speaker_id(cfg, 1) == 1
        with pytest.raises(ValueError, match="out of range"):
            resolve_speaker_id(cfg, 2)

    def test_single_speaker_model_needs_no_sid(self):
        class Inp:
            def __init__(self, name):
                self.name = name

        class Sess:
            def get_inputs(self):
                return [Inp("input"), Inp("input_lengths"), Inp("scales")]

        assert session_wants_sid(Sess()) is False

    def test_multi_speaker_model_is_detected(self):
        class Inp:
            def __init__(self, name):
                self.name = name

        class Sess:
            def get_inputs(self):
                return [Inp("input"), Inp("input_lengths"), Inp("scales"), Inp("sid")]

        assert session_wants_sid(Sess()) is True


class TestShapeProfiles:
    """Without profiles TensorRT recompiles per shape, and the stochastic
    duration predictor gives a new decoder length every request."""

    ERROR = (
        "FAIL : User needs to provide all the dynamic shape inputs with "
        "associated profiles. Following input(s) has no associated shape "
        "profiles provided: /Transpose_4_output_0,/Range_output_0"
    )

    def test_parses_the_names_onnxruntime_asks_for(self):
        assert missing_profile_inputs(self.ERROR) == [
            "/Transpose_4_output_0",
            "/Range_output_0",
        ]

    def test_unrelated_errors_yield_nothing(self):
        assert missing_profile_inputs("some other failure entirely") == []

    def test_only_the_marker_line_is_parsed(self):
        """ORT errors are multi-line; the trailing prose is not a tensor name."""
        message = (
            "Failed: "
            + _MISSING_PROFILE_MARKER
            + "/Range_output_0,/Unsqueeze_output_0"
            + chr(10)
            + " Please run shape inference on the model first."
        )

        names = missing_profile_inputs(message)

        assert names == ["/Range_output_0", "/Unsqueeze_output_0"]

    def test_an_accepted_probe_is_kept_not_discarded(self, tmp_path, monkeypatch):
        """The probe IS a profile set, so accepting it must not yield None.

        Regression: when TensorRT accepted the probe and asked for no further
        inputs, `names` stayed empty and discovery returned None - throwing away
        a working profile set and leaving the real session on implicit profiles,
        which is the per-request engine rebuild the profiles exist to prevent.
        """

        class FakeOrt:
            @staticmethod
            def InferenceSession(*_args, **_kwargs):
                return object()  # accepted: no exception, nothing demanded

        monkeypatch.setattr("piper_runtime.runner.tensor_dims", lambda _p: {})
        runner = _runner()

        profiles = runner._trt_profiles(
            FakeOrt, use_trt=True, cache_dir=tmp_path, timing_dir=tmp_path
        )

        assert profiles is not None, "an accepted probe must not be discarded"
        assert profiles["trt_profile_min_shapes"] == f"input:1x{PROFILE_PHONEMES[0]}"
        assert profiles["trt_profile_max_shapes"] == f"input:1x{PROFILE_PHONEMES[2]}"

    def test_use_trt_false_still_returns_none(self, tmp_path):
        """Without TensorRT there is nothing to profile."""
        runner = _runner()
        assert (
            runner._trt_profiles(
                object(), use_trt=False, cache_dir=tmp_path, timing_dir=tmp_path
            )
            is None
        )

    def test_symbolic_dims_resolve_by_meaning(self):
        dims = {
            "a": ["batch_size", "phonemes", 192],
            "b": ["batch_size*phonemes", 1],
            "c": ["batch_size", 1, 1, "phonemes"],
        }
        got = profile_for(["a", "b", "c"], dims, phonemes=128, frames=512)
        assert got == "a:1x128x192,b:128x1,c:1x1x1x128"

    def test_unknown_symbol_is_the_decoder_length(self):
        # /Range_output_0 carries an opaque symbol like "Range_5497_o0__d0";
        # that dimension is the frame count the duration predictor samples.
        dims = {"/Range_output_0": ["Range_5497_o0__d0"]}
        assert profile_for(["/Range_output_0"], dims, 128, 512) == (
            "/Range_output_0:512"
        )

    def test_missing_shape_returns_empty_so_caller_can_fall_back(self):
        # A partial profile measured SLOWER than none, so refusing to emit one
        # is the point.
        assert profile_for(["nope"], {"a": [1]}, 128, 512) == ""
