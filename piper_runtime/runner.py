"""Run the GLaDOS Piper VITS voice through ONNX Runtime.

This replaces the two-stage ForwardTacotron + HiFiGAN path and its
torch_tensorrt compilation. A single-stage VITS graph removes the TorchScript
wrapper, the engine-probe subprocess, and the manual mel windowing that the old
vocoder profile required.

Provider settings come from measurements on an RTX 4080 (see
`dataset_tools/README.md`):

* **FP32, not FP16.** FP32 was faster (7.5 vs 8.5 ms median) *and* avoids the
  layernorm overflow TensorRT warns about on this graph.
* **Explicit shape profiles, discovered at load.** This note used to say the
  opposite - that the internal tensors at partition boundaries "cannot be
  enumerated by hand" and that inferred profiles "do not cause per-inference
  engine rebuilds". Both were wrong, and being wrong confidently is why it went
  unexamined: inferred profiles rebuild on nearly every request, because the
  stochastic duration predictor hands the decoder a new length each time.
  Measured 2939 ms per utterance against 9.9 ms with profiles set. They cannot
  be enumerated by hand, but ONNX Runtime will name them on request - see
  `_trt_profiles`.
* **Engine cache on disk, keyed per model; timing cache shared.** With a warm
  timing cache a build is ~10s, against ~39s on a machine that has never built
  this graph. Engines are GPU-architecture specific and must be built on the
  deployment device. See `engine_cache_dir` for why the key is the model's
  content hash rather than a single shared directory, and `build_providers` for
  why the timing cache deliberately sits outside it.
* **`scales` stays a runtime input.** `scales[1]` is `length_scale`, which
  feeds the output-length computation, so TensorRT classifies `scales` as a
  shape tensor - and shape tensors must be Int32/Int64, not float. It logs
  "scales is a shape tensor but its data type is not allowed" and runs that one
  subgraph on CUDA. Freezing the scales as a constant silences that, but then
  TensorRT cannot build the engine at all (a Myelin convolution-bias assert)
  and the whole graph drops to CUDA, so the partial fallback is the cheaper
  failure. `dataset_tools/export_glados.sh` documents this under BAKE_SCALES.

Measured against the legacy ForwardTacotron + HiFiGAN path on the same GPU and
sentences: median 8 ms vs 49 ms per utterance, 209x vs 36-51x realtime,
536 MiB vs 1098 MiB of VRAM, and ~2s vs 41s to a serving-ready process.
Against the other providers on the same graph: CPU 85 ms, CUDA 140 ms. The
CUDA provider is slower than CPU here because it cannot take enough of the
graph to avoid constant host/device copies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2
CHANNELS = 1

# Piper's phoneme sequence markers (piper.const).
PAD = "_"
BOS = "^"
EOS = "$"

# Warmed with a real vowel, in preference order, so predicted durations -
# and therefore the decoder's input length - resemble live traffic. Falls
# back to the pad phoneme if a voice's map somehow has none of these.
WARMUP_PHONEMES = ("ɑ", "a", "ə", "ɪ", "i", "oʊ", "ˈoʊ")

# TensorRT builder effort, 0-5. ORT's default is 3, which spends minutes
# searching kernel tactics for a graph this size. Level 2 keeps most of the
# inference speed for a fraction of the build time, and since every millisecond
# saved at build time is paid back only once per voice while cold-start delay
# is felt on every deploy, the trade favours building fast.
BUILDER_OPTIMIZATION_LEVEL = 2

# VITS inference scales: [noise_scale, length_scale, noise_w].
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_NOISE_W = 0.8
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_ESPEAK_VOICE = "en-us"

_INT16_MAX = 32767.0


class Session(Protocol):
    """The slice of an onnxruntime InferenceSession this module uses."""

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, Any]
    ) -> list[Any]:
        """Execute the model."""
        ...  # pylint: disable=unnecessary-ellipsis

    def get_providers(self) -> list[str]:
        """Return the execution providers actually in use."""
        ...  # pylint: disable=unnecessary-ellipsis


# TensorRT optimization-profile bounds.
#
# Without explicit profiles ONNX Runtime compiles an engine for whatever shape
# it first sees, then RECOMPILES whenever a later shape falls outside it. VITS
# samples its decoder length from the stochastic duration predictor, so that
# length differs on every request even for identical input - measured at 2939
# ms per utterance, essentially a rebuild each time, against 9.9 ms once
# profiles are set.
#
# Phoneme bucketing plus repeated warmup passes were an expensive way of
# widening that profile by brute force. Declaring the range up front does it in
# one build.
PROFILE_PHONEMES = (8, 128, 512)  # min, opt, max
PROFILE_FRAMES = (8, 512, 4096)  # decoder frames: 4096 ~= 47s of audio
# Each round discovers one subgraph's inputs; the graph has two TRT
# subgraphs today, so this is headroom rather than a tuning knob.
_PROFILE_DISCOVERY_ROUNDS = 6

_MISSING_PROFILE_MARKER = (
    "Following input(s) has no associated shape profiles provided:"
)


def missing_profile_inputs(message: str) -> list[str]:
    """Tensor names ONNX Runtime says still need a shape profile.

    The graph is partitioned - the `scales` shape-tensor rejection splits it -
    so the TensorRT subgraphs' inputs are internal tensors like
    `/Range_output_0`, not the model's named inputs. Rather than hard-code
    names that change on every re-export, ask ONNX Runtime: a deliberately
    incomplete profile makes it fail immediately, listing exactly what it
    wants, without building anything.
    """
    if _MISSING_PROFILE_MARKER not in message:
        return []
    # Only the marker's own line. ONNX Runtime errors are multi-line - the
    # caller logs splitlines()[0] for exactly that reason - so reading to the
    # end of the message appends the following sentence to the last tensor
    # name. strip() does not remove it, so that name matches nothing, the
    # profile comes back empty, and the session falls back to rebuilding an
    # engine per request.
    # splitlines() on an empty tail is [], so index 0 would raise. Inside
    # _trt_profiles that IndexError is swallowed by the outer handler and
    # silently abandons every profile - the failure this parsing serves.
    tail_lines = message.split(_MISSING_PROFILE_MARKER, 1)[1].splitlines()
    tail = tail_lines[0] if tail_lines else ""
    return [name.strip() for name in tail.split(",") if name.strip()]


def tensor_dims(model_path: Path) -> dict[str, list[Any]]:
    """Declared dimensions of every tensor, symbolic names included."""
    import onnx  # pylint: disable=import-outside-toplevel

    model = onnx.load(str(model_path), load_external_data=False)
    dims: dict[str, list[Any]] = {}
    for info in list(model.graph.value_info) + list(model.graph.input):
        shape = info.type.tensor_type.shape
        dims[info.name] = [
            d.dim_value if d.HasField("dim_value") else (d.dim_param or "")
            for d in shape.dim
        ]
    return dims


def profile_for(
    names: Sequence[str],
    dims_by_name: dict[str, list[Any]],
    phonemes: int,
    frames: int,
) -> str:
    """One `name:dxdxd,...` profile string for the given bound.

    Symbolic dimensions are resolved by meaning, not position: anything
    mentioning `phonemes` scales with the input length (`batch_size*phonemes`
    included, since batch is 1), `batch_size` is 1, and any remaining symbol is
    the data-dependent decoder length - which is the whole reason this exists.
    Returns "" if a name has no declared shape, so the caller can fall back
    rather than emit a profile that is quietly wrong.
    """
    parts: list[str] = []
    for name in names:
        dims = dims_by_name.get(name)
        if dims is None:
            return ""
        resolved: list[int] = []
        for dim in dims:
            if isinstance(dim, int) and dim > 0:
                resolved.append(dim)
            elif "phonemes" in str(dim):
                resolved.append(phonemes)
            elif str(dim) == "batch_size":
                resolved.append(1)
            else:
                resolved.append(frames)
        parts.append(f"{name}:{'x'.join(str(d) for d in resolved)}")
    return ",".join(parts)


def build_providers(
    cache_dir: Path | None,
    *,
    use_trt: bool = True,
    timing_cache_dir: Path | None = None,
    builder_optimization_level: int = BUILDER_OPTIMIZATION_LEVEL,
    profiles: dict[str, str] | None = None,
) -> list[Any]:
    """Build the ONNX Runtime provider chain, TensorRT first when enabled.

    The chain degrades gracefully: if the TensorRT provider cannot load, or a
    subgraph is unsupported, ONNX Runtime falls back per-subgraph to CUDA and
    then CPU instead of aborting the process - unlike the torch_tensorrt path,
    where a stale engine could segfault the interpreter on deserialization.

    `timing_cache_dir` should be SHARED across models while `cache_dir` is
    per-model. The timing cache records how fast each candidate kernel ran on
    this GPU, which is a property of the hardware, not of the weights - so a
    new voice can reuse it and skip re-timing every kernel it has already
    measured. Nesting it inside the per-model directory throws that away on
    every retrain, which is most of what makes a cold build slow.
    """
    providers: list[Any] = []
    if use_trt and cache_dir is not None:
        timing_dir = timing_cache_dir if timing_cache_dir is not None else cache_dir
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache_dir),
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(timing_dir),
                    # Reuse a timing cache even if the GPU clocks drifted
                    # between runs; without this ORT discards it as unsafe and
                    # the shared cache never actually gets used.
                    "trt_force_timing_cache": True,
                    "trt_builder_optimization_level": builder_optimization_level,
                    # Deliberately FP32; see the module docstring.
                    "trt_fp16_enable": False,
                    **(profiles or {}),
                },
            )
        )
    providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert float audio in [-1, 1] to little-endian 16-bit PCM bytes."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32).flatten(), -1.0, 1.0)
    return np.round(clipped * _INT16_MAX).astype("<i2").tobytes()


def phonemes_to_ids(
    phonemes: Sequence[str], id_map: Mapping[str, Sequence[int]]
) -> list[int]:
    """Map phonemes to model input ids.

    Mirrors `piper.phoneme_ids.phonemes_to_ids`: the sequence opens with BOS
    then PAD, each phoneme is followed by a PAD, and it closes with EOS.
    Reimplemented here rather than imported so the runtime does not depend on
    the trainer package and so the convention stays covered by tests.
    Phonemes absent from the map are skipped, matching upstream.
    """
    ids: list[int] = []
    ids.extend(id_map[BOS])
    ids.extend(id_map[PAD])
    for phoneme in phonemes:
        mapped = id_map.get(phoneme)
        if mapped is None:
            _LOGGER.warning("Missing phoneme from id map: %s", phoneme)
            continue
        ids.extend(mapped)
        ids.extend(id_map[PAD])
    ids.extend(id_map[EOS])
    return ids


def build_scales(
    alpha: float,
    *,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    noise_w: float = DEFAULT_NOISE_W,
) -> np.ndarray:
    """Build the VITS `scales` input.

    `alpha` carries over from the old ForwardTacotron runner as a duration
    multiplier, which maps directly onto the VITS `length_scale`: above 1.0 is
    slower speech.
    """
    return np.array([noise_scale, alpha, noise_w], dtype=np.float32)


def model_fingerprint(model_path: Path, chunk_size: int = 1 << 20) -> str:
    """A short content hash of the model file."""
    digest = hashlib.sha256()
    with open(model_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def engine_cache_dir(models_dir: Path, model_path: Path) -> Path:
    """Per-model TensorRT engine cache directory.

    This must be keyed on the model's CONTENT, and the reason is a trap worth
    spelling out. ONNX Runtime names cached engines after a hash of the graph
    TOPOLOGY, not its weights - re-exporting the same architecture from a newer
    training checkpoint yields the identical engine filename. With one shared
    cache directory the stale engine is loaded and the server keeps speaking
    with the OLD weights, reporting no error: warmup is instant, the provider
    list still says TensorRT, and nothing in the logs looks wrong.

    Hashing the file puts each export in its own directory, so new weights
    always trigger a rebuild and rolling back to a previous voice reuses the
    engines it already had.
    """
    return models_dir / "trt_cache" / model_fingerprint(model_path)


def session_wants_scales(session: Any) -> bool:
    """True if the model takes `scales` at runtime rather than baked in.

    Voices are normally exported with `scales` as a live input, so this is
    usually True. A voice exported with BAKE_SCALES=1 freezes them into the
    graph instead and has no such input; see the note in
    `dataset_tools/export_glados.sh` for why that is not the default.

    Injected test doubles need not implement `get_inputs`.
    """
    get_inputs = getattr(session, "get_inputs", None)
    if get_inputs is None:
        return True
    return any(inp.name == "scales" for inp in get_inputs())


def session_wants_sid(session: Any) -> bool:
    """True if the model is multi-speaker and needs a speaker id per request.

    Piper's exporter only emits the `sid` input when the checkpoint has more
    than one speaker, so its presence is the reliable signal - the voice config
    can carry a speaker_id_map that the exported graph does not use.

    Injected test doubles need not implement `get_inputs`.
    """
    get_inputs = getattr(session, "get_inputs", None)
    if get_inputs is None:
        return False
    return any(inp.name == "sid" for inp in get_inputs())


def resolve_speaker_id(config: dict[str, Any], speaker: str | int | None) -> int:
    """Map a speaker name to its id using the voice config's speaker_id_map.

    Ids are assigned by first appearance in metadata.csv, so they are a
    property of the trained voice and must be read from its config rather than
    assumed. An unknown name is an error: silently falling back to 0 would
    serve a different character than asked for.
    """
    id_map: dict[str, int] = config.get("speaker_id_map") or {}
    if speaker is None:
        return 0
    if isinstance(speaker, str) and speaker in id_map:
        return int(id_map[speaker])
    if isinstance(speaker, int) or speaker.isdigit():
        return _checked_speaker_id(int(speaker), config, id_map)
    raise ValueError(
        f"Unknown speaker {speaker!r}; voice offers {sorted(id_map) or 'none'}"
    )


def _speaker_count(config: dict[str, Any], id_map: dict[str, int]) -> int | None:
    """How many speakers the voice has, or None if it does not say."""
    count = config.get("num_speakers")
    if isinstance(count, int) and count > 0:
        return count
    if id_map:
        return max(id_map.values()) + 1
    return None


def _checked_speaker_id(
    speaker_id: int, config: dict[str, Any], id_map: dict[str, int]
) -> int:
    """Reject an id the voice has no embedding for.

    An unknown *name* already raises. A numeric id used to bypass that check
    entirely and reach the model, where it indexes the speaker embedding out of
    bounds - surfacing as an inference error on the first request, or worse, a
    silent out-of-range read. Failing here makes it a startup error instead.
    """
    count = _speaker_count(config, id_map)
    if count is not None and not 0 <= speaker_id < count:
        raise ValueError(
            f"Speaker id {speaker_id} out of range; voice has {count} "
            f"speaker(s) (valid ids 0-{count - 1})"
        )
    return speaker_id


class PiperTTSRunner:
    """Synthesize GLaDOS speech from a Piper VITS ONNX model.

    Exposes `run_tts_stream`, the single seam `server.process.GladosProcess`
    depends on, so it drops into the existing Wyoming server unchanged.
    """

    def __init__(
        self,
        models_dir: Path = Path("models"),
        voice_name: str = "glados",
        *,
        use_trt: bool = True,
        session: Session | None = None,
        phonemizer: Any = None,
        phoneme_id_map: dict[str, Sequence[int]] | None = None,
        speaker: str | int | None = None,
    ) -> None:
        """Load the voice, or accept injected collaborators for testing."""
        self.models_dir = Path(models_dir)
        self.voice_name = voice_name
        self.model_path = self.models_dir / f"{voice_name}.onnx"
        self.config_path = self.models_dir / f"{voice_name}.onnx.json"

        self._phoneme_id_map = phoneme_id_map
        self._phonemizer = phonemizer
        # espeak-ng keeps its translator and voice state in C globals, so two
        # threads inside phonemize() interleave and corrupt each other's output.
        # The server pipelines three syntheses at once (handler keeps one
        # draining, one queued and one blocked on put) and runs each in the
        # default executor, all sharing this one runner - so overlap is the
        # steady state, not a rare race. Inference itself is thread-safe;
        # ONNX Runtime sessions are. Only phonemization needs serialising.
        self._phonemize_lock = threading.Lock()
        self.config: dict[str, Any] = {}
        self.session: Session | None = session

        # Defaults matching the shipped voice. _load overwrites them from the
        # voice config, which is the authority: a voice exported at a different
        # sample rate, phonemized with a different espeak voice, or trained
        # with different inference scales carries all of that in its .json, and
        # hardcoding these meant such a voice was streamed at the wrong rate,
        # phonemized against a language it was never trained on, or run with
        # scales its author did not choose - none of it visible in the logs.
        self.sample_rate = SAMPLE_RATE
        self.espeak_voice = DEFAULT_ESPEAK_VOICE
        self.noise_scale = DEFAULT_NOISE_SCALE
        self.noise_w = DEFAULT_NOISE_W
        self.length_scale = DEFAULT_LENGTH_SCALE

        if self.session is None:
            self._load(use_trt=use_trt)
        elif self._phoneme_id_map is None:
            raise ValueError("phoneme_id_map is required when injecting a session")

        self.wants_scales = session_wants_scales(self.session)
        self.wants_sid = session_wants_sid(self.session)
        # Resolve only when the graph actually takes a speaker id. --speaker is
        # documented as ignored by single-speaker voices, but resolving here
        # unconditionally made any unrecognized value a startup crash on a voice
        # that would never have used it.
        self.speaker_id = (
            resolve_speaker_id(self.config, speaker) if self.wants_sid else 0
        )
        if not self.wants_sid and speaker is not None:
            _LOGGER.warning(
                "Ignoring speaker %r: this voice is single-speaker", speaker
            )
        if self.wants_sid:
            # Say which speaker out loud. Asking for none on a multi-speaker
            # voice quietly serves id 0, and on this voice that is p1 - the
            # flattest one - so a misconfigured SPEAKER looks like a bad model
            # rather than a bad setting.
            names = {v: k for k, v in (self.config.get("speaker_id_map") or {}).items()}
            _LOGGER.info(
                "Multi-speaker voice: synthesizing as %s (id %d) of %d",
                names.get(self.speaker_id, "?"),
                self.speaker_id,
                self.config.get("num_speakers", 1),
            )
            if speaker is None:
                _LOGGER.warning(
                    "No --speaker/SPEAKER given; defaulting to id 0. Available: %s",
                    sorted(self.config.get("speaker_id_map") or {}),
                )

    def _trt_profiles(
        self,
        ort: Any,
        *,
        use_trt: bool,
        cache_dir: Path,
        timing_dir: Path,
    ) -> dict[str, str] | None:
        """Shape profiles for the TensorRT provider, or None to go without.

        Discovered rather than hard-coded. The profile inputs are internal
        tensors (`/Range_output_0` and friends), because the `scales`
        shape-tensor rejection partitions the graph - and their names change on
        every re-export, so asking ONNX Runtime is the only durable way.

        Discovery is iterative: ONNX Runtime names the missing inputs of ONE
        subgraph at a time, so a single pass under-reports. A partial profile
        is worse than none - it was measured slower than no profiles at all -
        so this loops until a probe stops complaining, and gives up entirely
        rather than applying a half-built one.

        Probes never compile: profile validation happens before engine build,
        and a session that is created but never run builds nothing.

        Every failure path returns None, which is exactly how this ran before
        profiles existed, so nothing here can stop the server from serving.
        """
        if not use_trt:
            return None

        # All three bounds are required to make validation fail loudly. Given
        # only a min, the provider logs "Profile shapes validation failed" and
        # silently falls back to implicit profiles, naming nothing.
        # Derived from PROFILE_PHONEMES, not written out. These literals used
        # to be throwaway - the probe existed only to make TensorRT name its
        # missing inputs. The probe is now RETURNED as the live profile set
        # when TensorRT accepts it, so its bounds are the bounds the engine is
        # built for, and they must be the same ones synthesize_ids checks
        # against. Hardcoding them meant widening PROFILE_PHONEMES would move
        # the warning threshold while leaving the engine at 512.
        probe = {
            "trt_profile_min_shapes": f"input:1x{PROFILE_PHONEMES[0]}",
            "trt_profile_opt_shapes": f"input:1x{PROFILE_PHONEMES[1]}",
            "trt_profile_max_shapes": f"input:1x{PROFILE_PHONEMES[2]}",
        }
        try:
            dims = tensor_dims(self.model_path)
            lo_p, opt_p, hi_p = PROFILE_PHONEMES
            lo_f, opt_f, hi_f = PROFILE_FRAMES
            names: list[str] = []

            for _ in range(_PROFILE_DISCOVERY_ROUNDS):
                candidate = probe
                if names:
                    # Prepend `input`. Rebuilding purely from the discovered
                    # names dropped the profile the probe had already supplied
                    # for `input`, so ONNX Runtime rejected the next round
                    # complaining about it again - burning a discovery round
                    # re-learning something already known, and risking the
                    # `if not fresh` bail-out that abandons profiles entirely.
                    profiled = ["input", *names]
                    candidate = {
                        "trt_profile_min_shapes": profile_for(
                            profiled, dims, lo_p, lo_f
                        ),
                        "trt_profile_opt_shapes": profile_for(
                            profiled, dims, opt_p, opt_f
                        ),
                        "trt_profile_max_shapes": profile_for(
                            profiled, dims, hi_p, hi_f
                        ),
                    }
                    if not all(candidate.values()):
                        _LOGGER.warning(
                            "No declared shape for one of %d profile inputs; "
                            "continuing without profiles",
                            len(names),
                        )
                        return None
                try:
                    # Mirror the real session's options. A bare probe builds
                    # without the engine cache, timing cache or builder level,
                    # and that build can fail ("Could not find any
                    # implementation for node") where the real one succeeds -
                    # making discovery flaky and silently dropping us back to
                    # per-shape recompiles. Whatever this validates is exactly
                    # what will run, and the engine it builds is cached for it.
                    ort.InferenceSession(
                        str(self.model_path),
                        providers=build_providers(
                            cache_dir,
                            use_trt=True,
                            timing_cache_dir=timing_dir,
                            profiles=candidate,
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - message is the payload
                    discovered = missing_profile_inputs(str(exc))
                    fresh = [n for n in discovered if n not in names]
                    if not fresh:
                        _LOGGER.warning(
                            "Shape-profile discovery stalled (%s); "
                            "continuing without profiles",
                            str(exc).splitlines()[0][:120],
                        )
                        return None
                    names.extend(fresh)
                    continue

                if not names:
                    # Reaching here means the probe was ACCEPTED, and the probe
                    # is itself a profile set (input 1x8/1x128/1x512) - not an
                    # empty one. So this is "TensorRT wanted no inputs beyond
                    # `input`", not "TensorRT wants no profiles".
                    #
                    # Returning None here threw that working set away and left
                    # the real session on implicit profiles, which is exactly
                    # the per-request engine rebuild this discovery exists to
                    # prevent (2939 ms per utterance against 9.9 ms). Keep it.
                    _LOGGER.info(
                        "TensorRT accepted the probe profile with no further "
                        "inputs required (phonemes %d-%d)",
                        lo_p,
                        hi_p,
                    )
                    return candidate
                _LOGGER.info(
                    "TensorRT shape profiles set for %d inputs "
                    "(phonemes %d-%d, decoder frames %d-%d)",
                    len(names),
                    lo_p,
                    hi_p,
                    lo_f,
                    hi_f,
                )
                return candidate

            _LOGGER.warning(
                "Shape-profile discovery did not converge in %d rounds; "
                "continuing without profiles",
                _PROFILE_DISCOVERY_ROUNDS,
            )
            return None
        except Exception as exc:  # noqa: BLE001 - never block startup
            _LOGGER.warning("Shape-profile setup failed (%s); continuing without", exc)
            return None

    def _load(self, *, use_trt: bool) -> None:
        """Create the ONNX Runtime session and load the voice config."""
        import onnxruntime as ort  # pylint: disable=import-outside-toplevel

        if not self.model_path.exists():
            raise FileNotFoundError(f"Voice model not found: {self.model_path}")

        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._phoneme_id_map = config["phoneme_id_map"]
        # Kept so speaker names can be resolved against speaker_id_map.
        self.config = config

        audio = config.get("audio") or {}
        espeak = config.get("espeak") or {}
        inference = config.get("inference") or {}
        self.sample_rate = int(audio.get("sample_rate", SAMPLE_RATE))
        self.espeak_voice = espeak.get("voice") or DEFAULT_ESPEAK_VOICE
        self.noise_scale = float(inference.get("noise_scale", DEFAULT_NOISE_SCALE))
        self.noise_w = float(inference.get("noise_w", DEFAULT_NOISE_W))
        self.length_scale = float(inference.get("length_scale", DEFAULT_LENGTH_SCALE))

        cache_dir = engine_cache_dir(self.models_dir, self.model_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        timing_dir = cache_dir.parent
        timing_dir.mkdir(parents=True, exist_ok=True)

        profiles = self._trt_profiles(
            ort, use_trt=use_trt, cache_dir=cache_dir, timing_dir=timing_dir
        )
        session = ort.InferenceSession(
            str(self.model_path),
            providers=build_providers(
                cache_dir,
                use_trt=use_trt,
                timing_cache_dir=timing_dir,
                profiles=profiles,
            ),
        )
        self.session = session
        _LOGGER.info("Piper voice loaded, providers: %s", session.get_providers())

        if self._phonemizer is None:
            # pylint: disable=import-outside-toplevel
            from piper.phonemize_espeak import EspeakPhonemizer

            self._phonemizer = EspeakPhonemizer()

    def phonemize(self, text: str) -> list[list[str]]:
        """Split text into sentences of phonemes.

        Serialised: see the note on _phonemize_lock. The call is short next to
        synthesis, so this costs little of the pipeline's overlap.
        """
        with self._phonemize_lock:
            return list(self._phonemizer.phonemize(self.espeak_voice, text))

    def to_ids(self, phonemes: list[str]) -> list[int]:
        """Map one sentence of phonemes to model input ids."""
        if self._phoneme_id_map is None:
            raise RuntimeError("Phoneme id map is not loaded")
        return phonemes_to_ids(phonemes, self._phoneme_id_map)

    @property
    def _pad_id(self) -> int:
        """Id of the padding phoneme."""
        if self._phoneme_id_map is None:
            raise RuntimeError("Phoneme id map is not loaded")
        return self._phoneme_id_map[PAD][0]

    def synthesize_ids(self, ids: Sequence[int], alpha: float = 1.0) -> bytes:
        """Run the model over one sentence of ids and return PCM bytes.

        Shapes are passed through as-is within the profile range. Padding to
        fixed buckets used to be necessary because TensorRT recompiled per
        unseen shape; explicit optimization profiles cover the range in one
        engine, so that padding is now pure waste - except below the profile's
        lower bound, where it is what keeps the shape inside the engine.
        """
        if self.session is None:
            raise RuntimeError("Session is not initialized")
        true_length = len(ids)
        if true_length > PROFILE_PHONEMES[2]:
            # Sentences are split upstream, so this is rare - but nothing caps
            # it, and a sentence with no terminal punctuation can run long. The
            # TensorRT engine is built for at most PROFILE_PHONEMES[2] phonemes,
            # so past that the provider has no valid profile. Say so here:
            # otherwise it surfaces as an opaque inference error with no hint
            # that a bound was crossed.
            _LOGGER.warning(
                "Sentence is %d phonemes, above the %d-phoneme shape profile; "
                "TensorRT may reject or rebuild for it",
                true_length,
                PROFILE_PHONEMES[2],
            )
        # The lower bound needs handling too, and its failure mode is worse
        # than the upper bound's: rather than an error, TensorRT silently
        # rebuilds the engine for the unseen shape - the 2939 ms per utterance
        # against 9.9 ms that the profiles exist to prevent. Short sentences
        # are not an edge case for a voice assistant ("Yes." is seven ids with
        # the BOS/EOS markers), so pad up to the minimum rather than warn.
        #
        # input_lengths stays at the true count, so the model masks the padding
        # exactly as it did under the old fixed-bucket scheme and the audio is
        # unchanged.
        padded = list(ids)
        if true_length < PROFILE_PHONEMES[0]:
            padded.extend([self._pad_id] * (PROFILE_PHONEMES[0] - true_length))

        feed: dict[str, np.ndarray] = {
            "input": np.array([padded], dtype=np.int64),
            "input_lengths": np.array([true_length], dtype=np.int64),
        }
        if self.wants_scales:
            # alpha multiplies the voice's own length_scale rather than
            # replacing it: a voice tuned to 1.3 spoke at 1.0 while the
            # other two inference values were read from its config.
            feed["scales"] = build_scales(
                alpha * self.length_scale,
                noise_scale=self.noise_scale,
                noise_w=self.noise_w,
            )
        if self.wants_sid:
            feed["sid"] = np.array([self.speaker_id], dtype=np.int64)
        audio = self.session.run(None, feed)[0]
        return float_to_pcm16(np.asarray(audio))

    @property
    def _warmup_phoneme_id(self) -> int:
        """A voiced phoneme to warm engines with, falling back to padding.

        Padding is nearly silent, so a tensor of pad ids predicts durations
        close to zero and produces a very short waveform. The decoder's input
        length comes from those durations, so warming with padding primes
        engines for output shapes no real utterance ever hits. A vowel gives
        representative durations.
        """
        if self._phoneme_id_map is None:
            raise RuntimeError("Phoneme id map is not loaded")
        for candidate in WARMUP_PHONEMES:
            ids = self._phoneme_id_map.get(candidate)
            if ids:
                return ids[0]
        return self._pad_id

    def warmup(self) -> None:
        """Run one synthesis so the first real request never pays a surprise.

        This used to walk seven phoneme buckets three times each under a
        five-minute budget, because TensorRT recompiled for every unseen shape
        and the stochastic duration predictor produced a new decoder length on
        every pass. That was a workaround for never declaring a shape range:
        with explicit optimization profiles the engine is built once during
        session creation and covers the lot. Measured 275.4s -> 0.4s.

        What remains is a smoke test. It proves inference actually runs before
        the server announces itself, which is worth one synthesis.
        """
        started = time.monotonic()
        phoneme_id = self._warmup_phoneme_id
        self.synthesize_ids([phoneme_id, self._pad_id] * 16)
        _LOGGER.info("Warmup synthesis in %.2fs", time.monotonic() - started)

    def run_tts_stream(self, text: str, alpha: float = 1.0) -> Iterator[bytes]:
        """Yield PCM for each sentence as it is synthesized.

        Sentence-level granularity replaces the old vocoder windowing: VITS
        synthesizes a whole utterance in one pass, so the earliest point audio
        can be emitted is per sentence. The server's sentence-boundary detector
        already feeds text in at this granularity when streaming.
        """
        if not text.strip():
            return
        for phonemes in self.phonemize(text):
            if not phonemes:
                continue
            pcm = self.synthesize_ids(self.to_ids(phonemes), alpha)
            if pcm:
                yield pcm
