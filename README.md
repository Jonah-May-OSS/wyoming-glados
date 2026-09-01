# Wyoming GLaDOS

[Wyoming protocol](https://github.com/rhasspy/wyoming) server providing a GLaDOS
text-to-speech voice, accelerated with NVIDIA TensorRT.

The voice is a [VITS](https://arxiv.org/abs/2106.06103) model trained with
[Piper](https://github.com/OHF-Voice/piper1-gpl) and served through ONNX Runtime
using the TensorRT execution provider. The server is a heavily stripped down
version of [wyoming-piper](https://github.com/rhasspy/wyoming-piper).

Earlier releases used R2D2FISH's two-stage
[glados-tts](https://github.com/R2D2FISH/glados-tts) (ForwardTacotron + HiFiGAN
via torch_tensorrt). That path has been removed. On the same GPU and sentences,
the VITS model measured:

| | ForwardTacotron + HiFiGAN | VITS + ONNX Runtime |
| --- | --- | --- |
| median synthesis | 49 ms | **8 ms** |
| realtime factor | 36-51x | **209x** |
| time to serving | 41 s | **~2 s** |
| VRAM (steady state) | 1098 MiB | **536 MiB** |
| model files on disk | ~267 MB | **~129 MB** |

### Execution providers

The same VITS model, same sentences, on an idle RTX 4080. Median and worst are
per utterance; realtime factor is audio seconds produced per second of compute.

| provider | median | worst | realtime | warm start |
| --- | --- | --- | --- | --- |
| **TensorRT** | **8 ms** | **30 ms** | **209x** | 0.6 s |
| CUDA | 140 ms | 436 ms | 15x | 3.1 s |
| CPU | 85 ms | 147 ms | 35x | 1.7 s |

**arm64 gets TensorRT too, but not from PyPI.** `onnxruntime-gpu` ships the
TensorRT provider only in its x86_64 wheels - the manylinux aarch64 wheel
carries `libonnxruntime_providers_cuda.so` and no TensorRT provider at all,
in the current release and in nightlies alike. So this project compiles one:
a workflow builds ONNX Runtime for aarch64 with `--use_tensorrt` once per
onnxruntime version and publishes the wheel, and the arm64 images install
that instead. The build never happens during a release, only on a version
bump. That matters because of the next paragraph - without it, arm64 would
fall back to CUDA, which is the slowest of the three for this model.

**Plain CUDA is slower than CPU for this model**, which is worth knowing before
treating it as the fallback. ONNX Runtime reports the reason: *"28 Memcpy nodes
are added to the graph for CUDAExecutionProvider"*. VITS has enough ops the
CUDA provider will not take that the graph ping-pongs between host and device,
and the transfers cost more than the kernels save. TensorRT compiles the whole
graph into a handful of engines and does not pay that.

So the provider chain is TensorRT, then CUDA, then CPU, but the middle rung is
a formality: on a machine where TensorRT cannot load, CPU is the better
outcome. Watch for it — a TensorRT failure is silent and looks only like a 10x
slowdown.

VRAM was measured as a delta on an otherwise idle GPU; measuring it while
training ran inflated the VITS figure to 951 MiB through allocator noise. The
536 MiB was measured with engines for all seven phoneme buckets resident, so it
is a real serving cost rather than a best case. Bucketing has since been
removed in favour of shape profiles, which need one engine rather than seven.

Disk counts what is actually fetched: 61 MB of ONNX plus a 68 MB engine cache
for VITS, against 267 MB of checkpoints and engines for the old path. The
legacy release also shipped a 111 MB `tacotron-trt.ts` that was a mislabeled
copy of `glados-new.pt` and was never downloaded.

It also drops torch and torch-tensorrt from the runtime image entirely, and the
single-stage graph removes the TorchScript wrapper, the engine-probe subprocess
and the manual mel windowing the old vocoder needed.

### Shape profiles, and why first start used to be slow

TensorRT compiles an engine for the input shape it first sees, then
**recompiles whenever a later shape falls outside it**. VITS samples its
decoder length from a stochastic duration predictor, so that length differs on
every request — even for identical input. Left to itself, the provider
therefore rebuilds almost every time:

| | per utterance |
| --- | --- |
| no shape profiles | **2939 ms** |
| explicit profiles | **9.9 ms** |

The fix is to declare the range up front, via `trt_profile_{min,opt,max}_shapes`.
One engine then covers phoneme counts 8—512 and decoder frames 8—4096
(about 47 seconds of audio). Anything outside that still works; it just pays a
rebuild, which is the old behaviour.

The profile inputs are **discovered, not hard-coded**. Because the `scales`
shape-tensor rejection partitions the graph, TensorRT's subgraph inputs are
internal tensors like `/Range_output_0`, whose names change on every re-export.
So the runtime hands ONNX Runtime a deliberately incomplete profile, which
makes it fail immediately and name exactly what it wants, and repeats that
until nothing is missing — one subgraph is named per attempt, so a single
pass under-reports. Probes never compile: validation happens before any build.
If discovery fails for any reason the server logs it and runs without profiles,
which is simply the slower path.

**What this replaced.** Earlier versions padded every utterance up to one of
seven phoneme buckets, warmed each bucket three times, and capped the whole
thing with a five-minute budget. That was three layers of workaround for never
declaring a shape range: the repeated passes were brute-forcing the profile
wider until common lengths stopped triggering rebuilds. It was slow and
unreliable — warming the largest bucket once took over two hours while a
training job held the card, since the builder benchmarks tactics and thrashes
when VRAM is squeezed. All of it is gone:

| | before | after |
| --- | --- | --- |
| warmup | 275.4 s | **0.4 s** |
| server cold start | 280 s | **70 s** |

Warmup is now a single synthesis, kept only to prove inference runs before the
server announces itself.

### Two caches, and why the second one matters more

Engines are cached per model, keyed by a content hash of the weights, so a
retrain always misses. The **timing** cache is different: it records how fast
each candidate kernel ran on this GPU, which is a property of the hardware
rather than the weights, so it survives a re-export. Keeping it out of the
per-model directory is worth more than any builder setting:

| engine build | |
| --- | --- |
| brand-new machine, both caches empty | 39.4 s |
| **retrain, timing cache warm** | **10.1 s** |

Both live in the mounted models directory, so keep that volume.

Builder optimization level is 2, against ONNX Runtime's default of 3. Measured
on this model with a warm timing cache: level 3 costs 37 s more build for no
measurable latency gain, and level 0 saves less than a second of build for
**2.7x slower inference** (27.8 ms against 13.2 ms). Narrowing the profile's
maximum decoder frames does not help either — 1024 or 2048 frames both built
*slower* than 4096 — so the range stays wide.

Warmup still runs on a background thread, so the server accepts connections
immediately rather than appearing unavailable to Home Assistant.

## Usage

### Pre-requisites
1. Install and configure Docker
2. Install and configure the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Docker Compose (recommended)
For AMD64/ARM64 with discrete GPUs:
```yaml
version: "3"
services:
  wyoming-glados:
    image: captnspdr/wyoming-glados:latest
    container_name: wyoming-glados
    ports:
      - 10201:10201
    volumes:
      - ./models:/usr/src/models:rw
    environment:
      - STREAMING=true
      - DEBUG=false
      # Optional. The GLaDOS voice already defaults to p2, the expressive
      # Portal 2 speaker; set this only to pick a different one. Ignored by
      # single-speaker voices.
      - SPEAKER=p2
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### Which image does a Jetson need?

It depends on the JetPack version, because that is where Jetson's CPU
architecture story changes:

| Device | Image | Why |
| --- | --- | --- |
| **JetPack 7 and newer** (Thor) | `captnspdr/wyoming-glados:latest` | JetPack 7 aligned Jetson with **SBSA**, the standard Arm server architecture, and moved to a unified CUDA 13 install across Arm targets. The generic `arm64` slice of the main image is built from the SBSA aarch64 wheels, so it is the correct build - use the same tag as any other machine. |
| **JetPack 6 and older** | `captnspdr/wyoming-glados:latest-igpu` | Pre-SBSA Tegra needs CUDA and TensorRT builds distinct from the SBSA aarch64 wheels pip ships, which is the entire reason this separate image exists. |

The `-igpu` image is **legacy** and is slated for removal once JetPack 6 is no
longer supported. It has also never been validated on Jetson hardware; if you
run it, please report what happens.

For JetPack 6 and older, use the `-igpu` image tag instead:
```yaml
version: "3"
services:
  wyoming-glados:
    image: captnspdr/wyoming-glados:latest-igpu
    container_name: wyoming-glados
    runtime: nvidia
    ports:
      - 10201:10201
    volumes:
      - ./models:/usr/src/models:rw
    environment:
      - STREAMING=true
      - DEBUG=false
      # Optional. The GLaDOS voice already defaults to p2, the expressive
      # Portal 2 speaker; set this only to pick a different one. Ignored by
      # single-speaker voices.
      - SPEAKER=p2
    restart: unless-stopped
```


### Docker (Latest tag on Docker Hub)
1. Clone this repository
2. Browse to the repository docker folder
3. Run the following command based on your platform:
   
For AMD64/ARM64 with dGPU:

```bash
docker run \
  --gpus all \                                # expose all NVIDIA GPUs
  --name wyoming-glados \                     # give the container a name
  -d \                                        # run in detached mode
  -v "$(pwd)/models":/usr/src/models:rw \     # set directory to download model files to so they persist for easy container updates
  -p 10201:10201 \                            # map port 10201 → 10201
  -e STREAMING=true \                         # Enable partial streaming
  captnspdr/wyoming-glados:latest
```

### Docker (Latest GitHub commit, AMD64/ARM64 with dGPU)
1. Clone this repository
2. Browse to the repository docker folder
3. Run ``docker compose -f docker-compose-github.yaml up -d``


## Connecting to Home Assistant
### Adding the TTS engine to Home Assistant
1. Go to Settings -> Devices & Services
2. Add Integration -> Wyoming Protocol
3. Enter the IP and Port, click Submit
4. Click Finish if it adds successfully

### Modify the Voice Assist Pipeline to use the new engine
1. Go to Settings -> Voice assistants
2. Select the assistant/pipeline
3. Under Text-to-speech select glados-tts


## Building the voice

The corpus and training pipeline live in [`dataset_tools/`](dataset_tools/).
Transcripts and audio come from [theportalwiki.com](https://theportalwiki.com),
which publishes hand-written transcripts alongside the original 44.1 kHz WAVs —
so no ASR pass is needed, and cut/unused lines that have no closed-caption entry
in the shipped games are still usable.

```bash
python -m dataset_tools fetch                      # wiki pages + audio
python -m dataset_tools build --max-seconds 20 \
    --multi-speaker --potato-speaker         # normalise, write metadata.csv
dataset_tools/setup_wsl.sh                         # one-time training env
dataset_tools/train_glados.sh                      # fine-tune from LJSpeech
dataset_tools/export_glados.sh                     # checkpoint -> TensorRT-ready ONNX
dataset_tools/serve_glados.sh                      # run it locally
```

Current corpus: **1704 clips / 125.8 minutes** across four speakers.

### Speakers

Pooling every source into one voice averages deliveries that are not alike.
Portal 1 GLaDOS is deliberately flat and affectless, Dota 2 is announcer mode,
and Portal 2 carries the emotional range most people mean by "GLaDOS". Pooled,
the model learns the mean of those and sounds flatter than Portal 2 ever is —
and with Dota 2 at a quarter of the clips, that pull is not small.

Multi-speaker training gives each source its own embedding, so the eras stay
distinct while still sharing one acoustic model across all 125 minutes.

| id | speaker | clips | what it is |
| --- | --- | --- | --- |
| 0 | `p1` | 210 | Portal 1: flat, affectless, early GLaDOS |
| 1 | `potato` | 193 | band-pass filtered — see below |
| 2 | `p2` | 896 | Portal 2 and co-op: the expressive one |
| 3 | `dota2` | 405 | Dota 2 announcer lines |

**The shipped voice defaults to `p2`**, and it says so itself: the config
carries a `default_speaker` field, so the default belongs to the voice rather
than to the server and a future voice with a different speaker layout cannot
inherit this one's answer. Nothing needs setting to get the Portal 2 GLaDOS.

To pick a different speaker, use `--speaker p1` at serve time or the `SPEAKER`
environment variable, which the Docker images read.

Without that default the server would fall back to id 0 - whichever speaker
came first in `metadata.csv`, here `p1` with 210 clips rather than `p2` with
896. Every evaluation of this voice up to 2026-09-01 unknowingly ran as `p1`,
and the occasional mangled word that produced read as an undertrained model.
Two training runs went into chasing it. CI now fails if the published config
carries no `default_speaker`, or if the server does not report synthesizing
as `p2`.

Ids are assigned by first appearance in `metadata.csv`, so they are a property
of the trained voice, not a constant. They are written into the voice config as
`speaker_id_map` and resolved from there **by name** — asking for an unknown
speaker is an error rather than a silent fall back to id 0, which would serve a
different character with nothing in the logs to say so. A numeric id is checked
against the voice's speaker count for the same reason: out of range, it would
otherwise index the speaker embedding out of bounds at the first request.

Both checks apply only to voices whose graph actually takes a speaker id. On a
single-speaker voice `--speaker` is ignored with a warning, never a startup
failure.

Omitting `--multi-speaker` pools everything into one voice and drops the
filtered clips, which is the older single-speaker behaviour.

### PotatOS: the band-pass filtered lines

For much of Portal 2 GLaDOS runs on a potato battery, and every line in that
stretch carries a heavy lo-fi filter. Measured against the clean lines it is a
band-pass: **−32 dB below 300 Hz** (the fundamental of her speaking voice),
**+9 to +12 dB** across 800–4000 Hz, and **−15 to −24 dB** above 8 kHz. Pooled
into one voice the model learns that timbre as a variation it can apply to any
word, and it leaks audibly into unrelated output.

These are detected **acoustically**, by the ratio of low-band to mid-band
energy, not by which wiki chapter they came from. Provenance does not work:
the same filtered scenes also appear on the co-op page and in Chapter 5, and
conversely two files Valve named `potatos_*` are clean because she is back in
her chassis by then. The gate drops **222 clips** and also catches Portal 1's
escape sequence, where she speaks over facility PA speakers — a different
filter with the same problem. Every exclusion is listed in
`filtered_clips.csv`. Tune with `--min-low-mid-db`.

The filter cannot be undone. The sub-300 Hz content is at the noise floor, so
inverting it means applying +32 dB of gain to noise.

**But they need not be thrown away.** The leaking happened because a single
pooled voice had nowhere to put that timbre; a speaker embedding is exactly the
place for it. With `--multi-speaker --potato-speaker` these clips become speaker
`potato` rather than being dropped, so PotatOS is selectable and the timbre
stays bound to that id instead of bleeding onto everything else.

The name is loose: the gate is acoustic, so **25 of the 193 clips are Portal 1's
escape sequence** over facility PA speakers — a different effect that measures
the same. The per-source audit counts those under their originating page, so its
totals will not match the metadata's speaker counts in this mode.

### Excluded: clips whose audio contains sound the transcript does not

The wiki marks non-speech inside the transcript - `[train horn]`,
`[gentle laughter]`, `(phone ringing)`, `(page flip)`. Stripping those markers
cleans the *text*, but the sound is still in the *audio*, so the transcript no
longer describes what is heard. The model then learns that a burst of
non-speech belongs to the surrounding words, and the damage lands on alignment
and the duration predictor rather than on timbre - which surfaces as artifacts
on unrelated output.

Parentheses carried worse cases than brackets, because the wiki uses them for
editorial notes as well: one transcript was `Alternate version of(subtitled as
" ")` over nine seconds of audio, another paired Spanish speech with an English
translation the audio never says. That clip was the corpus's single worst
text-to-audio outlier.

The gate is provenance-based here, unlike the potato filter: the wiki editor
who heard the clip is a better detector of a train horn than any spectral
statistic, and `[sic]` is allowlisted so text-only notes do not cost a clip.
Dashes framing an annotation are consumed with it, since `noises--[train
horn]--` otherwise leaves `-- --` mid-sentence; dashes elsewhere are kept,
because those are GLaDOS interrupting herself. **38 clips** are dropped.

The same pass raised the band-pass gate from -12.0 to -10.0 dB. At -12 the
closest survivor was `fgbgladostransfer15` ("GET YOUR HANDS OFF ME! NO! STOP!
No!") at -11.0 dB, with the next survivor 4 dB above it at -7.0. That gap is a
natural break, so -10.0 drops exactly that one clip.

### Additional voice sources (TODO)

- **Portal 2 beta / cut content.** The 2026 Valve leak includes 2009 Portal 2
  betas with cut GLaDOS lines. The wiki's "unused/alternate" sections already
  cover some of this; datamined audio would add more.
- **[Combine OverWiki: GLaDOS quotes from other games](https://combineoverwiki.net/wiki/GLaDOS/Quotes/Other_games)**
  — Lego Dimensions, Aperture Hand Lab and similar spinoffs, if audio can be
  sourced.
- **Portal: The Uncooperative Cake Acquisition Game** — the board game has
  GLaDOS lines.
- **Peer Review DLC** — covered by the wiki's "Leaderboard responses" section.

## Licence

**GPL-3.0-or-later.** This project links against
[`piper-tts`](https://github.com/OHF-Voice/piper1-gpl) (GPL-3.0-or-later) for
phonemization, and the published image ships it, so the distributed program is
a combined work and must be offered under the same terms. It was previously
labelled MIT, which was wrong for what we publish; the prior MIT notice is
preserved in [NOTICE.md](NOTICE.md) along with the voice-model and corpus
attributions.
