# Dataset tooling

Build-time tooling for the GLaDOS VITS training corpus. Nothing here is
imported by the Wyoming server or shipped in the runtime image.

## Why the wiki instead of the game files

[theportalwiki.com](https://theportalwiki.com) publishes GLaDOS voice lines
with **hand-written transcripts inline**, which removes the ASR pass the
pipeline would otherwise need. It also covers unused and cut lines, which have
no `closecaption_english` entry in the shipped games and therefore cannot be
recovered from a VPK extract at all.

The audio is 16-bit PCM at 44.1 kHz — original quality, not lossy re-encodes.
Portal 1-era files are mono, much of Portal 2 is stereo. 44.1 kHz downsamples
to Piper's 22.05 kHz at a clean 2:1 ratio.

## Source pages

| Page | Lines | Speaker |
| --- | --- | --- |
| `GLaDOS_voice_lines_(Portal)` | 228 | `p1` |
| `GLaDOS_voice_lines_(Portal_2)` | 327 | `p2` |
| `GLaDOS_voice_lines_(Cooperative_Testing_Initiative)` | 651 | `p2` |
| `GLaDOS_voice_lines_(Other)` | 589 | `p1` / `p2` / `dota2` |

**1,794 lines total**: `p1` 258, `p2` 1129, `dota2` 407.

## Speaker scheme

**Current decision: a single pooled voice.** All 1,546 clips train as one
speaker, and `metadata.csv` is written in Piper's two-column single-speaker
form (`wav|text`). The column count is not cosmetic — a single-speaker
checkpoint has no speaker embedding, so a three-column file would build one the
fine-tune checkpoint does not have.

`--multi-speaker` switches to one ID per recording era, three columns, no
re-fetch needed. The per-source audit below is reported in both modes so the
decision stays cheap to revisit if the pooled voice sounds blurred.

The known cost of pooling: `dota2` is 20 of the 123.5 minutes (16%), and it is
a separate session with different post-processing, so its character averages
into the Portal voice rather than staying selectable.

Sources, and the IDs they take under `--multi-speaker`:

- **`p1`** — Portal 1, plus the "Portal 1 Unused/alternate lines" section of
  the *Other* page. Heavier vocoder artifacting than Portal 2; kept separate so
  that character survives instead of averaging away.
- **`p2`** — Portal 2 single-player, co-op, and the Portal 2 cut content on the
  *Other* page. Co-op merges in because it is the same game, cast and session,
  and the corpus is small enough that splitting it would starve the main voice.
  The *Other* page's "Leaderboard responses" section (24 lines) is the
  **Peer Review DLC** — the section title does not say so, so it is recorded
  here to save the next person working this out.
- **`dota2`** — the Dota 2 announcer pack. A much later session with different
  post-processing, so it gets its own ID rather than blurring `p2`.

Lines whose transcript is only a wiki annotation (`[hums a tune]`,
`[fast gibberish]`, `[clap clap]`) are dropped — that is non-speech.

## Status

- [x] `portalwiki.py` — parse wiki pages into transcript/audio pairs
- [x] `fetch.py` — rate-limited, resumable fetch of the 1,794 `.wav` files
- [x] `audio.py` — mono downmix, 22.05 kHz resample, silence trim, peak normalize
- [x] `build.py` — `metadata.csv` plus the per-source duration audit
- [x] Duration audit — the Phase 0 gate, see below
- [ ] Transcript spot-check — wiki transcripts are fan-written; sample for errors
- [x] TensorRT spike — viable, ~3x, see findings below

## Audit

Built with `--max-seconds 20` (see the long-clip tail below):

```text
Speaker      clips   duration
  dota2        405     20.0 min
  p1           229     21.2 min
  p2           912     76.3 min
  TOTAL       1546    117.5 min
Excluded: lo-fi filtered 222, shorter than 0.7s 23, longer than 20.0s 3
```

**1.96 hours.** Enough to fine-tune from a Piper *medium* checkpoint; nowhere
near the ~10 hours from-scratch VITS needs. Only medium-quality checkpoints are
supported for fine-tuning without config changes.

Every output clip is 1ch / 22050 Hz / 16-bit, with metadata rows matching wavs
exactly.

Only 2 clips tripped the speaking-rate check, so the wiki transcripts appear
reliable and the manual spot-check is a small job.

### The band-pass gate

222 clips are dropped because they are filtered rather than plainly recorded:
the Portal 2 potato-battery scenes, and Portal 1's escape sequence where GLaDOS
speaks over facility PA. Both are band-limited, and pooled into one voice the
model learns that timbre as something it can apply to any word.

Detection is acoustic — the ratio of 100-300 Hz energy to 800-4000 Hz energy,
where clean lines sit near +7 dB and filtered ones below -30 dB. It is
deliberately not based on wiki chapter: the same scenes appear on the co-op
page and in Chapter 5, and two files Valve named `potatos_*` are clean because
she is back in her chassis by then. Exclusions are listed in
`filtered_clips.csv`; the threshold is `--min-low-mid-db`.

### The long-clip tail

26 clips exceed the 15s default. Raising `--max-seconds` recovers them:

| Cap | Clips recovered | Speech gained |
| --- | --- | --- |
| 16s | 12 | +3.1 min |
| 18s | 18 | +4.8 min |
| 20s | 23 | +6.4 min |
| 30s | 26 | +7.6 min |

`--max-seconds 20` is what the current dataset uses: 88% of the tail for
+6.4 min (~5% more corpus) without paying VRAM for the 27.2s outlier. These are
long monologues, which tend to carry the corpus's best prosody.

## Training

`train_glados.sh` runs the fine-tune; `setup_wsl.sh` provisions the environment.

**Base voice: `ljspeech` medium, not `lessac`.** Piper's docs use lessac as the
worked example, but the Blizzard 2013 Lessac corpus is licensed for research
only - it forbids commercial voice synthesis products and redistribution
without written consent. Fine-tuned weights carry the base weights, so that
restriction would follow into any published image. LJSpeech is public domain,
also a female single speaker at 22.05 kHz, and is the most widely used TTS
fine-tune base. `kristin` (LibriVox) is the other public-domain option;
`hfc_female` is CC BY-NC-SA, so non-commercial and share-alike.

Four failures worth knowing about, all encoded in the script:

* **Use `--model.warmstart_ckpt`, not `--ckpt_path`.** `--ckpt_path` resumes a
  run, restoring hyperparameters; the published checkpoints carry a
  `sample_bytes` hyperparameter current Piper rejects, so it fails with
  `Subcommand 'fit' does not accept option 'model.sample_bytes'`. warmstart
  copies weights only, which is what fine-tuning wants.
* **PyTorch 2.6+ refuses the checkpoint.** `torch.load` now defaults to
  `weights_only=True` and the checkpoints pickle a `pathlib.PosixPath`, raising
  `UnpicklingError`. A thin wrapper allowlists the path types.

* **Batch 24 hits CUDA OOM** on a 16 GB card with 20s clips, and gradient
  accumulation is not an escape hatch - VITS is a GAN using manual
  optimization, which Lightning refuses to auto-accumulate. Batch 12 is also
  ~3.5x faster per epoch, because 24 was thrashing the memory ceiling:

  | Batch | Steps/epoch | Rate | Time/epoch |
  | --- | --- | --- | --- |
  | 24 | 65 | 0.32 it/s | ~203 s |
  | 12 | 130 | ~2.4 it/s | ~55 s |

* **The `val_mos` checkpoint must be removed.** Piper registers a second
  `ModelCheckpoint` monitoring UTMOS perceptual quality, but its validation
  loop collects no MOS scores, so Lightning raises
  `MisconfigurationException` at the end of every validation epoch and kills
  the run. The wrapper strips that callback; the primary `val_mel` checkpoint
  is unaffected. (UTMOS also downloads from GitHub on first use, which can
  lose a race with the first validation epoch.)

Warmstart transfers 784 parameters with 0 skipped, confirming the architecture
matches. On an RTX 4080 at batch 16 the run is ~97 steps/epoch and roughly
40 s/epoch, so an overnight run passes 500 epochs. Baseline `val_mel` after the
first epoch is ~0.594. See `train_glados.sh` note 3 for why 16 and not 12 or
24, and set `RESUME_CKPT` to continue an interrupted run without losing the
epoch counter.

Keep the dataset on WSL's ext4 filesystem. Training from `/mnt/c` puts every
dataloader read across the 9p bridge and is dramatically slower.

## Serving, and connecting Home Assistant

Export a trained checkpoint, then serve it:

```bash
dataset_tools/export_glados.sh          # best checkpoint by val_mel
dataset_tools/serve_glados.sh           # Wyoming server on :10201
```

`export_glados.sh` runs `symbolic_shape_infer` after Piper's ONNX export. That
pass is **required**: the raw export leaves internal shapes unresolved and the
TensorRT provider refuses to partition it.

### Serve from Windows, not WSL, if you can

Home Assistant needs to reach the server over the LAN, and **WSL2 does not
share the Windows host IP by default** - it sits behind NAT on its own subnet,
so nothing on the LAN can reach it.

The tidy fix is WSL mirrored networking (`networkingMode=mirrored` in
`%USERPROFILE%\.wslconfig`), but applying it needs `wsl --shutdown`, which
kills any training run in progress. Do not do that mid-training.

So prefer running the server natively on Windows, where it binds the host IP
directly and Home Assistant just works. Training stays in WSL; the two are
independent, and the exported voice is reachable from both sides.

If you must serve from WSL while training runs, forward the port from Windows
instead of restarting WSL:

```powershell
# WSL_IP changes on every WSL restart, so re-run this after one.
netsh interface portproxy add v4tov4 `
  listenport=10201 listenaddress=0.0.0.0 `
  connectport=10201 connectaddress=<WSL_IP>
New-NetFirewallRule -DisplayName "Wyoming GLaDOS" -Direction Inbound `
  -LocalPort 10201 -Protocol TCP -Action Allow
```

### Home Assistant

Settings -> Devices & Services -> Add Integration -> **Wyoming Protocol**, then
enter the host IP and port `10201`. The voice appears as a TTS entity usable
from assist pipelines, `tts.speak`, and automations.

### Gotcha: onnxruntime CPU shadows the GPU build

`piper-tts` depends on CPU `onnxruntime`. Installing it alongside
`onnxruntime-gpu` leaves both present, and the CPU build wins on import - so
`get_available_providers()` returns only CPU and the TensorRT provider is
silently unavailable. Install piper-tts first, then:

```bash
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu
```

Verify with `get_available_providers()`; it must list
`TensorrtExecutionProvider`.

## TensorRT spike findings

Measured against a stock `en_US-lessac-medium` Piper voice (2,755 nodes) on an
RTX 4080, to de-risk Phase 3 before training.

**Verdict: viable, ~3x faster, but with caveats.**

```text
CUDA EP     min 15.3 ms   median 26.9 ms
TRT fp16    min  6.0 ms   median  8.5 ms
TRT fp32    min  5.5 ms   median  7.5 ms
```

### Use FP32, not FP16

FP32 is *faster* than FP16 here (7.5 vs 8.5 ms median), and TensorRT warns that
FP16 layernorm after self-attention may overflow, forcing those layers to FP32
anyway. FP16 buys nothing and carries numerical risk.

### Required setup, none of it optional

1. **Run `symbolic_shape_infer` on the ONNX export.** The raw Piper export fails
   TRT partitioning outright: `"TensorRT input: /enc_p/Split_output_0 has no
   shape specified"`. Needs `sympy`.
2. **Pin TensorRT to 10.x** (`tensorrt-cu13==10.16.1.11`). onnxruntime 1.29
   links `libnvinfer.so.10`; pip's default 11.2.x ships `.so.11` and the EP
   silently falls back.
3. **`LD_LIBRARY_PATH` must cover both TRT and CUDA/cuDNN libs** — and a login
   shell (`bash -lc`) strips it.

### Explicit shape profiles are essential (this section used to say the opposite)

It previously read: *"Those are not enumerable by hand... Letting the EP infer
profiles works, and does **not** cause per-inference engine rebuilds."* The
first half is true; the conclusion was wrong, and stated confidently enough
that nobody re-measured it for months.

Inferred profiles rebuild the engine on **nearly every request**. TensorRT
compiles for the shape it first sees and recompiles when a later one falls
outside, and the stochastic duration predictor hands the decoder a new length
every time — so the same input, run twice, forces a rebuild:

| | per utterance |
| --- | --- |
| inferred profiles | **2939 ms** |
| explicit profiles | **9.9 ms** |

`trt_profile_min/opt/max_shapes` does have to cover every TRT subgraph input,
including internal tensors at partition boundaries. Those are not enumerable by
hand — but ONNX Runtime will enumerate them for you. Hand it a deliberately
incomplete profile and it fails immediately, naming exactly what is missing,
without building anything. Repeat until it stops complaining: it names one
subgraph per attempt, so a single pass under-reports.

Do not hard-code the names. They are export artefacts: adding the `onnxslim`
pass renamed `/Range_5497_o0__d0` to `/Range_2297_o0__d0` within a single
afternoon.

This is also what phoneme bucketing, three warmup passes per bucket and the
warmup budget were all working around. Warmup went from 275.4s to 0.4s when the
profiles went in, and all three mitigations were deleted.

### Engine caching behaves as hoped

Engines are `sm89`-specific and non-portable, so they must be built on the
deployment GPU. Two caches, and the second matters more than it looks: the
engine cache is keyed by a content hash of the weights, so a retrain always
misses it, while the **timing** cache records per-kernel measurements for this
GPU and survives a re-export. With it warm a build is ~10s; without it ~39s.

### Output equivalence cannot be verified numerically

The graph contains 12 `NonZero` (data-dependent shapes) and 2
`RandomNormalLike` (stochastic duration predictor). TRT and CUDA draw different
noise, so they produce different phoneme durations:

| Comparison | Error vs signal |
| --- | --- |
| CUDA vs CUDA (control) | −33 dB |
| TRT vs CUDA | +0.9 dB |

Global statistics match closely (RMS 0.0798 vs 0.0795, peak 0.5855 vs 0.5858,
spectral cosine similarity 0.90), so TRT output is plausible speech, not
corruption — it is a different valid random realization. Waveform correlation
is only 0.38 and alignment does not fix it.

**This means A/B numerical validation is impossible while the stochastic
duration predictor is in the graph.** Quality must be judged by listening test,
or by switching to a deterministic duration path if reproducibility matters.

## TODO: additional sources

None of the sources below ship audio on any wiki — the
[Combine OverWiki other-games page](https://combineoverwiki.net/wiki/GLaDOS/Quotes/Other_games)
is transcripts only. Each needs the game itself plus an extraction pass, so
they are ordered by expected yield per unit of effort.

| Source | Engine / archive | Extraction | Notes |
| --- | --- | --- | --- |
| Bridge Constructor Portal | Unity asset bundles | AssetStudio / AssetRipper | **Best expected yield.** Ellen McLain narrates the whole game, dialogue written with Valve; Unity extraction is well-trodden |
| Aperture Hand Lab | Source 2 | Source 2 Viewer | Valve-made, straightforward extraction; modest volume |
| Poker Night 2 | Telltale `.ttarch2` | `ttarchext` | Solid volume — GLaDOS is a dealer with a lot of table banter |
| Portal Pinball | Zen Studios, proprietary | Unclear | Suspected to reuse shipped Portal audio; **dedupe before use** |
| Lego Dimensions | TT Games, proprietary | Thin tooling | Licensed title, stretch goal |
| Dota 2 Portal Pack | — | — | **Already in the corpus** as `dota2` (407 lines) |

Two things to settle before adding any of them:

1. **Speaker IDs.** Each of these is a separate recording session with its own
   post-processing, so by the scheme above each needs its own ID rather than
   merging into `p1`/`p2`. That fragments an already small corpus — worth it
   only where the volume justifies a usable speaker.
2. **Near-duplicate audio.** Portal Pinball in particular may ship recycled
   Portal 2 lines. The same utterance appearing twice at different mastering
   levels is actively harmful to a small fine-tune, so any new source needs a
   duplicate check against the existing corpus, not just a filename check.

### Portal 2 2009 beta lines

The August 2026 Valve leak includes 2009 Portal 2 betas with cut content. Once
datamined, these could extend the corpus. Deferred, with two caveats to resolve
first: (1) leak-derived weights are traceable to the breach, which matters for
anything published to Docker Hub, and (2) much beta VO is scratch/placeholder
recorded by Valve staff rather than Ellen McLain, and its post-processing
differs from shipped audio — so it needs its own speaker ID, never a merge into
`p1`/`p2`. Note that a large share of Portal 2 cut content is *already* covered
by the *Other* wiki page and is in the corpus today.
