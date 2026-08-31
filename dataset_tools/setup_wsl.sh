#!/usr/bin/env bash
# Provision a WSL2 (Ubuntu) environment for training the GLaDOS VITS voice.
#
# Training runs in WSL rather than natively on Windows: Piper's trainer needs
# build_monotonic_align.sh plus a Cython build, and espeak-ng for phonemization.
#
# Keep the dataset and checkpoints on the WSL ext4 filesystem, NOT under
# /mnt/c or /mnt/d - the 9p bridge is very slow for the many-small-files
# access pattern of a training dataloader.
set -euo pipefail

log() { echo "[$(date +%H:%M:%S)] $*"; }

PIPER_DIR="${PIPER_DIR:-$HOME/piper1-gpl}"

log "Installing system packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential g++ cmake ninja-build \
  espeak-ng libespeak-ng-dev \
  ffmpeg python3-pip python3-venv python3-dev git

log "Cloning piper1-gpl into $PIPER_DIR"
[ -d "$PIPER_DIR" ] || git clone --depth 1 https://github.com/OHF-voice/piper1-gpl.git "$PIPER_DIR"
cd "$PIPER_DIR"

log "Creating venv"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python3 -m pip install --upgrade pip -q

log "Installing piper training extras (downloads torch, several GB)"
python3 -m pip install -e '.[train]'

# scikit-build is NOT pulled in by '.[train]', but setup.py imports it. Without
# it the espeakbridge C extension never builds and phonemization fails at
# training time with an opaque "cannot import name 'espeakbridge'".
log "Installing build backend for the espeakbridge extension"
python3 -m pip install -q scikit-build cmake

# UTMOS - the perceptual-quality predictor behind the "val_mos" metric - is
# loaded from torch.hub and imports torchaudio. Piper itself never imports it,
# so it is not in '.[train]', and its absence is silent: MosPredictor catches
# the ImportError, logs "val_mos logging disabled" once, and training proceeds
# with no perceptual signal at all. That cost us a whole run's worth of
# visibility, so install it explicitly.
#
# --no-deps and the matching CUDA index are both load-bearing. torchaudio on
# PyPI trails torch, so a plain `pip install torchaudio` resolves an older
# release and DOWNGRADES torch to match it, silently rebuilding the training
# environment around the wrong version. --no-deps makes that impossible.
log "Installing torchaudio (required by the UTMOS val_mos metric)"
CU_TAG="$(python3 -c 'import torch; v=torch.version.cuda; print("cu"+v.replace(".","")) if v else print("cpu")')"
python3 -m pip install -q --no-deps   --index-url "https://download.pytorch.org/whl/${CU_TAG}" torchaudio

log "Building monotonic_align"
[ -f ./build_monotonic_align.sh ] && bash ./build_monotonic_align.sh

log "Building espeakbridge"
python3 setup.py build_ext --inplace

log "Verifying"
python3 - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"vram free: {free / 1024**3:.1f} / {total / 1024**3:.1f} GiB")

from piper.train.vits.monotonic_align import maximum_path  # noqa: F401
print("monotonic_align: OK")

from piper.phonemize_espeak import EspeakPhonemizer
print("phonemes:", "".join(EspeakPhonemizer().phonemize("en-us", "Hello, test subject.")[0]))
PY

log "DONE"
