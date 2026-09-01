#!/usr/bin/env bash
# Serve the trained GLaDOS voice over the Wyoming protocol, from WSL.
#
# Usage:
#   dataset_tools/serve_glados.sh [--cpu]
#
# Home Assistant connects to this with the Wyoming integration. See the
# "Connecting Home Assistant" section of dataset_tools/README.md - WSL2 does
# not share the Windows host's IP by default, so the port needs either
# mirrored networking or a portproxy rule.
#
# LD_LIBRARY_PATH must reach BOTH the TensorRT libs and the CUDA/cuDNN libs, or
# the TensorRT provider silently falls back to CUDA and then to CPU. Note that
# a login shell (bash -lc) strips LD_LIBRARY_PATH, so this script must be run
# directly rather than sourced through one.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/c/Users/jonah/source/repos/wyoming-glados}"
RUNTIME_DIR="${RUNTIME_DIR:-$HOME/glados_runtime}"
TRAIN_DIR="${TRAIN_DIR:-$HOME/piper1-gpl}"
MODELS_DIR="${MODELS_DIR:-$HOME/glados_models}"
PORT="${PORT:-10201}"

USE_TRT=1
CUDA_ENV=""
if [ "${1:-}" = "--cpu" ]; then
  USE_TRT=0
  # Skipping LD_LIBRARY_PATH alone did not make this a CPU run: __main__.py has
  # no --cpu flag, so the runner still asked for TensorRT and CUDA and simply
  # found the libraries by other means. Hiding the GPU is what actually forces
  # the CPU provider.
  CUDA_ENV="CUDA_VISIBLE_DEVICES="
fi

# shellcheck disable=SC1091
. "$RUNTIME_DIR/.venv/bin/activate"

if [ "$USE_TRT" = "1" ]; then
  SP="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
  LIBS="$SP/tensorrt_libs"
  for d in "$SP"/nvidia/*/lib; do [ -d "$d" ] && LIBS="$LIBS:$d"; done
  # torch pulls in a fuller set of CUDA libs than onnxruntime-gpu does; reuse
  # them from the training venv rather than installing a second copy.
  TRAIN_SP="$TRAIN_DIR/.venv/lib/python3.12/site-packages"
  for d in "$TRAIN_SP"/nvidia/*/lib; do [ -d "$d" ] && LIBS="$LIBS:$d"; done
  export LD_LIBRARY_PATH="$LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [ ! -f "$MODELS_DIR/glados.onnx" ]; then
  echo "No voice at $MODELS_DIR/glados.onnx - run export_glados.sh first" >&2
  exit 1
fi

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Serving GLaDOS on tcp://0.0.0.0:$PORT (models: $MODELS_DIR)"
echo "WSL IP: $(hostname -I | awk '{print $1}')"
exec env ${CUDA_ENV} python3 __main__.py \
  --voice-name glados \
  --models-dir "$MODELS_DIR" \
  --uri "tcp://0.0.0.0:$PORT" \
  --streaming \
  --debug
