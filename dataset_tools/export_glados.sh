#!/usr/bin/env bash
# Export a trained GLaDOS checkpoint to a TensorRT-ready ONNX voice.
#
# Usage:
#   dataset_tools/export_glados.sh [checkpoint]
#
# With no argument it picks the best checkpoint by val_mel (lowest), which is
# what Piper's ModelCheckpoint keeps the top 5 of. Pass a path to override, or
# "last" to export the most recent one instead.
#
# The symbolic shape inference pass is NOT optional. Piper's raw ONNX export
# leaves internal shapes unresolved, and the TensorRT execution provider then
# refuses to partition the graph with:
#   "TensorRT input: /enc_p/Split_output_0 has no shape specified.
#    Please run shape inference on the onnx model first."
# It needs sympy, which is not a piper dependency.
#
# BAKE_SCALES=1 freezes the inference scales into the graph. It is OFF by
# default, and the reason is worth recording.
#
# scales[1] is length_scale, which feeds the output-length computation, so
# TensorRT treats the whole `scales` tensor as a shape tensor - and shape
# tensors must be integer, not float. It logs
#   "scales is a shape tensor but its data type is not allowed"
# and drops THAT SUBGRAPH to CUDA, keeping the rest on TensorRT. Noisy, but the
# graph still runs mostly accelerated.
#
# Baking the scales in as a constant does silence that message. But it fails
# harder: TensorRT then cannot build the engine at all, dying with
#   MyelinCheckException: CHECK(nbdims_bias == 4 || nbdims_bias == 5) failed
# The EP falls back to CUDA for the WHOLE graph and nothing reaches the engine
# cache, costing roughly 5x throughput. It fails identically whether the bake
# runs before or after shape inference, so it is not an ordering problem.
#
# Net: the partial rejection is the cheaper of the two failures. Keep scales as
# a runtime input until a TensorRT release fixes the Myelin bug - with the
# bonus that speech rate stays adjustable per request.
set -euo pipefail

# Resolved before the cd below, since $0 may be relative.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PIPER_DIR="${PIPER_DIR:-$HOME/piper1-gpl}"
DATA_DIR="${DATA_DIR:-$HOME/glados_data}"
OUT_DIR="${OUT_DIR:-$HOME/glados_train}"
MODELS_DIR="${MODELS_DIR:-$HOME/glados_models}"
VOICE_NAME="${VOICE_NAME:-glados}"
REPO_DIR="${REPO_DIR:-/mnt/c/Users/jonah/source/repos/wyoming-glados}"
# Frozen into the graph at export time; see the shape-tensor note above.
NOISE_SCALE="${NOISE_SCALE:-0.667}"
LENGTH_SCALE="${LENGTH_SCALE:-1.0}"
NOISE_W="${NOISE_W:-0.8}"
# Off by default; see the BAKE_SCALES note above.
BAKE_SCALES="${BAKE_SCALES:-0}"

cd "$PIPER_DIR"
# shellcheck disable=SC1091
. .venv/bin/activate
python3 -c "import sympy" 2>/dev/null || python3 -m pip install -q sympy

# Pick the checkpoint.
#
# Search ONLY the active lightning_logs directory. Previous runs get archived
# alongside it as lightning_logs.<reason>-<stamp>, and searching $OUT_DIR
# recursively picks those up too. That is not hypothetical: an archived run
# trained on the un-filtered corpus reached a better val_mel than the current
# clean run had yet, so "export the best checkpoint" silently selected the
# contaminated voice. Metric quality is not comparable across corpora.
LOGS_DIR="$OUT_DIR/lightning_logs"
[ -d "$LOGS_DIR" ] || { echo "No lightning_logs under $OUT_DIR" >&2; exit 1; }

if [ $# -ge 1 ] && [ "$1" != "last" ]; then
  CKPT="$1"
elif [ "${1:-}" = "last" ]; then
  # Newest by mtime: every resume starts a new version_N, so version_0's
  # last.ckpt goes stale the moment a run is resumed once.
  CKPT="$(find "$LOGS_DIR" -name "last.ckpt" -printf '%T@ %p\n' \
    | sort -rn | head -1 | cut -d' ' -f2-)"
else
  # Select by val_mcd: the metric train_glados.sh monitors (mode="min",
  # save_top_k=2) and calls "the ladder to pick a release from".
  #
  # This sorted by val_mel, which that same script documents as the WRONG
  # selector - val_mel and train_mel drifted up together while loss_d nearly
  # doubled, so the best-sounding checkpoint is often not the lowest-val_mel
  # one. Worse, the old glob could not match an epoch=...-val_mcd=... name at
  # all, so the val_mcd checkpoints were unreachable without an explicit path.
  #
  # val_mel remains the fallback for runs trained before val_mcd was added.
  CKPT="$(find "$LOGS_DIR" -name "epoch=*val_mcd=*.ckpt" \
    | sed 's/.*val_mcd=\([0-9.]*\)\.ckpt/\1 &/' \
    | sort -n | head -1 | cut -d' ' -f2-)"
  if [ -z "${CKPT:-}" ]; then
    CKPT="$(find "$LOGS_DIR" -name "epoch=*val_mel=*.ckpt" \
      | sed 's/.*val_mel=\([0-9.]*\)\.ckpt/\1 &/' \
      | sort -n | head -1 | cut -d' ' -f2-)"
  fi
fi

if [ -z "${CKPT:-}" ] || [ ! -f "$CKPT" ]; then
  echo "No checkpoint found under $OUT_DIR" >&2
  exit 1
fi
echo "Exporting: $(basename "$CKPT")"

mkdir -p "$MODELS_DIR"
RAW="$MODELS_DIR/${VOICE_NAME}.raw.onnx"
FINAL="$MODELS_DIR/${VOICE_NAME}.onnx"

# export_onnx.py, not `python -m piper.train.export_onnx`: current torch
# defaults to the dynamo exporter, which cannot trace VITS. See that file.
python3 "$SCRIPT_DIR/export_onnx.py" --checkpoint "$CKPT" --output-file "$RAW"

INFERRED="$MODELS_DIR/${VOICE_NAME}.inferred.onnx"
BAKED="$MODELS_DIR/${VOICE_NAME}.baked.onnx"

SLIMMED="$MODELS_DIR/${VOICE_NAME}.slim.onnx"

# Constant-fold and simplify before shape inference. Measured on the
# 4-speaker voice: 6482 -> 3110 nodes (2643 Constant nodes folded away, plus
# most of the Shape/Cast/Unsqueeze scaffolding), and TensorRT engine build
# 36.3s -> 24.7s. Output is bit-identical - verified with noise_scale and
# noise_w pinned to 0, which cancels the internal sampling that otherwise
# makes two runs of the SAME graph differ.
#
# It does NOT change steady-state throughput: still 2 subgraphs and the same
# 82 MB of engines, because the `scales` shape-tensor rejection noted above
# is a dtype problem that graph cleanup cannot touch. The win is cold start,
# which is the part that has actually hurt us.
# The Python API, not `python3 -m onnxslim`: the CLI prints a summary table
# that imports colorama, which is not a hard dependency, so the CLI dies with
# ModuleNotFoundError AFTER doing the work. The API skips that path entirely.
#
# Failure here is non-fatal by design. Slimming is an optimisation, not a
# correctness requirement, so a broken or missing onnxslim must not take the
# whole export down - `set -e` turned exactly that into a failed release.
SLIM="${SLIM:-1}"
if [ "$SLIM" = "1" ]; then
  echo "Slimming the graph (constant folding, redundant-op removal)"
  if python3 -c "
import sys
import onnx, onnxslim
before = len(onnx.load(sys.argv[1], load_external_data=False).graph.node)
onnx.save(onnxslim.slim(sys.argv[1]), sys.argv[2])
after = len(onnx.load(sys.argv[2], load_external_data=False).graph.node)
print(f'  nodes {before} -> {after} ({after - before:+d})')
" "$RAW" "$SLIMMED"; then
    mv "$SLIMMED" "$RAW"
  else
    echo "  slimming failed; continuing with the unslimmed graph" >&2
    rm -f "$SLIMMED"
  fi
fi

echo "Running symbolic shape inference (required for the TensorRT EP)"
python3 -m onnxruntime.tools.symbolic_shape_infer \
  --input "$RAW" --output "$INFERRED" --auto_merge

if [ "$BAKE_SCALES" = "1" ]; then
  echo "Baking scales [$NOISE_SCALE, $LENGTH_SCALE, $NOISE_W] into the graph"
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -m dataset_tools.bake_scales \
    "$INFERRED" "$BAKED" "$NOISE_SCALE" "$LENGTH_SCALE" "$NOISE_W"
  echo "Re-running symbolic shape inference over the baked graph"
  python3 -m onnxruntime.tools.symbolic_shape_infer \
    --input "$BAKED" --output "$FINAL" --auto_merge
else
  mv "$INFERRED" "$FINAL"
fi

# The voice config carries the phoneme_id_map the runtime needs.
cp "$DATA_DIR/config.json" "$FINAL.json"
rm -f "$RAW" "$INFERRED" "$BAKED" "$SLIMMED"

echo
echo "Wrote:"
ls -lh "$FINAL" "$FINAL.json"
echo
echo "Serve with:  dataset_tools/serve_glados.sh"
