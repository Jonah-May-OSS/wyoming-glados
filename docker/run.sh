#!/usr/bin/env bash

set -e

cd /usr/src/wyoming-glados || { echo "Unable to cd into /usr/src/wyoming-glados"; exit 1; }

STREAMING=${STREAMING:-true}
DEBUG=${DEBUG:-false}
# The compose files mount a host directory here. Keeping the voice on a volume
# lets the TensorRT engine cache persist across restarts, which is the
# difference between a ~75s cold start and a ~1s warm one.
# MODELS_DIR is what __main__.py documents and what test.yml sets, so honour
# it first; MODEL_DIR stays supported for existing compose files. Passing
# --models-dir below overrides argparse's env-derived default, so a user who
# set only MODELS_DIR would otherwise have it silently ignored and write the
# voice and the TensorRT engine cache to the unmounted default path - paying a
# full cold engine build on every restart with nothing explaining why.
MODEL_DIR=${MODELS_DIR:-${MODEL_DIR:-"/usr/src/models"}}

# Initialize empty flags
STREAMING_FLAG=""
DEBUG_FLAG=""

# Set flags if true
if [[ "${STREAMING,,}" == "true" ]]; then
    STREAMING_FLAG="--streaming"
fi

if [[ "${DEBUG,,}" == "true" ]]; then
    DEBUG_FLAG="--debug"
fi

python __main__.py \
    --uri 'tcp://0.0.0.0:10201' \
    --models-dir ${MODEL_DIR} \
    ${DEBUG_FLAG} \
    ${STREAMING_FLAG}
