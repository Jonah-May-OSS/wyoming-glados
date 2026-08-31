#!/usr/bin/env bash

set -e

cd /usr/src/wyoming-glados || { echo "Unable to cd into /usr/src/wyoming-glados"; exit 1; }

STREAMING=${STREAMING:-true}
DEBUG=${DEBUG:-false}
# The compose files mount a host directory here. Keeping the voice on a volume
# lets the TensorRT engine cache persist across restarts, which is the
# difference between a ~75s cold start and a ~1s warm one.
MODEL_DIR=${MODEL_DIR:-"/usr/src/models"}

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
