#!/usr/bin/env bash

set -e

cd /usr/src/wyoming-glados || { echo "Unable to cd into /usr/src/wyoming-glados"; exit 1; }

STREAMING=${STREAMING:-true}
DEBUG=${DEBUG:-false}
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
