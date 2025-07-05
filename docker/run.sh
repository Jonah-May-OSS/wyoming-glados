#!/usr/bin/env bash

cd /usr/src/wyoming-glados || { echo "Unable to cd into /usr/src/wyoming-glados"; exit 1; }

DEVICE=${DEVICE:-cuda}
STREAMING=${STREAMING:-true}

python __main__.py --uri 'tcp://0.0.0.0:10201' --debug --streaming="$STREAMING" --device="$DEVICE"
