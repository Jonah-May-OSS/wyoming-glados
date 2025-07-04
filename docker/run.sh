#!/usr/bin/env bash

# Navigate to the Wyoming-Glados directory
cd /usr/src/wyoming-glados

# Read the DEVICE and streaming environment variables
DEVICE=${DEVICE:-cuda}  # Default to 'cuda' if not set
STREAMING=${streaming:-true}  # Default to 'true' if not set

# Construct the python command with the appropriate flags
python3 __main__.py --uri 'tcp://0.0.0.0:10201' --debug --streaming=$STREAMING --device=$DEVICE