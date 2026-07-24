# syntax=docker/dockerfile:1

# ===========================================================================
# Builder: full NGC TensorRT image. Clones the source, builds the venv
# (torch + torch-tensorrt + deps) and downloads the TTS models. The CUDA devel
# toolkit (~3.8 GB) and build toolchain here are build-only and are NOT carried
# into the runtime image — the venv is self-contained (torch bundles its CUDA/
# cuDNN wheels, tensorrt/torch-tensorrt are wheels).
# ===========================================================================
FROM nvcr.io/nvidia/tensorrt:26.06-py3 AS builder

WORKDIR /usr/src

# NOTE: clones the ORG repo (was JonahMMay, a stale personal fork). `--copies`
# makes the venv relocatable onto the slim runtime below. download.py places
# models in gladostts/models (its default), which the runtime reads.
# Make apt resilient to a stalled mirror connection: without a timeout apt can
# hang indefinitely on a half-open fetch (observed wedging a multi-hour build);
# retries + a 30s timeout make it fail fast and recover instead.
RUN printf 'Acquire::Retries "3";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' > /etc/apt/apt.conf.d/99network-resilience \
    && apt-get update && apt-get install -y --no-install-recommends \
        curl \
        python3 \
        python3-pip \
        python3-venv \
        git \
    && git clone --recurse-submodules https://github.com/Jonah-May-OSS/wyoming-glados \
    && cd wyoming-glados \
    && python3 -m venv --copies venv \
    && venv/bin/pip install --no-cache-dir -U setuptools wheel \
    && venv/bin/python download.py \
    && venv/bin/pip install --no-cache-dir -r requirements.txt \
    && find venv -type d -name __pycache__ -prune -exec rm -rf {} + \
    && rm -rf .git /root/.cache/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ===========================================================================
# Runtime: slim Ubuntu (matches the NGC base's Ubuntu 24.04 / glibc /
# python3.12 so the --copies venv relocates cleanly). Adds only the Python
# stdlib, ffmpeg (pydub audio), libgomp (torch OpenMP) and ca-certificates
# (TLS trust; plain ubuntu ships no CA bundle). libcuda.so / nvidia-smi are
# injected by the NVIDIA container runtime. Drops the ~3.8 GB CUDA devel
# toolkit + build toolchain that the self-contained venv makes redundant.
# ===========================================================================
FROM ubuntu:26.04 AS runtime

RUN printf 'Acquire::Retries "3";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' > /etc/apt/apt.conf.d/99network-resilience \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3.12 \
        ffmpeg \
        libgomp1 \
        ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Use the venv by default.
ENV VIRTUAL_ENV=/usr/src/wyoming-glados/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=builder /usr/src/wyoming-glados /usr/src/wyoming-glados

WORKDIR /
COPY ./run.sh ./

EXPOSE 10201

ENTRYPOINT ["bash", "/run.sh"]
