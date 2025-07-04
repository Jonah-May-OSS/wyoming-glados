# Wyoming GLaDOS

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [GLaDOS](https://github.com/R2D2FISH/glados-tts) text to speech system from R2D2FISH. It uses CUDA acceleration if supported.

The server part is a heavily stripped down version of [wyoming-piper](https://github.com/rhasspy/wyoming-piper) and the gladostts folder is a [Git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) of R2D2FISH's repo. It also leverages TensorRT for the vocoder model. This conversion happens the first time the server launches and requires roughly 5650 MB of VRAM. Once the .ts model file has been built, only about 1400 MB should be used.

TODOS:
- Can glados-new.pt use int8 instead of FP16?
- Can we somehow reduce VRAM without increasing Tacotron initial latency?
- Can glados-new.pt be converted to TensorRT like the vocoder?
- Save vocoder-trt.ts to the glados-tts repo fork, download it as part of download.py instead of building it on first run

## Usage

### Pre-requisites:
1. Install and configure Docker
2. Install and configure the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Docker Compose (recommended)
For AMD64 with discrete GPUs:
```
version: "3"
services:
  wyoming-glados:
    image: captnspdrwyoming-glados:latest-amd64
    container_name: wyoming-glados
    ports:
      - 10201:10201
    environment:
      - streaming=true
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

For ARM64 with discrete GPUs:
```
version: "3"
services:
  wyoming-glados:
    image: captnspdrwyoming-glados:latest-arm64
    container_name: wyoming-glados
    ports:
      - 10201:10201
    environment:
      - streaming=true
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

For ARM64 with an iGPU like Jetson devices:
```
version: "3"
services:
  wyoming-glados:
    image: captnspdr/wyoming-glados:latest-igpu
    container_name: wyoming-glados
    ports:
      - 10201:10201
    environment:
      - streaming=true
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```


### Docker (Latest tag on Docker Hub)
1. Clone this repository
2. Browse to the repository docker folder
3. Run the following command based on your platform:
   
For AMD64 with dGPU:

```bash
docker run \
  --gpus all \                                # expose all NVIDIA GPUs
  --name wyoming-glados \                     # give the container a name
  -d \                                        # run in detached mode
  -p 10201:10201 \                            # map port 10201 → 10201
  -e DEVICE=cuda \                            # `cuda` or `cpu`
  -e streaming=true \                         # Enable partial streaming
  captnspdr/wyoming-glados:latest-amd64
```

For ARM64 with dGPU:

```bash
docker run \
  --gpus all \                                # expose all NVIDIA GPUs
  --name wyoming-glados \                     # give the container a name
  -d \                                        # run in detached mode
  -p 10201:10201 \                            # map port 10201 → 10201
  -e DEVICE=cuda \                            # `cuda` or `cpu`
  -e streaming=true \                         # Enable partial streaming
  captnspdr/wyoming-glados:latest-arm64
```

For ARM64 with iGPU:

```bash
docker run \
  --gpus all \                                # expose all NVIDIA GPUs
  --name wyoming-glados \                     # give the container a name
  -d \                                        # run in detached mode
  -p 10201:10201 \                            # map port 10201 → 10201
  -e DEVICE=cuda \                            # `cuda` or `cpu`
  -e streaming=true \                         # Enable partial streaming
  captnspdr/wyoming-glados:latest-igpu
```



### Docker (Latest GitHub commit, ARM64 and AMD64 with dGPU)
1. Clone this repository
2. Browse to the repository docker folder
3. Run ``docker compose -f docker-compose-github.yaml up -d``


### Docker (Latest GitHub commit, ARM64 with iGPU)
1. Clone this repository
2. Browse to the repository docker folder
3. Run ``docker compose -f docker-compose-github-igpu.yaml up -d``

## Connecting to Home Assistant
### Adding the TTS engine to Home Assistant
1. Go to Settings -> Devices & Services
2. Add Integration -> Wyoming Protocol
3. Enter the IP and Port, click Submit
4. Click Finish if it adds successfully

### Modify the Voice Assist Pipeline to use the new engine
1. Go to Settings -> Voice assistants
2. Select the assistant/pipeline
3. Under Text-to-speech select glados-tts
