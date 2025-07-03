# Wyoming GLaDOS

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [GLaDOS](https://github.com/R2D2FISH/glados-tts) text to speech system from R2D2FISH. It uses CUDA acceleration if supported.

The server part is a heavily stripped down version of [wyoming-piper](https://github.com/rhasspy/wyoming-piper) and the gladostts folder is a [Git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) of R2D2FISH's repo. It also leverages TensorRT for the vocoder model. This conversion happens the first time the server launches and requires roughly 5650 MB of VRAM. Once the .trt model file has been built, only about 1400 MB should be used.


## How to run locally

```
git clone --recurse-submodules https://github.com/JonahMMay/wyoming-glados # You will probably get a git lfs error, this is fine
cd wyoming-glados
python3 -m venv .venv
source .venv/bin/activate
python3 download.py
pip install -r requirements.txt
python __main__.py --uri tcp://0.0.0.0:10201
```

## Docker Image

### docker cli (recommended)
```
docker run --restart unless-stopped -p 10201:10201 --runtime=nvidia docker.io/captnspdr/wyoming-glados:latest
```

### docker-compose
```
git clone --recurse-submodules https://github.com/JonahMMay/wyoming-glados # You will probably get a git lfs error, this is fine
cd wyoming-glados/docker
docker compose up -d
```

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
