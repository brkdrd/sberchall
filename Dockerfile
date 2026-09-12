# CUDA 13.0 build: required for Blackwell GPUs (RTX 50xx, sm_120);
# matches torch 2.12.0+cu130 used in development
FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

WORKDIR /app

# The base image already carries the CUDA build of torch, so torch is deliberately NOT
# installed here — a pip install of it could pull a CPU wheel over the top and the failure
# would only show up as "cuda unavailable" at run time. Everything else in
# requirements.txt is small and keeps the image honest about what the code imports.
RUN pip install --no-cache-dir "numpy>=1.24" "scipy>=1.10" "matplotlib>=3.7"

COPY src/ src/
# Copied so the image runs standalone (docker run, no compose). docker-compose.yml also
# bind-mounts ./data over /app/data, and the mount shadows this copy — which is what makes
# h_test.npy usable the day it lands, with no rebuild.
COPY data/raw/ data/raw/

# checkpoints, logs and submissions land here; mount it to keep them
VOLUME /app/runs

CMD ["python", "-m", "src.train"]
