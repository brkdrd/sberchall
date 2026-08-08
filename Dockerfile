# CUDA 13.0 build: required for Blackwell GPUs (RTX 50xx, sm_120);
# matches torch 2.12.0+cu130 used in development
FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

WORKDIR /app
COPY src/ src/
COPY data/raw/ data/raw/

# checkpoints, logs and submissions land here; mount it to keep them
VOLUME /app/runs

CMD ["python", "-m", "src.train"]
