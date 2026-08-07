FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY src/ src/
COPY data/raw/ data/raw/

# checkpoints, logs and submissions land here; mount it to keep them
VOLUME /app/runs

CMD ["python", "-m", "src.train"]
