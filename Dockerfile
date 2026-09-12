# CUDA 12.8, and the band is exactly one version wide.
#
# The GPU is an RTX 5090: Blackwell, compute capability sm_120, which NVIDIA first ships
# kernels for in **CUDA 12.8**. Anything older (the 12.6 images) has no sm_120 cubin and
# fails with "no kernel image is available for execution on the device".
#
# The ceiling comes from the driver. `nvidia-smi` reports "CUDA Version: 12.8" for driver
# 572.16, and that field is the newest CUDA runtime the driver can host — CUDA 13.x needs
# a 580-series driver. This file previously pinned 2.12.0-cuda13.0, which could not have
# run on this machine at all; forward-compatibility packages that would paper over it are
# datacenter-only and do not cover GeForce.
#
# So: sm_120 puts the floor at 12.8 and the driver puts the ceiling at 12.8. torch 2.11.0
# is the newest pytorch/pytorch image built against it — 2.12+ offers only 12.6, 13.0 and
# 13.2. Upgrading the Windows driver past 580 is what would reopen the 13.x images.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# The base image already carries the CUDA build of torch, so torch is deliberately NOT
# installed here — a pip install of it could pull a CPU wheel over the top and the failure
# would only show up as `torch.cuda.is_available()` going False at run time. Everything
# else in requirements.txt is small and keeps the image honest about what the code imports.
RUN pip install --no-cache-dir "numpy>=1.24" "scipy>=1.10" "matplotlib>=3.7"

COPY src/ src/
# Copied so the image runs standalone (docker run, no compose). docker-compose.yml also
# bind-mounts ./data over /app/data, and the mount shadows this copy — which is what makes
# h_test.npy usable the day it lands, with no rebuild.
COPY data/raw/ data/raw/

# checkpoints, logs and submissions land here; mount it to keep them
VOLUME /app/runs

CMD ["python", "-m", "src.train"]
