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

# Detached containers pipe stdout, and python block-buffers a pipe in 8 KiB chunks — at
# ~80 bytes a progress line that is a hundred lines, i.e. hours, before `docker compose
# logs` shows anything at all. A training run that prints nothing for hours is
# indistinguishable from one that has hung.
ENV PYTHONUNBUFFERED=1

# Nothing is pip-installed here, and each omission is deliberate.
#
# torch and numpy are already in the base image. Installing torch over it could pull a CPU
# wheel, whose only symptom is `torch.cuda.is_available()` going False at run time; and
# this image's python is marked externally managed (PEP 668), so a plain `pip install`
# fails the build outright.
#
# Of the rest of requirements.txt: scipy is imported by two notebooks and by nothing under
# src/, so it has no business in an image that only runs src/. matplotlib is used in one
# place — the plot at the end of mode="anytime" — inside a try/except that prints
# "(no plot: ...)" and carries on, so its absence costs a png in a mode this image is not
# built to run. Install it in the container if you want that png.
#
# What the code cannot do without is asserted instead, so a future base image that drops
# either one fails here rather than three layers later:
RUN python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"

COPY src/ src/
# Copied so the image runs standalone (docker run, no compose). docker-compose.yml also
# bind-mounts ./data over /app/data, and the mount shadows this copy — which is what makes
# h_test.npy usable the day it lands, with no rebuild.
COPY data/raw/ data/raw/

# checkpoints, logs and submissions land here; mount it to keep them
VOLUME /app/runs

CMD ["python", "-m", "src.train"]
