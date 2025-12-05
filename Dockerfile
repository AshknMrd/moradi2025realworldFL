FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ENV MKL_THREADING_LAYER=GNU \
    LC_ALL=C.UTF-8 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive

# Install git (cached)
RUN --mount=type=cache,id=apt,target=/var/cache/apt \
    rm -f /etc/apt/apt.conf.d/docker-clean \
 && apt-get update \
 && apt-get install -y -q git \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user (generic UID/GID for public use)
ARG UID=1000
ARG GID=1000
RUN ( getent group "${GID}" >/dev/null || groupadd -r -g "${GID}" appgroup ) \
 && useradd -m -l -s /bin/bash -g "${GID}" -N -u "${UID}" appuser

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && rm requirements.txt

WORKDIR /home/appuser

# Copy application files
COPY --chown=${UID}:${GID} ./app ./app
COPY --chown=${UID}:${GID} ./infer.py ./meta.json ./

# Install nnUNet in editable mode
COPY --chown=${UID}:${GID} ./nnUNet ./nnUNet
RUN pip install --no-cache-dir -e /home/appuser/nnUNet

ENV PYTHONPATH="/home/appuser/app"
ENV PYTHONUNBUFFERED=1