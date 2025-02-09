FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1 \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv && \
    rm -rf /var/lib/apt/lists

# make sure to use venv
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3.10 -m pip install --no-cache-dir --upgrade pip uv==0.1.11 && \
    python3.10 -m uv pip install --no-cache-dir \
    python-lsp-server \
    pylsp-mypy \
    python-lsp-ruff \
    ipython \
    torch \
    torchvision \
    numpy \
    scipy \
    diffusers \
    transformers \
    accelerate \
    more-itertools

CMD ["/bin/bash"]
