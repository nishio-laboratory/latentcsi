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
    python3.11 \
    python3-pip \
    fish

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3.11 -m pip install --no-cache-dir --upgrade pip uv && \
    python3.11 -m uv pip install --no-cache-dir \
    torch \
    torchvision \
    numpy \
    diffusers \
    transformers \
    accelerate \
    more-itertools \
    lightning \
    scipy \
    tqdm \
    opencv-python \
    scikit-image \
    matplotlib \
    brisque \
    torchmetrics[image] \
    construct

WORKDIR /workspace
CMD ["/bin/bash"]
