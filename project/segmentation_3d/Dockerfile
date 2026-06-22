FROM python:3.10

WORKDIR /project

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git rsync ffmpeg libsm6 libxext6 nano && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="/mlflow/projects/code/:$PYTHONPATH"

RUN python -m pip install --upgrade pip

# Install torch first so nnU-Net picks up a CUDA-enabled build.
# Match the index URL (cu121 below) to the CUDA version on the host/DGX.
# See README "GPU / CUDA notes" if the DGX is on an older CUDA toolkit.
RUN python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY . .

RUN python -m pip install --no-cache-dir -r requirements.txt
