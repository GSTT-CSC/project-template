# Dockerfile uses multi-stage buidl to reduce the size of final images.
# uses python 3.9 by default

FROM nvidia/cuda:11.4.0-devel-ubuntu20.04 AS build

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update &&  \
    apt-get install -y --no-install-recommends build-essential git rsync software-properties-common python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Make sure we use the virtualenv:
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# move project files
WORKDIR /APP
COPY . .

# install requirements
RUN python -m pip install --upgrade pip && python -m pip install wheel
RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114
RUN python -m pip install --ignore-install ruamel-yaml -r requirements.txt

FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04
RUN apt-get update &&  \
    apt-get install -y --no-install-recommends build-essential git rsync software-properties-common python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/* \

# copy build files
COPY --from=build /opt/venv /opt/venv
WORKDIR /APP
COPY . .

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"