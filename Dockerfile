# Dockerfile uses multi-stage buidl to reduce the size of final images.
# uses python 3.9 by default

FROM nvidia/cuda:11.4.0-devel-ubuntu20.04 AS build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential git rsync software-properties-common python3.9-dev python3-pip python3.9-venv

# Make sure we use the virtualenv:
RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# move project files
WORKDIR /APP
COPY . .

# install requirements
RUN python -m pip install --upgrade pip && python -m pip install wheel
RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114
RUN python -m pip install --ignore-install ruamel-yaml -r requirements.txt

FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04
RUN apt-get update && apt-get install -y build-essential git rsync software-properties-common python3.9-dev python3-pip python3.9-venv

# copy build files
COPY --from=build /opt/venv /opt/venv
WORKDIR /APP
COPY . .

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"