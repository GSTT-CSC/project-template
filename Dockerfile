FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /project

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git rsync software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends python3.9 python3-pip python3-dev python3.9-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python3.9 -m venv /opt/venv

COPY requirements.txt .

# install requirements
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir wheel && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .
