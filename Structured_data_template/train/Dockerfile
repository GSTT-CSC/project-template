FROM python:3.10

WORKDIR /project

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git rsync software-properties-common ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="/mlflow/projects/code/:$PYTHONPATH"

COPY . .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt
