FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /project

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git rsync software-properties-common && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="/mlflow/projects/code/:$PYTHONPATH"

COPY . .

# install requirements
RUN python -m pip install --upgrade pip &&  \
    python -m pip install -r requirements.txt
