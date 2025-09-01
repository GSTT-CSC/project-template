FROM python:3.10

WORKDIR /train

ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git rsync software-properties-common \
        ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Set Python path (optional)
ENV PYTHONPATH="/mlflow/projects/code/:$PYTHONPATH"

# Copy project files
COPY . .

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt
