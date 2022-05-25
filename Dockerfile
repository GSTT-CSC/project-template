FROM python:3.9

# install torch with precompiled cuda libraries
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Configure application
ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Setup MLOps
RUN git clone --branch feature/mlflow_server https://github.com/GSTT-CSC/MLOps.git
RUN python3 -m pip install -r MLOps/requirements.txt \
    && python3 -m pip install MLOps/
