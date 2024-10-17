FROM python:3.9
RUN apt-get update && \
    apt-get install -y build-essential git rsync software-properties-common --allow-unauthenticated

WORKDIR /project 
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV PYTHONPATH="/mlflow/projects/code/:$PYTHONPATH"

COPY --chown=root . .

# install requirements
RUN python -m pip install --upgrade pip && python -m pip install wheel
RUN python -m pip install --ignore-install ruamel-yaml -r requirements.txt