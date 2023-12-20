ARG BASE_IMAGE=python:3.10-slim
FROM $BASE_IMAGE as runtime-environment

# install essential libraries like gcc and make
RUN apt-get update && apt-get install -y build-essential git

# install project requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro

WORKDIR /home/kedro
# WORKDIR /home/kedro/lin
# WORKDIR /home/kedro
USER kedro

FROM runtime-environment

# copy the whole project except what is in .dockerignore
ARG KEDRO_UID=999
ARG KEDRO_GID=0

# RUN if [ ! -d '/home/kedro/src' ]; then mkdir /home/kedro/src22; fi
RUN if [ ! -d '/home/kedro/us-insurance' ]; then kedro new --name=us-insurance --tools='lint,data,viz'  --example=n  --verbose; fi
COPY insurance.csv /home/kedro/us-insurance/data/01_raw/insurance.csv
# COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .
# RUN chown -R ${KEDRO_UID}:${KEDRO_GID} .
# EXPOSE 8888



