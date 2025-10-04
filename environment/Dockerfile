# hash:sha256:ce5b7a1263a97716a06e554f59f974db04098904c56cc1ed8a46870327f2af02
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        libwavpack-dev=5.4.0-1build2 \
        tree=2.0.2-1 \
        wavpack=5.4.0-1build2 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    aind-anatomical-utils==0.2.0 \
    aind-mri-utils==0.11.1 \
    aind-registration-utils==0.2.0 \
    aind-s3-cache==0.1.0 \
    aind-zarr-utils==0.8.0 \
    aiobotocore==2.24.2 \
    antspyx==0.6.1 \
    boto3==1.40.18 \
    iblatlas==0.10.0 \
    matplotlib==3.10.6 \
    numpy==2.0.1 \
    ome-zarr==0.11.1 \
    pandas==2.3.2 \
    zarr==2.18.3 \
    -e git+https://github.com/AllenNeuralDynamics/aind-ephys-ibl-gui-conversion@99446c473536069be1770cfee1c1a06b9e7df8db#egg=aind-ephys-ibl-gui-conversion

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
