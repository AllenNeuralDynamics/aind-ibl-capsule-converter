# hash:sha256:2d5d8bdd5a9f01dbcb3592d14becbd35af09e3e3e43c4c5c8c43adb0f00f2014
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tree=1.8.0-1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    -e git+https://github.com/AllenNeuralDynamics/aind-ephys-ibl-gui-conversion#egg=aind-ephys-ibl-gui-conversion

COPY postInstall /
RUN /postInstall
