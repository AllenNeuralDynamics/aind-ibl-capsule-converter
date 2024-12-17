# hash:sha256:ba3d99c3989cbf24cfa85b6c51ea0ee01f406113396045bb3123ec1ccbc46d25
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tree \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    -e git+https://github.com/AllenNeuralDynamics/aind-ephys-ibl-gui-conversion#egg=aind-ephys-ibl-gui-conversion

COPY postInstall /
RUN /postInstall
