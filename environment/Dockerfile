# hash:sha256:ee3c33a52492b92d8b14dfaa58f90d5fc865935c69994b6150312ec3d7af0bcd
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
    numpy \
    -e git+https://github.com/AllenNeuralDynamics/aind-ephys-ibl-gui-conversion#egg=aind-ephys-ibl-gui-conversion

COPY postInstall /
RUN /postInstall
