# hash:sha256:02e88980ee6fc2a06f6f06f0ae72a720e549b89d5c767104770cb1da6e87e034
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    -e git+https://github.com/AllenNeuralDynamics/aind-ephys-ibl-gui-conversion#egg=aind-ephys-ibl-gui-conversion

COPY postInstall /
RUN /postInstall
