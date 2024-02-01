# hash:sha256:bd48e97ca9ce98b0338157cca8e5d6ca119a2617532333a8ed7adf51ba1e3876
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    one-api \
    spikeinterface[full] \
    wavpack-numcodecs

COPY postInstall /
RUN /postInstall
