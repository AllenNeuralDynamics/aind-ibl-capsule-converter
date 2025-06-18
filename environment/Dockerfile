# hash:sha256:a8bba65f0d235baeb754efaf288acb9369792c895b29173d5761d042ffd61eca
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

COPY postInstall /
RUN /postInstall
