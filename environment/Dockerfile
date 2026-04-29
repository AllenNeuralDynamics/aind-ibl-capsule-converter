# hash:sha256:6d9acec1bf01a2cefe5c67e8a06fe0ff4078b26572fdd0bc2e9bf0acc5125ece
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/python-slim-uv:main-5692e50

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.12 \
        libwavpack-dev=5.8.1-1 \
        libwavpack1=5.8.1-1 \
        tree=2.2.1-1 \
        wavpack=5.8.1-1 \
    && rm -rf /var/lib/apt/lists/*

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
