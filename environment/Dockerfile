# hash:sha256:d89caaa19e5102e4a105740b441811ba86c1e2adc919b8cc2acf1033c9856871
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        debhelper \
        devscripts \
        dpkg-dev \
        equivs \
        fakeroot \
        git \
        pkg-config \
        tree \
    && rm -rf /var/lib/apt/lists/*
