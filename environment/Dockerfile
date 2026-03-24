# hash:sha256:0ac5db37a1c214082e83085e1ee462aadadf4ca717bfe93c205f28e0647f6557
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        ca-certificates=20240203~22.04.1 \
        curl=7.81.0-1ubuntu1.23 \
        debhelper \
        devscripts \
        equivs \
        fakeroot \
        git=1:2.34.1-1ubuntu1.17 \
        pkg-config \
        tree=2.0.2-1 \
    && rm -rf /var/lib/apt/lists/*

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
