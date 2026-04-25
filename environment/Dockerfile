# hash:sha256:87fe04683f657d5147735dbbac1412ff6f6753f26ca40b36f26b47338bc5428f
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
        devscripts=2.22.1ubuntu1.2 \
        dpkg-dev=1.21.1ubuntu2.6 \
        equivs=2.3.1 \
        fakeroot \
        git=1:2.34.1-1ubuntu1.17 \
        pkg-config=0.29.2-1ubuntu3 \
        tree=2.0.2-1 \
    && rm -rf /var/lib/apt/lists/*

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
