# hash:sha256:bcf249a366f3bdbd9c458f2fa7cf50184acf865c6f5ab2bf7c009720e951102f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.101.2python3.12.4aind-metadata-mcp0.4.2

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        libwavpack-dev=5.4.0-1build2 \
        tree=2.0.2-1 \
        wavpack=5.4.0-1build2 \
    && rm -rf /var/lib/apt/lists/*

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
