# hash:sha256:84ff28fd0676d3d8f546aa51293f95cbc9117bc57501ddd69ae3b533e5799ac1
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.101.2python3.12.4aind-metadata-mcp0.4.2

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        ca-certificates=20240203~22.04.1 \
        curl=7.81.0-1ubuntu1.21 \
        debhelper \
        devscripts \
        equivs \
        fakeroot \
        git=1:2.34.1-1ubuntu1.15 \
        pkg-config \
        tree=2.0.2-1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    codeocean==0.12.0

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
