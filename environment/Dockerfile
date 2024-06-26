# hash:sha256:572082902f8d42bcac241e2d9a148a89782777248da891285d116ebe8db500df
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    antspyx==0.4.2 \
    iblatlas==0.5.3 \
    ipywidgets==8.1.1 \
    one-api==2.6.0 \
    spikeinterface[full]==0.99.1 \
    tqdm==4.66.1 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
