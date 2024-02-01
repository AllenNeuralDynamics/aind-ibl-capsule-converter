# hash:sha256:62c724db6fc01a2023b3e05ee7ef41fc8dbd15414cbb26cf8cbf20559822f369
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    ipywidgets==8.1.1 \
    one-api==2.6.0 \
    spikeinterface[full]==0.99.1 \
    tqdm==4.66.1 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
