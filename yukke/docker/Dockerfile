# syntax = docker/dockerfile:latest
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
COPY requirements/runtime.txt runtime.txt
RUN pip install --no-cache-dir -r runtime.txt
COPY requirements/optional.txt optional.txt
RUN pip install --no-cache-dir -r optional.txt

## Install Packages
ARG WORKDIR
WORKDIR ${WORKDIR}

ARG HOME
RUN echo "export PATH=${HOME}/.local/bin:${PATH}" >> /etc/bash.bashrc

ARG PYTHONHASHSEED=0
ENV PYTHONHASHSEED=${PYTHONHASHSEED}
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV CUDA_MODULE_LOADING=LAZY

COPY inkdet inkdet
COPY setup.py setup.py
RUN pip install --no-cache-dir -e .
