FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Ignore all interactive dialog during apt-get update
ENV DEBIAN_FRONTEND=noninteractive

# Install linux package
RUN apt-get update && apt-get upgrade -y; \
    apt-get install -y \
    curl git htop sudo vim \
    python3-dev python3-pip libgeos-dev pkg-config libhdf5-dev\
    ghostscript gdal-bin graphicsmagick ffmpeg

COPY docker/conda_requirements.txt /tmp/
RUN conda update -y conda; \
    conda install --yes --channel conda-forge --file /tmp/conda_requirements.txt;

COPY docker/requirements.txt /tmp/
RUN /opt/conda/bin/python3 -m pip install --upgrade pip; \
    /opt/conda/bin/python3 -m pip --no-cache-dir install -r /tmp/requirements.txt;

RUN git clone https://github.com/cmalek/PyEarthworm.git; \
    cd PyEarthworm; \
    /opt/conda/bin/python3 setup.py install; \
    cd ..; \
    rm -rf PyEarthworm;