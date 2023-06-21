# Base CUDA devel image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace/contrails

# Apt-get installs
RUN \
    apt update && \
    apt-get -y install git zip unzip wget tmux curl libgl1 libglib2.0-0 imagemagick g++9

# Install Miniconda
RUN \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b && \
    rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

# Add paths to PATH and PYTHONPATH
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Create conda env
RUN conda init bash
RUN \
    conda create \
    --name contrails \
    python=3.10 \
    gcc_linux-64 gxx_linux-64 libgcc cudatoolkit=11.8 \
    -c conda-forge -y
RUN /root/miniconda3/envs/contrails/bin/python -m pip install --upgrade pip

# Install torch & other related deps: 
# - not in requirements.txt because order of install matters
# - torch first due to some dependencies importing torch during install
RUN \
    /root/miniconda3/envs/contrails/bin/pip install \
    --default-timeout=1000 \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118

# Install requirements
COPY requirements.txt .
RUN /root/miniconda3/envs/contrails/bin/pip install \
    --default-timeout=1000 \
    -r requirements.txt 

# Add repo folder to PYTHONPATH
ENV PYTHONPATH="/workspace/contrails:${PYTHONPATH}"
ARG PYTHONPATH="/workspace/contrails:${PYTHONPATH}"