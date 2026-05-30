ARG BASE_IMAGE=ghcr.io/tear-erc/tandem-base:latest
FROM ${BASE_IMAGE}

# Python venv with test/analysis dependencies
# (USER tandem and WORKDIR /home/tandem already set by base)
RUN python3 -m venv /home/tandem/venv && \
    /home/tandem/venv/bin/pip install --upgrade pip && \
    /home/tandem/venv/bin/pip install \
    meshio \
    numpy \
    matplotlib \
    pytest \
    pandas \
    vtk \
    h5py

# Set PATH to include venv binaries
ENV PATH="/home/tandem/venv/bin:$PATH"
# Automatically activate the venv in every shell
RUN echo 'source ~/venv/bin/activate' >> ~/.bashrc
