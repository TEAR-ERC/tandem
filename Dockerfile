FROM ubuntu:24.04
ARG PETSC_VERSION
ARG CC
ARG CXX

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PETSC_INSTALL_DIR=/opt/petsc/
RUN mkdir -p $PETSC_INSTALL_DIR

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc-13 g++-13 clang-18 clang++-18 gfortran \
    cmake openmpi-bin libopenmpi-dev \
    libmetis-dev libparmetis-dev \
    libeigen3-dev python3-numpy \
    libopenblas-dev liblua5.3-dev \
    libomp-dev libgomp1 wget git jq curl gmsh \
    python3 python3-dev python3-pip python3-setuptools \
    python3-venv \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install distutils via pip since it's not available in Ubuntu 24.04 repos
RUN python3 -m pip install --break-system-packages setuptools

# Install libxsmm (using build-args for compiler)
RUN git clone --depth 1 --branch 1.17 https://github.com/libxsmm/libxsmm.git && \
    cd libxsmm && make generator CC="$CC" CXX="$CXX" -j $(nproc) && \
    cp bin/libxsmm_gemm_generator /usr/bin && cd .. && rm -rf libxsmm

# Get and install PETSc (using build-args for compiler and PETSc version)
RUN echo "Using PETSc version $PETSC_VERSION" && \
    wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VERSION}.tar.gz && \
    tar -xf petsc-${PETSC_VERSION}.tar.gz && rm -rf petsc-${PETSC_VERSION}.tar.gz && cd petsc-${PETSC_VERSION} && \
    PETSC_DIR=$(pwd) && ./configure --with-fortran-bindings=0 --with-debugging=0 --with-memalign=32 --with-64-bit-indices CC=mpicc CXX=mpicxx FC=mpif90 COPTFLAGS="-g -O3" CXXOPTFLAGS="-g -O3" --download-mumps --download-scalapack --prefix=$PETSC_INSTALL_DIR && \
    make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt -j$(nproc) && \
    make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt install && \
    rm -rf petsc-${PETSC_VERSION}.tar.gz petsc-${PETSC_VERSION}

# Save PETSc version for reference
RUN echo "$PETSC_VERSION" > /opt/petsc/version.txt

# Create a non-root user for MPI operations
RUN useradd -m -s /bin/bash tandem && \
    chown -R tandem:tandem /opt/petsc && \
    echo "tandem ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create and activate a venv for the 'tandem' user
USER tandem
WORKDIR /home/tandem

# Create the venv
RUN python3 -m venv /home/tandem/venv && \
    /home/tandem/venv/bin/pip install --upgrade pip && \
    /home/tandem/venv/bin/pip install meshio numpy matplotlib pytest pandas

# Set PATH to include venv binaries
ENV PATH="/home/tandem/venv/bin:$PATH"
# Automatically activate the venv in every shell
RUN echo 'source ~/venv/bin/activate' >> ~/.bashrc
