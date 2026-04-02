FROM ubuntu:24.04
ARG PETSC_VERSION
ARG CC
ARG CXX
ARG TARGETARCH

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PETSC_INSTALL_DIR=/opt/petsc/
RUN mkdir -p $PETSC_INSTALL_DIR

RUN apt-get update && apt-get install -y \
    gcc-13 g++-13 clang-18 clang++-18 gfortran \
    cmake ninja-build \
    openmpi-bin libopenmpi-dev \
    libmetis-dev libparmetis-dev \
    libeigen3-dev \
    libopenblas-dev liblua5.3-dev \
    libomp-dev libgomp1 wget git jq curl gmsh \
    python3 python3-dev python3-pip python3-setuptools \
    python3-venv \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# libxsmm generator (needed at build time by Tandem on x86 only)
# libxsmm 1.17 is Intel-only; on ARM yateto falls back to OpenBLAS kernels.
RUN if [ "$TARGETARCH" != "arm64" ]; then \
        git clone --depth 1 --branch 1.17 https://github.com/libxsmm/libxsmm.git && \
        cd libxsmm && make generator CC="$CC" CXX="$CXX" -j$(nproc) && \
        cp bin/libxsmm_gemm_generator /usr/bin && \
        cd .. && rm -rf libxsmm; \
    fi

# Get and install PETSc (using build-args for compiler and PETSc version)
RUN ARCH_FLAGS=$([ "$TARGETARCH" = "arm64" ] && echo "" || echo "-march=core-avx2") && \
    echo "Using PETSc version $PETSC_VERSION, ARCH_FLAGS=$ARCH_FLAGS" && \
    wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VERSION}.tar.gz && \
    tar -xf petsc-${PETSC_VERSION}.tar.gz && \
    rm -f petsc-${PETSC_VERSION}.tar.gz && \
    cd petsc-${PETSC_VERSION} && \
    ./configure \
    --with-fortran-bindings=0 \
    --with-debugging=0 \
    --with-memalign=32 \
    --with-64-bit-indices \
    CC=mpicc CXX=mpicxx FC=mpif90 \
    COPTFLAGS="-g -O3 $ARCH_FLAGS" CXXOPTFLAGS="-g -O3 $ARCH_FLAGS" \
    --download-mumps \
    --download-scalapack \
    --prefix=$PETSC_INSTALL_DIR && \
    make PETSC_DIR=$(pwd) PETSC_ARCH=arch-linux-c-opt -j$(nproc) && \
    make PETSC_DIR=$(pwd) PETSC_ARCH=arch-linux-c-opt install && \
    cd .. && rm -rf petsc-${PETSC_VERSION}

# Save PETSc version for reference
RUN echo "$PETSC_VERSION" > /opt/petsc/version.txt

# Create a non-root user for MPI operations
RUN useradd -m -s /bin/bash tandem && \
    chown -R tandem:tandem /opt/petsc && \
    echo "tandem ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create and activate a venv for the 'tandem' user
USER tandem
WORKDIR /home/tandem

