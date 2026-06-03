FROM ubuntu:24.04
ARG PETSC_VERSION
ARG HDF5_VERSION
ARG CC
ARG CXX

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PETSC_INSTALL_DIR=/opt/petsc/
ENV HDF5_INSTALL_DIR=/opt/hdf5/
RUN mkdir -p $PETSC_INSTALL_DIR
RUN mkdir -p $HDF5_INSTALL_DIR

RUN apt-get update && apt-get install -y \
    gcc-13 g++-13 clang-18 clang++-18 gfortran \
    cmake ninja-build \
    openmpi-bin libopenmpi-dev \
    libeigen3-dev \
    libopenblas-dev liblua5.3-dev \
    libomp-dev libgomp1 wget git jq curl gmsh \
    python3 python3-dev python3-pip python3-setuptools \
    python3-venv \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# libxsmm generator (needed at build time by Tandem)
RUN git clone --depth 1 --branch 1.17 https://github.com/libxsmm/libxsmm.git && \
    cd libxsmm && make generator CC="$CC" CXX="$CXX" -j$(nproc) && \
    cp bin/libxsmm_gemm_generator /usr/bin && \
    cd .. && rm -rf libxsmm

# Get and install PETSc (using build-args for compiler and PETSc version)
RUN echo "Using PETSc version $PETSC_VERSION" && \
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
    COPTFLAGS="-g -O3" CXXOPTFLAGS="-g -O3" \
    --download-mumps \
    --download-scalapack \
    --download-metis \ 
    --download-parmetis \
    --prefix=$PETSC_INSTALL_DIR && \
    make PETSC_DIR=$(pwd) PETSC_ARCH=arch-linux-c-opt -j$(nproc) && \
    make PETSC_DIR=$(pwd) PETSC_ARCH=arch-linux-c-opt install && \
    cd .. && rm -rf petsc-${PETSC_VERSION}

# Get and install HDF5
RUN echo "Using HDF5 version $HDF5_VERSION" && \
    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-$(echo $HDF5_VERSION | cut -d. -f1,2)/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.gz && \
    tar -xzf hdf5-${HDF5_VERSION}.tar.gz && \
    rm -f hdf5-${HDF5_VERSION}.tar.gz && \
    cmake -G Ninja -B hdf5-build -S hdf5-${HDF5_VERSION} \
    -DCMAKE_INSTALL_PREFIX=$HDF5_INSTALL_DIR \
    -DHDF5_ENABLE_PARALLEL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DBUILD_TESTING=OFF \
    -DHDF5_BUILD_EXAMPLES=OFF && \
    ninja -C hdf5-build -j$(nproc) && \
    ninja -C hdf5-build install && \
    rm -rf hdf5-build hdf5-${HDF5_VERSION}

# Save versions for reference
RUN echo "$PETSC_VERSION" > /opt/petsc/version.txt
RUN echo "$HDF5_VERSION" > /opt/hdf5/version.txt

# Create a non-root user for MPI operations
RUN useradd -m -s /bin/bash tandem && \
    chown -R tandem:tandem /opt/petsc /opt/hdf5 && \
    echo "tandem ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create and activate a venv for the 'tandem' user
USER tandem
WORKDIR /home/tandem
