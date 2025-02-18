ARG DEPENDENCIES_IMAGE
FROM ${DEPENDENCIES_IMAGE}

# Define build arguments
ARG CC
ARG CXX

# Set environment variables (using build args)
ENV CC=$CC
ENV CXX=$CXX
ENV PETSC_INSTALL_DIR=/opt/petsc/

# Create build directory
WORKDIR /app

# Copy the Tandem source code into the container
COPY . /app

# Build Tandem (using build args for compiler)
RUN mkdir build_2d_p3 && cd build_2d_p3 && \
    cmake .. \
        -DCMAKE_PREFIX_PATH=${PETSC_INSTALL_DIR} \
        -DDOMAIN_DIMENSION=2 \
        -DPOLYNOMIAL_DEGREE=3 \
        -DCMAKE_C_COMPILER="$CC" \ 
        -DCMAKE_CXX_COMPILER="$CXX" && \
    make -j$(nproc) && \
    make test

# Build Tandem (using build args for compiler)
RUN mkdir build_3d_p3 && cd build_3d_p3 && \
    cmake .. \
        -DCMAKE_PREFIX_PATH=${PETSC_INSTALL_DIR} \
        -DDOMAIN_DIMENSION=3 \
        -DPOLYNOMIAL_DEGREE=3 \
        -DCMAKE_C_COMPILER="$CC" \ 
        -DCMAKE_CXX_COMPILER="$CXX" && \
    make -j$(nproc) && \
    make test

ENTRYPOINT ["/bin/bash"]
