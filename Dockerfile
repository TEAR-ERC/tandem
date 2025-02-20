ARG DEPENDENCIES_IMAGE
FROM ${DEPENDENCIES_IMAGE}

# Define build arguments
ARG CC
ARG CXX

# Set environment variables (using build args)
ENV CC=$CC
ENV CXX=$CXX
ENV PETSC_INSTALL_DIR=/opt/petsc/
# Create a non-root user (replace "myuser" with your desired username)
# this is to avoid mpirun error 
RUN useradd -ms /bin/bash myuser


# Create build directory
WORKDIR /app

# Copy the Tandem source code into the container
COPY . /app

# Build Tandem (using build args for compiler)
WORKDIR /app
RUN mkdir build_2d_p3 build_3d_p3

WORKDIR /app/build_2d_p3
RUN cmake .. \
        -DCMAKE_PREFIX_PATH=${PETSC_INSTALL_DIR} \
        -DDOMAIN_DIMENSION=2 \
        -DPOLYNOMIAL_DEGREE=3 \
        -DCMAKE_C_COMPILER="$CC" \ 
        -DCMAKE_CXX_COMPILER="$CXX" && \
    make -j$(nproc) && \
    make test

WORKDIR /app/build_3d_p3
RUN cmake .. \
        -DCMAKE_PREFIX_PATH=${PETSC_INSTALL_DIR} \
        -DDOMAIN_DIMENSION=3 \
        -DPOLYNOMIAL_DEGREE=3 \
        -DCMAKE_C_COMPILER="$CC" \ 
        -DCMAKE_CXX_COMPILER="$CXX" && \
    make -j$(nproc) && \
    make test

WORKDIR /app
RUN ls -lah /app

# Change ownership of relevant directories (if needed)
RUN chown -R myuser:myuser /app

USER myuser

ENTRYPOINT ["/bin/bash"]
