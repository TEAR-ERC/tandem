FROM debian:stable-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget ca-certificates gcc g++ gfortran libgomp1 make cmake cmake-curses-gui libopenblas-dev libopenblas-base libopenmpi-dev libopenmpi3 git libeigen3-dev python3 python3-numpy python3-distutils liblua5.3-0 liblua5.3-dev vim zlib1g zlib1g-dev \
    && rm -rf /var/lib/apt/lists/

RUN cd /tmp \
    && wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz \
    && tar -xvf metis-5.1.0.tar.gz \
    && cd metis-5.1.0 \
    && make config && make && make install

RUN cd /tmp \
    && wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz \
    && tar -xvf parmetis-4.0.3.tar.gz \
    && cd parmetis-4.0.3 \
    && make config && make && make install

RUN cd /tmp \
    && wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.13.3.tar.gz \
    && tar -xvf petsc-lite-3.13.3.tar.gz \
    && cd petsc-3.13.3 \
    && ./configure --with-debugging=0 CC=mpicc CXX=mpicxx FC=mpif90 --prefix=/usr/local/ \
    && make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt -j4 \
    && make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt install

WORKDIR /home
