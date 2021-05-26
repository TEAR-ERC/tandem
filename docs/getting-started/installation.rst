Installation
============

In order to compile tandem natively you need to install dependencies.
The following dependencies are likely available via your package manager:

- A recent C++-17 capable compiler (we recommend GCC ≥ 8.0 or clang ≥ 8)
- MPI (e.g. OpenMPI)
- zlib (≥ 1.2)
- `Eigen <https://eigen.tuxfamily.org/>`_ (≥ 3.3)
- Python (≥ 3.5) with NumPy (≥ 1.12.0)
- `Lua <https://www.lua.org/>`_ (≥ 5.3)

The following dependencies likely need to be installed manually:

- `METIS <http://glaros.dtc.umn.edu/gkhome/metis/metis/overview>`_ (≥ 5.1) and `ParMETIS <http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview>`_ (≥ 4.0)
- `PETSc <https://www.mcs.anl.gov/petsc/>`_ (≥ 3.13)
- (Optional) `libxsmm <https://github.com/hfp/libxsmm>`_ (= 1.16.1)

Dependencies via package manager
--------------------------------

The following instructions are valid for Debian buster and might also work
for Ubuntu.
Consult your package manager's documentation for other operating systems.

.. code:: console
   
   # apt-get install -y gcc g++ gfortran libgomp1 \
         make cmake libopenblas-dev libopenblas-base \
         libopenmpi-dev libopenmpi3 git libeigen3-dev \
         python3 python3-distutils python3-numpy \
         liblua5.3-0 liblua5.3-dev zlib1g zlib1g-dev

Install METIS and ParMETIS
--------------------------

.. code:: console

    # wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
    # wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
    # tar -xvf metis-5.1.0.tar.gz
    # tar -xvf parmetis-4.0.3.tar.gz
    # cd metis-5.1.0
    # make config && make && make install
    # cd ../parmetis-4.0.3
    # make config && make && make install

Install PETSc
-------------

.. code:: console

    # wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.13.3.tar.gz
    # tar -xvf petsc-lite-3.13.3.tar.gz
    # cd petsc-3.13.3
    # ./configure --with-fortran-bindings=0 --with-debugging=0 \
         --with-memalign=32 --with-64-bit-indices \
         CC=mpicc CXX=mpicxx FC=mpif90 --prefix=/usr/local/ \
         --download-mumps --download-scalapack \
         COPTFLAGS="-g -O3" CXXOPTFLAGS="-g -O3"
    # make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt -j
    # make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt install

(Optional) Install libxsmm
--------------------------

.. code:: console

    # wget https://github.com/hfp/libxsmm/archive/refs/tags/1.16.1.tar.gz
    # tar -xvf 1.16.1.tar.gz
    # cd libxsmm-1.16.1
    # make -j generator
    # cp bin/libxsmm_gemm_generator /usr/local/bin/

Compile tandem
--------------

.. code:: console

   $ git clone https://github.com/TEAR-ERC/tandem.git
   $ cd tandem/
   $ git submodule update --init
   $ mkdir build
   $ cd build
   $ cmake .. -DPOLYNOMIAL_DEGREE=6
   $ make -j

If you installed libraries to a folder different from :code:`/usr` or :code:`/usr/local` and CMake
cannot find your libraries, try to set the `CMAKE_PREFIX_PATH <https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html>`_, e.g.

.. code:: console

   $ cmake .. -DPOLYNOMIAL_DEGREE=6 -DCMAKE_PREFIX_PATH=/path/to/your/libs
