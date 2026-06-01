Manual installation
===================

Tandem and its dependencies can be installed manually by following the instructions below.
Alternatively, tandem can be installed using :doc:`Spack <spack>` or on :doc:`specific HPC systems <hpc/index>`.

The following dependencies are likely available via your package manager:

- A recent C++-17 capable compiler (we recommend GCC ≥ 8.0 or clang ≥ 8)
- MPI (e.g. OpenMPI)
- zlib (≥ 1.2)
- `Eigen <https://eigen.tuxfamily.org/>`_ (≥ 3.3)
- Python (≥ 3.5) with NumPy (≥ 1.12.0)
- `Lua <https://www.lua.org/>`_ (≥ 5.3)
- `CMake <https://cmake.org/>`_ (≥ 3.18)

The following dependencies likely need to be installed manually:

- `METIS <http://glaros.dtc.umn.edu/gkhome/metis/metis/overview>`_ (≥ 5.1) and `ParMETIS <http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview>`_ (≥ 4.0)
- `PETSc <https://www.mcs.anl.gov/petsc/>`_ (≥ 3.13)
- (Optional) `libxsmm <https://github.com/hfp/libxsmm>`_ (= 1.16.1)
- (Optional) `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ (= 1.12.3)

Dependencies via package manager
--------------------------------

The following instructions are valid for Debian buster and might also work
for Ubuntu.
Consult your package manager's documentation for other operating systems.

.. code:: console
   
   $ apt-get install -y gcc g++ gfortran libgomp1 \
         make cmake libopenblas-dev libopenblas-base \
         libopenmpi-dev libopenmpi3 git libeigen3-dev \
         python3 python3-distutils python3-numpy \
         liblua5.3-0 liblua5.3-dev zlib1g zlib1g-dev


Install PETSc METIS and ParMETIS
--------------------------------

.. code-block:: bash

   mkdir petsc && cd petsc
   wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.22.3.tar.gz
   tar -xvf petsc-3.22.3.tar.gz
   cd petsc-3.22.3
   ./configure --with-fortran-bindings=0 --with-debugging=0 --with-memalign=32 \
               --with-64-bit-indices CC=mpicc CXX=mpicxx FC=mpif90 \
               COPTFLAGS="-g -O3" CXXOPTFLAGS="-g -O3" --download-mumps \
               --download-scalapack --download-metis --download-parmetis \
               --download-fblaslapack=1 --prefix=$HOME/petsc/build

Build and install:

.. code-block:: bash

   make PETSC_DIR=$PWD PETSC_ARCH=arch-linux-c-opt all
   make PETSC_DIR=$PWD PETSC_ARCH=arch-linux-c-opt install
    $ cd ..

(Optional) Install libxsmm
--------------------------

.. code:: console

    $ wget https://github.com/hfp/libxsmm/archive/refs/tags/1.16.1.tar.gz
    $ tar -xvf 1.16.1.tar.gz
    $ cd libxsmm-1.16.1
    $ make -j generator
    $ cp bin/libxsmm_gemm_generator /usr/local/bin/
    $ cd ..

(Optional) Install HDF5
--------------------------

.. code:: console

    $ wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.3/src/hdf5-1.12.3.tar.gz && \
    $ tar -xzf hdf5-1.12.3.tar.gz && \
    $ rm -f hdf5-1.12.3.tar.gz && \
    $ export CC=mpicc
    $ export CXX=mpicxx
    $ export FC=mpifort
    $ ./configure --prefix=$PREFIX_PATH \
                  --enable-parallel \
                  --enable-fortran \
                  CC=mpicc CXX=mpicxx FC=mpifort
    $ make -j$(nproc)    
    $ make check            
    $ make install

(Optional) Installing Lua Modules
---------------------------------

Additional Lua modules can be installed through ``luarocks``. This is often required for specific benchmark setups that parse external files or interact with the system environment—for example, the Ridgecrest setup of Uphoff et al. (2023):

.. code-block:: bash

   luarocks install csv
   luarocks install luaposix

Compile tandem
--------------

You can provide polynomical degree and domain dimension during the compilation stage using :code:`-DPOLYNOMIAL_DEGREE` and :code:`-DDOMAIN_DIMENSION`, respectively. 
If not provided, both polynomical degree and domain dimension are set to 2 by default.

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

If you require multiple paths to CMake, the syntax is as follows :code:`-DCMAKE_PREFIX_PATH="/usr/local/path_1;/usr/local/path_2"`

If you want to use the optional HDF5 support, you need to set the CMake variable :code:`-DENABLE_HDF5=ON` and provide the path to your HDF5 installation using :code:`-DCMAKE_PREFIX_PATH`, e.g. :code:`cmake .. -DPOLYNOMIAL_DEGREE=6 -DENABLE_HDF5=ON -DCMAKE_PREFIX_PATH=/path/to/your/hdf5/installation`.
