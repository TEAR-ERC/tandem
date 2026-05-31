Frontera
========

Environment Setup
-----------------

Load the necessary modules to set up the environment:

.. code-block:: bash

   module load gcc/9.1.0
   module load hdf5/1.14.3
   module load netcdf/4.9.2
   module load python3/3.9.2

Install PETSc, METIS, and ParMETIS
----------------------------------

The version of PETSc should be chosen based on the Tandem version you wish to use. Use **PETSc 3.22.3** for the latest Tandem version (e.g., ``dmay/staging``). For older versions, use **PETSc 3.13.6**.

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

.. note::
   ``--download-fblaslapack=1`` is required for proper PETSc configuration.

Build and install:

.. code-block:: bash

   make PETSC_DIR=$PWD PETSC_ARCH=arch-linux-c-opt all
   make PETSC_DIR=$PWD PETSC_ARCH=arch-linux-c-opt install

Install Eigen
-------------

.. code-block:: bash

   wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
   tar -zxvf eigen-3.4.0.tar.gz
   cd eigen-3.4.0/
   mkdir build && cd build
   cmake ../ -DCMAKE_INSTALL_PREFIX=$HOME/petsc/eigen-3.4.0/build
   make install

Clone the Tandem Repository
---------------------------

.. code-block:: bash

   cd ~
   mkdir tandem_rep && cd tandem_rep
   git clone https://github.com/TEAR-ERC/tandem.git
   cd tandem
   git submodule update --init

If you need a specific branch (e.g., ``dmay/staging``):

.. code-block:: bash

   git checkout dmay/staging

Build Tandem
------------

Create a build directory (e.g., for 2D, polynomial degree 6) and run CMake:

.. code-block:: bash

   cd ..
   mkdir build_2d_6p_staging && cd build_2d_6p_staging
   cmake $HOME/tandem/tandem \
         -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
         -DCMAKE_PREFIX_PATH=$HOME/petsc/build \
         -DDOMAIN_DIMENSION=2 -DPOLYNOMIAL_DEGREE=6

Compile Tandem:

.. code-block:: bash

   make -j 20 tandem
