Installation
============

Tandem and its dependencies can be installed automatically with `Spack <https://github.com/spack/spack/wiki>`_, or manually.

Spack installation
------------------

`Spack <https://github.com/spack/spack/wiki>`_ is an HPC software package manager.
It automates the process of installing, upgrading, configuring, and removing computer programs.
In particular, the spack package ``tandem`` allows automatically installing tandem and all its dependencies, and creating environment modules.
First, install spack with, e.g.

.. code-block:: bash

    cd $HOME
    git clone --depth 1 https://github.com/spack/spack.git
    cd spack
    echo "export SPACK_ROOT=$PWD" >> $HOME/.bashrc
    echo "export PATH=\$SPACK_ROOT/bin:\$PATH" >> $HOME/.bashrc

Then install tandem with:

.. code-block:: bash

    spack install tandem@main polynomial_degree=3 domain_dimension=2

tandem can then be loaded with ``spack load tandem``.
Alternatively, we might prefer loading tandem from environment modules. We therefore now detail the procedure to generate such module(s).
You may want to update ``~/.spack/modules.yaml``, to specify the path where the module file(s) should be installed (if e.g. if want to share your installation with other users and they cannot access your $HOME), and to generate module files with more readable names:

.. code-block:: yaml

    modules:
      default:
        roots:
         tcl: your_custom_path_2_modules
      default:
        tcl:
          all:
            suffixes:
              domain_dimension=2: 'd2'
              domain_dimension=3: 'd3'
              polynomial_degree=1: 'p1'
              polynomial_degree=2: 'p2'
              polynomial_degree=3: 'p3'
              polynomial_degree=4: 'p4'
              polynomial_degree=5: 'p5'
              polynomial_degree=6: 'p6'

Note that a custom install directory for spack packages can also be set, by changing ``~/.spack/config.yaml``:

.. code-block:: yaml

    config:
      install_tree: path_2_packages

We can then generate a tandem module file with:

.. code-block:: bash

    spack module tcl refresh tandem

to access the module at start up, add to your ``~/.bashrc``:

.. code-block:: bash

    module use your_custom_path_2_modules/your_spack_arch_string

e.g.:

.. code-block:: bash

    module use $HOME/spack/modules/x86_avx512/linux-sles15-skylake_avx512/

SuperMUC-NG installation
------------------------

First, have a look at `this page <https://seissol.readthedocs.io/en/latest/behind_firewall.html>`_ to best configure git on SuperMUC-NG.

The software stack on SuperMUC-NG has been installed with spack.
Yet, spack on SuperMUC-NG is not recent enough to natively know how to compile tandem. 
The recipe for compiling spack should then be added from a repository:

.. code-block:: bash

    # load spack
    module load user_spack
    # clone seissol-spack-aid and add the repository
    git clone --branch supermuc_NG https://github.com/SeisSol/seissol-spack-aid.git
    cd seissol-spack-aid
    spack repo add ./spack

tandem can be then installed, e.g. with:

.. code-block:: bash

    spack install tandem@main polynomial_degree=3 domain_dimension=2 target=skylake_avx512

The procedure to create an environment module is the same as detailed above.

Manual installation
-------------------

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
    # cd ..

Install PETSc
-------------

.. code:: console

    # wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.14.6.tar.gz
    # tar -xvf petsc-lite-3.14.6.tar.gz
    # cd petsc-3.14.6
    # ./configure --with-fortran-bindings=0 --with-debugging=0 \
         --with-memalign=32 --with-64-bit-indices \
         CC=mpicc CXX=mpicxx FC=mpif90 --prefix=/usr/local/ \
         --download-mumps --download-scalapack \
         COPTFLAGS="-g -O3" CXXOPTFLAGS="-g -O3"
    # make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt -j
    # make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt install
    # cd ..

(Optional) Install libxsmm
--------------------------

.. code:: console

    # wget https://github.com/hfp/libxsmm/archive/refs/tags/1.16.1.tar.gz
    # tar -xvf 1.16.1.tar.gz
    # cd libxsmm-1.16.1
    # make -j generator
    # cp bin/libxsmm_gemm_generator /usr/local/bin/
    # cd ..

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
