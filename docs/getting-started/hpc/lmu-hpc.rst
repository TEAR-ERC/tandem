LMU-HPC (Heisenbug and Kernelpanic)
===================================

The LMU-HPC cluster consists of several nodes and used by the SIO-LMU Computational Earthquake group. Although the LMU-HPC cluster are not available to the public, the instructions provided here can be adapted for similar HPC environments.

Environment Setup
-----------------

Add the following lines to your ``~/.bashrc`` file or equivalent to set up the compiler and MPI environment.



.. code-block:: bash

   module load gcc/12.2.0
   module load mpi.ompi-gcc/12.2.0
   export OMP_NUM_THREADS=1


Installation
------------

1. **Clone the tandem Git repository:**

.. code-block:: bash

   mkdir project-tandem
   cd project-tandem
   git clone https://github.com/TEAR-ERC/tandem.git
   cd tandem
   git submodule update --init

2. **(Optional) Checkout a specific branch:**

.. code-block:: bash

   git checkout BRANCH/NAME
   export PATH_TO_TANDEM=$PWD
   cd ..

3. **Compile Tandem:**

Create a build directory and run CMake. You can specify the polynomial degree and domain dimension.

.. code-block:: bash

   mkdir build-2d-p6
   cd build-2d-p6
   cmake $PATH_TO_TANDEM \
         -DCMAKE_CXX_COMPILER=mpicxx \
         -DCMAKE_C_COMPILER=mpicc \
         -DCMAKE_PREFIX_PATH="/PATH/TO/YOUR/DEPENDENCIES" \
         -DDOMAIN_DIMENSION=2 \
         -DPOLYNOMIAL_DEGREE=6

.. tip::
   You can combine multiple directories for ``CMAKE_PREFIX_PATH`` by separating them with a semicolon, e.g., ``-DCMAKE_PREFIX_PATH="/path/to/petsc/arch-linux-c-opt/;/path/to/other/deps"``. The order matters; your desired directory should come first.

4. **Build Tandem:**

.. code-block:: bash

   make tandem -j20

5. **Verify the installation:**

Use ``ldd`` to ensure the executable is linked to the correct libraries.

.. code-block:: bash

   ldd app/tandem

Running Tandem
--------------

Before running multiple processor jobs, check the number of free (idle) processors using ``top`` or ``htop``.

**Basic syntax:**

.. code-block:: bash

   tandem parameters.toml --petsc -options_file petsc_options.cfg

**With multiple processors using MPI (e.g., with NCORES number of cores):**

.. code-block:: bash

   mpiexec -bind-to core -n NCORES tparameters.toml --petsc -options_file petsc_options.cfg

.. tip::
   Use ``htop -u YOUR_USER_NAME`` to see only your running jobs.
