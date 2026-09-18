LMU-HPC
=======

Here, information about installing and running tandem on the LMU High Performance Computing (HPC) Facilities are provided.
`This internal wiki entry <https://wiki.geophysik.uni-muenchen.de/dokuwiki/doku.php?id=hpc>`__ provides more details on the available HPC systems. 
Although these HPC infrastructures are not available to the public, the instructions provided here can be adapted for similar systems.

LMU SMP clusters (e.g. Heisenbug and Kernelpanic)
-------------------------------------------------

The LMU-HPC platform consists of several Symmetric Multi Processor (SMP) clusters used by the SIO-LMU Computational Earthquake group. 

Environment Setup
^^^^^^^^^^^^^^^^^

Add the following lines to your ``~/.bashrc`` file or equivalent to set up the compiler and MPI environment.



.. code-block:: bash

   module load gcc/12.2.0
   module load mpi.ompi-gcc/12.2.0
   export OMP_NUM_THREADS=1


Installation
^^^^^^^^^^^^

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
^^^^^^^^^^^^^^

Before running multiple processor jobs, check the number of free (idle) processors using ``top`` or ``htop``.

**Basic syntax:**

.. code-block:: bash

   tandem parameters.toml --petsc -options_file petsc_options.cfg

**With multiple processors using MPI (e.g., with NCORES number of cores):**

.. code-block:: bash

   mpiexec -bind-to core -n NCORES parameters.toml --petsc -options_file petsc_options.cfg

.. tip::
   Use ``htop -u YOUR_USER_NAME`` to see only your running jobs.


TETHYS-3G
---------

The compute nodes of tethys-3g reside in a network that is separated from the main network of Geophysics and can only be accessed from the cluster head.
`This internal wiki entry <https://seissol.readthedocs.io/en/latest/behind-firewall.html>`__ gives more details on the cluster.


Installation
^^^^^^^^^^^^

.. code-block:: bash

   module load spack/1.1
   module load intel/2025 mpi.intel/2025
   export OMP_NUM_THREADS=1


Update ``~/.spack/modules.yaml`` to configure the module generation paths and naming conventions:

.. code-block:: yaml

   modules:
     default:
       roots:
         tcl: /DATA/<<your_user_name>>/spack_install/modules
       tcl:
         all:
           autoload: none
           suffixes:
             domain_dimension=2: 'd2'
             domain_dimension=3: 'd3'
             polynomial_degree=1: 'p1'
             polynomial_degree=2: 'p2'
             polynomial_degree=3: 'p3'
             polynomial_degree=4: 'p4'
             polynomial_degree=5: 'p5'
             polynomial_degree=6: 'p6'

And update ``~/.spack/config.yaml`` to move the installation tree outside your home directory:

.. code-block:: yaml

   config:
     install_tree:
       root: /DATA/<<your_user_name>>/spack_install
     source_cache: /DATA/<<your_user_name>>/spack_install/source_cache


Run the automatic detection tools:

.. code-block:: bash

   spack external find
   spack compiler find


Manually edit ``~/.spack/packages.yaml`` to register the cluster's Intel MPI, remove the `zlib` entry (the system library causing compilation failures), and enforce dependencies:

.. code-block:: yaml

   packages:
     all:
       providers:
         mpi: [intel-oneapi-mpi]

     intel-oneapi-mpi:
       externals:
       - spec: intel-oneapi-mpi@2021.16%intel-oneapi-compilers
         prefix: /opt/software/intel/oneapi/2025.2
       buildable: false


Finally, install tandem.

.. code-block:: bash

   spack install tandem@1.2.0-rc polynomial_degree=2 domain_dimension=3 +libxsmm %intel-oneapi-compilers

