LUMI
====

Installation of tandem on LUMI-C/G can be done either using spack or manually.
Unfortunately, the spack module based on the 23.09 programming environment only have rocm 5.6.1, which is unstable with petsc when running on GPUs
(because of this issue https://gitlab.com/petsc/petsc/-/issues/1529, fixed in rocm-6.0.2).
Therefore, petsc and tandem should be installed manually for LUMI-G.

Step by step installation
-------------------------

We first compile PETSc with:

```
module load cray-mpich/8.1.29
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load rocm/6.0.3
export CPATH=$ROCM_PATH/include/rocm-core:$CPATH
export blas_dir=/opt/cray/pe/libsci/24.03.0/CRAYCLANG/17.0/x86_64
git clone --branch v3.22.1 --single-branch https://gitlab.com/petsc/petsc
cd petsc
export PETSC_DIR=$(pwd) 
export PETSC_ARCH=arch-cray-c-rocm-hip-tandem-32-v3.22.1

./configure --download-c2html=0 --download-cmake --with-debugging=no  --download-hwloc=0 --download-metis --download-parmetis --download-sowing=0 --with-64-bit-indices --with-fortran-bindings=0 --with-hip --with-hip-arch=gfx90a --with-hipc=hipcc --with-memalign=32 --with-mpi-dir=${MPICH_DIR} --with-x=0 PETSC_ARCH=${PETSC_ARCH} --with-blaslapack-lib="${blas_dir}/lib/libsci_cray.a ${blas_dir}/lib/libsci_cray.so ${blas_dir}/lib/libsci_cray_mpi.a ${blas_dir}/lib/libsci_cray_mpi.so"

make -j 30 all

```

Eigen and lua then need to be installed.

Then we can proceed with tandem

```
git clone --branch dmay/staging --recursive https://github.com/TEAR-ERC/tandem
cd tandem
mkdir build_gpu
CC=/opt/rocm-6.0.3/bin/amdclang CXX=/opt/rocm-6.0.3/bin/amdclang++ cmake .. -DCMAKE_PREFIX_PATH=${PETSC_DIR}/${PETSC_ARCH} -DDOMAIN_DIMENSION=3 -DCMAKE_CXX_FLAGS="-I${MPICH_DIR}/include" -DCMAKE_C_FLAGS="-I${MPICH_DIR}/include" -DCMAKE_EXE_LINKER_FLAGS="-L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"
make -j 30 all

```

spack installation
------------------

We can use spack on LUMI to leverage the preinstalled spack software stack there when installing tandem.
It requires setting up a spack environment and installing tandem there.

Preparing spack for installation
--------------------------------

Spack requires disk space to install packages, bookkeeping, etc. On LUMI this is controlled by the environment variable ``$SPACK_USER_PREFIX``, which needs to be set before the Spack module is loaded.
It is recommended to point this variable to a directory that is readable by your whole LUMI compute project. 
Therefore we add to ~/.bashrc:

.. code-block:: bash

    export PROJECT_ID=465000831
    export WORK=/project/project_${PROJECT_ID}
    export SCRATCH=/scratch/project_${PROJECT_ID}
    export SPACK_USER_PREFIX=/project/project_${PROJECT_ID}/spack_tandem
    module load spack/spack/23.09
    # eval $(spack env activate --sh $SPACK_USER_PREFIX)

Note that spack_tandem could be changed to another name.
Next step is to source ``~/.bashrc``:

.. code-block:: bash

    source ~/.bashrc

Installation of the spack environment
-------------------------------------

At the first usage of spack, we bootstrap spack with

.. code-block:: bash

    spack bootstrap now

and create the spack environment folder:

.. code-block:: bash

    mkdir $SPACK_USER_PREFIX && cd $SPACK_USER_PREFIX

Then we add `spack.yaml` in the folder:

.. code-block:: yaml

    spack:
      concretizer:
        unify: when_possible
      view: false
      modules:
        default:
          roots:
            tcl: $SPACK_USER_PREFIX/spack-modules
          arch_folder: false
          tcl:
            hash_length: 3
            #naming_scheme: '{name}/{version}-{compiler.name}'
            hierarchy: []    # or just remove the hierarchy line completely
            all:
              suffixes:
                domain_dimension=2: d2
                domain_dimension=3: d3
                polynomial_degree=1: p1
                polynomial_degree=2: p2
                polynomial_degree=3: p3
                polynomial_degree=4: p4
                polynomial_degree=5: p5
                polynomial_degree=6: p6
                cuda: cuda
                rocm: rocm
                debug: debug

We then source the environment with:

.. code-block:: bash

    eval $(spack env activate --sh $SPACK_USER_PREFIX)

(you can also now uncomment the line in ``~/.bashrc``).

We then add the seissol-spack-aid repository which contains the latest version of the tandem spack package:

.. code-block:: bash

    git clone --branch spack_0.22 https://github.com/SeisSol/seissol-spack-aid
    spack repo add seissol-spack-aid/spack/

Next step is to discover the more recent compilers (e.g. gcc-13)

.. code-block:: bash

    spack compiler find

After that we load find python and cmake, to avoid rebuilding them:

.. code-block:: bash

    module load cray-python/3.11.7
    spack external find python cmake

We tag this added packages as non buildable in spack_tandem/spack.yaml, with `add buildable: false`. That is the files now looks like:

.. code-block:: yaml

  packages:
    cmake:
      externals:
      - spec: cmake@3.20.4
        prefix: /usr
        buildable: false
    python:
      externals:
      - spec: python@3.11.7+bz2+crypt+ctypes+dbm+lzma+nis+pyexpat+pythoncmd+readline+sqlite3+ssl~tkinter+uuid+zlib
        prefix: /opt/cray/pe/python/3.11.7
        buildable: false


Installation of tandem on LUMI-C
--------------------------------

We can now install tandem, e.g. with:

.. code-block:: yaml

    spack install -j 20 --add tandem@main polynomial_degree=4 domain_dimension=3 %gcc@13


Installation of tandem on LUMI-G
--------------------------------

We can install the GPU version of tandem, with:

.. code-block:: yaml

    spack install -j 20 --add tandem@main%gcc+rocm amdgpu_target=gfx90a domain_dimension=3 polynomial_degree=4

Using modules
-------------

Modules can be created, with:

.. code-block:: bash

    spack module tcl refresh

Once you have installed tandem with spack, you can use modules instead of spack to load tandem, e.g. by changing the ``~/.bashrc`` to, e.g.:

.. code-block:: bash

    export PROJECT_ID=465000831
    export WORK=/project/project_${PROJECT_ID}
    export SCRATCH=/scratch/project_${PROJECT_ID}
    export SPACK_USER_PREFIX=/project/project_${PROJECT_ID}/spack_tandem
    #module load spack/23.09
    #eval $(spack env activate --sh $SPACK_USER_PREFIX)
    export MODULEPATH=$SPACK_USER_PREFIX/spack-modules:$MODULEPATH
    module load gcc tandem

Running tandem
--------------

Here is an example of slurm job file for running static on LUMI-C:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=static
    #SBATCH --account=project_465000831
    #SBATCH --time=00:30:00
    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=128
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=220G
    #SBATCH --partition=standard
    ##SBATCH --mem=4000G
    ##SBATCH --partition=largemem

    #module load gcc/12.2.0  tandem
    export MP_SINGLE_THREAD=yes
    ulimit -Ss 2097152
    srun static ridge.toml  --output ridgecrest  --mg_strategy twolevel --mg_coarse_level 1  --petsc -ksp_max_it 400 -pc_type mg -mg_levels_ksp_max_it 4 -mg_levels_ksp_type cg -mg_levels_pc_type bjacobi -ksp_rtol 1.0e-6 -mg_coarse_pc_type gamg -mg_coarse_ksp_type cg -mg_coarse_ksp_rtol 1.0e-1 -ksp_type gcr -log_view


Here is an example of slurm job file for running static on LUMI-G:

.. code-block:: bash

    #!/bin/bash -l
    #SBATCH --job-name=static
    #SBATCH --partition=dev-g  # partition name
    #SBATCH --nodes=4              # Total number of nodes 
    #SBATCH --ntasks-per-node=8     # 8 MPI ranks per node, 16 total (2x8)
    #SBATCH --gpus-per-node=8      # Allocate one gpu per MPI rank
    #SBATCH --account=project_465000831
    #SBATCH --time=00:30:00
    #SBATCH --cpus-per-task=1
    #SBATCH --exclusive
    #SBATCH --mem=224G #debug, standard
    #SBATCH --export=ALL

    export MPICH_GPU_SUPPORT_ENABLED=1

    module load cray-mpich/8.1.29
    module load craype-x86-trento
    module load craype-accel-amd-gfx90a
    module load rocm/6.0.3
    module load lua

    echo "Allocated nodes: ${SLURM_JOB_NUM_NODES:-1}"
    echo "Tasks per node: ${SLURM_TASKS_PER_NODE}"
    echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
    #echo "Allocated GPUs in node: $SLURM_JOB_GPUS"
    CPU_BIND="core"

    time -p srun --cpu-bind=$CPU_BIND static ridge.toml --mg_strategy twolevel --mg_coarse_level 1  --petsc -ksp_max_it 400 -pc_type mg -mg_levels_ksp_max_it 4 -mg_levels_ksp_type cg -mg_levels_pc_type bjacobi -ksp_rtol 1.0e-6 -mg_coarse_pc_type gamg -mg_coarse_ksp_type cg -mg_coarse_ksp_rtol 1.0e-1 -ksp_type gcr -log_view  -vec_type hip -mat_type aijhipsparse -log_view_gpu_time
