LUMI-G
======

LUMI-G is the GPU-accelerated partition of the LUMI supercomputer in Finland, powered by AMD Instinct MI250X GPUs. Here, we detail a manual installation procedure for tandem, as a Spack-based installation is hindered by an outdated version of Spack provided on the cluster.

Step by step installation
-------------------------

We first compile PETSc (and parmetis) with:

.. code-block:: bash

    module load LUMI partition/G
    module load rocm
    module load cpeAMD

    module load Eigen
    module load cray-hdf5-parallel
    module load cray-netcdf-hdf5parallel
    module load cray-python

    export CPATH=$ROCM_PATH/include/rocm-core:$CPATH
    export blas_dir=/opt/cray/pe/libsci/25.03.0/CRAYCLANG/17.0/x86_64/
    export PETSC_ARCH=arch-cray-c-rocm-hip-tandem-32-v3.25.1

    wget https://fossies.org/linux/misc/petsc-3.25.1.tar.gz
    cd petsc
    export PETSC_DIR=$(pwd)

    ./configure --download-c2html=0 --download-cmake --with-debugging=no  --download-hwloc=0 --download-metis --download-parmetis --download-sowing=0 --with-64-bit-indices --with-fortran-bindings=0 --with-hip --with-hip-arch=gfx90a --with-hipc=hipcc --with-memalign=32 --with-mpi-dir=${MPICH_DIR} --with-x=0 PETSC_ARCH=${PETSC_ARCH} --with-blaslapack-lib="${blas_dir}/lib/libsci_cray.a ${blas_dir}/lib/libsci_cray.so ${blas_dir}/lib/libsci_cray_mpi.a ${blas_dir}/lib/libsci_cray_mpi.so"

    make -j 30 all

Next we install lua and luarocks:

.. code-block:: bash

    # define TANDEM_DEP, e.g.
    # export SCRATCH=/scratch/project_$your_project_number$/$your_name$
    # export TANDEM_DEP=$SCRATCH/local


    # install lua
    wget https://www.lua.org/ftp/lua-5.4.6.tar.gz
    tar -xf lua-5.4.6.tar.gz
    cd lua-5.4.6
    make all install INSTALL_TOP=$TANDEM_DEP
    cd ..

    # install luarocks
    wget https://luarocks.org/releases/luarocks-3.11.0.tar.gz
    tar zxpf luarocks-3.11.0.tar.gz
    cd luarocks-3.11.0
    ./configure --prefix=$TANDEM_DEP --with-lua=$TANDEM_DEP
    $TANDEM_DEP/bin/luarocks install csv
    $TANDEM_DEP/bin/luarocks install luaposix


Then we can proceed with tandem

.. code-block:: bash

    git clone --branch thomas/petsc_v3.23.2 --recursive https://github.com/TEAR-ERC/tandem
    cd tandem
    mkdir build_gpu && cd build_gpu
    CC=amdclang CXX=amdclang++ cmake .. -DCMAKE_PREFIX_PATH="${PETSC_DIR}/${PETSC_ARCH};$TANDEM_DEP" -DDOMAIN_DIMENSION=3 -DCMAKE_CXX_FLAGS="-I${MPICH_DIR}/include" -DCMAKE_C_FLAGS="-I${MPICH_DIR}/include" -DCMAKE_EXE_LINKER_FLAGS="-L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"
    make -j 30 all


Running tandem
--------------

Here is an example of slurm job file for running static on LUMI-G:

.. code-block:: bash

    #!/bin/bash -l
    #SBATCH --job-name=tandem
    #SBATCH --partition=dev-g  # partition name
    #SBATCH --nodes=1              # Total number of nodes 
    #SBATCH --ntasks-per-node=8     # 8 MPI ranks per node, 16 total (2x8)
    #SBATCH --gpus-per-node=8      # Allocate one gpu per MPI rank
    #SBATCH --account=project_465002391
    #SBATCH --time=00:30:00
    #SBATCH --cpus-per-task=7
    #SBATCH --exclusive
    #SBATCH --mem=224G #debug, standard
    #SBATCH --export=ALL

    export MPICH_GPU_SUPPORT_ENABLED=1

    echo "Allocated nodes: ${SLURM_JOB_NUM_NODES:-1}"
    echo "Tasks per node: ${SLURM_TASKS_PER_NODE}"
    echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
    #echo "Allocated GPUs in node: $SLURM_JOB_GPUS"
    CPU_BIND="core"

    tandem_exe=/pfs/lustrep4/scratch/project_465002391/ulrich/tandem_base/tandem/build_gpu/app/tandem
    time -p srun --cpu-bind=$CPU_BIND $tandem_exe bp5.toml --mg_strategy twolevel --mg_coarse_level 1 --petsc -options_file options_LUMI-G.cfg


with ``options_LUMI-G.cfg`` specifying ``-vec_type hip`` and ``-mat_type aijhipsparse``, as well as ``-mg_levels_pc_type sor``:

.. code-block:: bash

    -ksp_max_it 400
    -pc_type mg
    -mg_levels_ksp_max_it 4
    -mg_levels_ksp_type cg

    -ksp_rtol 1.0e-6
    -mg_coarse_pc_type gamg
    -mg_coarse_ksp_type cg
    -mg_coarse_ksp_rtol 1.0e-1
    -ksp_type gcr
    -log_view

    -ts_type rk
    -ts_rk_type 5dp
    -ts_rtol 1.0e-7
    -ts_atol 1e-20
    -ts_adapt_wnormtype infinity

    -ts_monitor
    -ts_dt 1.0e-8

    # GPU specfic
    -vec_type hip
    -mat_type aijhipsparse
    -log_view_gpu_time
    # bjacobi is buggy with ROCM
    -mg_levels_pc_type sor

