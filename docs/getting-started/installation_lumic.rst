Installation on LUMI-C
======================

Installation on LUMI-C leverages the preinstalled spack software stack there.
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
        unify: true
      modules:
        default:
          roots:
            tcl: $SPACK_USER_PREFIX/spack-modules
          arch_folder: False
          tcl:
            hash_length: 3
            naming_scheme: '{name}/{version}-{compiler.name}'
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
                debug: debug
            core_compilers:
            - gcc@=12.2.0

We then source the environment with:

.. code-block:: bash

    eval $(spack env activate --sh $SPACK_USER_PREFIX)

(you can also now uncomment the line in ``~/.bashrc``).

We then add the seissol-spack-aid repository which contains the latest version of the tandem spack package:

.. code-block:: bash

    git clone --branch NG https://github.com/SeisSol/seissol-spack-aid
    spack repo add seissol-spack-aid/spack/


Installation of tandem
----------------------

We can now install tandem, e.g. with:

.. code-block:: yaml

    spack install -j 20 --add tandem@main polynomial_degree=4 domain_dimension=3 %gcc@12.2.0


Using modules
-------------

Modules can be created, with:

.. code-block:: bash

    spack module tcl refresh --upstream-modules $(spack find -d --format "{name}{/hash:5}" tandem) 

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

Note that this may not work with spack/23.09 because the module folder of the preinstalled modules is currently not accessible for reading.

Running tandem
--------------

Here is an example of slurm job file for running static on LUMI:

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


