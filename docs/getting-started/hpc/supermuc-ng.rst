SuperMUC-NG
===========

First, have a look at `this page <https://seissol.readthedocs.io/en/latest/behind-firewall.html>`__ to best configure git on SuperMUC-NG.

The software stack on SuperMUC-NG has been installed with spack.
Yet, spack on SuperMUC-NG is not recent enough to natively know how to compile tandem. 
The recipe for compiling spack should then be added from a repository.

Environment Setup
-----------------

Add the following lines to your ``~/.bashrc`` file or equivalent:

.. code-block:: bash

    module load cmake gcc/11.2.0 python/3.8.11-extended
    module load intel-mpi intel
    module load user_spack

Install Spack
-------------

.. code-block:: bash

    # clone seissol-spack-aid and add the repository
    git clone --branch supermuc_NG https://github.com/SeisSol/seissol-spack-aid.git
    cd seissol-spack-aid
    spack repo add ./spack

Note: to accomplish this step, you need to set up access to GitHub behind the firewall. Detailed instructions can be found `here <https://seissol.readthedocs.io/en/latest/behind_firewall.html#pypi-behind-firewall>`_.

Add the Python module to spack by adding the following lines to your ``~/.spack/packages.yaml`` file:

.. code-block:: yaml

    packages:
        python:
            externals:
            - spec: python@3.8.11
              buildable: False
              modules:
               - python/3.8.11-extended

Compiling Tandem
----------------

tandem can be then installed (e.g., 2d, polynomial degree 6) with:

.. code-block:: bash

    spack install tandem@main domain_dimension=2 polynomial_degree=6 target=skylake_avx512

Note that it is important to specify the target as ``skylake_avx512``.

The procedure to create an environment module is the same as detailed in :doc:`Spack installation <../spack>`.

Using other branches
~~~~~~~~~~~~~~~~~~~~

If you want to install a branch other than ``main``, you need to modify the ``seissol-spack-aid/spack/package/tandem/package.py`` file:

* **git**: If you want to use your own fork, update the ``git`` URL (Line 17).
* **version**: Specify the name of the branch and the name for this build (Line 21). E.g., ``version("TScheckpoint", branch="dmay/seas-checkpoint", submodules=True)``.
* **patch**: If your branch is stale, you may need to include a patch (Line 22). E.g., ``patch("fix_v1.0_compilation.diff", when="@TScheckpoint")``. (Note: only include this when your branch is STALE. Check if the modifications are already applied in ``app/CMakeLists.txt`` in your branch).

Then install using the build name after ``tandem@``:

.. code-block:: bash

    spack install tandem@BUILD_NAME domain_dimension=2 polynomial_degree=6 target=skylake_avx512

Running Tandem
--------------

In SuperMUC-NG, you need to submit a job using SLURM. An example job submission script (e.g., ``run_Tandem_supermuc.sh``) resourcing 40 nodes in the general partition with a time limit of 3 hours would look like the following:

.. code-block:: bash

    #!/bin/bash

    # Job Name and Files
    #SBATCH -J <job name>
    #Output and error:
    #SBATCH -o ./%j.%x.out
    #SBATCH -e ./%j.%x.err

    #Initial working directory:
    #SBATCH --chdir=<work directory>

    #Notification and type
    #SBATCH --mail-type=END
    #SBATCH --mail-user=<your email address>

    # Wall clock limit:
    #SBATCH --time=3:00:00
    #SBATCH --no-requeue

    #Setup of execution environment
    #SBATCH --export=ALL
    #SBATCH --account=<your account name>

    #SBATCH --partition=<desired queue name>

    #EAR may impact code performance
    #SBATCH --ear=off

    #Number of nodes and MPI tasks per node:
    #SBATCH --nodes=<nodes number>
    #SBATCH --ntasks-per-node=1

    module load slurm_setup

    #Run the program:
    export OMP_NUM_THREADS=1
    source /etc/profile.d/modules.sh

    echo 'num_nodes:' $SLURM_JOB_NUM_NODES 'ntasks:' $SLURM_NTASKS 'cpus_per_task:' $SLURM_CPUS_PER_TASK
    ulimit -Ss 2097152

    mpiexec -n $SLURM_NTASKS tandem parameters.toml --petsc -options_file petsc_options.cfg

note::
   Replace the placeholders (e.g., ``<job name>``, ``<work directory>``, ``<your account name>``, ``<desired queue name>``, ``<nodes number>``) with your desired values.

Submit the script using ``sbatch``:

.. code-block:: bash

    sbatch run_Tandem_supermuc.sh

More information about the job submission process on SuperMUC-NG can be found on `this page <https://doku.lrz.de/job-processing-with-slurm-on-supermuc-ng-10746568.html>`__.
