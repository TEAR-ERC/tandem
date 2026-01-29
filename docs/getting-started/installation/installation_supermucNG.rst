SuperMUC-NG
===========

Preinstalled modules
--------------------

A few versions of tandem have been preinstalled on supermuc-NG.
They can be discovered, assuming access to project pn49ha, with:

.. code-block:: bash

    module purge
    module load spack/23.1.0
    module use /hppfs/work/pn49ha/di73yeq4/user_spack23.1/modules/
    module avail tandem 


Installation procedure
----------------------

Alternatively (and for reference), to compile tandem on SuperMUC-NG, follow the procedure below:

First, have a look at `this page <https://seissol.readthedocs.io/en/latest/behind_firewall.html>`_ to best configure git on SuperMUC-NG, and at the general :doc:`Spack installation documentation <spack_installation>`.

The software stack on SuperMUC-NG has been installed with spack.
Yet, spack on SuperMUC-NG needs small modifications to be able to compile tandem properly.
The recipe for compiling spack should then be added from a repository:

.. code-block:: bash

    # load the most recent spack version
    module purge
    module load user_spack/23.1.0
    module load spack/23.1.0
    module load intel intel-mpi intel-mkl python/3.10.10-extended

    # clone seissol-spack-aid and add the repository
    git clone --branch NG https://github.com/SeisSol/seissol-spack-aid.git
    cd seissol-spack-aid
    spack repo add ./spack


Configure and start a proxy, as described in `this page <https://seissol.readthedocs.io/en/latest/behind_firewall.html>`_ (the same procedure used to access PyPI is used to access the package data in spack).

tandem can be then installed, e.g. with:

.. code-block:: bash

    # change ddddd to the port numbers used in the RemoteForward directive
    export https_proxy="http://localhost:dddddd/"
    spack install tandem@main polynomial_degree=3 domain_dimension=2

The procedure to create an environment module is the same as detailed above.
