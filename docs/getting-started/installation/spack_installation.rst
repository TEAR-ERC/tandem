Spack installation
==================

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
              cuda: cuda
              rocm: rocm
              debug: debug

Note that a custom install directory for spack packages can also be set, by changing ``~/.spack/config.yaml``:

.. code-block:: yaml

    config:
      install_tree: path_2_packages

We can then generate a tandem module file with:

.. code-block:: bash

    spack module tcl refresh --upstream-modules $(spack find -d --format "{name}{/hash:5}" tandem) 

to access the module at start up, add to your ``~/.bashrc``:

.. code-block:: bash

    module use your_custom_path_2_modules/your_spack_arch_string

e.g.:

.. code-block:: bash

    module use $HOME/spack/modules/x86_avx512/linux-sles15-skylake_avx512/


Installing various branches using Spack
---------------------------------------

Spack installs the main branch by default. 
Compiling tandem from other branches using Spack can be done by modifying the file :code:`package.py` under directory :code:`spack/var/spack/repos/builtin/packages/tandem` or :code:`seissol-spack-aid/spack/packages/tandem` in SuperMUC.
The lines one may want to alter include:

.. code-block:: python

    git = "https://github.com/TEAR-ERC/tandem.git"
    version("main", branch="main", submodules=True)

If one aims to install other branches in tandem repository, one may change the version information, e.g.: 

.. code-block:: python

    version("TSckp", branch="dmay/seas-checkpoint", submodules=True)

The installation can be done by, e.g.:

.. code-block:: bash

    spack install tandem@TSckp polynomial_degree=3 domain_dimension=2 target=skylake_avx512

If one aims to install their own fork of tandem, one may alter the git address, e.g.:

.. code-block:: python

    git = "https://github.com/USER_REPO/fork_of_tandem.git"

