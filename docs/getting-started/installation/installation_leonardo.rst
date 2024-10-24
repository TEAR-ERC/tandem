Leonardo
========

available modules
-----------------

Already installed tandem modules can be discovered with ``module use /leonardo/pub/userexternal/tulrich0/spack-0.23.0-dev0/modules/linux-rhel8-icelake``.
You can then check the available modules with ``module avail tandem``.

Installation
------------

Installation on Leonardo leverages the preinstalled software, while relying on the latest spack.
First we "patch" the cuda module so that petsc can be installed (https://github.com/spack/spack/issues/47163):

.. code-block:: bash

    module load cuda/12.1
    cd /leonardo/pub/userexternal/tulrich0/
    cp -r $CUDA_HOME .
    mv none/ cuda_12.1/
    cd cuda_12.1/
    ln -s /leonardo/pub/userexternal/tulrich0/cuda_12.1/compat/libcuda.so.1 lib64/


We then extract the preinstalled packages from the system stack

.. code-block:: bash

    mkdir .spack
    module load spack
    spack config get packages >  ~/.spack/packages.yaml
    module unload spack


and update the cuda module with:

.. code-block:: yaml

    - spec: cuda@12.1.66
      prefix: /leonardo/pub/userexternal/$USER/cuda_12.1

Then we create a file `~/.spack/config.yaml`, setting up the path to install:

.. code-block:: yaml

    config:
      install_tree: /leonardo/pub/userexternal/$USER/spack-0.23.0-dev0/install

Then we create a file  `~/.spack/modules.yaml`, setting up the modules:

.. code-block:: yaml

    modules:
        default:
          roots:
            tcl: /leonardo/pub/userexternal/$USER/spack-0.23.0-dev0/modules
          tcl:
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
          enable:
          - tcl

Then we install the latest spack:

.. code-block:: bash

    git clone https://github.com/spack/spack
    cd spack
    git remote add thomas-fork https://github.com/Thomas-ulrich/spack.git
    git fetch thomas-fork
    git checkout thomas/tandem_manyarch
    echo "export SPACK_ROOT=$PWD" >> $HOME/.bashrc
    echo "export PATH=\$SPACK_ROOT/bin:\$PATH" >> $HOME/.bashrc

And discover system programs (avoiding having to install them) and the gcc compiler:

.. code-block:: bash

    spack external find
    module load gcc
    spack compiler find

We then launch the tandem installation with:

.. code-block:: bash

    spack install -j 30 tandem@develop polynomial_degree=4 domain_dimension=3 %gcc@12 +cuda cuda_arch=80 arch=linux-rhel8-icelake +python +libxsmm ^cuda@12.6.2 ^petsc@main

Finally we can create and discover the modules with:

.. code-block:: bash

    spack module tcl refresh
    echo "module use /leonardo/pub/userexternal/$USER/spack-0.23.0-dev0/modules/linux-rhel8-icelake" >> $HOME/.bashrc

And install some lua modules if required:

.. code-block:: bash

    source ~/.bashrc
    module load tandem
    luarocks install luaposix
    luarocks install csv

