Tandem with GPUs
================

installation with spack
-----------------------

First add to you `~/.spack/packages.yaml`

.. code-block:: yaml

    packages:
      all:
        variants: cuda_arch=xxx

With xxx you cuda architecture (e.g. 86). For AMD GPUs, use the keyword `amdgpu_target` (e.g. =gfx90a).
This will ensure that the architecture is propagated to all dependencies.

Then, installing the GPU version tandem is (in theory) as simple as adding the GPU variant (+cuda for Nvidia GPUs or +rocm for AMD GPUs), for example:

.. code-block:: bash

    spack install -j 20 tandem@main polynomial_degree=4 domain_dimension=3 +cuda


Using a GPU version
-------------------

To ensure that tandem utilizes GPUs for computation, additionnal PETSc arguments are required when running the application.
For Cuda, use:

.. code-block:: bash

    -mg_levels_mat_type aijcusparse -vec_type cuda -mat_type aijcusparse

For ROCM, use:

.. code-block:: bash

    -vec_type hip -mat_type aijhipsparse
