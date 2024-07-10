Tandem with GPUs
================

installation with spack
-----------------------

Installing the GPU version tandem is (in theory) as simple as adding the GPU variant (+cuda for Nvidia GPUs or +rocm for AMD GPUs), for example:

.. code-block:: bash

    spack install -j 20 tandem@main polynomial_degree=4 domain_dimension=3 +cuda


Using a GPU version
-------------------


To ensure that tandem utilizes GPUs for computation, the following PETSc arguments are required when running the application:

.. code-block:: bash

    -mg_levels_mat_type aijcusparse -vec_type cuda -mat_type aijcusparse


