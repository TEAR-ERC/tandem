Quick start with Docker
=======================

.. attention::
   Please install `Docker <https://www.docker.com/>`_ to follow the
   quick start procedure.
   You may use a drop-in replacement such as `podman <https://podman.io/>`_, too.

Step 1: Enter the Docker container
----------------------------------

Open a terminal and pull the tandem-env image:

.. code:: console

   $ docker pull uphoffc/tandem-env

Activate the Docker container with

.. code:: console

   $ docker run -it -v $(pwd):/home -u $(id -u):$(id -g) uphoffc/tandem-env

You should now see something like

.. code::

   I have no name!@<random string>:/home$

.. note::

   In the docker run command, the option :code:`-it` opens an interactive terminal,
   :code:`-v $(pwd):/home` maps your current working directory to the home folder
   inside the container, and :code:`-u $(id -u):$(id -g)` fixes the file permissions
   for files you create inside the container.

Step 2: Compile tandem
----------------------

Inside the Docker container, clone the tandem repository and load submodules:

.. code:: console

   $ git clone https://github.com/TEAR-ERC/tandem.git
   $ cd tandem/
   $ git submodule update --init

Tandem uses CMake, which is contained in the Docker image.
We do not run CMake from the tandem directory but create a build directory
to not pollute our workspace.
From the build directory we run CMake followed by make:

.. code:: console

   $ mkdir build
   $ cd build
   $ cmake .. -DPOLYNOMIAL_DEGREE=6
   $ make -j

Note that we specified the polynomial degree of the finite element spaces
with the :code:`POLYNOMIAL_DEGREE` variable.
Another important variable is :code:`DOMAIN_DIMENSION`, which should be set
to 2 or 3, depending on whether you want to run 2D or 3D models.
You can also use :code:`ccmake ..` to enter a GUI which shows all available
compilation variables.

Step 3: Run tests
-----------------

To check that everything works, run

.. code:: console

   $ make test

from the build folder. At the end of the tests, you should see

.. code:: none

   100% tests passed, 0 tests failed out of 21

You are now set to run the :doc:`examples <examples>`.
