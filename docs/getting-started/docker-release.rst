Running Tandem with Docker Release Images
==========================================

Overview
--------

Pre-built Docker images are available on
`GitHub Container Registry (GHCR) <https://github.com/TEAR-ERC/tandem/pkgs/container/tandem>`_
for users who want to run Tandem without compiling it themselves.

Images are built for ``linux/amd64`` (Intel/x86) and run on:

- Linux (native)
- macOS, including Apple Silicon, via emulation
- Any Kubernetes cluster

.. note::

   On Apple Silicon Macs (M1/M2/M3), the images run under ``linux/amd64``
   emulation. This works for testing and small runs, but is significantly
   slower than native. It is not recommended for large production simulations.

Prerequisites
-------------

- `Docker Desktop <https://docs.docker.com/get-docker/>`_ installed and running.
- `Gmsh <https://gmsh.info/>`_ installed on your **host machine** for mesh generation.
  The Docker image does not include Gmsh — meshes must be generated before
  launching the container.

Available Images
----------------

The all-in-one image contains all 12 Tandem builds and exposes each as a
named command alias (e.g. ``tandem_2d_6p_tandem``, ``tandem_3d_3p_tandem``,
etc.). This lets you switch between configurations without pulling multiple
images.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tag
     - Description
   * - ``ghcr.io/tear-erc/tandem:all``
     - All 12 builds; use named aliases to select a configuration.

Versioning
----------

Image tags follow this strategy:

- ``:all`` always points to the latest release.
- ``:all-1.2.0`` pins to a specific release for reproducibility.

For research workflows or production runs, pin to a version tag so your
results are not affected by future updates:

.. code-block:: sh

   docker pull ghcr.io/tear-erc/tandem:all-1.2.0

Mounting Input and Output Files
--------------------------------

The container's default working directory ``/work`` already contains the
bundled example files. To avoid shadowing them, mount your host directory
to a **different path** such as ``/host``:

.. code-block:: sh

   docker run --platform linux/amd64 -it --rm \
     -v $(pwd):/host \
     ghcr.io/tear-erc/tandem:all

Files written by Tandem under ``/host`` will appear in your current
directory on the host after the run.

Running with MPI
----------------

Set the number of MPI ranks with ``-n``. Choose a value that matches the
number of CPU cores available:

.. code-block:: sh

   mpirun -n 8 tandem_2d_6p_tandem simulation.toml

.. note::

   Tandem input files use the ``.toml`` format, not ``.lua``.

Examples
--------

The image ships with example input files at ``/work/examples/`` inside the
container. Available subdirectories:

- ``elasticity/``
- ``poisson/``
- ``tandem/``
- ``options/``

A typical workflow for running a bundled example:

1. **Copy example files to your host directory** from inside the container:

   .. code-block:: sh

      # Inside the container:
      cp /work/examples/tandem/2d/bp1_sym.toml /host/
      cp /work/examples/tandem/2d/bp1_sym.geo  /host/

2. **Generate the mesh on the host** (Gmsh is not inside the container):

   .. code-block:: sh

      # On the host, outside the container:
      gmsh -2 bp1_sym.geo

   This produces ``bp1_sym.msh`` in the same directory.

3. **Run Tandem inside the container**:

   .. code-block:: sh

      # Inside the container:
      mpirun -n 4 tandem_2d_3p_tandem /host/bp1_sym.toml

Building from Source
--------------------

If you need a custom build, see the :doc:`installation <installation>` guide.
