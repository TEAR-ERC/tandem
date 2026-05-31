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

Available Images
----------------

A single image contains all 12 Tandem builds (2D/3D × degrees 1–6). Each
configuration is available as two binaries on ``$PATH``:

- ``tandem_Xd_Yp`` — the main Tandem solver
- ``static_Xd_Yp`` — the static solver

For example: ``tandem_2d_6p``, ``static_3d_3p``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tag
     - Description
   * - ``ghcr.io/tear-erc/tandem:latest``
     - All 12 builds; always points to the latest release.

Versioning
----------

Image tags follow this strategy:

- ``:latest`` always points to the latest release.
- ``:1.2.0`` pins to a specific release for reproducibility.

For research workflows or production runs, pin to a version tag so your
results are not affected by future updates:

.. code-block:: sh

   docker pull ghcr.io/tear-erc/tandem:1.2.0

Mounting Input and Output Files
--------------------------------

The container's default working directory ``/work`` already contains the
bundled example files. To avoid shadowing them, mount your host directory
to a **different path** such as ``/host``:

.. code-block:: sh

   docker run --platform linux/amd64 -it --rm \
     -v $(pwd):/host \
     ghcr.io/tear-erc/tandem:latest

Files written by Tandem under ``/host`` will appear in your current
directory on the host after the run.

Running with MPI
----------------

Set the number of MPI ranks with ``-n``. Choose a value that matches the
number of CPU cores available:

.. code-block:: sh

   mpirun -n 8 tandem_2d_6p simulation.toml

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
      cd /host
      cp /work/examples/tandem/2d/bp1_sym.toml .
      cp /work/examples/tandem/2d/bp1_sym.geo  .
      cp /work/examples/tandem/2d/bp1.lua .

2. **Generate the mesh inside the container**:

   .. code-block:: sh

      # Inside the container:
      gmsh -2 ./bp1_sym.geo

   This produces ``bp1_sym.msh`` in the same directory.

3. **Run Tandem inside the container**:

   .. code-block:: sh

      # Inside the container:
      mpirun -n 4 tandem_2d_3p ./bp1_sym.toml

   .. warning::

      Tandem writes output files to the path specified in the ``.toml``
      configuration. If ``/host`` is not bound to your host machine as shown in
      the mount step above, all output will be written inside the container and
      lost when it exits. Simulation output files can be very large; writing them
      inside the container may also exhaust the container's writable layer and
      cause unexpected failures.

Building from Source
--------------------

If you need a custom build, see the :doc:`installation <installation>` guide.
