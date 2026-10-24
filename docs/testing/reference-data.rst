Reference data
==============

Only two tests compare against stored outputs of tandem:
the static regression test and the SEAS regression test (see :doc:`integration-tests`).
All other integration tests compare against an exact solution, a convergence rate, or another
run of the same build, and need no stored results.

The stored data is in a separate repository,
`TEAR-ERC/tandem_test_data <https://github.com/TEAR-ERC/tandem_test_data>`_.
``-DRUN_CI_TESTS=ON`` clones its ``master`` branch (``--depth 1``) into ``test/test_data``.
A shallow clone has only the newest commit, so to see the history of a file you must clone
the repository again without ``--depth``.


Contents
--------

.. list-table::
   :header-rows: 1
   :widths: 42 28 30

   * - file
     - used by
     - what it is
   * - ``reference_results/p<deg>/ref_regression_2D_output.vtu``
     - static regression (2D)
     - ``app/static`` output of ``circular_hole``
   * - ``reference_results/p<deg>/ref_regression_3D_output.vtu``
     - static regression (3D)
     - ``app/static`` output of ``spherical_hole``
   * - ``reference_results/p<deg>/vmax_ref_QD.csv``
     - SEAS regression
     - ``VMax(t)`` of ``bp1_ref``, QD mode
   * - ``reference_results/p<deg>/vmax_ref_QDGreen.csv``
     - SEAS regression
     - ``VMax(t)`` of ``bp1_ref``, QDGreen mode
   * - ``reference_results/p<deg>/gf_ref/``
     - SEAS regression (QDGreen run)
     - Green's functions of ``bp1_ref``, an *input*: the QDGreen run loads them instead of
       computing them
   * - ``reference_meshes/2D/{bp1_ref,circular_hole}.msh``
     - SEAS and static regression (2D)
     - stored meshes, so that the result does not depend on the gmsh version
   * - ``reference_meshes/3D/spherical_hole.msh``
     - static regression (3D)
     - stored mesh
   * - ``reference_meshes/3D/bp5_mesh_parser_test.{h5,msh}``
     - mesh parser integration (doctest)
     - the same mesh in both formats

``<deg>`` is ``p2`` or ``p3``. There is no data for other degrees.
The configurations and generator scripts that made these files are in this repository
(``test/reference_configs/`` and ``test/scripts/``). They were moved here from the data repository
in March 2026 (data repository commits ``830900e`` to ``85ddd2c``, "only data in this repo").


Provenance
----------

**Summary.** The reference results were generated with tandem built from the branch
``dmay/staging`` (tip ``f8be6045``, 2024-11-08, "Update pure petsc code for petsc v3.22")
and then committed to ``tandem_test_data``. All of them are older than the viscoelasticity work.

The person who made the data gives this account. The git history of both repositories agrees
with it, as the evidence below shows.

What the history shows
^^^^^^^^^^^^^^^^^^^^^^

The commits that wrote reference results (``tandem_test_data``, branch ``master``):

.. list-table::
   :header-rows: 1
   :widths: 13 13 74

   * - commit
     - date
     - change
   * - ``c8c6145``
     - 2025-07-01
     - Initial commit: first ``vmax_ref_*.csv``, ``ref_regression_{2,3}D_output.vtu``, ``gf_ref/``
       (degree 3, then without a ``p3/`` folder).
   * - ``299a4e3``
     - 2025-07-02
     - 3D VTU updated.
   * - ``e6aa64c``
     - 2025-07-14
     - SEAS VMax replaced by a **serial** run ("Better agreement in testing").
   * - ``2b7e7d4``
     - 2026-03-14
     - SEAS VMax updated.
   * - ``c2949a5``, ``830900e``
     - 2026-03-14, 03-26
     - ``gf_ref/`` regenerated or moved with the reorganisation.
   * - ``b9e9d45``
     - 2026-04-08
     - Meshes added to ``reference_meshes/``. 2D VTU and ``gf_ref`` updated to the stored mesh.
   * - ``4bc8421``
     - 2026-04-09
     - 3D VTU updated "according to saved meshes".
   * - ``6111efa``
     - 2026-04-11
     - Degree-2 data added. Degree-3 data moved to ``p3/`` (rename, same content).
   * - ``4a317e7``
     - 2026-04-11
     - ``p2/ref_regression_3D_output.vtu`` updated.

The newest reference result is from 2026-04-11. The later merge ``bcfa070`` (2026-05-30) added only
mesh parser test meshes.

Checks against the tandem history:

* The test suite was developed on top of ``dmay/staging``: the merge base of the test branch and
  ``origin/dmay/staging`` is the tip ``f8be6045``, and the first test commits (2025-07-01) follow it.
* For each date above, the newest commit of the test branch on that date has the **same**
  ``src/``, ``app/``, ``external/``, ``cmake/``, top-level ``CMakeLists.txt`` and submodule pointers as
  ``f8be6045`` (``git diff f8be6045 <commit> -- src app external cmake CMakeLists.txt submodules`` is empty).
  The differences in that period are only in ``test/``, documentation, Docker and CI files.
* ``dmay/staging`` was merged into ``release-v1.2.0`` on 2026-03-10 (``a08d672c``).
  It has had no new commits since 2024-11-08.
* The first viscoelastic commit (``e53feda9``) is from 2026-04-28, after the last reference result.
* PETSc 3.22.x, the version in ``test/README.md``, is the version that ``f8be6045`` adapted the code to.

What the history cannot show
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The files carry no tandem version. The VTU and CSV files have no version field, and the logs with the
  ``code_version=`` line were not committed. That the binary was built from ``dmay/staging`` is the
  author's account. The history shows only that the sources available at the time were the ``dmay/staging``
  sources.
* Machine, compiler, PETSc patch release, gmsh version, and the number of MPI processes are not recorded,
  except that the SEAS data of ``e6aa64c`` is from a serial run.
  tandem is not bit-reproducible across process counts, so this matters for tight tolerances.

To repeat these checks:

.. code-block:: bash

   git clone --filter=blob:none https://github.com/TEAR-ERC/tandem_test_data
   cd tandem_test_data
   git log --follow --format='%h %ad %s' --date=short -- reference_results/p3/vmax_ref_QD.csv

   cd /path/to/tandem
   git merge-base origin/dmay/staging HEAD                 # f8be6045
   c=$(git rev-list -1 --before="2026-04-11 23:59" HEAD)
   git diff --stat f8be6045 $c -- src app external cmake CMakeLists.txt submodules   # empty

Is the data still valid after the viscoelastic changes?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The data was made before the viscoelasticity work, which changed shared code
(the DG operator, the QD driver, the time step control). The regression tests still pass against the
old data with the current code (2D p2, 2D p3 and 3D p2 on 2026-10-23, see :doc:`results`).
So the viscoelastic changes did not change the elasticity and Poisson results beyond the
regression tolerances. The data was not regenerated.


Regenerating the data
---------------------

Regenerate only on purpose: when a change is *meant* to change the regression results
(a new discretisation, a fixed bug), and the change has been verified by the
exact-solution tests. Say why in the commit message, and record the tandem commit, the PETSc version and
the number of processes.

#. Build tandem with ``-DRUN_CI_TESTS=ON`` for each ``(dimension, degree)`` that has data:
   2D and 3D, degrees 2 and 3.
#. Run only the generator step. It keeps the outputs:

   .. code-block:: bash

      ctest -R generate-test-outputs       # writes test/temp_test_results/

#. Copy the new outputs over the reference files:

   .. list-table::
      :header-rows: 1

      * - output in ``test/temp_test_results/``
        - reference file in ``reference_results/p<deg>/``
      * - ``output2D_0.vtu``
        - ``ref_regression_2D_output.vtu``
      * - ``output3D_0.vtu``
        - ``ref_regression_3D_output.vtu``
      * - ``vmax_output_QD.csv``
        - ``vmax_ref_QD.csv``
      * - ``vmax_output_QDGreen.csv``
        - ``vmax_ref_QDGreen.csv``

#. Green's functions: ``bp1_ref_QDGreen.toml`` loads them from ``test_data/reference_results/p<deg>/gf_ref/``.
   If the change affects them, delete that folder in your data checkout first. tandem then computes them
   and writes them there. The file records the process count. A run with another count repartitions them
   when it loads them (``load_discrete_greens_operator``), so the test would not load them the way they
   were written. Generate them with one process, as the test runs.
#. Commit to a branch of ``tandem_test_data`` and test it with
   ``cmake -DTEST_DATA_BRANCH=<branch>`` before you merge it to ``master``.

Run the SEAS reference serially (one process), as ``e6aa64c`` did. This gives the best agreement
with the CI runs, which are also serial.
