Test suite
==========

.. toctree::
   :hidden:

   unit-tests
   integration-tests
   analytical-solutions
   reference-data
   results

This part of the documentation is the reference for the tandem test suite:
what each test checks, where its ground truth comes from, and what it takes to pass.
The test sources are in ``test/`` and the generator scripts in ``test/scripts/``.
When a number here and the code disagree, the code is right:
the tolerances live in ``test/conftest.py``.

* :doc:`unit-tests`: the C++ (doctest) tests of single components.
* :doc:`integration-tests`: the pytest tests that run ``app/static`` or ``app/tandem``,
  one section per test.
* :doc:`analytical-solutions`: the exact solutions the tests compare against.
* :doc:`reference-data`: the stored reference results, how they were made and how to remake them.
* :doc:`results`: the pass criteria in one table, measured values, run times and the result plots.


Two kinds of tests
------------------

Unit tests
   Fast C++ tests built with every configuration. ``make test`` (or ``ctest``) runs them.

Integration tests
   They run the solvers on small problems and check the output with pytest.
   They are built only with ``-DRUN_CI_TESTS=ON``, which also clones the
   `tandem_test_data <https://github.com/TEAR-ERC/tandem_test_data>`_ repository
   into ``test/test_data``.

The integration tests check the output in one of three ways:

Against an exact solution
   The test knows the answer in closed form. Most of these problems have a solution
   that the discrete space can represent exactly, so the error must be at round-off.
   Examples: free slip, traction, the viscoelastic tests.

Against a convergence rate
   The error must fall at the expected rate when the mesh (or the time step) is refined.
   Examples: static convergence, viscoelastic manufactured solution.

Against stored reference data
   The output must match a stored earlier output of tandem (a *regression* test).
   This finds changes, not errors: the reference itself is not proven correct.
   Examples: static regression, SEAS regression.
   See :doc:`reference-data`.

Some tests compare tandem with itself instead: serial against parallel runs, or
CSV against HDF5 output.


Running the tests
-----------------

.. code-block:: bash

   mkdir build && cd build
   cmake .. -DDOMAIN_DIMENSION=2 -DPOLYNOMIAL_DEGREE=2 -DRUN_CI_TESTS=ON
   make -j
   ctest --output-on-failure

The domain dimension and the polynomial degree are compile-time options,
so one build runs the tests of one ``(dimension, degree)`` pair.
CI builds the matrix ``dimension = {2, 3}`` x ``degree = {2, 3}``, with and without HDF5.

Useful CMake options:

``-DTEST_DATA_BRANCH=<branch>``
   Use another branch of ``tandem_test_data`` (default ``master``).
   Use it to test new reference data before it is merged, see :doc:`reference-data`.

``-DPLOT_CI_TESTS=OFF``
   Do not make the result plots (on by default, see :doc:`results`).

To run one pytest test by hand, first make the outputs with the ``generate-test-outputs`` step,
then call pytest from ``test/``:

.. code-block:: bash

   cd build && ctest -R generate-test-outputs     # writes test/temp_test_results/
   cd ../test
   python -m pytest test_free_slip_bc.py --domain_dimension=2 --polynomial_degree=2

``ctest`` deletes ``test/temp_test_results`` again at the end of a full run
(``cleanup-test-outputs``).


How a CI run is ordered
-----------------------

``ctest`` uses fixtures to order the integration tests:

.. code-block:: text

   pre-clean-test-outputs      delete stale outputs                  (fixture PreClean)
           |
   generate-test-outputs       run every generator script once       (fixture TestData)
           |                   -> test/temp_test_results/
           |
   +-------+--------------------------------------------+
   | pytest tests (read the outputs, run no solver)     |
   | plot-test-results (PNGs for the CI artifacts)      |
   +-------+--------------------------------------------+
           |
   cleanup-test-outputs        delete test/temp_test_results

All simulations run in ``generate-test-outputs``
(``test/scripts/generate_test_outputs.sh``, which calls
``test/scripts/<dim>D/generate_<dim>D_outputs.sh``).
The pytest tests only read logs, VTU, CSV and HDF5 files.
This is why the generator step takes nearly all of the run time (see :doc:`results`).


Which tests run where
---------------------

.. list-table::
   :header-rows: 1
   :widths: 34 8 8 10 40

   * - ctest name
     - 2D
     - 3D
     - degree
     - ground truth
   * - ``static-regression-test``
     - yes
     - yes
     - 2, 3
     - stored VTU (reference data)
   * - ``SEAS-regression-test``
     - yes
     -
     - 2, 3
     - stored VMax CSV (reference data)
   * - ``convergence-test``
     - yes
     - yes
     - any
     - L2 / H1 rates (exact solution)
   * - ``parallel-consistency-test``
     - yes
     - yes
     - any
     - serial run
   * - ``probe-writer-consistency-test``
     - HDF5
     -
     - any
     - CSV output, one stored number
   * - ``mesh-parser-integration`` (doctest)
     -
     - HDF5
     - any
     - GMSH parser
   * - ``volume-tagging-correctness-test``
     - yes
     -
     - any
     - exact solution
   * - ``volume-tagging-parallel-consistency-test``
     - yes
     -
     - any
     - serial run
   * - ``free-slip-bc-test``
     - yes
     - yes
     - any
     - exact solution
   * - ``traction-patch-bc-test``
     - yes
     -
     - any
     - exact solution
   * - ``traction-bc-test``
     -
     - yes
     - any
     - closed-form end displacement
   * - ``viscoelastic-simple-shear-test``
     - yes
     -
     - any
     - closed form
   * - ``viscoelastic-mms-test``
     - yes
     -
     - any (time part: p >= 2)
     - manufactured solution
   * - ``viscoelastic-cantilever-test``
     -
     - yes
     - any
     - Zener closed form

The regression tests need stored data for the degree. Data exists for degrees 2 and 3
(``ORDERS_WITH_REFERENCE_DATA`` in ``test/CMakeLists.txt``). For other degrees ctest runs
the placeholder ``regression-tests-skipped-p<N>`` instead.

Software versions used for the suite: PETSc 3.22.x, gmsh with mesh format 2.2.


Adding a test
-------------

``test/README.md`` gives the steps. Please also add the test to these pages:
a section in :doc:`integration-tests`, its exact solution in :doc:`analytical-solutions`
(if it has one) and its criterion in :doc:`results`.
