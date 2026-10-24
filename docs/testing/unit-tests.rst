Unit tests
==========

The unit tests are C++ tests written with `doctest <https://github.com/doctest/doctest>`_.
Each source file in ``test/`` builds one executable ``test-<name>``, and
``doctest_discover_tests`` registers every ``TEST_CASE`` as its own ctest entry.
They run in every build, also without ``-DRUN_CI_TESTS``.
Their reference values are written into the source (most are exact values or values from Maple),
so they need no external data.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - file
     - test cases
     - what is checked
   * - ``basis.cpp``
     - Basis, Nodes
     - The singularity-free Jacobi polynomials agree with the standard ones.
       Dubiner polynomials and their gradients on the triangle and the tetrahedron.
       LGL and Gauss-Legendre nodes. Warp-and-blend and equidistant nodes on the
       triangle (P0 to P6) and the tetrahedron (P0 to P8), and their permutations.
   * - ``form.cpp``
     - Reference element, Projection
     - The mass matrix and its inverse on the reference element, and the L2 projection.
   * - ``geometry.cpp``
     - Curvilinear, Simplex distance, Point locator
     - The curvilinear map, ``det(J)`` and the normals. The distance to a line,
       a triangle and a tetrahedron. Finding the element that holds a point,
       inside the mesh and on its boundary. Uses MPI.
   * - ``quadrules.cpp``
     - Quadrature
     - Gauss-Jacobi and triangle rules against Maple values (tests from SeisSol).
       The polynomial order of the interval, triangle and tetrahedron rules.
       The Jaskowiec-Sukumar and Witherden-Vincent rules and the automatic rule selection.
   * - ``simplex.cpp``
     - Simplex
     - Access, downward maps (dimension 0, 1, 2) and the reference vertices of a simplex.
   * - ``tensor.cpp``
     - Subtensor
     - Packed subtensors: shape, offset and stride, values, and full, slice and vector copies.
   * - ``util.cpp``
     - Combinatorics, Root finding, Allocator, Algorithm
     - Binomials and "choose k" sets, the sets of all integer sums, root finding,
       the aligned allocator, finding blocks and applying a permutation.
   * - ``parallel.cpp``
     - parallel
     - ``SortedDistributionToRank``: which rank owns which part of a distributed range.
   * - ``io.cpp``
     - GMSH
     - Parsing of a GMSH mesh.
   * - ``script.cpp``
     - Lua functions
     - Loading a Lua script and calling its functions.
   * - ``volume_tagging.cpp``
     - Volume tagging
     - ``VolumeTagData`` survives a round trip, sends ``-1`` for out-of-range local ids,
       and ``make_volume_functional`` selects by volume tag. Uses MPI.
   * - ``HDF5MeshParser.cpp``
     - HDF5MeshParser (5 cases)
     - One tetrahedron with all faces tagged, removal of duplicate faces between two
       tetrahedra, a partly tagged boundary, error handling, and the fallback when
       tandem is built without HDF5.
   * - ``HDF5Writer.cpp``
     - HDF5Writer (6 cases)
     - File creation, extendible datasets, the file is closed after destruction,
       checkpoint mode appends and checkpoint off overwrites, read-back of the data,
       and the fallback without HDF5.
   * - ``MeshParserIntegration.cpp``
     - HDF5 and GMSH parsers produce equivalent meshes
     - Only with ``-DRUN_CI_TESTS=ON``, HDF5 and 3D: the HDF5 and the GMSH parser read
       ``test_data/reference_meshes/3D/bp5_mesh_parser_test.{h5,msh}`` and must give the
       same elements and facets. This is the only unit test that needs the test data repository.
