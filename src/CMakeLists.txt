find_package(Eigen3 REQUIRED NO_MODULE)

# Add element-local stuff here
add_library(tandem-fem-lib
    basis/Equidistant.cpp
    basis/Functions.cpp
    basis/GaussLegendre.cpp
    basis/Nodal.cpp
    basis/WarpAndBlend.cpp
    form/FiniteElementFunction.cpp
    form/RefElement.cpp
    quadrules/GaussJacobi.cpp
    quadrules/IntervalQuadratureRule.cpp
    quadrules/JaskowiecSukumar2020.cpp
    quadrules/WitherdenVincent2015_Tri.cpp
    quadrules/WitherdenVincent2015_Tet.cpp
    quadrules/TensorProductRule.cpp
    util/Math.cpp
)

target_compile_features(tandem-fem-lib PUBLIC cxx_std_17)
target_compile_options(tandem-fem-lib PRIVATE ${CPU_ARCH_FLAGS})
target_include_directories(tandem-fem-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tandem-fem-lib PUBLIC
    Eigen3::Eigen
)

# Add everything else here
add_library(tandem-lib
    form/BoundaryMap.cpp
    form/DGCurvilinearCommon.cpp
    form/DGOperatorTopo.cpp
    form/Error.cpp
    io/BoundaryProbeWriter.cpp
    io/ProbeWriter.cpp
    io/ScalarWriter.cpp
    io/GlobalSimplexMeshBuilder.cpp
    io/GMSHLexer.cpp
    io/GMSHParser.cpp
    io/PVDWriter.cpp
    io/VTUWriter.cpp
    io/HDF5Writer.cpp
    io/HDF5BoundaryProbeWriter.cpp
    io/VTUWriter.cpp
    mesh/GenMesh.cpp
    mesh/GlobalSimplexMesh.cpp
    geometry/Curvilinear.cpp
    geometry/PointLocator.cpp
    parallel/Affinity.cpp
    parallel/CommPattern.cpp
    parallel/MetisPartitioner.cpp
    parallel/Profile.cpp
    parallel/ScatterPlan.cpp
    parallel/SortedDistribution.cpp
    parallel/Summary.cpp
    script/LuaLib.cpp
    util/TablePrinter.cpp
    util/Zero.cpp
)

find_package(Lua REQUIRED)
find_package(MPI REQUIRED)
find_package(ParMETIS REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)
include(../cmake/filesystem.cmake)
find_package(HDF5 COMPONENTS C HL REQUIRED)
message(STATUS "Found HDF5: ${HDF5_VERSION} at ${HDF5_INCLUDE_DIRS}")
target_compile_options(tandem-lib PRIVATE ${CPU_ARCH_FLAGS})
target_include_directories(tandem-lib PUBLIC 
    ${HDF5_INCLUDE_DIRS}
    ../submodules/
    ../submodules/mneme/include/
    ${LUA_INCLUDE_DIR}
)
target_link_libraries(tandem-lib PUBLIC
    ${HDF5_LIBRARIES}    
    tandem-fem-lib
    ${FILESYSTEM_LIBRARIES}
    ${LUA_LIBRARIES}
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    PARMETIS::PARMETIS
    tinyxml2
    toml
    ZLIB::ZLIB
)
