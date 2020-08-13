find_package(Eigen3 REQUIRED NO_MODULE)

# Add element-local stuff here
add_library(tandem-fem-lib
    basis/Equidistant.cpp
    basis/Functions.cpp
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
)

target_compile_features(tandem-fem-lib PUBLIC cxx_std_17)
target_include_directories(tandem-fem-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tandem-fem-lib PUBLIC
    Eigen3::Eigen
)

# Add everything else here
add_library(tandem-lib
    form/DG.cpp
    form/Error.cpp
    io/VTUWriter.cpp
    io/VTUWriter.cpp
    mesh/GenMesh.cpp
    mesh/GlobalSimplexMesh.cpp
    geometry/Curvilinear.cpp
    parallel/CommPattern.cpp
    parallel/MetisPartitioner.cpp
    parallel/SortedDistribution.cpp
    script/LuaLib.cpp
)

find_package(Lua REQUIRED)
find_package(MPI REQUIRED)
find_package(ParMETIS REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)

target_include_directories(tandem-lib PUBLIC 
    ../submodules/
    ../submodules/mneme/include/
    ${LUA_INCLUDE_DIR}
)
target_link_libraries(tandem-lib PUBLIC
    tandem-fem-lib
    ${LUA_LIBRARIES}
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    PARMETIS::PARMETIS
    tinyxml2
    toml
    ZLIB::ZLIB
)
