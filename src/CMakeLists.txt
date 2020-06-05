add_library(tandem-lib
    basis/Functions.cpp
    basis/Nodal.cpp
    basis/Quadrature.cpp
    basis/WarpAndBlend.cpp
    mesh/GenMesh.cpp
    geometry/Curvilinear.cpp
    parallel/CommPattern.cpp
    parallel/MetisPartitioner.cpp
    parallel/SortedDistribution.cpp
)

find_package (MPI REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package (ParMETIS REQUIRED)
find_package (OpenMP REQUIRED)

target_compile_features(tandem-lib PUBLIC cxx_std_17)
target_include_directories(tandem-lib PUBLIC .
                                          ../external/
                                          ../submodules/
                                          ../submodules/mneme/include/
                                          ${PARMETIS_INCLUDE_DIRS}
)
target_link_libraries(tandem-lib PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    ${PARMETIS_LIBRARIES}
)
