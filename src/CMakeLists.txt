set(SOURCES "main.cpp"
            "mesh/GenMesh.cpp"
            "parallel/CommPattern.cpp"
            "parallel/MetisPartitioner.cpp"
)

find_package (MPI REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package (ParMETIS REQUIRED)

add_executable(tandem ${SOURCES})
target_compile_features(tandem PUBLIC cxx_std_17)
target_include_directories(tandem PRIVATE .
                                          ../submodules/
                                          ../submodules/mneme/include/
                                          ${PARMETIS_INCLUDE_DIRS})
target_compile_definitions(tandem PRIVATE PARALLEL)
target_link_libraries(tandem PRIVATE
    MPI::MPI_CXX
    Eigen3::Eigen
    ${PARMETIS_LIBRARIES})
