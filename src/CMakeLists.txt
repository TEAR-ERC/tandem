add_custom_target(
    make-kernels-directory
    COMMAND
    ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels")

find_package(Python3 COMPONENTS Interpreter)

add_custom_command(
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/generate.py
        "--arch" "dnoarch"
        "--degree" 3
        "--quadPoints" 5
        "--outputDir" ${CMAKE_CURRENT_BINARY_DIR}/kernels
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    DEPENDS
        make-kernels-directory
        kernels/generate.py
    OUTPUT
        kernels/init.h
        kernels/init.cpp
        kernels/kernel.h
        kernels/kernel.cpp
        kernels/subroutine.h
        kernels/subroutine.cpp
        kernels/tensor.h
        kernels/tensor.cpp
    COMMENT "Kernel generation script.")

add_library(tandem-lib
    basis/Functions.cpp
    basis/Nodal.cpp
    basis/WarpAndBlend.cpp
    form/DG.cpp
    kernels/init.cpp
    kernels/kernel.cpp
    kernels/subroutine.cpp
    kernels/tensor.cpp
    mesh/GenMesh.cpp
    geometry/Curvilinear.cpp
    parallel/CommPattern.cpp
    parallel/MetisPartitioner.cpp
    parallel/SortedDistribution.cpp
    quadrules/GaussJacobi.cpp
    quadrules/IntervalQuadratureRule.cpp
    quadrules/JaskowiecSukumar2020.cpp
    quadrules/WitherdenVincent2015_Tri.cpp
    quadrules/WitherdenVincent2015_Tet.cpp
    quadrules/TensorProductRule.cpp
)

find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(ParMETIS REQUIRED)
find_package(OpenMP REQUIRED)

target_compile_features(tandem-lib PUBLIC cxx_std_17)
target_include_directories(tandem-lib PUBLIC .
                                          ../external/
                                          ../submodules/
                                          ../submodules/mneme/include/
                                          ../submodules/yateto/include/
                                          ${PARMETIS_INCLUDE_DIRS}
)
target_link_libraries(tandem-lib PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    ${PARMETIS_LIBRARIES}
)
