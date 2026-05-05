include(CTest)
include(../cmake/doctest.cmake)

add_library(test-runner main.cpp)
target_compile_options(test-runner PUBLIC ${CPU_ARCH_FLAGS})
target_include_directories(test-runner PUBLIC
    ../external
)
target_link_libraries(test-runner PUBLIC tandem-lib)

add_library(test-runner-mpi main-mpi.cpp)
target_compile_options(test-runner-mpi PUBLIC ${CPU_ARCH_FLAGS})
target_include_directories(test-runner-mpi PUBLIC
    ../external
)
target_link_libraries(test-runner-mpi PUBLIC tandem-lib)

add_executable(test-basis basis.cpp)
target_link_libraries(test-basis test-runner)
doctest_discover_tests(test-basis)

add_executable(test-form form.cpp)
target_link_libraries(test-form test-runner)
doctest_discover_tests(test-form)

add_executable(test-geometry geometry.cpp)
target_link_libraries(test-geometry test-runner-mpi)
doctest_discover_tests(test-geometry)

add_executable(test-parallel parallel.cpp)
target_link_libraries(test-parallel test-runner)
doctest_discover_tests(test-parallel)

add_executable(test-quadrules quadrules.cpp)
target_link_libraries(test-quadrules test-runner)
doctest_discover_tests(test-quadrules)

add_executable(test-io io.cpp)
target_link_libraries(test-io test-runner)
doctest_discover_tests(test-io)

add_executable(test-script script.cpp)
target_link_libraries(test-script test-runner)
doctest_discover_tests(test-script)

add_executable(test-simplex simplex.cpp)
target_link_libraries(test-simplex test-runner)
doctest_discover_tests(test-simplex)

add_executable(test-tensor tensor.cpp)
target_link_libraries(test-tensor test-runner)
doctest_discover_tests(test-tensor)

add_executable(test-util util.cpp)
target_link_libraries(test-util test-runner)
doctest_discover_tests(test-util)

# H-matrix validation test: requires PETSc (H2Opus).
# This test defines its own main() (with PetscInitialize/PetscFinalize) so it
# does NOT link against test-runner or test-runner-mpi (both of which supply
# their own main).  app-common is linked to pull in PETSc and app-layer objects;
# tandem-lib (transitively included via app-common) supplies MPI/OpenMP etc.
# app/CMakeLists.txt runs find_package(PETSc REQUIRED GLOBAL) before this
# directory is processed, making PETSc::PETSc available here.
add_executable(test-h-matrix h_matrix_validation.cpp)
target_compile_options(test-h-matrix PRIVATE ${CPU_ARCH_FLAGS})
target_include_directories(test-h-matrix PRIVATE
    ../external
    ${CMAKE_BINARY_DIR}/app   # for generated config.h
    ${CMAKE_SOURCE_DIR}/app   # for app/ headers
)
target_link_libraries(test-h-matrix PRIVATE app-common)
doctest_discover_tests(test-h-matrix)
