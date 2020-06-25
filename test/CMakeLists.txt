include(CTest)
include(../cmake/doctest.cmake)

add_library(test-runner main.cpp)
target_include_directories(test-runner PUBLIC
    ../external
)
target_link_libraries(test-runner PUBLIC tandem-lib)

add_library(test-runner-mpi main-mpi.cpp)
target_include_directories(test-runner-mpi PUBLIC
    ../external
)
target_link_libraries(test-runner-mpi PUBLIC tandem-lib)

add_executable(test-basis basis.cpp)
target_link_libraries(test-basis test-runner)
doctest_discover_tests(test-basis)

add_executable(test-geometry geometry.cpp)
target_link_libraries(test-geometry test-runner-mpi)
doctest_discover_tests(test-geometry)

add_executable(test-parallel parallel.cpp)
target_link_libraries(test-parallel test-runner)
doctest_discover_tests(test-parallel)

add_executable(test-quadrules quadrules.cpp)
target_link_libraries(test-quadrules test-runner)
doctest_discover_tests(test-quadrules)

add_executable(test-simplex simplex.cpp)
target_link_libraries(test-simplex test-runner)
doctest_discover_tests(test-simplex)

add_executable(test-tensor tensor.cpp)
target_link_libraries(test-tensor test-runner)
doctest_discover_tests(test-tensor)

add_executable(test-util util.cpp)
target_link_libraries(test-util test-runner)
doctest_discover_tests(test-util)
