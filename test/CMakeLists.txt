include(CTest)
include(../cmake/doctest.cmake)

option(RUN_CI_TESTS "Enable CI integration tests" OFF)

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

add_executable(test-volume-tagging volume_tagging.cpp)
target_link_libraries(test-volume-tagging test-runner-mpi)
doctest_discover_tests(test-volume-tagging)

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

if(RUN_CI_TESTS)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(Git REQUIRED)

    # Fetch test data only when needed
    set(TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_data")
    if(NOT EXISTS "${TEST_DATA_DIR}/reference_results")
        message(STATUS "Fetching test data...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone
            --depth 1
            --branch master
            https://github.com/piyushkarki/tandem_test_data.git
            ${TEST_DATA_DIR}
            RESULT_VARIABLE GIT_CLONE_RESULT
        )
        if(NOT GIT_CLONE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to clone test data repository")
        endif()
    endif()

    # Pre-clean to remove stale outputs from any previous test run
    add_test(NAME pre-clean-test-outputs
        COMMAND bash -c
        "source ${CMAKE_CURRENT_SOURCE_DIR}/test_data/scripts/cleanup_test_outputs.sh ${CMAKE_SOURCE_DIR}"
    )
    set_tests_properties(pre-clean-test-outputs PROPERTIES
        FIXTURES_SETUP PreClean
    )

    # Generate test outputs before any pytest test runs
    add_test(NAME generate-test-outputs
        COMMAND bash -c
        "source ${CMAKE_CURRENT_SOURCE_DIR}/test_data/scripts/generate_test_outputs.sh ${DOMAIN_DIMENSION} ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}"
    )
    set_tests_properties(generate-test-outputs PROPERTIES
        FIXTURES_REQUIRED PreClean
        FIXTURES_SETUP TestData
    )

    # Helper macro for adding Python pytest-based tests
    macro(add_pytest_test test_name file_name)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file_name}")
            add_test(
                NAME ${test_name}
                COMMAND ${Python_EXECUTABLE} -m pytest ${file_name} --domain_dimension=${DOMAIN_DIMENSION}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )
            set_tests_properties(${test_name} PROPERTIES
                FIXTURES_REQUIRED TestData
            )
        else()
            message(WARNING "Python test file ${file_name} not found. Skipping test ${test_name}.")
        endif()
    endmacro()

    add_pytest_test("static-regression-test-${DOMAIN_DIMENSION}D" "test_regression_static.py")
    add_pytest_test("parallel-consistency-test-${DOMAIN_DIMENSION}D" "test_parallel_consistency.py")
    add_pytest_test("convergence-test-${DOMAIN_DIMENSION}D" "test_static_convergence.py")
    if(DOMAIN_DIMENSION EQUAL 2)
        add_pytest_test("SEAS-regression-test-${DOMAIN_DIMENSION}D" "test_regression_SEAS.py")
    endif()

    # Remove test outputs after all pytest tests finish
    add_test(NAME cleanup-test-outputs
        COMMAND bash -c
        "source ${CMAKE_CURRENT_SOURCE_DIR}/test_data/scripts/cleanup_test_outputs.sh ${CMAKE_SOURCE_DIR}"
    )
    set_tests_properties(cleanup-test-outputs PROPERTIES
        FIXTURES_CLEANUP TestData
    )

endif()
