# Apps without generated kernels

add_executable(preprocess preprocess.cpp)
target_include_directories(preprocess PRIVATE ../external/)
target_link_libraries(preprocess PRIVATE tandem-fem-lib)

add_executable(tensors tensors.cpp)
target_link_libraries(tensors PRIVATE tandem-fem-lib)

add_executable(warp-blend-opt warp-blend-opt.cpp)
target_include_directories(warp-blend-opt PRIVATE ../external/)
target_link_libraries(warp-blend-opt PRIVATE tandem-fem-lib)

add_executable(tandem tandem.cpp)
target_link_libraries(tandem PRIVATE tandem-lib tinyxml2)
target_compile_definitions(tandem PUBLIC USE_MPI)

add_executable(test-gauss test-gauss.cpp)
target_include_directories(test-gauss PRIVATE ../external/)
target_link_libraries(test-gauss PRIVATE tandem-lib)

# Preprocessing for kernel generation

set(OPTIONS_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/options.json)

add_custom_command(
    COMMAND
        preprocess
        -Q ${MIN_QUADRATURE_ORDER}
        -o ${OPTIONS_FILE_NAME}
        ${DOMAIN_DIMENSION}
        ${POLYNOMIAL_DEGREE}
    DEPENDS
        preprocess
    OUTPUT
        ${OPTIONS_FILE_NAME}
)

# Generate kernels

add_custom_target(
    make-kernels-directory
    COMMAND
    ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/kernels")

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
else()
    find_package(PythonInterp 3 REQUIRED)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

add_custom_command(
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/generate.py
        "--arch" "dnoarch"
        "--options" ${OPTIONS_FILE_NAME}
        "--outputDir" ${CMAKE_CURRENT_BINARY_DIR}/kernels
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    DEPENDS
        make-kernels-directory
        ${OPTIONS_FILE_NAME}
        kernels/generate.py
        kernels/poisson.py
    OUTPUT
        kernels/init.h
        kernels/init.cpp
        kernels/kernel.h
        kernels/kernel.cpp
        kernels/test-kernel.cpp
        kernels/subroutine.h
        kernels/subroutine.cpp
        kernels/tensor.h
        kernels/tensor.cpp
    COMMENT "Kernel generation script.")

find_package(Eigen3 REQUIRED NO_MODULE)

add_library(kernels-lib
    kernels/init.cpp
    kernels/kernel.cpp
    kernels/subroutine.cpp
    kernels/tensor.cpp
)
target_include_directories(kernels-lib PUBLIC
    ../external/
    ../submodules/yateto/include/
)
target_link_libraries(kernels-lib PUBLIC Eigen3::Eigen)

include(CTest)
include(../cmake/doctest.cmake)
add_executable(test-kernel kernels/test-kernel.cpp)
target_link_libraries(test-kernel kernels-lib test-runner)
doctest_discover_tests(test-kernel)

# Apps using generated kernels

configure_file( config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h )

add_executable(poisson
    poisson/Poisson.cpp
    poisson.cpp)
target_link_libraries(poisson PRIVATE tandem-lib kernels-lib)
target_include_directories(poisson PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(poisson PUBLIC USE_MPI)
