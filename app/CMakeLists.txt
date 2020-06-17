# Apps without generated kernels

add_executable(tandem tandem.cpp)
target_link_libraries(tandem PRIVATE tandem-lib)
target_compile_definitions(tandem PUBLIC PARALLEL)

add_executable(tensors tensors.cpp)
target_link_libraries(tensors PRIVATE tandem-lib)

add_executable(test-gauss test-gauss.cpp)
target_link_libraries(test-gauss PRIVATE tandem-lib)

add_executable(warp-blend-opt warp-blend-opt.cpp)
target_link_libraries(warp-blend-opt PRIVATE tandem-lib)

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
        "--dim" ${DOMAIN_DIMENSION}
        "--degree" ${POLYNOMIAL_DEGREE}
        "--quadPoints" 5
        "--outputDir" ${CMAKE_CURRENT_BINARY_DIR}/kernels
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    DEPENDS
        make-kernels-directory
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
    kernels/test-kernel.cpp
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

