# Apps without generated kernels

add_executable(preprocess preprocess.cpp)
target_include_directories(preprocess PRIVATE ../external/)
target_link_libraries(preprocess PRIVATE tandem-fem-lib)

add_executable(tensors tensors.cpp)
target_link_libraries(tensors PRIVATE tandem-fem-lib)

add_executable(warp-blend-opt warp-blend-opt.cpp)
target_include_directories(warp-blend-opt PRIVATE ../external/)
target_link_libraries(warp-blend-opt PRIVATE tandem-fem-lib)

add_executable(test-mesh test-mesh.cpp)
target_include_directories(test-mesh PRIVATE ../external/)
target_link_libraries(test-mesh PRIVATE tandem-lib tinyxml2)

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

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
else()
    find_package(PythonInterp 3 REQUIRED)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
include(CTest)
include(../cmake/doctest.cmake)

function(generate_code APP)
    set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernels/${APP})
    file(MAKE_DIRECTORY ${OUTPUT_DIR})
    add_custom_command(
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/generate.py
            "--app" ${APP}
            "--arch" "dnoarch"
            "--options" ${OPTIONS_FILE_NAME}
            "--outputDir" ${OUTPUT_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        DEPENDS
            ${OPTIONS_FILE_NAME}
            kernels/generate.py
            kernels/${APP}.py
        OUTPUT
            kernels/${APP}/init.h
            kernels/${APP}/init.cpp
            kernels/${APP}/kernel.h
            kernels/${APP}/kernel.cpp
            kernels/${APP}/test-kernel.cpp
            kernels/${APP}/subroutine.h
            kernels/${APP}/subroutine.cpp
            kernels/${APP}/tensor.h
            kernels/${APP}/tensor.cpp
        COMMENT "Kernel generation script.")
    add_library(${APP}-kernels-lib
        kernels/${APP}/init.cpp
        kernels/${APP}/kernel.cpp
        kernels/${APP}/subroutine.cpp
        kernels/${APP}/tensor.cpp
    )
    # Use C++-17 in order to enforce inline "constexpr static" variables 
    target_compile_features(${APP}-kernels-lib PUBLIC cxx_std_17)
    target_include_directories(${APP}-kernels-lib PUBLIC
        ../external/
        ../submodules/yateto/include/
    )
    target_link_libraries(${APP}-kernels-lib PUBLIC Eigen3::Eigen)

    add_executable(test-${APP}-kernel kernels/${APP}/test-kernel.cpp)
    target_link_libraries(test-${APP}-kernel ${APP}-kernels-lib test-runner)
    doctest_discover_tests(test-${APP}-kernel)
endfunction()

# Apps using generated kernels

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

## Poisson example
generate_code(poisson)

add_executable(poisson
    poisson/Poisson.cpp
    poisson.cpp)
target_link_libraries(poisson PRIVATE tandem-lib poisson-kernels-lib PkgConfig::PETSC toml)
target_include_directories(poisson PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

## Tandem
generate_code(tandem)

add_executable(tandem
    tandem/Elasticity.cpp
    tandem.cpp)
target_link_libraries(tandem PRIVATE tandem-lib tandem-kernels-lib PkgConfig::PETSC)
target_include_directories(tandem PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
