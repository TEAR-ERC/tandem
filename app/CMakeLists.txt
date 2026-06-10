# Apps without generated kernels

add_executable(preprocess preprocess.cpp)
target_compile_options(preprocess PRIVATE ${CPU_ARCH_FLAGS})
target_include_directories(preprocess PRIVATE ../external/)
target_link_libraries(preprocess PRIVATE tandem-fem-lib)

add_executable(tensors tensors.cpp)
target_compile_options(tensors PRIVATE ${CPU_ARCH_FLAGS})
target_link_libraries(tensors PRIVATE tandem-fem-lib)

add_executable(warp-blend-opt warp-blend-opt.cpp)
target_compile_options(warp-blend-opt PRIVATE ${CPU_ARCH_FLAGS})
target_include_directories(warp-blend-opt PRIVATE ../external/)
target_link_libraries(warp-blend-opt PRIVATE tandem-fem-lib)

add_executable(check-bc check-bc.cpp)
target_compile_options(check-bc PRIVATE ${CPU_ARCH_FLAGS})
target_include_directories(check-bc PRIVATE ../external/)
target_link_libraries(check-bc PRIVATE tandem-lib)

add_executable(test-mesh test-mesh.cpp)
target_compile_options(test-mesh PRIVATE ${CPU_ARCH_FLAGS})
target_include_directories(test-mesh PRIVATE ../external/)
target_link_libraries(test-mesh PRIVATE tandem-lib tinyxml2)

add_executable(test-gauss test-gauss.cpp)
target_compile_options(test-gauss PRIVATE ${CPU_ARCH_FLAGS})
target_include_directories(test-gauss PRIVATE ../external/)
target_link_libraries(test-gauss PRIVATE tandem-lib)

add_executable(test-scatter test-scatter.cpp)
target_compile_options(test-scatter PRIVATE ${CPU_ARCH_FLAGS})
target_include_directories(test-scatter PRIVATE ../external/)
target_link_libraries(test-scatter PRIVATE tandem-lib)

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

# Define options for state evolution law
set(FRICTION_LAW "dieterich_ruina_ageing" CACHE STRING "Define the state variable evolution law")
if (FRICTION_LAW STREQUAL "dieterich_ruina_ageing")
    add_definitions(-DDR_AGEING_LAW)
    message(STATUS "State evolution law: Dieterich-Ruina Ageing Law")
elseif (FRICTION_LAW STREQUAL "dieterich_ruina_slip")
    add_definitions(-DDR_SLIP_LAW)
    message(STATUS "State evolution law: Dieterich-Ruina Slip Law")
else()
    message(FATAL_ERROR "Cannot recognize the given option. Your option should be either [dieterich_ruina_ageing] or [dieterich_ruina_slip].")
endif()

# Generate kernels

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
else()
    find_package(PythonInterp 3 REQUIRED)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(LibxsmmGenerator)
include(CTest)
include(../cmake/doctest.cmake)

find_package(PETSc REQUIRED)

if(PETSC_MEMALIGN LESS ALIGNMENT)
    message(SEND_ERROR "The memory alignment of PETSc is ${PETSC_MEMALIGN} bytes but an alignment of "
                       "at least ${ALIGNMENT} bytes is required for ARCH=${ARCH}. "
                       "Please compile PETSc with --with-memalign=${ALIGNMENT}.")
endif()

add_library(global-flop-counter kernels/flop_counter.cpp)

function(generate_code APP)
    set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernels/${APP})
    file(MAKE_DIRECTORY ${OUTPUT_DIR})
    set(YATETO_ARCH ${ARCH})
    if("${YATETO_ARCH}" MATCHES "skl|naples|rome")
        set(YATETO_ARCH "hsw")
    endif()
    set(WITH_LIBXSMM "\'\'")
    if(${LibxsmmGenerator_FOUND})
        set(WITH_LIBXSMM "\'${LibxsmmGeneratorExecutable}\'")
    endif()
    add_custom_command(
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/generate.py
            "--app" ${APP}
            "--arch" "d${YATETO_ARCH}"
            "--options" ${OPTIONS_FILE_NAME}
            "--outputDir" ${OUTPUT_DIR}
            "--with_libxsmm" ${WITH_LIBXSMM}
            "--petsc_memalign" ${PETSC_MEMALIGN}
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
    if(NO_REDZONE)
        set_source_files_properties(kernels/${APP}/subroutine.cpp PROPERTIES COMPILE_FLAGS -mno-red-zone)
    endif()
    # Use C++-17 in order to enforce inline "constexpr static" variables 
    target_compile_definitions(${APP}-kernels-lib PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=2097152)
    target_compile_features(${APP}-kernels-lib PUBLIC cxx_std_17)
    target_compile_options(${APP}-kernels-lib PUBLIC ${CPU_ARCH_FLAGS})
    target_include_directories(${APP}-kernels-lib PUBLIC
        ../external/
        ../submodules/yateto/include/
    )
    target_link_libraries(${APP}-kernels-lib PRIVATE global-flop-counter)
    target_link_libraries(${APP}-kernels-lib PUBLIC Eigen3::Eigen)

    add_executable(test-${APP}-kernel kernels/${APP}/test-kernel.cpp)
    target_link_libraries(test-${APP}-kernel ${APP}-kernels-lib test-runner)
    doctest_discover_tests(test-${APP}-kernel)
endfunction()

# Apps using generated kernels

include(GetGitRevisionDescription)
get_git_head_revision(refspec hash)
git_describe(PACKAGE_GIT_VERSION --always)
message(STATUS "Current Git description: " ${PACKAGE_GIT_VERSION})

configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

## Generate code
generate_code(poisson)
generate_code(poisson_adapter)
generate_code(elasticity)
generate_code(elasticity_adapter)

## LAPACK interface
include(CheckLanguage)

check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    include(FortranCInterface)
    FortranCInterface_HEADER(FC.h MACRO_NAMESPACE "FC_")

    find_package(LAPACK)
    if(${LAPACK_FOUND})
        add_definitions(-DHAVE_LAPACK)
    endif()
endif()

## Common
set(APP_COMMON_SRCS
    common/Banner.cpp
    common/MeshConfig.cpp
    common/MGConfig.cpp
    common/PetscDGMatrix.cpp
    common/PetscDGShell.cpp
    common/PetscInterplMatrix.cpp
    common/PetscLinearSolver.cpp
    common/PetscVector.cpp
    common/PetscTimeSolver.cpp
    common/PetscLoggingUtils.h
    form/SeasFDOperator.cpp
    form/SeasQDOperator.cpp
    form/SeasQDDiscreteGreenOperator.cpp
    localoperator/AdapterBase.cpp
    localoperator/NodalInterpolation.cpp
    localoperator/Elasticity.cpp
    localoperator/ElasticityAdapter.cpp
    localoperator/Poisson.cpp
    localoperator/PoissonAdapter.cpp
    localoperator/RateAndStateBase.cpp
    pc/lspoly.c
    pc/register.cpp
    common/ts_util.c
    common/vecnest_util.c
)
if(${LAPACK_FOUND})
    list(APPEND APP_COMMON_SRCS
         pc/eigdeflate.c
         pc/reig_aux.c
    )
endif()
add_library(app-common ${APP_COMMON_SRCS})
target_compile_definitions(app-common PUBLIC "ALIGNMENT=${ALIGNMENT}")
target_link_libraries(app-common PUBLIC
    ${LAPACK_LIBRARIES}
    poisson-kernels-lib
    poisson_adapter-kernels-lib
    elasticity-kernels-lib
    elasticity_adapter-kernels-lib
    tandem-lib
    PETSc::PETSc)
target_include_directories(app-common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

## Static solver

add_executable(static static.cpp)
target_link_libraries(static PRIVATE app-common)
target_include_directories(static PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

## Tandem

add_executable(tandem
    tandem/Monitor.cpp
    tandem/SeasConfig.cpp
    tandem/SEAS.cpp
    tandem.cpp)
target_link_libraries(tandem PRIVATE app-common)
target_include_directories(tandem PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
