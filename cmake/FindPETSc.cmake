find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET GLOBAL PETSc)
add_library(PETSc::PETSc ALIAS PkgConfig::PETSc)

# --- Version check -------------------------------------------------------------
set(TANDEM_PETSC_MIN_VERSION "3.22" CACHE STRING "Minimum supported PETSc version (inclusive)")
set(TANDEM_PETSC_MAX_VERSION "3.23" CACHE STRING "First unsupported PETSc version (exclusive)")
option(TANDEM_IGNORE_PETSC_VERSION "Only warn about an unsupported PETSc version" OFF)
mark_as_advanced(TANDEM_PETSC_MIN_VERSION TANDEM_PETSC_MAX_VERSION)

if(PETSc_VERSION VERSION_LESS TANDEM_PETSC_MIN_VERSION OR
   NOT PETSc_VERSION VERSION_LESS TANDEM_PETSC_MAX_VERSION)
    set(_petsc_msg
        "PETSc ${PETSc_VERSION} is not supported. "
        "tandem requires PETSc >= ${TANDEM_PETSC_MIN_VERSION} and < ${TANDEM_PETSC_MAX_VERSION}.\n"
        "Use a supported PETSc, e.g. via\n"
        "  PKG_CONFIG_PATH=/path/to/petsc-3.22/lib/pkgconfig cmake -B build ...\n"
        "or pass -DTANDEM_IGNORE_PETSC_VERSION=ON to continue anyway.")
    if(TANDEM_IGNORE_PETSC_VERSION)
        message(WARNING ${_petsc_msg})
    else()
        message(FATAL_ERROR ${_petsc_msg})
    endif()
else()
    message(STATUS "Found PETSc ${PETSc_VERSION} (${PETSc_PREFIX})")
endif()

set(PETSC_MEMALIGN "PETSC_MEMALIGN-NOTFOUND")
find_file(PETSC_CONF_H "petscconf.h" HINTS ${PETSc_INCLUDE_DIRS})
if(NOT ${PETSC_CONF_H} MATCHES "PETSC_CONF_H-NOTFOUND")
    file(READ ${PETSC_CONF_H} PETSC_CONF_CONTENT)
    if(${PETSC_CONF_CONTENT} MATCHES "#[ \t]*define[ \t]+PETSC_MEMALIGN[ \t]+([0-9]+)")
        set(PETSC_MEMALIGN ${CMAKE_MATCH_1})
    endif()
endif()
if(PETSC_MEMALIGN MATCHES "PETSC_MEMALIGN-NOTFOUND")
    message(STATUS "Could not determine PETSC_MEMALIGN. Default to 8.")
    set(PETSC_MEMALIGN 8)
endif()

