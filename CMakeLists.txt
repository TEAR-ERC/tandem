cmake_minimum_required(VERSION 3.13)

project(tandem VERSION 1.0
               LANGUAGES C CXX)

include(cmake/user_options.cmake)
include(cmake/cpu_arch_flags.cmake)

get_arch_flags(${ARCH} ${CMAKE_CXX_COMPILER_ID})

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


add_subdirectory(app)
add_subdirectory(external)
add_subdirectory(src)

enable_testing()
add_subdirectory(test)
