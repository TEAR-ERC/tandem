cmake_minimum_required(VERSION 3.1)

project(tandem VERSION 1.0
               LANGUAGES C CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})


add_subdirectory(app)
add_subdirectory(src)

enable_testing()
add_subdirectory(test)
