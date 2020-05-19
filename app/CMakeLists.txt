add_executable(tandem main.cpp)

target_link_libraries(tandem PRIVATE tandem-lib)
target_compile_definitions(tandem PUBLIC PARALLEL)

