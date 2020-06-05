add_executable(tandem tandem.cpp)
target_link_libraries(tandem PRIVATE tandem-lib)
target_compile_definitions(tandem PUBLIC PARALLEL)

add_executable(tensors tensors.cpp)
target_link_libraries(tensors PRIVATE tandem-lib)

add_executable(test-gauss test-gauss.cpp)
target_link_libraries(test-gauss PRIVATE tandem-lib)

add_executable(warp-blend-opt warp-blend-opt.cpp)
target_link_libraries(warp-blend-opt PRIVATE tandem-lib)
