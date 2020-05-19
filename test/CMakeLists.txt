
include(CTest)
include(../cmake/doctest.cmake)

add_library(test-runner main.cpp)
target_include_directories(test-runner PUBLIC
    ../external
)
target_link_libraries(test-runner PUBLIC tandem-lib)

#add_executable(test-displs displs.cpp)
#target_link_libraries(test-displs test-runner)
#doctest_discover_tests(test-displs)
