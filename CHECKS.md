# Static code checks

Clang tidy

    cmake .. -DCMAKE_CXX_CLANG_TIDY=clang-tidy-12

CppCheck

    cmake .. -DCMAKE_CXX_CPPCHECK="cppcheck;--std=c++17"

IWYU

    cmake .. -DCMAKE_CXX_INCLUDE_WHAT_YOU_USE="include-what-you-use;-Xiwyu;--mapping_file=$(pwd)/../tandem.imp"
