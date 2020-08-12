add_library(tinyxml2
    tinyxml2/tinyxml2.cpp
)
target_include_directories(tinyxml2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tinyxml2)

add_library(toml
    toml/toml.cpp
)
target_compile_features(toml PRIVATE cxx_std_17)
target_include_directories(toml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/toml)
