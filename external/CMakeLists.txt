add_library(tinyxml2
    tinyxml2/tinyxml2.cpp
)

target_include_directories(tinyxml2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tinyxml2)
