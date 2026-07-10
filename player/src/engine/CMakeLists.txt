add_library(ccmf_engine STATIC
    chunk.cpp
    resync.cpp
    index.cpp
    video.cpp
    render.cpp
    audio.cpp
    engine.cpp
    controls_math.cpp
)

target_include_directories(ccmf_engine PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ccmf_engine PRIVATE player_warnings)

if(PLAYER_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(ccmf_engine PUBLIC --coverage -O0 -g)
    target_link_options(ccmf_engine PUBLIC --coverage)
endif()
