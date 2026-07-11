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
# brotli/bzip2 static decoders (their PUBLIC include dirs propagate to chunk.cpp).
target_link_libraries(ccmf_engine PRIVATE player_warnings brotlidec_static bz2_static)

if(PLAYER_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(ccmf_engine PUBLIC --coverage -O0 -g)
    target_link_options(ccmf_engine PUBLIC --coverage)
endif()
