add_executable(ccmf_engine_tests
    test_support.cpp
    test_chunk.cpp
    test_index.cpp
    test_video.cpp
    test_render.cpp
    test_audio.cpp
    test_engine.cpp
    test_controls_math.cpp
)

target_link_libraries(ccmf_engine_tests PRIVATE ccmf_engine GTest::gtest_main ccmf_player_warnings)

# Fixtures (small .ccmf files + expected-output vectors) are read relative to
# this directory at test run time.
target_compile_definitions(ccmf_engine_tests PRIVATE
    CCMF_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

gtest_discover_tests(ccmf_engine_tests)
