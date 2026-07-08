cmake_minimum_required(VERSION 3.21)
project(player LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Emit compile_commands.json for editor tooling (clangd etc.); harmless if unused.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PLAYER_BUILD_TESTS "Build the ccmf_engine unit test suite" ON)
option(PLAYER_COVERAGE "Instrument ccmf_engine with --coverage (GCC/Clang only)" OFF)

include(FetchContent)

# --- raylib: window/graphics/audio for the app layer only (never linked by
# ccmf_engine, which stays headless and unit-testable). --------------------
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        5.5
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(raylib)

# Warning flags for OUR code only (never applied globally, so they don't turn
# into noise/failures inside fetched third-party sources like raylib/gtest).
add_library(player_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(player_warnings INTERFACE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(player_warnings INTERFACE /W4)
endif()

add_subdirectory(src/engine)
add_subdirectory(src/app)

if(PLAYER_BUILD_TESTS)
    # Match the runtime library GoogleTest links against to the rest of the
    # project on MSVC, or gtest and our own targets disagree on /MD vs /MT
    # and fail to link (a classic FetchContent+MSVC gotcha).
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif()
