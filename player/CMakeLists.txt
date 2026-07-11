cmake_minimum_required(VERSION 3.21)
# C is enabled for the fetched brotli/bzip2 sources (both are C libraries).
project(player LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimized build on single-config generators (Ninja, Makefiles):
# an unset CMAKE_BUILD_TYPE there means NO optimization flags at all (~ -O0), so a
# plain `cmake ..` would silently ship an unoptimized player. Multi-config
# generators (Visual Studio, Xcode) pick the config at build time, so leave them.
get_property(_player_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _player_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "player: no CMAKE_BUILD_TYPE set, defaulting to Release.")
endif()

# Emit compile_commands.json for editor tooling (clangd etc.); harmless if unused.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PLAYER_BUILD_TESTS "Build the ccmf_engine unit test suite" ON)
option(PLAYER_COVERAGE "Instrument ccmf_engine with --coverage (GCC/Clang only)" OFF)

# Probe once whether the toolchain can do interprocedural (link-time) optimization;
# enabled per-target on release configs after the targets are defined (see below).
include(CheckIPOSupported)
check_ipo_supported(RESULT player_ipo_supported OUTPUT player_ipo_error LANGUAGES CXX)

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

# --- brotli (decoder) + bzip2: native-only chunk compression (spec 4.1.2). The
# CC/Lua client can't inflate these (no native bit ops); the desktop player can.
# Fetched and compiled as small static libs straight from upstream sources --
# SOURCE_SUBDIR points at a dir with no CMakeLists.txt, so their own build never
# runs and we own the target (and keep our /W4 -Werror off their code). --------
FetchContent_Declare(
    brotli
    GIT_REPOSITORY https://github.com/google/brotli.git
    GIT_TAG        v1.1.0
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  do-not-configure
)
FetchContent_MakeAvailable(brotli)
file(GLOB _brotli_dec_srcs
     "${brotli_SOURCE_DIR}/c/common/*.c"
     "${brotli_SOURCE_DIR}/c/dec/*.c")
add_library(brotlidec_static STATIC ${_brotli_dec_srcs})
target_include_directories(brotlidec_static PUBLIC "${brotli_SOURCE_DIR}/c/include")

FetchContent_Declare(
    bzip2
    GIT_REPOSITORY https://gitlab.com/bzip2/bzip2.git
    GIT_TAG        bzip2-1.0.8
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  do-not-configure
)
FetchContent_MakeAvailable(bzip2)
add_library(bz2_static STATIC
    "${bzip2_SOURCE_DIR}/blocksort.c"
    "${bzip2_SOURCE_DIR}/huffman.c"
    "${bzip2_SOURCE_DIR}/crctable.c"
    "${bzip2_SOURCE_DIR}/randtable.c"
    "${bzip2_SOURCE_DIR}/compress.c"
    "${bzip2_SOURCE_DIR}/decompress.c"
    "${bzip2_SOURCE_DIR}/bzlib.c")
target_include_directories(bz2_static PUBLIC "${bzip2_SOURCE_DIR}")

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

# Interprocedural / link-time optimization for RELEASE-class builds only.
# ccmf_engine is a static library split across translation units, and the hot
# decode/render path crosses those boundaries (engine.cpp -> video/audio/chunk/
# index/resync.cpp; app -> engine), so cross-TU inlining and dead-code stripping
# only happen with IPO/LTO. Debug builds deliberately trade this away for build
# speed and debuggability. Set per-target so it never disturbs the fetched
# raylib/gtest builds.
if(player_ipo_supported)
    foreach(_player_tgt ccmf_engine player)
        set_property(TARGET ${_player_tgt}
                     PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        set_property(TARGET ${_player_tgt}
                     PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    endforeach()
else()
    message(STATUS "player: IPO/LTO unavailable; release builds will not use it "
                   "(${player_ipo_error})")
endif()

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
