cmake_minimum_required(VERSION 3.16)

project(ml-core
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Emit compile_commands.json for editor/tooling support.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to a Debug build when nothing was specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

option(ML_CORE_BUILD_TESTS "Build the ml-core test suite" ON)
option(ML_CORE_BUILD_BENCHMARKS "Build the ml-core benchmarks" OFF)

add_subdirectory(src)

if(ML_CORE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ML_CORE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
