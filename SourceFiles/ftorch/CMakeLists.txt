cmake_minimum_required(VERSION 3.16)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(spel_ftorch_demo LANGUAGES Fortran C)

# Assume FTorch is installed and provides a CMake package config.
# Adjust package name / dirs to match your setup.
find_package(FTORCH REQUIRED)

enable_language(Fortran)

set(CMAKE_Fortran_STANDARD 2008)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

add_library(spel_emulator
    src/spel_emulator.F90
)

target_link_libraries(spel_emulator
    PRIVATE
        FTORCH::ftorch
)

add_executable(test_spel_emulator
    src/test_spel_emulator.F90
)

target_link_libraries(test_spel_emulator
    PRIVATE
        spel_emulator
        FTORCH::ftorch
)

# Optional: where to place binaries
set_target_properties(test_spel_emulator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
