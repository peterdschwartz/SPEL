cmake_minimum_required(VERSION 3.18)
project(spel_ftorch LANGUAGES Fortran C)

find_package(FTorch REQUIRED)

enable_language(Fortran)

set(CMAKE_Fortran_STANDARD 2008)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

add_library(spel_emulator
    src/emulator.F90
    src/field_mod.F90
    src/kinds.F90
)

target_link_libraries(spel_emulator
    PUBLIC
    FTorch::ftorch
)

set(_spel_emulator_mods "${CMAKE_CURRENT_BINARY_DIR}/modules")
# Place generated Fortran .mod files in a predictable directory so that
# downstream Fortran targets (e.g. the driver test) can find them.
set_target_properties(spel_emulator PROPERTIES
  Fortran_MODULE_DIRECTORY "${_spel_emulator_mods}")


target_include_directories(spel_emulator PUBLIC
    $<BUILD_INTERFACE:${_spel_emulator_mods}>
)

enable_testing()

add_executable(test_tiny
    tests/test_tiny_emulator.F90
)

target_link_libraries(test_tiny
    PRIVATE
    spel_emulator
)

add_test(
    NAME test_tiny
    COMMAND test_tiny
)
