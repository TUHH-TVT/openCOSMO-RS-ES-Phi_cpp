cmake_minimum_required(VERSION 3.15)
project(openCOSMORS)
set(CMAKE_CXX_STANDARD 14)

# Release by default for single-configuration generators (Makefiles, Ninja)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The code selects its vector instructions with the compiler macros __SSE3__, __AVX__ and __FMA__,
# so the backend is chosen through the compiler flags. One of the three is required.
set(OPENCOMSORS_SIMD "AUTO" CACHE STRING "SIMD instruction set: AUTO, SSE3, AVX, FMA")
string(TOUPPER "${OPENCOMSORS_SIMD}" OPENCOMSORS_SIMD)
if(NOT OPENCOMSORS_SIMD MATCHES "^(AUTO|SSE3|AVX|FMA)$")
  message(FATAL_ERROR "OPENCOMSORS_SIMD must be one of AUTO, SSE3, AVX, FMA")
endif()
if(OPENCOMSORS_SIMD STREQUAL "AUTO")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    set(OPENCOMSORS_SELECTED_SIMD "FMA")
  else()
    message(FATAL_ERROR "Only x86 targets are supported (SSE3, AVX or FMA).")
  endif()
else()
  set(OPENCOMSORS_SELECTED_SIMD "${OPENCOMSORS_SIMD}")
endif()
message("openCOSMORS SIMD backend: ${OPENCOMSORS_SELECTED_SIMD}")

# Python module openCOSMORS
add_subdirectory(pybind11)
pybind11_add_module(openCOSMORS "${CMAKE_CURRENT_SOURCE_DIR}/code/bindings_forPython.cpp")
target_include_directories(openCOSMORS PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/code")
# the repository root provides <nlohmann/json.hpp>
target_include_directories(openCOSMORS SYSTEM PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/eigen"
)

if(OPENCOMSORS_SELECTED_SIMD STREQUAL "SSE3")
  if(MSVC)
    target_compile_definitions(openCOSMORS PRIVATE __SSE3__)
  else()
    target_compile_options(openCOSMORS PRIVATE -msse3)
  endif()
elseif(OPENCOMSORS_SELECTED_SIMD STREQUAL "AVX")
  if(MSVC)
    target_compile_options(openCOSMORS PRIVATE /arch:AVX)
  else()
    target_compile_options(openCOSMORS PRIVATE -mavx)
  endif()
elseif(OPENCOMSORS_SELECTED_SIMD STREQUAL "FMA")
  if(MSVC)
    target_compile_options(openCOSMORS PRIVATE /arch:AVX2)
  else()
    # AVX2 as with MSVC; the pinned Eigen does not compile with AVX but without AVX2
    target_compile_options(openCOSMORS PRIVATE -mavx2 -mfma)
  endif()
endif()

# OpenMP for release builds if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(openCOSMORS PUBLIC $<$<CONFIG:RELEASE>:OpenMP::OpenMP_CXX>)
else()
  message("OpenMP was not found, to improve performance for release builds use a compiler supporting it.")
endif()
