cmake_minimum_required(VERSION 3.28)
project(render3d VERSION 0.0.1 LANGUAGES C CXX)
# C is the project language; CXX exists solely for the vendored imgui/cimgui.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_definitions(_GNU_SOURCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
  message(WARNING "render3d targets clang/LLVM; got ${CMAKE_C_COMPILER_ID}")
endif()

option(R3D_WERROR "Treat warnings as errors" ON)
set(R3D_SANITIZE "" CACHE STRING "Sanitizer set: '' | asan-ubsan | tsan | msan")

add_library(r3d_warnings INTERFACE)
target_compile_options(r3d_warnings INTERFACE
  -Wall -Wextra -Wconversion -Wshadow -Wimplicit-fallthrough
  -Wformat=2 -Wvla -Wdouble-promotion
  $<$<BOOL:${R3D_WERROR}>:-Werror>)

add_library(r3d_sanitize INTERFACE)
if(R3D_SANITIZE STREQUAL "asan-ubsan")
  target_compile_options(r3d_sanitize INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer
    -fsanitize-address-use-after-scope)
  target_link_options(r3d_sanitize INTERFACE -fsanitize=address,undefined)
elseif(R3D_SANITIZE STREQUAL "tsan")
  target_compile_options(r3d_sanitize INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(r3d_sanitize INTERFACE -fsanitize=thread)
elseif(R3D_SANITIZE STREQUAL "msan")
  target_compile_options(r3d_sanitize INTERFACE
    -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer)
  target_link_options(r3d_sanitize INTERFACE -fsanitize=memory)
endif()

# Hardening for release binaries (OpenSSF baseline; skipped under sanitizers).
add_library(r3d_harden INTERFACE)
if(R3D_SANITIZE STREQUAL "" AND CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(r3d_harden INTERFACE
    -D_FORTIFY_SOURCE=3 -fstack-protector-strong -fstack-clash-protection
    -fstrict-flex-arrays=3 -mbranch-protection=standard)
  target_link_options(r3d_harden INTERFACE
    -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -Wl,--as-needed)
endif()

find_package(SDL3 REQUIRED CONFIG)
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# --- core (backend-agnostic: math, camera, volume, transfer, stats) ---
add_library(r3d_core STATIC
  src/core/stats.c
  src/core/volume.c
  src/core/camera.c
  src/core/transfer.c
  src/core/input.c
  src/core/screenshot.c)
target_include_directories(r3d_core PUBLIC src)
target_link_libraries(r3d_core PRIVATE r3d_warnings r3d_sanitize r3d_harden
                               PUBLIC m Threads::Threads SDL3::SDL3)

# --- c5d codec (sibling working tree, read-only dependency) ---
# Sources compiled directly (not add_subdirectory) to keep test/target
# namespaces separate; list mirrors the c5d lib in ${R3D_C5D_DIR}/CMakeLists.
set(R3D_C5D_DIR "$ENV{HOME}/compressor" CACHE PATH "c5d compressor checkout")
if(NOT EXISTS "${R3D_C5D_DIR}/src/brick.h")
  message(FATAL_ERROR "c5d not found at ${R3D_C5D_DIR} (set R3D_C5D_DIR)")
endif()
add_library(r3d_c5d STATIC
  ${R3D_C5D_DIR}/src/common/c5d_core.c
  ${R3D_C5D_DIR}/src/common/pool.c
  ${R3D_C5D_DIR}/src/transform/dct16.c
  ${R3D_C5D_DIR}/src/entropy/rans.c
  ${R3D_C5D_DIR}/src/entropy/priors.c
  ${R3D_C5D_DIR}/src/brick.c
  ${R3D_C5D_DIR}/src/gpu/host_entropy.c
  ${R3D_C5D_DIR}/src/shard.c
  ${R3D_C5D_DIR}/src/cache.c)
target_include_directories(r3d_c5d SYSTEM PUBLIC ${R3D_C5D_DIR}/src)
target_compile_options(r3d_c5d PRIVATE -ffast-math -fno-math-errno -O3)
target_link_libraries(r3d_c5d PUBLIC m Threads::Threads PRIVATE r3d_sanitize)

# --- shard store: vendored dct3d codec + zarr-v3 shard reader ---
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/tools/3ddct/dct3d.c")
  message(FATAL_ERROR "3ddct not vendored — run tools/fetch_3ddct.sh first")
endif()
add_library(r3d_shard STATIC
  tools/3ddct/dct3d.c
  src/shard/shardio.c)
target_include_directories(r3d_shard PUBLIC src)
target_include_directories(r3d_shard SYSTEM PUBLIC tools/3ddct)
# codec kernels are fast-math-first (c5d convention); shardio itself is exact
set_source_files_properties(tools/3ddct/dct3d.c PROPERTIES
  COMPILE_OPTIONS "-ffast-math;-fno-math-errno;-O3;-Wno-nan-infinity-disabled")
target_link_libraries(r3d_shard PUBLIC m Threads::Threads
                                PRIVATE r3d_sanitize r3d_harden)

# --- vendored cimgui/imgui (C++; the only non-C code in the tree) ---
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/tools/cimgui/cimgui.cpp")
  message(FATAL_ERROR "cimgui not vendored — run tools/fetch_cimgui.sh first")
endif()
add_library(r3d_imgui STATIC
  tools/cimgui/cimgui.cpp
  tools/cimgui/cimgui_impl.cpp
  tools/cimgui/imgui/imgui.cpp
  tools/cimgui/imgui/imgui_demo.cpp
  tools/cimgui/imgui/imgui_draw.cpp
  tools/cimgui/imgui/imgui_tables.cpp
  tools/cimgui/imgui/imgui_widgets.cpp
  tools/cimgui/imgui/backends/imgui_impl_sdl3.cpp
  tools/cimgui/imgui/backends/imgui_impl_vulkan.cpp)
target_include_directories(r3d_imgui SYSTEM PUBLIC
  tools/cimgui tools/cimgui/imgui tools/cimgui/imgui/backends)
target_compile_definitions(r3d_imgui
  PRIVATE "IMGUI_IMPL_API=extern \"C\""
  PUBLIC CIMGUI_USE_SDL3 CIMGUI_USE_VULKAN IMGUI_DISABLE_OBSOLETE_FUNCTIONS=1
         $<$<COMPILE_LANGUAGE:C>:CIMGUI_DEFINE_ENUMS_AND_STRUCTS=1>)
target_link_libraries(r3d_imgui PUBLIC SDL3::SDL3 Vulkan::Vulkan
                                PRIVATE r3d_sanitize)

# --- vulkan backend ---
add_library(r3d_vk STATIC
  src/vk/vkctx.c
  src/vk/vkswap.c
  src/vk/vkres.c
  src/vk/vkbackend.c
  src/vk/vkclip.c
  src/vk/vkc5d.c
  src/vk/vkgui.c)
target_include_directories(r3d_vk PUBLIC src)
target_link_libraries(r3d_vk PUBLIC Vulkan::Vulkan SDL3::SDL3 r3d_imgui r3d_shard r3d_c5d
                             PRIVATE r3d_warnings r3d_sanitize r3d_harden)

# --- shaders (Slang -> SPIR-V) ---
include(src/shaders/shaders.cmake)

# --- app ---
add_executable(render3d src/main.c)
target_compile_definitions(render3d PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(render3d r3d_shaders)
target_link_libraries(render3d PRIVATE r3d_core r3d_vk SDL3::SDL3
                                       r3d_warnings r3d_sanitize r3d_harden)

# --- tools ---
add_executable(assemble tools/assemble/main.c)
target_link_libraries(assemble PRIVATE m r3d_warnings r3d_sanitize r3d_harden)
add_executable(c5dpack tools/c5dpack/main.c)
target_link_libraries(c5dpack PRIVATE r3d_c5d r3d_shard r3d_warnings r3d_sanitize r3d_harden)
add_executable(mkpyramid tools/mkpyramid/main.c)
target_link_libraries(mkpyramid PRIVATE r3d_shard r3d_warnings r3d_sanitize r3d_harden)
add_executable(mkvol tools/mkvol/main.c)
target_link_libraries(mkvol PRIVATE m r3d_warnings r3d_sanitize r3d_harden)

# --- tests ---
enable_testing()
add_executable(test_quick tests/test_quick.c)
target_link_libraries(test_quick PRIVATE r3d_core r3d_warnings r3d_sanitize r3d_harden)
add_test(NAME quick COMMAND test_quick)
set_tests_properties(quick PROPERTIES LABELS quick TIMEOUT 30)

add_executable(test_shard tests/test_shard.c)
target_link_libraries(test_shard PRIVATE r3d_shard r3d_warnings r3d_sanitize r3d_harden)
add_test(NAME shard COMMAND test_shard ${CMAKE_SOURCE_DIR}/band)
set_tests_properties(shard PROPERTIES LABELS data TIMEOUT 60)

add_executable(test_c5dgpu tests/test_c5dgpu.c)
target_link_libraries(test_c5dgpu PRIVATE r3d_vk r3d_c5d r3d_warnings r3d_sanitize r3d_harden)
add_dependencies(test_c5dgpu r3d_shaders)
target_compile_definitions(test_c5dgpu PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_test(NAME c5dgpu COMMAND test_c5dgpu)
set_tests_properties(c5dgpu PROPERTIES LABELS gpu TIMEOUT 300
  ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/lsan.supp")

add_executable(test_gpu tests/test_gpu.c)
target_link_libraries(test_gpu PRIVATE r3d_core r3d_warnings r3d_sanitize r3d_harden)
add_test(NAME gpu COMMAND test_gpu $<TARGET_FILE:render3d>)
set_tests_properties(gpu PROPERTIES LABELS gpu TIMEOUT 120
  ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/lsan.supp")
