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

find_package(SDL3 REQUIRED CONFIG)
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(TIFF REQUIRED) # tifxyz segment surfaces (BigTIFF/LZW/predictor-3)
find_package(ZLIB REQUIRED) # grayscale PNG writer (ink maps, rendseg)
find_library(BLOSC_LIB blosc REQUIRED)
find_path(BLOSC_INCLUDE_DIR blosc.h REQUIRED)
include_directories(SYSTEM ${BLOSC_INCLUDE_DIR}) # zarr v2 chunk decode (zarr2volcomp + net ingest)

# --- core (backend-agnostic: math, camera, volume, transfer, stats) ---
add_library(r3d_core STATIC
  src/core/stats.c
  src/core/volume.c
  src/core/camera.c
  src/core/transfer.c
  src/core/umbilicus.c
  src/core/annot.c
  src/core/input.c
  src/core/tifxyz.c
  src/core/segtrace.c
  src/core/segstore.c
  src/core/cpuvol.c
  src/core/pngw.c
  src/core/tracer.c
  src/core/bsurf.c
  src/core/labelvol.c
  src/core/regvol.c
  src/core/flatten.c
  src/core/inklive.c
  src/core/surfpred.c
  src/core/odbrowse.c
  src/core/screenshot.c)
# hot numeric TUs: fast-math (finite data by construction; the tracer's
# guards are explicit range checks, not NaN probes)
set_source_files_properties(src/core/tracer.c src/core/cpuvol.c
  src/core/flatten.c src/core/bsurf.c
  PROPERTIES COMPILE_OPTIONS "-ffast-math;-funroll-loops")
target_include_directories(r3d_core PUBLIC src)
target_link_libraries(r3d_core PRIVATE r3d_warnings r3d_sanitize
                               PUBLIC m Threads::Threads SDL3::SDL3 TIFF::TIFF
                                      CURL::libcurl ZLIB::ZLIB ${BLOSC_LIB})

# --- volume-compressor (pinned portable CPU codec) ---
include(FetchContent)
set(R3D_VOLCOMP_PIN "70320c58c922c6c3bcd62f9c209eeb3fa3a332b8")
set(R3D_VOLCOMP_DIR "" CACHE PATH "Local volume-compressor source override")
if(NOT R3D_VOLCOMP_DIR)
  FetchContent_Declare(r3d_volcomp_source
    GIT_REPOSITORY https://github.com/SuperOptimizer/volume-compressor.git
    GIT_TAG ${R3D_VOLCOMP_PIN}
    SOURCE_SUBDIR __render3d_dependency_sources_only)
  FetchContent_MakeAvailable(r3d_volcomp_source)
  set(R3D_VOLCOMP_DIR "${r3d_volcomp_source_SOURCE_DIR}")
endif()
if(NOT EXISTS "${R3D_VOLCOMP_DIR}/volcomp.h")
  message(FATAL_ERROR "volume-compressor headers missing at ${R3D_VOLCOMP_DIR}")
endif()
set(R3D_VOLCOMP_REV "local override")
execute_process(COMMAND git -C "${R3D_VOLCOMP_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE R3D_VOLCOMP_REV OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

# --- fysics registration kernels (pinned source dependency) ---
# Volume registration (core/regvol.c): affine resampler, coarse-to-fine NCC
# registration, sub-voxel phase correlation. Same pinning discipline as volcomp.
set(R3D_FYSICS_PIN "cf9b2c482996e904fc537876ce8d46f072e0b91f")
set(R3D_FYSICS_DIR "" CACHE PATH "Override the pinned fysics checkout")
if(NOT R3D_FYSICS_DIR)
  include(FetchContent)
  FetchContent_Declare(r3d_fysics_source
    GIT_REPOSITORY https://github.com/SuperOptimizer/fysics.git
    GIT_TAG ${R3D_FYSICS_PIN}
    GIT_SHALLOW FALSE
    SOURCE_SUBDIR __render3d_dependency_sources_only)
  FetchContent_MakeAvailable(r3d_fysics_source)
  set(R3D_FYSICS_DIR "${r3d_fysics_source_SOURCE_DIR}")
endif()
if(NOT EXISTS "${R3D_FYSICS_DIR}/fysics.h")
  message(FATAL_ERROR "fysics not found at ${R3D_FYSICS_DIR} (set R3D_FYSICS_DIR)")
endif()
add_library(r3d_fysics STATIC
  ${R3D_FYSICS_DIR}/register.c
  ${R3D_FYSICS_DIR}/phasecorr.c
  ${R3D_FYSICS_DIR}/fft.c)
target_include_directories(r3d_fysics SYSTEM PUBLIC ${R3D_FYSICS_DIR})
target_compile_options(r3d_fysics PRIVATE -ffast-math -fno-math-errno -O3)
target_link_libraries(r3d_fysics PUBLIC m PRIVATE r3d_sanitize)
add_library(r3d_volcomp STATIC src/codec/brick.c src/codec/shard.c
  src/codec/label.c src/codec/tifxyz.c)
target_include_directories(r3d_volcomp PUBLIC src/codec)
target_include_directories(r3d_volcomp SYSTEM PRIVATE ${R3D_VOLCOMP_DIR})
target_compile_definitions(r3d_volcomp PUBLIC R3D_VOLCOMP_REV="${R3D_VOLCOMP_REV}")
target_link_libraries(r3d_volcomp PUBLIC m ZLIB::ZLIB TIFF::TIFF Threads::Threads PRIVATE r3d_warnings r3d_sanitize)
target_link_libraries(r3d_core PUBLIC r3d_volcomp r3d_fysics)

# --- stable headless ABI (no viewer/Vulkan dependency) ---
add_library(render3d_headless SHARED src/headless/headless.c)
target_include_directories(render3d_headless PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(render3d_headless PRIVATE src)
target_link_libraries(render3d_headless PRIVATE r3d_core r3d_volcomp r3d_warnings r3d_sanitize)
set_target_properties(render3d_headless PROPERTIES
  VERSION 2.0.0 SOVERSION 2 C_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
include(GNUInstallDirs)
install(TARGETS render3d_headless
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/render3d/headless.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/render3d)

# --- shard store: vendored dct3d codec + zarr-v3 shard reader ---
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/tools/3ddct/dct3d.c")
  message(FATAL_ERROR "3ddct not vendored — run tools/fetch_3ddct.sh first")
endif()
add_library(r3d_shard STATIC
  tools/3ddct/dct3d.c
  src/shard/shardio.c)
target_include_directories(r3d_shard PUBLIC src)
target_include_directories(r3d_shard SYSTEM PUBLIC tools/3ddct)
# codec kernels are fast-math-first (volcomp convention); shardio itself is exact
set_source_files_properties(tools/3ddct/dct3d.c PROPERTIES
  COMPILE_OPTIONS "-ffast-math;-fno-math-errno;-O3;-Wno-nan-infinity-disabled")
target_link_libraries(r3d_shard PUBLIC m Threads::Threads
                                PRIVATE r3d_sanitize)

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
  src/vk/vkgui.c)
target_include_directories(r3d_vk PUBLIC src)
target_link_libraries(r3d_vk PUBLIC Vulkan::Vulkan SDL3::SDL3 CURL::libcurl r3d_imgui r3d_shard r3d_volcomp
                             PRIVATE ${BLOSC_LIB} r3d_warnings r3d_sanitize)

# --- shaders (Slang -> SPIR-V) ---
include(src/shaders/shaders.cmake)

# --- app ---
add_executable(render3d src/main.c src/annotui.c)
target_compile_definitions(render3d PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(render3d r3d_shaders)
target_link_libraries(render3d PRIVATE r3d_core r3d_vk r3d_imgui SDL3::SDL3
                                       r3d_warnings r3d_sanitize)

# --- tools ---
add_executable(assemble tools/assemble/main.c)
target_link_libraries(assemble PRIVATE m r3d_warnings r3d_sanitize)
add_executable(volcomppack tools/volcomppack/main.c)
target_link_libraries(volcomppack PRIVATE r3d_volcomp r3d_shard r3d_warnings r3d_sanitize)
add_executable(lodpack tools/lodpack/main.c)
target_link_libraries(lodpack PRIVATE r3d_volcomp r3d_shard r3d_warnings r3d_sanitize)
add_executable(lodcheck tools/lodcheck/main.c)
target_link_libraries(lodcheck PRIVATE r3d_volcomp r3d_shard r3d_warnings r3d_sanitize)
add_executable(mkpyramid tools/mkpyramid/main.c)
target_link_libraries(mkpyramid PRIVATE r3d_shard r3d_warnings r3d_sanitize)
add_executable(bandcut tools/bandcut/main.c)
target_link_libraries(bandcut PRIVATE r3d_shard r3d_warnings r3d_sanitize)
add_executable(mkvol tools/mkvol/main.c)
target_link_libraries(mkvol PRIVATE m r3d_warnings r3d_sanitize)
add_executable(tifxyz-transform tools/tifxyz-transform/main.c)
target_link_libraries(tifxyz-transform PRIVATE r3d_core m r3d_warnings r3d_sanitize)
add_executable(segpack tools/segpack/main.c)
target_link_libraries(segpack PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_executable(mkumb tools/mkumb/main.c)
target_link_libraries(mkumb PRIVATE r3d_core r3d_warnings r3d_sanitize)

add_executable(tracecli tools/tracecli/main.c)
target_link_libraries(tracecli PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_executable(rendseg tools/rendseg/main.c)
target_link_libraries(rendseg PRIVATE r3d_core z m r3d_warnings r3d_sanitize)
add_executable(zarr2volcomp tools/zarr2volcomp/main.c)
target_link_libraries(zarr2volcomp PRIVATE r3d_core r3d_volcomp ${BLOSC_LIB} CURL::libcurl
                                       r3d_warnings r3d_sanitize)

# --- tests ---
enable_testing()
add_executable(test_headless tests/test_headless.c)
target_link_libraries(test_headless PRIVATE render3d_headless m r3d_warnings r3d_sanitize)
add_test(NAME headless COMMAND test_headless)
set_tests_properties(headless PROPERTIES LABELS quick TIMEOUT 30)

add_executable(test_annot tests/test_annot.c)
target_link_libraries(test_annot PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME annot COMMAND test_annot $<TARGET_FILE:render3d>)
set_tests_properties(annot PROPERTIES LABELS quick TIMEOUT 60)

add_executable(test_quick tests/test_quick.c)
target_link_libraries(test_quick PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME quick COMMAND test_quick)
set_tests_properties(quick PROPERTIES LABELS quick TIMEOUT 30)

add_executable(test_persist tests/test_persist.c)
target_link_libraries(test_persist PRIVATE r3d_core r3d_volcomp r3d_warnings r3d_sanitize)
add_test(NAME persist COMMAND test_persist)
set_tests_properties(persist PROPERTIES LABELS quick TIMEOUT 300)

add_executable(test_stack tests/test_stack.c)
target_link_libraries(test_stack PRIVATE r3d_core r3d_volcomp r3d_warnings r3d_sanitize)
add_test(NAME stack COMMAND test_stack $<TARGET_FILE:rendseg>)
set_tests_properties(stack PROPERTIES LABELS quick TIMEOUT 120)

add_executable(test_ingest tests/test_ingest.c)
target_link_libraries(test_ingest PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME ingest COMMAND test_ingest)
set_tests_properties(ingest PROPERTIES LABELS quick TIMEOUT 120)

add_executable(test_odlist tests/test_odlist.c)
target_link_libraries(test_odlist PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME odlist COMMAND test_odlist)
set_tests_properties(odlist PROPERTIES LABELS quick TIMEOUT 60)

add_executable(test_zarr2volcomp tests/test_zarr2volcomp.c)
target_link_libraries(test_zarr2volcomp PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME zarr2volcomp COMMAND test_zarr2volcomp $<TARGET_FILE:zarr2volcomp>)
set_tests_properties(zarr2volcomp PROPERTIES LABELS quick TIMEOUT 180)

add_executable(test_overlay tests/test_overlay.c)
target_link_libraries(test_overlay PRIVATE r3d_volcomp r3d_warnings r3d_sanitize m)
add_test(NAME overlay COMMAND test_overlay $<TARGET_FILE:render3d>)
set_tests_properties(overlay PROPERTIES LABELS gpu TIMEOUT 300 SKIP_RETURN_CODE 77
  ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/lsan.supp")

add_executable(test_exercise tests/test_exercise.c)
target_link_libraries(test_exercise PRIVATE r3d_volcomp r3d_warnings r3d_sanitize m)
add_test(NAME exercise COMMAND test_exercise $<TARGET_FILE:render3d>)
set_tests_properties(exercise PROPERTIES LABELS gpu TIMEOUT 600 SKIP_RETURN_CODE 77
  ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/lsan.supp")

add_executable(test_flatpane tests/test_flatpane.c)
target_link_libraries(test_flatpane PRIVATE r3d_core r3d_volcomp r3d_warnings r3d_sanitize)
add_test(NAME flatpane COMMAND test_flatpane $<TARGET_FILE:render3d>)
set_tests_properties(flatpane PROPERTIES LABELS gpu TIMEOUT 300 SKIP_RETURN_CODE 77
  ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/lsan.supp")

add_executable(test_netstream tests/test_netstream.c)
target_link_libraries(test_netstream PRIVATE r3d_volcomp r3d_warnings r3d_sanitize m)
add_test(NAME netstream COMMAND test_netstream $<TARGET_FILE:render3d>)
set_tests_properties(netstream PROPERTIES LABELS gpu TIMEOUT 300 SKIP_RETURN_CODE 77
  ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/lsan.supp")

add_executable(test_postfilt tests/test_postfilt.c)
target_link_libraries(test_postfilt PRIVATE r3d_volcomp r3d_warnings r3d_sanitize m)
add_test(NAME postfilt COMMAND test_postfilt $<TARGET_FILE:render3d>)
set_tests_properties(postfilt PROPERTIES LABELS gpu TIMEOUT 600 SKIP_RETURN_CODE 77
  ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/lsan.supp")

add_executable(test_seedreload tests/test_seedreload.c)
target_link_libraries(test_seedreload PRIVATE r3d_volcomp r3d_warnings r3d_sanitize m)
add_test(NAME seedreload COMMAND test_seedreload $<TARGET_FILE:render3d>)
set_tests_properties(seedreload PROPERTIES LABELS gpu TIMEOUT 300 SKIP_RETURN_CODE 77
  ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/lsan.supp")

add_executable(test_shard_safety tests/test_shard_safety.c)
target_link_libraries(test_shard_safety PRIVATE r3d_shard r3d_warnings r3d_sanitize)
add_test(NAME shard_safety COMMAND test_shard_safety)
set_tests_properties(shard_safety PROPERTIES LABELS quick TIMEOUT 30)

add_executable(test_shard tests/test_shard.c)
target_link_libraries(test_shard PRIVATE r3d_shard r3d_warnings r3d_sanitize)
add_test(NAME shard COMMAND test_shard ${CMAKE_SOURCE_DIR}/band)
set_tests_properties(shard PROPERTIES LABELS data TIMEOUT 60 SKIP_RETURN_CODE 77)

add_executable(test_gpu tests/test_gpu.c)
target_link_libraries(test_gpu PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME gpu COMMAND test_gpu $<TARGET_FILE:render3d>)
set_tests_properties(gpu PROPERTIES LABELS gpu TIMEOUT 120 SKIP_RETURN_CODE 77
  ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/lsan.supp")

add_executable(test_blocks tests/test_blocks.c)
target_include_directories(test_blocks SYSTEM PRIVATE ${R3D_VOLCOMP_DIR})
target_link_libraries(test_blocks PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME blocks COMMAND test_blocks)
set_tests_properties(blocks PROPERTIES LABELS quick TIMEOUT 60)

add_executable(test_block_atlas tests/test_block_atlas.c)
target_link_libraries(test_block_atlas PRIVATE r3d_core r3d_vk r3d_warnings r3d_sanitize)
target_compile_definitions(test_block_atlas PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(test_block_atlas r3d_shaders)
add_test(NAME block_atlas COMMAND test_block_atlas)
set_tests_properties(block_atlas PROPERTIES LABELS gpu TIMEOUT 60 SKIP_RETURN_CODE 77)
