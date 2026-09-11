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

# Portable floating-point surface codec, pinned by snapshot.json so a clean
# checkout builds without fetching a separate repository.
set(R3D_SURFCOMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tools/surface-compressor" CACHE PATH
    "surface-compressor source override")
add_library(r3d_surfcomp STATIC ${R3D_SURFCOMP_DIR}/src/surfcomp.c)
target_include_directories(r3d_surfcomp SYSTEM PUBLIC ${R3D_SURFCOMP_DIR})
target_compile_options(r3d_surfcomp PRIVATE -ffast-math -fno-finite-math-only)
target_link_libraries(r3d_surfcomp PUBLIC m PRIVATE r3d_sanitize)

# --- core (backend-agnostic: math, camera, volume, transfer, stats) ---
add_library(r3d_core STATIC
  src/core/stats.c
  src/core/volume.c
  src/core/camera.c
  src/core/transfer.c
  src/core/umbilicus.c
  src/core/annot.c
  src/core/view.c
  src/core/input.c
  src/core/tifxyz.c
  src/core/surface.c
  src/core/surfconv.c
  src/core/segtrace.c
  src/core/segstore.c
  src/core/cpuvol.c
  src/core/deblock.c
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
                               PUBLIC r3d_surfcomp m Threads::Threads SDL3::SDL3 TIFF::TIFF
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
  src/codec/label.c src/codec/tifxyz.c src/codec/native_source.c)
target_include_directories(r3d_volcomp PUBLIC src/codec)
target_include_directories(r3d_volcomp SYSTEM PRIVATE ${R3D_VOLCOMP_DIR})
target_compile_definitions(r3d_volcomp PUBLIC R3D_VOLCOMP_REV="${R3D_VOLCOMP_REV}")
target_link_libraries(r3d_volcomp PUBLIC m ZLIB::ZLIB TIFF::TIFF Threads::Threads CURL::libcurl PRIVATE r3d_warnings r3d_sanitize)
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
target_link_libraries(r3d_vk PUBLIC Vulkan::Vulkan SDL3::SDL3 CURL::libcurl r3d_imgui r3d_shard r3d_volcomp r3d_core
                             PRIVATE ${BLOSC_LIB} r3d_warnings r3d_sanitize)

# --- shaders (Slang -> SPIR-V) ---
include(src/shaders/shaders.cmake)

# --- app ---
add_executable(render3d src/main.c src/annotui.c src/viewui.c)
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
add_executable(surfconv tools/surfconv/main.c)
target_link_libraries(surfconv PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_executable(surfrefine tools/surfrefine/main.c)
target_link_libraries(surfrefine PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_executable(surfsamples tools/surfrefine/surfsamples.c)
target_link_libraries(surfsamples PRIVATE r3d_core TIFF::TIFF m r3d_warnings r3d_sanitize)
add_executable(tifxyz2obj tools/surfrefine/tifxyz2obj.c)
target_link_libraries(tifxyz2obj PRIVATE r3d_core m r3d_warnings r3d_sanitize)
add_executable(tifxyz-resample tools/tifxyz-resample/main.c)
target_link_libraries(tifxyz-resample PRIVATE r3d_core TIFF::TIFF m r3d_warnings r3d_sanitize)
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
add_executable(test_blockmap tests/test_blockmap.c)
target_include_directories(test_blockmap PRIVATE src)
target_link_libraries(test_blockmap PRIVATE r3d_warnings r3d_sanitize)
add_test(NAME blockmap COMMAND test_blockmap)
set_tests_properties(blockmap PROPERTIES LABELS quick TIMEOUT 60)

add_executable(test_headless tests/test_headless.c)
target_link_libraries(test_headless PRIVATE render3d_headless m r3d_warnings r3d_sanitize)
add_test(NAME headless COMMAND test_headless)
set_tests_properties(headless PROPERTIES LABELS quick TIMEOUT 30)

add_executable(test_annot tests/test_annot.c)
target_link_libraries(test_annot PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME annot COMMAND test_annot $<TARGET_FILE:render3d>)
set_tests_properties(annot PROPERTIES LABELS quick TIMEOUT 60)
add_test(NAME annot_gui COMMAND test_annot $<TARGET_FILE:render3d> gui)
set_tests_properties(annot_gui PROPERTIES LABELS gpu TIMEOUT 300)

add_executable(test_view tests/test_view.c)
target_link_libraries(test_view PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME view COMMAND test_view $<TARGET_FILE:render3d>)
set_tests_properties(view PROPERTIES LABELS quick TIMEOUT 120)
add_test(NAME view_gui COMMAND test_view $<TARGET_FILE:render3d> gui)
set_tests_properties(view_gui PROPERTIES LABELS gpu TIMEOUT 300)
add_executable(test_surface_stream tests/test_surface_stream.c)
target_link_libraries(test_surface_stream PRIVATE render3d_headless r3d_volcomp r3d_warnings r3d_sanitize)
add_test(NAME surface_stream COMMAND test_surface_stream)
set_tests_properties(surface_stream PROPERTIES LABELS quick TIMEOUT 30)

add_executable(test_lodpack tests/test_lodpack.c)
target_link_libraries(test_lodpack PRIVATE r3d_volcomp r3d_shard r3d_warnings r3d_sanitize)
add_test(NAME lodpack COMMAND test_lodpack)
set_tests_properties(lodpack PROPERTIES LABELS quick TIMEOUT 60)

add_executable(test_headless_roi tests/test_headless_roi.c)
target_include_directories(test_headless_roi PRIVATE include src)
target_link_libraries(test_headless_roi PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME headless_roi COMMAND test_headless_roi)
set_tests_properties(headless_roi PROPERTIES LABELS quick TIMEOUT 30)

add_executable(test_reg_retry tests/test_reg_retry.c)
target_link_libraries(test_reg_retry PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME reg_retry COMMAND test_reg_retry)
set_tests_properties(reg_retry PROPERTIES LABELS quick TIMEOUT 30)

add_executable(test_app_perf tests/test_app_perf.c)
target_link_libraries(test_app_perf PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME app_perf COMMAND test_app_perf)
set_tests_properties(app_perf PROPERTIES LABELS quick TIMEOUT 60)

add_executable(test_app_ui tests/test_app_ui.c src/annotui.c src/viewui.c)
target_compile_definitions(test_app_ui PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(test_app_ui r3d_shaders)
target_link_libraries(test_app_ui PRIVATE r3d_core r3d_vk r3d_warnings r3d_sanitize)
add_test(NAME app_ui COMMAND test_app_ui)
set_tests_properties(app_ui PROPERTIES LABELS quick TIMEOUT 60)

foreach(tool_test packtools mkpyramid)
  add_executable(test_${tool_test} tests/test_${tool_test}.c)
  target_link_libraries(test_${tool_test} PRIVATE r3d_volcomp r3d_shard r3d_warnings r3d_sanitize)
  add_test(NAME ${tool_test} COMMAND test_${tool_test})
  set_tests_properties(${tool_test} PROPERTIES LABELS quick TIMEOUT 60)
endforeach()

add_executable(test_quick tests/test_quick.c)
target_link_libraries(test_quick PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME quick COMMAND test_quick)
set_tests_properties(quick PROPERTIES LABELS quick TIMEOUT 30)

add_executable(test_persist tests/test_persist.c)
target_link_libraries(test_persist PRIVATE r3d_core r3d_volcomp r3d_warnings r3d_sanitize)
add_test(NAME persist COMMAND test_persist)
set_tests_properties(persist PROPERTIES LABELS quick TIMEOUT 300)

add_executable(test_stack tests/test_stack.c)
add_dependencies(test_stack rendseg)
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
add_dependencies(test_zarr2volcomp zarr2volcomp)
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

add_executable(bench_blocks tools/bench_blocks/main.c)
target_include_directories(bench_blocks PRIVATE tests)
target_include_directories(bench_blocks SYSTEM PRIVATE ${R3D_VOLCOMP_DIR})
target_link_libraries(bench_blocks PRIVATE r3d_volcomp r3d_warnings r3d_sanitize)

add_executable(test_blocks tests/test_blocks.c)
target_include_directories(test_blocks SYSTEM PRIVATE ${R3D_VOLCOMP_DIR})
target_link_libraries(test_blocks PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME blocks COMMAND test_blocks)
set_tests_properties(blocks PROPERTIES LABELS quick TIMEOUT 60)

add_executable(test_block_atlas tests/test_block_atlas.c)
target_link_libraries(test_block_atlas PRIVATE r3d_vk r3d_core r3d_warnings r3d_sanitize)
target_compile_definitions(test_block_atlas PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(test_block_atlas r3d_shaders)
add_test(NAME block_atlas COMMAND test_block_atlas)

add_executable(test_viewcache tests/test_viewcache.c)
target_link_libraries(test_viewcache PRIVATE r3d_core r3d_vk r3d_warnings r3d_sanitize)
target_compile_definitions(test_viewcache PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(test_viewcache r3d_shaders)
add_test(NAME viewcache COMMAND test_viewcache)
set_tests_properties(viewcache PROPERTIES LABELS gpu TIMEOUT 120 SKIP_RETURN_CODE 77)

add_executable(test_pagehash_gpu tests/test_pagehash_gpu.c)
target_link_libraries(test_pagehash_gpu PRIVATE r3d_vk r3d_warnings r3d_sanitize)
target_compile_definitions(test_pagehash_gpu PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(test_pagehash_gpu r3d_shaders)
add_test(NAME pagehash_gpu COMMAND test_pagehash_gpu)
set_tests_properties(pagehash_gpu PROPERTIES LABELS gpu TIMEOUT 60 SKIP_RETURN_CODE 77)

add_executable(test_sparse_zero tests/test_sparse_zero.c)
target_link_libraries(test_sparse_zero PRIVATE r3d_core r3d_vk r3d_warnings r3d_sanitize)
target_compile_definitions(test_sparse_zero PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(test_sparse_zero r3d_shaders)
add_test(NAME sparse_zero COMMAND test_sparse_zero)
set_tests_properties(sparse_zero PROPERTIES LABELS gpu TIMEOUT 120 SKIP_RETURN_CODE 77)

add_executable(test_gpu_updates tests/test_gpu_updates.c)
target_link_libraries(test_gpu_updates PRIVATE r3d_core r3d_vk r3d_warnings r3d_sanitize)
target_compile_definitions(test_gpu_updates PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(test_gpu_updates r3d_shaders)
add_test(NAME gpu_updates COMMAND test_gpu_updates)
set_tests_properties(gpu_updates PROPERTIES LABELS gpu TIMEOUT 120 SKIP_RETURN_CODE 77)
set_tests_properties(block_atlas PROPERTIES LABELS gpu TIMEOUT 60 SKIP_RETURN_CODE 77)

add_executable(test_blockpost tests/test_blockpost.c)
target_link_libraries(test_blockpost PRIVATE r3d_vk r3d_warnings r3d_sanitize)
target_compile_definitions(test_blockpost PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(test_blockpost r3d_shaders)
add_test(NAME blockpost COMMAND test_blockpost)
set_tests_properties(blockpost PROPERTIES LABELS gpu TIMEOUT 60 SKIP_RETURN_CODE 77)

# Browser native-volume bootstrap lives beside the renderer and packer.
configure_file(tools/bootstrap_volcomp.py bootstrap_volcomp.py COPYONLY)
configure_file(tools/fetch_volcomp.py fetch_volcomp.py COPYONLY)
add_dependencies(render3d volcomppack zarr2volcomp)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_executable(native_source_probe tests/native_source_probe.c)
  target_link_libraries(native_source_probe PRIVATE r3d_core r3d_warnings r3d_sanitize)
  add_test(NAME native_source COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_native_source.py $<TARGET_FILE:native_source_probe>
    $<TARGET_FILE:volcomppack>)
  add_dependencies(native_source_probe volcomppack)
  set_tests_properties(native_source PROPERTIES LABELS quick TIMEOUT 60)
endif()

add_executable(test_dataset_swap tests/test_dataset_swap.c)
target_link_libraries(test_dataset_swap PRIVATE r3d_volcomp r3d_warnings r3d_sanitize)
add_dependencies(test_dataset_swap render3d)
add_test(NAME dataset_swap COMMAND test_dataset_swap $<TARGET_FILE:render3d>)
set_tests_properties(dataset_swap PROPERTIES LABELS gpu TIMEOUT 120 SKIP_RETURN_CODE 77)
target_link_libraries(test_blockmap PRIVATE Threads::Threads)

add_executable(test_deblock tests/test_deblock.c)
target_include_directories(test_deblock SYSTEM PRIVATE ${R3D_VOLCOMP_DIR})
target_link_libraries(test_deblock PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME deblock COMMAND test_deblock)
set_tests_properties(deblock PROPERTIES LABELS quick TIMEOUT 60)
add_executable(test_deblock_render tests/test_deblock_render.c)
target_link_libraries(test_deblock_render PRIVATE r3d_volcomp r3d_warnings r3d_sanitize)
add_dependencies(test_deblock_render render3d)
add_test(NAME deblock_render COMMAND test_deblock_render $<TARGET_FILE:render3d>)
set_tests_properties(deblock_render PROPERTIES LABELS gpu TIMEOUT 120 SKIP_RETURN_CODE 77)

add_executable(test_coarse_pin tests/test_coarse_pin.c)
target_link_libraries(test_coarse_pin PRIVATE r3d_vk r3d_warnings r3d_sanitize)
target_compile_definitions(test_coarse_pin PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(test_coarse_pin r3d_shaders)
add_test(NAME coarse_pin COMMAND test_coarse_pin)
set_tests_properties(coarse_pin PROPERTIES LABELS gpu TIMEOUT 60 SKIP_RETURN_CODE 77)

add_executable(test_lod_pipeline tests/test_lod_pipeline.c)
target_link_libraries(test_lod_pipeline PRIVATE r3d_volcomp r3d_warnings r3d_sanitize)
add_dependencies(test_lod_pipeline render3d)
add_test(NAME lod_pipeline COMMAND test_lod_pipeline $<TARGET_FILE:render3d>)
set_tests_properties(lod_pipeline PROPERTIES LABELS gpu TIMEOUT 120 SKIP_RETURN_CODE 77)

add_executable(test_lod_cache tests/test_lod_cache.c)
target_link_libraries(test_lod_cache PRIVATE r3d_vk r3d_warnings r3d_sanitize)
target_compile_definitions(test_lod_cache PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(test_lod_cache r3d_shaders)
add_test(NAME lod_cache COMMAND test_lod_cache)
set_tests_properties(lod_cache PROPERTIES LABELS gpu TIMEOUT 60 SKIP_RETURN_CODE 77)
add_test(NAME lod_cache_air COMMAND test_lod_cache)
set_tests_properties(lod_cache_air PROPERTIES LABELS gpu TIMEOUT 60 SKIP_RETURN_CODE 77
  ENVIRONMENT "R3D_TEST_COARSE_AIR=1")

add_executable(test_lod_gradient tests/test_lod_gradient.c)
target_link_libraries(test_lod_gradient PRIVATE r3d_vk r3d_warnings r3d_sanitize)
target_compile_definitions(test_lod_gradient PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(test_lod_gradient r3d_shaders)
add_test(NAME lod_gradient COMMAND test_lod_gradient)
set_tests_properties(lod_gradient PROPERTIES LABELS gpu TIMEOUT 60)

add_executable(test_stream_teardown tests/test_stream_teardown.c)
target_link_libraries(test_stream_teardown PRIVATE r3d_vk r3d_warnings r3d_sanitize)
target_compile_definitions(test_stream_teardown PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
add_dependencies(test_stream_teardown r3d_shaders)
add_test(NAME stream_teardown COMMAND test_stream_teardown)
set_tests_properties(stream_teardown PROPERTIES LABELS gpu TIMEOUT 60)

if(Python3_Interpreter_FOUND)
  add_executable(native_renderer_probe tests/native_renderer_probe.c)
  target_link_libraries(native_renderer_probe PRIVATE r3d_vk r3d_warnings r3d_sanitize)
  target_compile_definitions(native_renderer_probe PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
  add_dependencies(native_renderer_probe r3d_shaders)
  add_test(NAME native_renderer COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_native_source.py $<TARGET_FILE:native_source_probe>
    $<TARGET_FILE:volcomppack> $<TARGET_FILE:native_renderer_probe>)
  set_tests_properties(native_renderer PROPERTIES LABELS gpu TIMEOUT 60)
endif()

add_executable(test_decode_restart tests/test_decode_restart.c)
target_include_directories(test_decode_restart SYSTEM PRIVATE ${R3D_VOLCOMP_DIR})
target_link_libraries(test_decode_restart PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME decode_restart COMMAND test_decode_restart)
set_tests_properties(decode_restart PROPERTIES LABELS quick TIMEOUT 120)

if(Python3_Interpreter_FOUND)
  add_executable(inference_renderer_probe tests/inference_renderer_probe.c)
  target_link_libraries(inference_renderer_probe PRIVATE r3d_vk r3d_warnings r3d_sanitize)
  target_compile_definitions(inference_renderer_probe PRIVATE R3D_SPV_DIR="${R3D_SPV_DIR}")
  add_dependencies(inference_renderer_probe r3d_shaders)
  add_test(NAME inference_stream COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_inference_stream.py $<TARGET_FILE:inference_renderer_probe>)
  set_tests_properties(inference_stream PROPERTIES LABELS gpu TIMEOUT 30)
endif()

add_executable(test_surface_cache tests/test_surface_cache.c)
target_link_libraries(test_surface_cache PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME surface_cache COMMAND test_surface_cache)
set_tests_properties(surface_cache PROPERTIES LABELS quick)
add_executable(test_surfrefine_export tests/test_surfrefine_export.c)
target_link_libraries(test_surfrefine_export PRIVATE r3d_core TIFF::TIFF m r3d_warnings r3d_sanitize)
add_dependencies(test_surfrefine_export tifxyz2obj surfsamples)
add_test(NAME surfrefine_export COMMAND test_surfrefine_export $<TARGET_FILE:tifxyz2obj> $<TARGET_FILE:surfsamples>)
set_tests_properties(surfrefine_export PROPERTIES LABELS quick TIMEOUT 120)
add_executable(test_surfrefine tests/test_surfrefine.c)
target_link_libraries(test_surfrefine PRIVATE r3d_core r3d_volcomp TIFF::TIFF r3d_warnings r3d_sanitize)
add_dependencies(test_surfrefine surfrefine)
add_test(NAME surfrefine COMMAND test_surfrefine $<TARGET_FILE:surfrefine>)
set_tests_properties(surfrefine PROPERTIES LABELS quick TIMEOUT 300)
add_test(NAME surfrefine_gui COMMAND test_surfrefine $<TARGET_FILE:surfrefine> $<TARGET_FILE:render3d>)
set_tests_properties(surfrefine_gui PROPERTIES LABELS gpu TIMEOUT 600)

add_executable(test_tracer_group tests/test_tracer_group.c)
target_include_directories(test_tracer_group SYSTEM PRIVATE ${R3D_VOLCOMP_DIR})
target_link_libraries(test_tracer_group PRIVATE r3d_core r3d_warnings r3d_sanitize)
add_test(NAME tracer_group COMMAND test_tracer_group)
set_tests_properties(tracer_group PROPERTIES LABELS quick TIMEOUT 300)
add_executable(test_surfconv tests/test_surfconv.c)
target_link_libraries(test_surfconv PRIVATE r3d_core TIFF::TIFF r3d_warnings r3d_sanitize)
add_dependencies(test_surfconv surfconv)
add_test(NAME surfconv COMMAND test_surfconv $<TARGET_FILE:surfconv>)
set_tests_properties(surfconv PROPERTIES LABELS quick TIMEOUT 120)
