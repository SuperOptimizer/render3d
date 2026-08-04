cmake_minimum_required(VERSION 3.16)
project(dct3d_downscale C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -march=native -funroll-loops")

find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

add_executable(dct3d_downscale
  main.c
  decode.c
  downsample.c
  srcfetch.c
  sftpget.c
  localio.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../export/shard.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../export/upload.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../../dct3d.c
)
target_include_directories(dct3d_downscale PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
target_link_libraries(dct3d_downscale PRIVATE CURL::libcurl Threads::Threads m)

find_package(mimalloc QUIET)
if(mimalloc_FOUND)
  target_link_libraries(dct3d_downscale PRIVATE mimalloc)
  message(STATUS "dct3d_downscale: using mimalloc (package)")
else()
  find_library(MIMALLOC_LIB NAMES mimalloc mimalloc-secure)
  if(MIMALLOC_LIB)
    target_link_libraries(dct3d_downscale PRIVATE ${MIMALLOC_LIB})
    message(STATUS "dct3d_downscale: using mimalloc (${MIMALLOC_LIB})")
  else()
    message(WARNING "dct3d_downscale: mimalloc not found — using system allocator")
  endif()
endif()

include(CheckCSourceCompiles)
if(NOT WIN32)
  find_library(MVEC_LIB mvec)
  if(MVEC_LIB)
    target_link_libraries(dct3d_downscale PRIVATE ${MVEC_LIB})
  endif()
endif()
