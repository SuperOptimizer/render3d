cmake_minimum_required(VERSION 3.16)
project(dct3d_export C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math")

find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

add_executable(dct3d_export
  main.c
  shard.c
  fetch.c
  oracle.c
  upload.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../../dct3d.c
  vendor/libs3.c
)
target_include_directories(dct3d_export PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor
)
target_link_libraries(dct3d_export PRIVATE CURL::libcurl Threads::Threads m)

# mimalloc: replace the system allocator to cut malloc/free churn and page-fault
# overhead on the 1 GiB shard buffers. Prefer a CMake package; fall back to a
# plain libmimalloc; skip cleanly if neither is installed (build still works).
find_package(mimalloc QUIET)
if(mimalloc_FOUND)
  target_link_libraries(dct3d_export PRIVATE mimalloc)
  message(STATUS "dct3d_export: using mimalloc (package)")
else()
  find_library(MIMALLOC_LIB NAMES mimalloc mimalloc-secure)
  if(MIMALLOC_LIB)
    target_link_libraries(dct3d_export PRIVATE ${MIMALLOC_LIB})
    message(STATUS "dct3d_export: using mimalloc (${MIMALLOC_LIB})")
  else()
    message(WARNING "dct3d_export: mimalloc not found — using system allocator")
  endif()
endif()

# gcc+glibc auto-vectorizes powf under -ffast-math into a libmvec call; link it.
include(CheckCSourceCompiles)
if(NOT WIN32)
  find_library(MVEC_LIB mvec)
  if(MVEC_LIB)
    target_link_libraries(dct3d_export PRIVATE ${MVEC_LIB})
  endif()
endif()
