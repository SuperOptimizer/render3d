# Slang -> SPIR-V at build time (pattern: c5d src/gpu/gpu.cmake).
# slangc comes from tools/slang/bin (tools/fetch_slang.sh) or PATH.

find_program(R3D_SLANGC NAMES slangc
             HINTS "${CMAKE_SOURCE_DIR}/tools/slang/bin")
if(NOT R3D_SLANGC)
  message(FATAL_ERROR "slangc not found — run tools/fetch_slang.sh first")
endif()

set(_shader_dir "${CMAKE_CURRENT_LIST_DIR}")
set(_spv_dir "${CMAKE_BINARY_DIR}/spv")
file(MAKE_DIRECTORY "${_spv_dir}")

set(_spv_outputs "")
foreach(_shader raycast)
  set(_src "${_shader_dir}/${_shader}.slang")
  set(_out "${_spv_dir}/${_shader}.spv")
  add_custom_command(
    OUTPUT "${_out}"
    COMMAND "${R3D_SLANGC}" "${_src}" -target spirv -profile spirv_1_5 -O2
            -entry main -o "${_out}"
    DEPENDS "${_src}" "${_shader_dir}/common.slang"
    COMMENT "slangc ${_shader}.slang -> ${_shader}.spv"
    VERBATIM)
  list(APPEND _spv_outputs "${_out}")
endforeach()
add_custom_target(r3d_shaders DEPENDS ${_spv_outputs})

set(R3D_SPV_DIR "${_spv_dir}")
