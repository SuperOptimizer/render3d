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
# name : source : extra slangc defines
# per-mode specialization (register pressure: see docs/measured.md 2026-08-04)
set(_variants
  "raycast_cube:raycast:-DR3D_MODE=0 -DWG_X=16 -DWG_Y=8"
  "raycast_slab:raycast:-DR3D_MODE=1 -DWG_X=16 -DWG_Y=8"
  "raycast_clip:raycast:-DR3D_MODE=2 -DWG_X=16 -DWG_Y=8"
  "raycast_bricks:raycast:-DR3D_MODE=3 -DWG_X=16 -DWG_Y=8"
  "raycast_vslab:raycast:-DR3D_MODE=4 -DWG_X=16 -DWG_Y=8"
  "raycast_8x8:raycast:-DR3D_MODE=0 -DWG_X=8 -DWG_Y=8"
  "raycast_16x16:raycast:-DR3D_MODE=0 -DWG_X=16 -DWG_Y=16")
foreach(_v ${_variants})
  string(REPLACE ":" ";" _parts "${_v}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _shader)
  list(GET _parts 2 _defs)
  separate_arguments(_defs)
  set(_src "${_shader_dir}/${_shader}.slang")
  set(_out "${_spv_dir}/${_name}.spv")
  add_custom_command(
    OUTPUT "${_out}"
    COMMAND "${R3D_SLANGC}" "${_src}" -target spirv -profile spirv_1_5 -O2
            -entry main ${_defs} -o "${_out}"
    DEPENDS "${_src}" "${_shader_dir}/common.slang"
    COMMENT "slangc ${_shader}.slang -> ${_name}.spv"
    VERBATIM)
  list(APPEND _spv_outputs "${_out}")
endforeach()
# c5d decode kernels (GLSL, from the compressor working tree) + our pack kernel
find_program(R3D_GLSLC NAMES glslc REQUIRED)
foreach(_ck entropy dequant_idct deblock corrections)
  set(_src "${R3D_C5D_DIR}/src/gpu/kernels/${_ck}.comp")
  set(_out "${_spv_dir}/c5d_${_ck}.spv")
  add_custom_command(
    OUTPUT "${_out}"
    COMMAND "${R3D_GLSLC}" -O --target-env=vulkan1.1 -o "${_out}" "${_src}"
    DEPENDS "${_src}"
    COMMENT "glslc c5d ${_ck}.comp"
    VERBATIM)
  list(APPEND _spv_outputs "${_out}")
endforeach()
foreach(_gk pack occmax occdilate apron)
  add_custom_command(
    OUTPUT "${_spv_dir}/${_gk}.spv"
    COMMAND "${R3D_GLSLC}" -O --target-env=vulkan1.1 -o "${_spv_dir}/${_gk}.spv"
            "${_shader_dir}/${_gk}.comp"
    DEPENDS "${_shader_dir}/${_gk}.comp"
    COMMENT "glslc ${_gk}.comp"
    VERBATIM)
  list(APPEND _spv_outputs "${_spv_dir}/${_gk}.spv")
endforeach()

add_custom_target(r3d_shaders DEPENDS ${_spv_outputs})

set(R3D_SPV_DIR "${_spv_dir}")
