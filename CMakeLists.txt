cmake_minimum_required(VERSION 3.20)
project(FiniteCore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Vulkan REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
find_package(glfw3 3.3 QUIET)

if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
  set(GLSLANG_VALIDATOR ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})
else()
  find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
endif()

# ---- shaders: GLSL -> SPIR-V embedded as C arrays ---------------------------------
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT ${CMAKE_CURRENT_BINARY_DIR}/generated/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT})
set(SHADERS
  explicit_step.comp cg_init.comp cg_init_check.comp cg_spmv.comp cg_update.comp
  cg_direction.comp apply_delta.comp
  field.vert field.frag line.vert line.frag text.vert text.frag
  surface.vert slice.vert scene.frag line3d.vert line3d.frag)
set(SHADER_HEADERS)
foreach(SH ${SHADERS})
  string(REPLACE "." "_" VAR ${SH})
  set(OUT ${SHADER_OUT}/${VAR}.h)
  add_custom_command(
    OUTPUT ${OUT}
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.3 --vn spv_${VAR} -o ${OUT} ${SHADER_DIR}/${SH}
    DEPENDS ${SHADER_DIR}/${SH} ${SHADER_DIR}/common.glsl ${SHADER_DIR}/colormaps.glsl ${SHADER_DIR}/scene.glsl
    COMMENT "Compiling shader ${SH}"
    VERBATIM)
  list(APPEND SHADER_HEADERS ${OUT})
endforeach()

# ---- executable ----------------------------------------------------------------------
add_executable(finitecore
  src/main.cpp
  src/config.cpp
  src/expr.cpp
  src/mesh.cpp
  src/problem.cpp
  src/fem.cpp
  src/cpu_solver.cpp
  src/gpu_solver.cpp
  src/renderer.cpp
  src/presenter.cpp
  src/image_io.cpp
  src/output.cpp
  src/simulation.cpp
  src/vk_context.cpp
  ${SHADER_HEADERS})
target_include_directories(finitecore PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(finitecore PRIVATE Vulkan::Vulkan ZLIB::ZLIB Threads::Threads)
if(MSVC)
  target_compile_options(finitecore PRIVATE /W4)
else()
  target_compile_options(finitecore PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
endif()

if(glfw3_FOUND)
  target_link_libraries(finitecore PRIVATE glfw)
  target_compile_definitions(finitecore PRIVATE FC_HAVE_GLFW=1)
  message(STATUS "FiniteCore: GLFW found - live window enabled")
else()
  message(STATUS "FiniteCore: GLFW not found - building headless only")
endif()
