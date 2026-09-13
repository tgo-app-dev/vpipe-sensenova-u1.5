cmake_minimum_required(VERSION 3.20)
project(vpipe_sensenova_u1_5 LANGUAGES CXX)

# Build against an INSTALLED vpipe, e.g.
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/vpipe-install
#   cmake --build build -j
# then load it:
#   vpipe --plugin build/vpipe-sensenova-u1.5.so ...
#
# A plugin must be rebuilt against the vpipe it deploys with: the ABI
# handshake is strict equality, so a mismatch is refused at load rather
# than crashing. This plugin needs ABI 4 -- exactly, not "or newer".
# The number moves whenever the host's plugin surface does, and this
# plugin tracks it by REBUILDING rather than by source change: it is an
# image family with its own stage, so the host's video-registry structs
# never reach it.
find_package(vpipe REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- the plugin's own kernels ----------------------------------------
# Two dtype twins of one source, exactly as the in-tree tree does it. The
# backbone runs bf16; the f32 twin exists because the two scalar
# embedders and the pixel head are F32 in the checkpoint and stay F32 --
# they are a few hundred kilobytes and they drive every modulation and
# every output pixel.
vpipe_add_metal_library(u15_kernels_bf16_obj u15_kernels_bf16
    SRC src/u15-kernels.metal DEFINES VPIPE_ELT=bfloat)
vpipe_add_metal_library(u15_kernels_f32_obj u15_kernels_f32
    SRC src/u15-kernels.metal DEFINES VPIPE_ELT=float)

# ---- the model layer --------------------------------------------------
# One OBJECT library so every source is COMPILED even before it has a
# caller, and cannot rot silently.
add_library(u15_model_obj OBJECT
    src/u15-config.cc
    src/u15-ref.cc
    src/u15-weights.cc
    src/u15-metal-ops.cc
    src/u15-backbone.cc
    src/u15-prompt.cc
    src/u15-image.cc
    src/u15-generator.cc
    src/u15-model-config-stage.cc
    src/u15-generate-stage.cc
    src/u15-catalog.cc)
target_include_directories(u15_model_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(u15_model_obj PRIVATE vpipe::vpipe)
target_compile_features(u15_model_obj PRIVATE cxx_std_20)
set_target_properties(u15_model_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    # Same bar as the host tree: warnings are errors.
    target_compile_options(u15_model_obj PRIVATE -Wall -Werror)
endif()

# ---- the plugin -------------------------------------------------------
# No METAL / COREML flags: the plugin reaches for no Metal escape hatch.
# Its kernels are registered by NAME through the public runtime API and
# dispatched through libvpipe's ComputeEncoder, so the frameworks stay
# inside libvpipe.
vpipe_add_plugin(vpipe-sensenova-u1.5
    SOURCES
        src/u15-plugin.cc
        $<TARGET_OBJECTS:u15_model_obj>
        $<TARGET_OBJECTS:u15_kernels_bf16_obj>
        $<TARGET_OBJECTS:u15_kernels_f32_obj>
)
target_include_directories(vpipe-sensenova-u1.5 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
    target_compile_options(vpipe-sensenova-u1.5 PRIVATE -Wall -Werror)
endif()

# ---- tests ------------------------------------------------------------
enable_testing()

# The config + detection layer. Runs everywhere; the checkpoint half is
# gated on VPIPE_U15_TEST_MODEL_PATH and SKIPS when unset, so read the
# output rather than the exit code -- a vacuous skip looks like a pass.
add_executable(u15-config-test
    tests/u15-config-test.cc
    src/u15-config.cc)
target_include_directories(u15-config-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-config-test PRIVATE vpipe::vpipe)
target_compile_features(u15-config-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-config-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-config COMMAND u15-config-test)

# The CPU reference against the reference implementation's own goldens.
# Needs no vpipe symbols -- deliberately: this layer is pure arithmetic,
# so it builds and runs even where libvpipe cannot.
add_executable(u15-ref-test
    tests/u15-ref-test.cc
    src/u15-ref.cc)
target_include_directories(u15-ref-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_features(u15-ref-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-ref-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-ref COMMAND u15-ref-test)

# Binding the real checkpoint. Trunk + one layer only -- see the header
# comment for why one layer answers everything 42 would.
add_executable(u15-weights-test
    tests/u15-weights-test.cc
    src/u15-config.cc
    src/u15-weights.cc)
target_include_directories(u15-weights-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-weights-test PRIVATE vpipe::vpipe)
target_compile_features(u15-weights-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-weights-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-weights COMMAND u15-weights-test)

# The plugin's own kernels on the GPU, against the CPU reference. Links
# the bf16 metallib object directly and registers it through the same
# public API the plugin uses.
add_executable(u15-m5-test
    tests/u15-m5-test.cc
    src/u15-config.cc
    src/u15-metal-ops.cc
    $<TARGET_OBJECTS:u15_kernels_bf16_obj>)
target_include_directories(u15-m5-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-m5-test PRIVATE vpipe::vpipe)
target_compile_features(u15-m5-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-m5-test PRIVATE -Wall -Werror)
endif()

add_executable(u15-kernels-test
    tests/u15-kernels-test.cc
    src/u15-metal-ops.cc
    src/u15-ref.cc
    $<TARGET_OBJECTS:u15_kernels_bf16_obj>)
target_include_directories(u15-kernels-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-kernels-test PRIVATE vpipe::vpipe)
target_compile_features(u15-kernels-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-kernels-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-kernels COMMAND u15-kernels-test)

# SageAttention wired into the backbone. Asserts the PLAN rather than the
# arithmetic, so it needs no checkpoint -- and says on every run which
# half of itself the GPU let it check.
add_executable(u15-sage-test
    tests/u15-sage-test.cc
    src/u15-metal-ops.cc
    $<TARGET_OBJECTS:u15_kernels_bf16_obj>)
target_include_directories(u15-sage-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-sage-test PRIVATE vpipe::vpipe)
target_compile_features(u15-sage-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-sage-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-sage COMMAND u15-sage-test)

# Layer 0 of the REAL checkpoint on the GPU, against the CPU reference at
# full width. The check that says the whole chain holds.
add_executable(u15-layer0-test
    tests/u15-layer0-test.cc
    src/u15-config.cc
    src/u15-weights.cc
    src/u15-metal-ops.cc
    src/u15-backbone.cc
    src/u15-ref.cc
    $<TARGET_OBJECTS:u15_kernels_bf16_obj>)
target_include_directories(u15-layer0-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-layer0-test PRIVATE vpipe::vpipe)
target_compile_features(u15-layer0-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-layer0-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-layer0 COMMAND u15-layer0-test)

# Tokenisation + the chat template, against ids from the reference's own
# AutoTokenizer. Must be EXACT -- see the header comment.
add_executable(u15-prompt-test
    tests/u15-prompt-test.cc
    src/u15-prompt.cc)
target_include_directories(u15-prompt-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-prompt-test PRIVATE vpipe::vpipe)
target_compile_features(u15-prompt-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-prompt-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-prompt COMMAND u15-prompt-test)

# End to end: prompt in, image out, on the full 42-layer checkpoint.
# Loads ~35 GB -- slow, and gated on VPIPE_U15_TEST_MODEL_PATH.
add_executable(u15-generate-test
    tests/u15-generate-test.cc
    src/u15-config.cc
    src/u15-weights.cc
    src/u15-metal-ops.cc
    src/u15-backbone.cc
    src/u15-image.cc
    src/u15-prompt.cc
    src/u15-generator.cc
    src/u15-ref.cc
    $<TARGET_OBJECTS:u15_kernels_bf16_obj>)
target_include_directories(u15-generate-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-generate-test PRIVATE vpipe::vpipe)
target_compile_features(u15-generate-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-generate-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-generate COMMAND u15-generate-test)

# Image EDITING end to end. Generates its own reference image first, so
# the reference is in-distribution and the test is self-contained.
add_executable(u15-edit-test
    tests/u15-edit-test.cc
    src/u15-config.cc
    src/u15-weights.cc
    src/u15-metal-ops.cc
    src/u15-backbone.cc
    src/u15-image.cc
    src/u15-prompt.cc
    src/u15-generator.cc
    src/u15-ref.cc
    $<TARGET_OBJECTS:u15_kernels_bf16_obj>)
target_include_directories(u15-edit-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-edit-test PRIVATE vpipe::vpipe)
target_compile_features(u15-edit-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-edit-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-edit COMMAND u15-edit-test)

# Where the time goes, at the shapes a real render uses. Reports rates;
# asserts only the A/B facts (a fast kernel must not be slower than the
# one it replaces), because absolute speed is a property of the box.
add_executable(u15-perf-test
    tests/u15-perf-test.cc
    src/u15-metal-ops.cc
    $<TARGET_OBJECTS:u15_kernels_bf16_obj>)
target_include_directories(u15-perf-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-perf-test PRIVATE vpipe::vpipe)
target_compile_features(u15-perf-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-perf-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-perf COMMAND u15-perf-test)

# The patch embedder and the pixel head on the GPU with real weights,
# against the CPU reference. The last two GPU paths that were only ever
# exercised end to end.
add_executable(u15-image-test
    tests/u15-image-test.cc
    src/u15-config.cc
    src/u15-weights.cc
    src/u15-metal-ops.cc
    src/u15-image.cc
    src/u15-ref.cc
    $<TARGET_OBJECTS:u15_kernels_bf16_obj>)
target_include_directories(u15-image-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-image-test PRIVATE vpipe::vpipe)
target_compile_features(u15-image-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-image-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-image COMMAND u15-image-test)

# A QUANTIZED pack against the dense one it was made from. Needs both
# paths; renders once from each and compares, because "does it load" is
# not the question a quantization test has to answer.
add_executable(u15-quant-test
    tests/u15-quant-test.cc
    src/u15-config.cc
    src/u15-weights.cc
    src/u15-metal-ops.cc
    src/u15-backbone.cc
    src/u15-image.cc
    src/u15-prompt.cc
    src/u15-generator.cc
    src/u15-ref.cc
    $<TARGET_OBJECTS:u15_kernels_bf16_obj>)
target_include_directories(u15-quant-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-quant-test PRIVATE vpipe::vpipe)
target_compile_features(u15-quant-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-quant-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-quant COMMAND u15-quant-test)

# LAYER STREAMING: the same picture out of a fraction of the memory.
# Renders once preloaded and once streamed and requires the two to be
# BIT-IDENTICAL -- streaming moves where the bytes live and must change
# nothing else. Also exercises the stream/preload verdict at box sizes
# this machine may not have, via VPIPE_RAM_LIMIT_MB.
add_executable(u15-stream-test
    tests/u15-stream-test.cc
    src/u15-config.cc
    src/u15-weights.cc
    src/u15-metal-ops.cc
    src/u15-backbone.cc
    src/u15-image.cc
    src/u15-prompt.cc
    src/u15-generator.cc
    src/u15-ref.cc
    $<TARGET_OBJECTS:u15_kernels_bf16_obj>)
target_include_directories(u15-stream-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-stream-test PRIVATE vpipe::vpipe)
target_compile_features(u15-stream-test PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-stream-test PRIVATE -Wall -Werror)
endif()
add_test(NAME u15-stream COMMAND u15-stream-test)

# HOW TO GET A TENSOR OFF DISK: mmap+memcpy against pread, on whatever
# drives are passed. A benchmark, not a gate -- it reports rates and
# asserts only that the arms read the same bytes.
add_executable(u15-io-bench tests/u15-io-bench.cc)
target_include_directories(u15-io-bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(u15-io-bench PRIVATE vpipe::vpipe)
target_compile_features(u15-io-bench PRIVATE cxx_std_20)
if(NOT MSVC)
    target_compile_options(u15-io-bench PRIVATE -Wall -Werror)
endif()
