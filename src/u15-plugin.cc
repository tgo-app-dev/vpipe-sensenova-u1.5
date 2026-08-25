// vpipe-sensenova-u1.5 -- SenseNova-U1.5-8B-MoT as a vpipe PLUGIN.
//
// The extension points used here, and why each one:
//
//   register_stage           `sensenova-u1.5-generate` and
//                            `sensenova-u1.5-model-config`. The generate
//                            stage is NOT a registry entry because there
//                            is no host stage doing this job for other
//                            families: the in-tree image path is
//                            conditioner -> generate-image -> vae-decode,
//                            and this model has no VAE, no separate text
//                            encoder, and a conditioning that is a KV
//                            cache rather than a tensor. See the header
//                            comment on u15-generate-stage.h.
//   register_metal_library   the plugin's own kernels, before anything
//                            can dispatch them.
//   register_catalog_entries so the checkpoint is discoverable and
//                            fetchable the way a built-in model is.
//
// Notably NOT used: register_vae_family (there is no VAE) and
// register_video_family (this is an image model). No host change was
// needed to ship this plugin -- ABI 2 as it stands was sufficient.
//
// The model weights are Apache-2.0, as is this plugin.

#include "plugin/plugin-abi.h"
#include "plugin/plugin-context.h"

#include "u15-catalog.h"
#include "u15-config.h"
#include "u15-generate-stage.h"
#include "u15-model-config-stage.h"

// Emitted by vpipe_add_metal_library (see CMakeLists.txt).
extern "C" const unsigned char u15_kernels_bf16_metallib[];
extern "C" const unsigned long u15_kernels_bf16_metallib_len;
extern "C" const unsigned char u15_kernels_f32_metallib[];
extern "C" const unsigned long u15_kernels_f32_metallib_len;

static const VpipePluginInfo kInfo = {
    VPIPE_PLUGIN_INFO_SCHEMA,
    "sensenova-u1.5",
    "0.1.0",
    "T-Go LLC",
    "Apache-2.0 (plugin and weights)",
    "SenseNova-U1.5-8B-MoT: unified MoT image generation and editing",
};

static void
u15_register(vpipe::VpipePluginContext* ctx)
{
  if (ctx == nullptr) { return; }

  // The kernels FIRST: a stage that loads before its metallib is
  // registered fails at MetalOps::init with a missing-library error
  // rather than at a dispatch, but registering first makes the ordering
  // a fact rather than a race to reason about.
  ctx->register_metal_library(u15::kMetalLibBf16, u15_kernels_bf16_metallib,
                              u15_kernels_bf16_metallib_len);
  ctx->register_metal_library(u15::kMetalLibF32, u15_kernels_f32_metallib,
                              u15_kernels_f32_metallib_len);

  // Registered through the plugin path rather than VPIPE_REGISTER_STAGE
  // so the StageSpec is attached and the stages show up in
  // /api/stage-types and the web-ui composer.
  ctx->register_stage<u15::U15GenerateStage>(
      u15::U15GenerateStage::stage_spec());
  ctx->register_stage<u15::U15ModelConfigStage>(
      u15::U15ModelConfigStage::stage_spec());

  ctx->register_catalog_entries(u15::catalog_entries());
}

VPIPE_PLUGIN_DEFINE(&kInfo, u15_register)
