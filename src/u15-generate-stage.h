#ifndef VPIPE_U15_GENERATE_STAGE_H
#define VPIPE_U15_GENERATE_STAGE_H

// SenseNova-U1.5 text-to-image, as ONE stage.
//
// WHY ONE STAGE AND NOT THE STOCK THREE. The in-tree image path is
// `diffusion-conditioner -> generate-image -> vae-decode -> save-image`,
// and none of the first three fit:
//
//   * `diffusion-conditioner` emits a conditioning TENSOR. U1.5's
//     conditioning is a KV CACHE produced by the same weights that
//     denoise, so it cannot cross a stage boundary.
//   * `generate-image` emits a LATENT for a VAE. U1.5 emits pixels.
//   * `vae-decode` has nothing to do -- there is no VAE in the
//     checkpoint at all.
//
// So this stage owns prefill + denoise + pixel head, and emits the
// exact payload `vae-decode` would have: a planar U8 RGB TensorBeat
// tagged `rgb-frames`. The stock `save-image`, `compare-image` and the
// web-ui preview consume it unchanged.
//
// That is the registry-vs-stage rule applied rather than dodged: a
// family registry is right when the host already has a stage doing the
// job for other families, and here it does not.

#include "u15-backbone.h"
#include "u15-config.h"
#include "u15-generator.h"
#include "u15-image.h"
#include "u15-metal-ops.h"
#include "u15-prompt.h"
#include "u15-weights.h"

#include "common/flex-data.h"
#include "pipeline/stage-spec.h"
#include "pipeline/typed-stage.h"
#include "stages/model-memory.h"

#include <memory>
#include <string>
#include <vector>

namespace u15 {

class U15GenerateStage : public vpipe::TypedStage<U15GenerateStage> {
public:
  static constexpr const char* kTypeName = "sensenova-u1.5-generate";

  U15GenerateStage(const vpipe::SessionContextIntf* session,
                   std::string                      id,
                   std::vector<vpipe::InEdge>       iports,
                   vpipe::FlexData                  config);

  const vpipe::StageSpec& spec() const noexcept override;
  static const vpipe::StageSpec* stage_spec() noexcept;

  // Declared BEFORE anything loads, so peers sizing themselves against
  // this stage see a real number rather than whatever happens to be
  // resident when they look.
  std::vector<vpipe::ResourceClaim> declare_resources() const override;

  vpipe::Job process(vpipe::RuntimeContext& ctx) override;

private:
  bool ensure_loaded_();
  void unload_();

  // Resolve `auto` from the box, once, after the first beat -- where
  // every peer has loaded and real bytes are authoritative. Before the
  // init barrier the answer would be taken against whatever happened to
  // have loaded first.
  void resolve_policy_();

  std::string _hf_dir;
  GenParams   _params;
  vpipe::model_memory::UnloadPolicy _policy =
      vpipe::model_memory::UnloadPolicy::kKeep;
  // What `auto` resolved to; equal to _policy for every other value.
  vpipe::model_memory::UnloadPolicy _idle =
      vpipe::model_memory::UnloadPolicy::kKeep;
  bool _policy_resolved = false;

  // Everything below is built on the first beat, not in the
  // constructor: the model directory can arrive on a port, and a
  // constructor cannot see the graph.
  std::unique_ptr<MetalOps>     _ops;
  std::unique_ptr<U15Weights>   _weights;
  std::unique_ptr<U15Backbone>  _backbone;
  std::unique_ptr<ImagePath>    _image;
  std::unique_ptr<Prompt>       _prompt;
  std::unique_ptr<Generator>    _gen;
  U15Config                     _cfg;
  bool                          _load_failed = false;
  int                           _emitted = 0;
};

}  // namespace u15

#endif  // VPIPE_U15_GENERATE_STAGE_H
