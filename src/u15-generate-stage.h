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

  // Pre-launch twin of the model-port latch in process(): the same beat
  // and the same parse, early enough that declare_resources() sees the
  // model.
  //
  // Without it a graph that names its checkpoint through a model-select
  // stage -- rather than in this stage's own config -- declares NOTHING,
  // and 33 GB of weights never reach the ledger the streaming decision
  // is taken against. That is the quiet direction of wrong: the plan
  // reads as roomy and the graph is admitted.
  //
  // Bookkeeping only; nothing loads here (see Stage::apply_constant).
  void apply_constant(unsigned iport, const vpipe::FlexData& beat) override;

private:
  // `_hf_dir` as a DIRECTORY on disk.
  //
  // What the user configures is a model REFERENCE, which may be a
  // registry key (`sensenova/SenseNova-U1.5-8B-MoT`, what model-fetch
  // wrote) as easily as a path. Everything downstream -- detect,
  // parse_config, the weight set, the tokenizer, and every claim keyed
  // on a directory -- walks the filesystem, so each of those has to ask
  // for the resolved form. resolve_model_dir() returns a plain path
  // unchanged, so this is safe on both.
  //
  // The raw reference is KEPT rather than overwritten: it is what the
  // user named, and it is the right thing to put in a message.
  std::string model_dir_() const;

  bool ensure_loaded_();
  void unload_();

  // Move the ledger to the checkpoint `_hf_dir` now names, after a
  // model-port switch: release `prev_dir`'s claim and declare the new
  // one BEFORE it loads. revise_declaration() refuses to create an
  // entry, so a checkpoint nobody declared would otherwise never be
  // counted at all.
  void switch_declaration_(const std::string& prev_dir);

  // Resolve `auto` from the box, once, after the first beat -- where
  // every peer has loaded and real bytes are authoritative. Before the
  // init barrier the answer would be taken against whatever happened to
  // have loaded first.
  void resolve_policy_();

  std::string _hf_dir;
  GenParams   _params;
  // Accelerated mode, handed to MetalOps once it is initialized. Held
  // here rather than in GenParams because it is a property of how the
  // GEMMs run, not of the picture being asked for.
  bool        _i8_gemm = false;
  // SageAttention, the second accelerated tier. Independent of
  // `_i8_gemm` beside it and settable with it: that one changes how
  // a weight is multiplied, this one how a score is computed.
  bool _sage_attn{};
  int  _sage_dense_layers{};
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
