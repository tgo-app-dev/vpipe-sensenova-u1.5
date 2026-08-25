#ifndef VPIPE_U15_GENERATOR_H
#define VPIPE_U15_GENERATOR_H

// Text-to-image: prefill, denoise, decode.
//
// THE SHAPE OF THE LOOP. The two prefixes (conditional and
// unconditional) are prefilled ONCE into two KV caches. Every denoise
// step then runs only the L image tokens against those frozen prefixes
// -- bidirectionally, and WITHOUT extending the cache -- so a 50-step
// run costs 50 x L tokens of attention rather than a growing context.
// That is the reference's own structure (`update_cache=False`), not an
// optimisation invented here.
//
// THE HEAD PREDICTS x0, NOT VELOCITY. v = (x_pred - z) / (1 - t), with
// the denominator clamped at t_eps. Treating the head's output as a
// velocity produces structured noise, not an error.
//
// The sampler arithmetic runs on the HOST in f32. It is a few million
// elementwise operations against a forward pass of ~16 GFLOP per step
// per branch, and it is where the reference is explicit about
// precision.

#include "u15-backbone.h"
#include "u15-config.h"
#include "u15-image.h"
#include "u15-metal-ops.h"
#include "u15-prompt.h"
#include "u15-weights.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe { class FlexData; }

namespace u15 {

// How CFG is renormalised. The reference's four modes.
enum class CfgNorm { None, Global, Channel, CfgZeroStar };

// A reference image for the EDIT path, as channel-last u8 RGB at its
// native size. The model layer does its own smart_resize + bicubic +
// ImageNet normalisation: the geometry it needs is a multiple of 32
// within the model's pixel bounds, which is a model fact rather than
// something a graph should have to know.
struct RefImage {
  std::vector<std::uint8_t> rgb;
  int width = 0;
  int height = 0;
};

struct GenParams {
  int    width  = 1024;
  int    height = 1024;
  int    steps  = 50;
  double cfg_scale = 4.0;
  // The SECOND guidance axis, for editing: how hard the result is pushed
  // away from "these images, no instruction". 1 disables it, and then
  // the unconditional prefix is never built.
  double img_cfg_scale = 1.0;
  double timestep_shift = 3.0;
  CfgNorm cfg_norm = CfgNorm::None;
  double cfg_interval_lo = 0.0;
  double cfg_interval_hi = 1.0;
  // The reference example's default is 42, not 0.
  std::uint64_t seed = 42;
  double t_eps = 0.05;

  // Debug / repro: raw f32 [H][W][3] channel-last initial noise, in
  // MODEL space and ALREADY scaled. Exists so an end-to-end run can be
  // compared against the reference, whose torch RNG this cannot
  // reproduce -- see the note in generate().
  std::string init_noise_path;

  // Read the knobs this family owns off a model-config beat.
  static GenParams from_flex(const vpipe::FlexData& fd, const GenParams& d);
};

class Generator {
 public:
  struct Deps {
    MetalOps*         ops = nullptr;
    // NOT const: a streaming model reads its layers during the run and
    // grows a resident set as it goes, and the generator is what knows
    // the schedule to pace that growth against.
    U15Weights* weights = nullptr;
    U15Backbone*      backbone = nullptr;
    ImagePath*        image = nullptr;
    const Prompt*     prompt = nullptr;
  };

  static std::unique_ptr<Generator> create(const Deps& d,
                                           const U15Config& cfg,
                                           std::string* err);

  // Generate one image. `out_u8` receives planar u8 RGB [3][H][W].
  //
  // `progress` is called with (step, total) before each step so a stage
  // can report the phase people actually wait through.
  // `refs` empty => text-to-image. Non-empty => editing, which changes
  // the prefill (reference images enter as UNDERSTANDING tokens under
  // block-causal attention) and may add a third guidance branch.
  bool generate(const std::string& prompt,
                const std::vector<RefImage>& refs, const GenParams& p,
                std::vector<std::uint8_t>* out_u8,
                const std::function<void(int, int)>& progress,
                std::string* err);

  // What a run of this geometry will need, so a stage can declare it
  // before anything loads.
  std::size_t kv_bytes(const GenParams& p) const;

 private:
  Generator() = default;

  Deps      _d;
  U15Config _cfg;
};

}  // namespace u15

#endif  // VPIPE_U15_GENERATOR_H
