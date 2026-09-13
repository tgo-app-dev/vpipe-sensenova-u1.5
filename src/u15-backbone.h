#ifndef VPIPE_U15_BACKBONE_H
#define VPIPE_U15_BACKBONE_H

// The 42-layer MoT backbone on the GPU.
//
// TWO HOMOGENEOUS PATHS, never a mixed one. The reference's own
// per-token dispatch raises NotImplementedError for the mixed case and
// tells callers to split the sequence at token-type boundaries, so a
// forward here is either all-understanding or all-generation and selects
// ONE weight set for the whole pass. There is no gather.
//
// ONE PASS PER STEP, NOT ONE PER BRANCH. The CFG branches differ only
// in their prefix and their positions -- they run the SAME 42 layers
// over the same weights -- so forward_many() puts the branch loop
// INSIDE the layer loop. On a preloaded model that is a reordering
// worth nothing; on a streamed one it is the difference between reading
// the checkpoint once per step and once per branch, which for an edit
// (three branches) is 3x the disk traffic of the whole run.
//
// It is exactly a reordering: each branch's arithmetic is independent
// of the others, the encoders stay ordered on one queue, and the number
// of command buffers is unchanged. The streaming test pins that by
// requiring the same seed to produce the same bytes.
//
// THE KV CACHE IS THE CONDITIONING. The reference prefills the text
// prefix once, then runs every denoise step over only the image tokens
// against that frozen prefix with update_cache=False. So the cache is
// sized [kv_heads][prefix + image_tokens][head_dim] once, the prefix
// written at position 0 and never touched again, and each step
// OVERWRITES the image block in place. That is why a 50-step run costs
// 50 x L tokens of attention rather than a growing cache.

#include "u15-config.h"
#include "u15-metal-ops.h"
#include "u15-weights.h"

#include "apple-silicon/metal-compute/command-stream.h"

#include <memory>
#include <functional>
#include <string>
#include <vector>

namespace u15 {

enum class Expert { Und, Gen };

// How a pass attends.
//
//   Causal         the t2i prefill. Every text token has a distinct t,
//                  so the reference's block-causal rule REDUCES to plain
//                  causal and libvpipe's mask-free kernel is exact.
//   BlockCausal    the EDIT prefill. A reference image's tokens all
//                  share one t, so the rule stops collapsing and the
//                  pass needs the t-index vector.
//   Bidirectional  the denoise. Image tokens attend the whole prefix and
//                  each other, with no horizon at all.
enum class Attn { Causal, BlockCausal, Bidirectional };

// One layer's K and V, sized for the whole run.
struct KvCache {
  std::vector<vpipe::metal_compute::SharedBuffer> k, v;   // per layer
  int capacity   = 0;   // prefix_cap + gen_len
  int prefix_len = 0;   // filled by prefill()
  bool valid() const { return !k.empty() && capacity > 0; }
};

class U15Backbone {
 public:
  // `w` is NOT const: a streaming model reads its layers during the
  // pass and grows a resident set as it goes, so a forward mutates it.
  static std::unique_ptr<U15Backbone> create(MetalOps* ops,
                                             const U15Config& cfg,
                                             U15Weights* w,
                                             std::string* err);

  // Unwires the activation arena through `w`, so the weights must still
  // be alive -- see the definition.
  ~U15Backbone();

  // Allocate a cache for `capacity` total tokens across all layers.
  KvCache make_cache(int capacity, std::string* err) const;

  // Bytes a cache of `capacity` tokens costs. Reported rather than
  // estimated by the caller, because a wrong KV number is the one that
  // reads healthy right up until a long prompt exhausts the box.
  std::size_t kv_bytes(int capacity) const;

  // What the shared activation arena costs at `n` tokens. Answerable
  // without allocating anything, so the stage can declare it during the
  // planning phase and the residency policy can reserve it before the
  // first pass has built it.
  std::size_t scratch_bytes(int n) const;

  // What the arena is ACTUALLY holding right now. Zero until the first
  // forward sizes it.
  std::size_t scratch_resident_bytes() const;

  // The arena's buffers, for the wired pool.
  //
  // The activations are what a pass cannot proceed without, so they
  // have a better claim on the pool than any resident layer -- which is
  // an optimisation the model can always shed and stream instead.
  // Wiring the optional half first is how a run ends up with 30 GB of
  // wired layers beside an activation buffer the compressor is free to
  // take.
  std::vector<vpipe::metal_compute::SharedBuffer*> scratch_buffers();

  // One branch's state through a stack pass. Everything here is
  // per-branch; `n`, the expert and the attention rule are shared and
  // are arguments to forward_many().
  struct PassSlot {
    // [n][hidden] bf16, updated IN PLACE.
    const vpipe::metal_compute::SharedBuffer* x = nullptr;
    KvCache* kv = nullptr;
    int kv_off = 0;   // where this pass writes its K/V
    int kv_len = 0;   // how many keys to attend over
    const std::vector<int>* pos_t = nullptr;
    const std::vector<int>* pos_h = nullptr;
    const std::vector<int>* pos_w = nullptr;
    // Required for Attn::BlockCausal: an i32 buffer of `kv_len` entries
    // giving each cached token's t.
    const vpipe::metal_compute::SharedBuffer* t_index = nullptr;
  };

  // A COOPERATIVE STOP, checked once per LAYER.
  //
  // Per layer and not per forward, because on a streamed model a single
  // forward is the whole 42-layer checkpoint read off disk -- minutes at
  // a large geometry. A stop that is only noticed between forwards is a
  // Stop button that does nothing for the length of a step, which is
  // indistinguishable from one that does not work.
  //
  // Returning true means ABANDON. Both forwards then return false with
  // `err` set to kStopped, which the caller tells apart from a real
  // failure -- a cancelled run is not something to warn about.
  void set_stop(std::function<bool()> fn) { _stop = std::move(fn); }
  static constexpr const char* kStopped = "stopped";

  // Run `n` tokens of EVERY slot through all layers, layer-outermost.
  // See the note at the top of this file for why the loops are this way
  // round.
  bool forward_many(vpipe::metal_compute::CommandStream& stream,
                    const std::vector<PassSlot>& slots, int n, Expert e,
                    Attn attn, std::string* err);

  // Run `n` tokens through all layers.
  //
  //   x        [n][hidden] bf16, updated IN PLACE
  //   kv_off   where this pass writes its K/V
  //   kv_len   how many keys to attend over (>= kv_off + n)
  //   causal   und prefill: true. generation: false (bidirectional).
  //
  // The final norm is NOT applied here -- prefill does not need it and
  // the generation path needs the gen-expert one, so it is the caller's.
  bool forward(vpipe::metal_compute::CommandStream& stream,
               const vpipe::metal_compute::SharedBuffer& x, int n,
               Expert e, KvCache& kv, int kv_off, int kv_len,
               const std::vector<int>& pos_t,
               const std::vector<int>& pos_h,
               const std::vector<int>& pos_w, Attn attn,
               std::string* err,
               // Required for Attn::BlockCausal: an i32 buffer of
               // `kv_len` entries giving each cached token's t.
               const vpipe::metal_compute::SharedBuffer* t_index = nullptr);

  // Apply the final RMSNorm for one expert.
  void final_norm(vpipe::metal_compute::ComputeEncoder& enc,
                  const vpipe::metal_compute::SharedBuffer& x, int n,
                  Expert e) const;

  const U15Config& cfg() const { return _cfg; }

 private:
  std::function<bool()> _stop;
  U15Backbone() = default;
  bool ensure_scratch_(int n, std::string* err);

  // What one slot does at one layer. Encodes only -- the commit and, on
  // a streamed model, the wait that lets the layer go are the caller's.
  void layer_step_(vpipe::metal_compute::CommandStream& stream,
                   const MotLayer& L, const PassSlot& s, int li, int n,
                   Expert e, Attn attn,
                   const vpipe::metal_compute::SharedBuffer& rope_cos,
                   const vpipe::metal_compute::SharedBuffer& rope_sin,
                   const MetalOps::SteelAttn* steel);

  MetalOps*        _ops = nullptr;
  U15Weights*      _w   = nullptr;
  U15Config        _cfg;

  // ONE shared scratch arena, sized to the largest pass and reused by
  // every layer. Per-layer scratch is how a 42-layer stack quietly
  // allocates more than the checkpoint -- the activations are bigger
  // than they look once the intermediate width is 12288.
  // The steel attention plans for the BIDIRECTIONAL path, one per slot.
  // A plan depends only on (heads, tq, tkv, head_dim, capacity), which
  // are constant across a pass's 42 layers -- so they are built once per
  // pass rather than 42 times. PER SLOT because tkv is the branch's own
  // prefix plus the image block, and the branches' prefixes differ.
  std::vector<MetalOps::SteelAttn> _steel;

  int _scratch_tokens = 0;
  vpipe::metal_compute::SharedBuffer _normed, _q, _k, _v, _attn, _proj;
  vpipe::metal_compute::SharedBuffer _attn_t;
  vpipe::metal_compute::SharedBuffer _qh, _gate, _up, _ff;
};

}  // namespace u15

#endif  // VPIPE_U15_BACKBONE_H
