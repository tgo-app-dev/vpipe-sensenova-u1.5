#ifndef VPIPE_U15_WEIGHTS_H
#define VPIPE_U15_WEIGHTS_H

// Binding the released checkpoint to GPU buffers.
//
// THE DTYPE SPLIT IS THE WHOLE PROBLEM HERE. The base (understanding)
// expert ships BF16; every `_mot_gen` twin ships F32. That is why 8B
// parameters per expert is a 50 GB file, and it is not a signal: the
// reference declares torch_dtype bfloat16 and casts on load, so bf16 is
// the intended compute precision for BOTH experts. Converting the F32
// twins down at bind is faithful, and takes the resident set from ~50 GB
// to ~35 GB.
//
// The conversion goes through derived(), which caches only the PRODUCT.
// The f32 source is read UNCACHED inside the builder and dies when the
// builder returns, so the two never coexist beyond one tensor. Doing it
// the other way -- caching the source and converting after -- would peak
// at the sum and defeat the point.
//
// STREAMING. The layers are the repeating unit and 97% of the bytes, so
// they are what a box too small to hold this model gives up: the trunk
// stays resident and each layer is re-read from the checkpoint as the
// stack runs. That is the same shape the in-tree DiTs use, with one
// difference that matters here -- the gen expert ships F32, so a
// streamed layer cannot use the raw-refill fast path for half its bytes
// and pays a conversion per pass. See STREAMING COSTS in the .cc.
//
// What streams is decided ABOVE this class (the stage asks
// model_memory::plan_streaming). What grows back is decided BELOW it,
// per pass, by BlockResidency -- so a streamed run on a roomy box
// converges on holding the whole stack anyway.
//
// RESIDENCY: OWNED, NEVER MAPPED -- and not for want of asking.
//
// The host's rule is conditional, not absolute:
// `weights_may_be_mapped(stream_blocks, wire_resident)` in
// shared/wired-pool.h allows a mapped tensor only when the model
// neither streams nor wires. A mapped tensor is a SUBVIEW of a
// whole-shard mmap, so it can be neither wired (mlock on file-backed
// pages is refused well short of the pool's ceiling -- MEASURED at
// ~4 GB on MiniMax-H3) nor parked (mark_inactive refuses a handle that
// does not own its allocation). Both are the whole point of keeping a
// resident set. This model streams and it wires, so that rule alone
// settles it.
//
// A THIRD reason applies here that holds even when the first two do
// not, which is why there is no residency switch at all below.
// Mapping is per SHARD, not per tensor -- load_mapped() wraps the whole
// file -- so it only pays for a loader that converts almost nothing.
// Half of this checkpoint is F32 that becomes bf16 at bind, and those
// tensors sit in the same shards as the BF16 ones. Mapping would
// therefore hold ~46.8 GB of shard beside the ~15.8 GB of converted
// generation-expert buffers it still had to allocate: ~62.6 GB against
// 31.6 GB for reading everything Copied. (That is arithmetic over the
// checkpoint, not a measurement. The measurement vpipe already has is
// the same shape: switching the LM reads to Mapped took peak footprint
// 3.62 -> 5.00 GB, +38%.)
//
// The one configuration where mapping might genuinely have paid is a
// QUANTIZED pack with the wired pool off -- its codes are U32
// pass-through, so it is conversion-light. NOT taken: the pool is on by
// default, which makes the question moot in every configuration anyone
// actually runs, and a second residency path would exist only to be
// wrong in.
//
// GOING THROUGH THE WEIGHT SET is not optional (vpipe's
// docs/MODEL-MEMORY.md): the manager owns the checkpoint and a model
// borrows it, so two stages over one checkpoint share it and the manager
// can see what is resident. A 35 GB model that opened its own mmap would
// be invisible to exactly the accounting it most needs.

#include "u15-config.h"
#include "u15-metal-ops.h"

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/block-residency.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/shared/wired-pool.h"
#include "generative-models/weight-set.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe { namespace metal_compute { class MetalCompute; } }

namespace u15 {

// One expert's weights for one layer. Both experts have identical
// structure and differ only in which tensors are bound -- which is
// exactly why one struct serves both, and why binding the same set
// twice is a bug the layer test checks for.
struct ExpertLayer {
  vpipe::metal_compute::SharedBuffer input_ln;
  vpipe::metal_compute::SharedBuffer post_ln;
  // The seven MATRICES, which model-quantize takes and the norms it
  // leaves alone. QWeight carries either form, so one loader and one
  // forward read a dense pack and a quantized one.
  QWeight q, k, v, o;
  // The four head-dim norms. q_norm / k_norm cover the t half; the _hw
  // pair covers the hw half BEFORE it splits into h and w, so all four
  // are head_dim/2 wide, not head_dim/4.
  vpipe::metal_compute::SharedBuffer q_norm, k_norm;
  vpipe::metal_compute::SharedBuffer q_norm_hw, k_norm_hw;
  QWeight gate, up, down;

  bool complete() const;
};

struct MotLayer {
  ExpertLayer und;   // text + reference images
  ExpertLayer gen;   // the image being denoised
};

// Everything that is not a transformer layer.
struct Trunk {
  vpipe::metal_compute::SharedBuffer embed_tokens;
  vpipe::metal_compute::SharedBuffer norm;       // final, und
  vpipe::metal_compute::SharedBuffer norm_gen;   // final, gen
  vpipe::metal_compute::SharedBuffer lm_head;    // may be unbound

  // The two patch embedders: same structure, separate weights.
  vpipe::metal_compute::SharedBuffer und_patch_w, und_patch_b;
  vpipe::metal_compute::SharedBuffer und_dense_w, und_dense_b;
  vpipe::metal_compute::SharedBuffer gen_patch_w, gen_patch_b;
  vpipe::metal_compute::SharedBuffer gen_dense_w, gen_dense_b;

  // The pixel head (ConvDecoder): two convolutions, nothing else.
  vpipe::metal_compute::SharedBuffer conv1_w, conv1_b;
  vpipe::metal_compute::SharedBuffer conv2_w, conv2_b;

  // The two scalar embedders. Identical shape; one takes the timestep,
  // the other the resolution-dependent noise scale.
  vpipe::metal_compute::SharedBuffer tstep_w0, tstep_b0;
  vpipe::metal_compute::SharedBuffer tstep_w2, tstep_b2;
  vpipe::metal_compute::SharedBuffer nscale_w0, nscale_b0;
  vpipe::metal_compute::SharedBuffer nscale_w2, nscale_b2;
};

class U15Weights {
 public:
  struct Options {
    // The lm_head is only reachable in think-mode, where the model
    // samples text before generating. A pure T2I run never touches it,
    // and it is [151936, 4096] -- 1.2 GB in bf16.
    bool with_lm_head = false;

    // Bind only the first N layers. For tests: binding all 42 is 35 GB,
    // and what a bind test checks (names, shapes, dtypes, the expert
    // split) is answered by one layer.
    int max_layers = -1;

    // STREAM the layers instead of holding them: bind the trunk plus a
    // leading prefix of `pinned_layers`, and read the rest per pass.
    //
    // The caller decides this, not the loader -- it is the stage that
    // can see the whole graph (model_memory::plan_streaming), and the
    // decision is irreversible in the sense that changing it means
    // rebuilding the model. Defaulting to false keeps every existing
    // caller, including all the tests, on the path they were verified
    // against.
    bool stream_layers = false;

    // How many LEADING layers stay resident when streaming. Sized by
    // the caller from genai::stream_pin_count(); 0 streams everything
    // and is always correct, just slower.
    int pinned_layers = 0;

    // Overwrite the reusable slot with pread(2) instead of rebuilding
    // it with an allocate-and-copy-from-mmap. On by default; the switch
    // exists so the two can be A/B'd on a real checkpoint, and as an
    // escape hatch for a pack the fast path mishandles.
    bool refill_streamed = true;

    // Whether to try to WIRE the layers this model keeps resident.
    // Off for the offline tools and the tests, which have no session
    // and no pool to ask.
    bool wire_resident = false;
  };

  // `ws` is HELD for this object's lifetime -- mapped tensors point into
  // the set's mmap and cached ones are refcounted aliases of buffers it
  // owns, so the checkpoint must not unmap under us.
  static std::unique_ptr<U15Weights> load(
      std::shared_ptr<vpipe::genai::WeightSet> ws,
      vpipe::metal_compute::MetalCompute* mc, const U15Config& cfg,
      const Options& opt, std::string* err);

  const Trunk& trunk() const { return _trunk; }
  const std::vector<MotLayer>& layers() const { return _layers; }
  int n_layers() const { return (int)_layers.size(); }

  // Bytes this object is holding RIGHT NOW, for declare_resources() and
  // for reporting what the dtype conversion actually cost.
  //
  // Not a load-time constant: a streaming model starts at its trunk plus
  // the pinned prefix and GROWS as residency admits layers, and on a
  // roomy box it converges on the whole stack within a pass or two. A
  // figure frozen at load would tell every peer that sizes after us
  // there is 26 GB of room that this model has since taken back -- the
  // one direction of error that thrashes.
  std::size_t resident_bytes() const;

  // What was held at LOAD, before any growth. What declare_resources()
  // is being corrected to on the first revision.
  std::size_t loaded_bytes() const { return _bytes; }
  std::size_t converted_bytes() const { return _converted; }
  // 0 when the checkpoint is dense; otherwise the bits every quantized
  // matrix was packed at (they must agree, and the loader refuses a pack
  // where they do not).
  int quant_bits() const { return _qbits; }
  int quant_group() const { return _qgroup; }

  // ---- streaming ----------------------------------------------------

  bool streaming() const { return _streaming; }

  // The RESIDENT bytes of one layer, measured from a real one rather
  // than estimated from the config -- the two disagree by the whole F32
  // conversion, and this figure sizes admission decisions.
  //
  // Before any layer has been built it falls back to the checkpoint's
  // widest layer ON DISK, which for this model over-states bf16 by the
  // conversion ratio. That is the safe direction for a budget.
  std::size_t layer_bytes() const;

  // How many layers are held right now: the pinned prefix plus whatever
  // residency growth has kept.
  int resident_layers() const;

  // Bytes read from the checkpoint since load, and how many layer reads
  // that took. Both zero on a preloaded model.
  std::size_t streamed_bytes() const { return _streamed_bytes; }
  int streamed_layers() const { return _streamed_layers; }
  // How many layer reads had to REBUILD -- allocate a fresh set of
  // buffers -- rather than refill the slots in place. Only the first
  // read of a run and any promotion should; a run that rebuilds every
  // layer has lost the fast path and should say so rather than just
  // being slow.
  std::size_t rebuilt_layers() const { return _rebuilt_layers; }
  // Prefetches issued and prefetches that arrived in time.
  int prefetch_started() const { return _slots.prefetch_started(); }
  int prefetch_hits() const { return _slots.prefetch_hits(); }

  // ---- the stack-pass protocol --------------------------------------
  //
  // A "pass" is ONE traversal of the 42 layers, whatever runs inside it.
  // With the branch loop inverted (see u15-backbone.h) a denoise step is
  // one pass covering every CFG branch, which is what makes streaming
  // cost one read of the checkpoint per step rather than one per branch.
  //
  //   begin_pass()  once, at the top
  //   layer(i)      per layer; the returned layer is valid until
  //                 end_layer(i), which the caller must not run before
  //                 the GPU work reading it has retired
  //   end_layer(i)  after that work has retired
  //
  // On a preloaded model all three are nearly free, so the caller does
  // not branch on `streaming()`.
  void begin_pass();
  // Layer `i` for the pass now running. A resident layer comes back as
  // it sits; a streamed one is read into one of two REUSABLE SLOTS and
  // stays valid until the next acquire of that slot.
  const MotLayer* layer(int i, std::string* err);
  // Issue the read of the next STREAMED layer after `after`, to run
  // under the GPU work just committed. A no-op when everything ahead is
  // resident, when a read is already outstanding, or when there is only
  // one slot.
  void prefetch_after(int after);
  void end_layer(int i);
  // Join any outstanding read. MUST be called before this object or its
  // slots can go away -- see block-slots.h.
  void join_reads();

  // The activation arena's buffers, wired ahead of the trunk and ahead
  // of any layer admission -- see U15Backbone::scratch_buffers(). The
  // backbone owns them, so it hands them over rather than this reaching
  // for them.
  //
  // `unwire_scratch` must be called BEFORE the arena is replaced: only
  // unwire_from_pool() decrements the pool's counter, so a wired buffer
  // that is simply dropped leaks its bytes for the rest of the run.
  // Unwire the weights for an idle model. Wired and parked are
  // opposites -- mark_inactive() refuses a wired buffer -- so parking
  // without this reclaims nothing. Returns the bytes given back.
  std::size_t wire_down();

  void wire_scratch(
      const std::vector<vpipe::metal_compute::SharedBuffer*>& bufs);
  void unwire_scratch(
      const std::vector<vpipe::metal_compute::SharedBuffer*>& bufs);

  // What must stay free for the rest of a pass, and how much of it the
  // caller has ALREADY allocated.
  //
  // Growth stays OFF until this is called once, so a caller that has
  // not been taught what its activations cost keeps exactly the
  // behaviour it had. Reserving what is already allocated is asking for
  // the same headroom twice -- the budget admission reads has already
  // subtracted it -- which is how a run streams everything to protect
  // room that is sitting in its own hands.
  // The reserve has TWO terms with two different owners -- the KV
  // caches, allocated by the generator, and the activation arena, sized
  // by the backbone -- and neither knows about the other. So each
  // reports its own, absolutely, and this adds them up.
  void set_residency_reserve(std::size_t bytes);
  void note_kv_allocated(std::size_t bytes);
  void note_arena_allocated(std::size_t bytes);

  // How many passes the run will make, so growth can be paced to it: a
  // probe sized for a 30-step schedule reaches full residency only near
  // the end of a 5-step one, where nothing is left to use it.
  void set_residency_schedule(int passes);

  // Give back every layer residency has promoted, because this model
  // has been told to let go between beats.
  //
  // Parking cannot do this. `park_weights()` reclaims what the weight
  // SET cached, and a promoted layer is the model's own buffer -- it
  // was read uncached precisely so the set would not hold it. So an
  // idle streaming model that is not asked to release stays at whatever
  // the residency policy grew to, which on a roomy box is the whole
  // checkpoint.
  //
  // Resets the growth ratchet, and that is not optional: release()
  // ratchets the ceiling down to what is left, so giving everything
  // back would cap the NEXT beat at one layer forever. The ratchet
  // exists to react to memory PRESSURE, and this is not pressure -- it
  // is a decision this model took while nothing was wrong. Returns what
  // was released; 0 on a preloaded model, which has nothing of the
  // policy's to shed.
  std::size_t release_at_idle();

 private:
  U15Weights() = default;

  bool load_layer_(int i, MotLayer* into, bool streamed, std::string* err);
  // Every (checkpoint name, destination buffer) pair of one layer, in a
  // FIXED order. Used only by the refill path; load_layer_ builds its
  // own names because it also has to derive the quantization metadata
  // from the shapes, which needs the checkpoint rather than the layer.
  // A name here that does not match one there simply misses, and the
  // refill falls back to a rebuild -- slower, never wrong.
  void each_layer_tensor_(
      int i, MotLayer& L,
      const vpipe::genai::BlockSlots<MotLayer>::TensorFn& fn) const;
  // Overwrite an already-built layer's buffers in place. False means
  // the slot may be PARTLY WRITTEN and has to be rebuilt wholesale.
  bool clone_layer_(const MotLayer& src, MotLayer* dst, bool copy) const;
  void configure_slots_();
  // The one dtype no raw refill can place: F32 is twice the width of
  // its bf16 destination, so there is nowhere to put the bytes. Preads
  // into a scratch buffer and converts out of it, which keeps even this
  // half off the mmap path.
  bool convert_into_bf16_(const std::string& name,
                          const vpipe::metal_compute::SharedBuffer& dst);
  std::size_t layer_resident_bytes_(const MotLayer& L) const;
  std::size_t evict_tail_layer_();
  std::size_t wire_layer_(MotLayer& L, bool on);
  std::size_t wire_trunk_(bool on);
  void resident_pages_(std::size_t* examined, std::size_t* incore,
                       std::size_t* paged_out) const;

  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  Trunk                                    _trunk;
  std::vector<MotLayer>                    _layers;
  vpipe::metal_compute::MetalCompute*      _mc = nullptr;
  U15Config                                _cfg;
  std::size_t                              _bytes = 0;
  std::size_t                              _converted = 0;
  int                                      _qbits = 0;
  int                                      _qgroup = 0;

  // ---- streaming state ----------------------------------------------
  bool        _streaming = false;
  int         _pinned = 0;            // leading layers held at load
  std::size_t _layer_bytes = 0;       // measured; 0 until one is built
  std::size_t _disk_layer_bytes = 0;  // the fallback, from the table
  std::size_t _streamed_bytes = 0;
  int         _streamed_layers = 0;
  bool        _wire_allowed = false;
  std::size_t _res_kv = 0;            // reserve term: the KV caches
  std::size_t _res_arena = 0;         // reserve term: the activations

  // TWO REUSABLE READ DESTINATIONS plus the prefetch, from the host's
  // shared policy (block-slots.h). Legal here only because all 42 MoT
  // layers are identical in shape, which is the property that makes one
  // destination serve every read.
  vpipe::genai::BlockSlots<MotLayer> _slots;
  bool        _refill = true;
  // Where an F32 tensor is pread before being converted into the slot.
  // Grown on demand to the largest such tensor.
  vpipe::metal_compute::SharedBuffer _f32_scratch;
  std::size_t _rebuilt_layers = 0;    // read through the allocate path

  // Which streamed layers residency growth has promoted, in admission
  // order -- so eviction gives back the most recently taken, which on a
  // cyclic scan is the one furthest from being wanted again.
  std::vector<int>            _promoted;
  vpipe::genai::BlockResidency _resid;
  vpipe::genai::WiredPool      _wire;
};

}  // namespace u15

#endif  // VPIPE_U15_WEIGHTS_H
