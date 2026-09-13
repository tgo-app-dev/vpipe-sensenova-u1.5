#include "u15-backbone.h"

#include <string>

using vpipe::metal_compute::CommandStream;
using vpipe::metal_compute::ComputeEncoder;
using vpipe::metal_compute::SharedBuffer;

namespace u15 {

namespace {

constexpr std::size_t kElt = 2;   // bf16

}  // namespace

std::unique_ptr<U15Backbone>
U15Backbone::create(MetalOps* ops, const U15Config& cfg, U15Weights* w,
                    std::string* err)
{
  const auto fail = [err](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<U15Backbone>{};
  };
  if (ops == nullptr || w == nullptr) { return fail("no ops or weights"); }
  if (w->n_layers() != cfg.llm.num_hidden_layers) {
    return fail("weights hold " + std::to_string(w->n_layers()) +
                " layers but the config says " +
                std::to_string(cfg.llm.num_hidden_layers));
  }
  auto b = std::unique_ptr<U15Backbone>(new U15Backbone());
  b->_ops = ops;
  b->_w = w;
  b->_cfg = cfg;
  return b;
}

// THE ARENA GOES BACK TO THE POOL WITH THE BACKBONE. forward_many() wires
// it and ensure_scratch_() unwires it before REPLACING it, and destroying
// it is the same case: freeing a wired buffer unwires it in the kernel
// but does not decrement the pool's counter, which only
// unwire_from_pool() does. The stage destroys the backbone on every
// unload, the park branch included, so without this each idle cycle
// would leak an arena's worth of the budget for the rest of the run.
//
// Through the weights, which hold the pool handle, so they must outlive
// this: the stage resets the backbone before them, and every holder
// declares the weights first so member destruction agrees.
U15Backbone::~U15Backbone()
{
  if (_w != nullptr) { _w->unwire_scratch(scratch_buffers()); }
}

std::size_t
U15Backbone::kv_bytes(int capacity) const
{
  const std::size_t per =
      (std::size_t)_cfg.llm.num_key_value_heads *
      (std::size_t)capacity * (std::size_t)_cfg.llm.head_dim * kElt;
  return per * 2 * (std::size_t)_cfg.llm.num_hidden_layers;   // k and v
}

KvCache
U15Backbone::make_cache(int capacity, std::string* err) const
{
  KvCache kv;
  if (capacity <= 0) {
    if (err != nullptr) { *err = "capacity must be positive"; }
    return kv;
  }
  const std::size_t per =
      (std::size_t)_cfg.llm.num_key_value_heads * (std::size_t)capacity *
      (std::size_t)_cfg.llm.head_dim * kElt;
  const int L = _cfg.llm.num_hidden_layers;
  kv.k.resize((std::size_t)L);
  kv.v.resize((std::size_t)L);
  for (int i = 0; i < L; ++i) {
    kv.k[(std::size_t)i] = _ops->mc()->make_shared_buffer(per);
    kv.v[(std::size_t)i] = _ops->mc()->make_shared_buffer(per);
    if (kv.k[(std::size_t)i].empty() || kv.v[(std::size_t)i].empty()) {
      if (err != nullptr) {
        *err = "out of memory allocating the KV cache (" +
               std::to_string(kv_bytes(capacity) / (1024 * 1024)) + " MB)";
      }
      return KvCache{};
    }
  }
  kv.capacity = capacity;
  return kv;
}

std::vector<SharedBuffer*>
U15Backbone::scratch_buffers()
{
  return {&_normed, &_q, &_qh, &_k, &_v, &_attn, &_attn_t, &_proj,
          &_gate, &_up, &_ff};
}

bool
U15Backbone::ensure_scratch_(int n, std::string* err)
{
  if (n <= _scratch_tokens) { return true; }
  // UNWIRE BEFORE REPLACING. Every buffer below is replaced wholesale,
  // and only unwire_from_pool() decrements the pool's counter -- so a
  // buffer dropped while wired leaks its bytes from the pool for the
  // rest of the run, and the leak compounds each time the arena grows.
  if (_w != nullptr) { _w->unwire_scratch(scratch_buffers()); }
  const std::size_t H  = (std::size_t)_cfg.llm.hidden_size;
  const std::size_t QD = (std::size_t)_cfg.llm.num_attention_heads *
                         (std::size_t)_cfg.llm.head_dim;
  const std::size_t KD = (std::size_t)_cfg.llm.num_key_value_heads *
                         (std::size_t)_cfg.llm.head_dim;
  const std::size_t I  = (std::size_t)_cfg.llm.intermediate_size;
  const std::size_t t  = (std::size_t)n;

  auto* mc = _ops->mc();
  _normed = mc->make_shared_buffer(t * H * kElt);
  _q      = mc->make_shared_buffer(t * QD * kElt);
  _qh     = mc->make_shared_buffer(t * QD * kElt);
  _k      = mc->make_shared_buffer(t * KD * kElt);
  _v      = mc->make_shared_buffer(t * KD * kElt);
  _attn   = mc->make_shared_buffer(t * QD * kElt);
  _attn_t = mc->make_shared_buffer(t * QD * kElt);
  _proj   = mc->make_shared_buffer(t * H * kElt);
  _gate   = mc->make_shared_buffer(t * I * kElt);
  _up     = mc->make_shared_buffer(t * I * kElt);
  _ff     = mc->make_shared_buffer(t * I * kElt);
  if (_normed.empty() || _q.empty() || _qh.empty() || _k.empty() ||
      _v.empty() || _attn.empty() || _attn_t.empty() ||
      _proj.empty() || _gate.empty() ||
      _up.empty() || _ff.empty()) {
    if (err != nullptr) { *err = "out of memory allocating scratch"; }
    _scratch_tokens = 0;
    return false;
  }
  _scratch_tokens = n;
  return true;
}

void
U15Backbone::final_norm(ComputeEncoder& enc, const SharedBuffer& x, int n,
                        Expert e) const
{
  const SharedBuffer& w =
      (e == Expert::Gen) ? _w->trunk().norm_gen : _w->trunk().norm;
  _ops->rms_norm(enc, x, w, x, n, _cfg.llm.hidden_size,
                 (float)_cfg.llm.rms_norm_eps);
}

std::size_t
U15Backbone::scratch_bytes(int n) const
{
  if (n <= 0) { return 0; }
  const std::size_t H  = (std::size_t)_cfg.llm.hidden_size;
  const std::size_t QD = (std::size_t)_cfg.llm.num_attention_heads *
                         (std::size_t)_cfg.llm.head_dim;
  const std::size_t KD = (std::size_t)_cfg.llm.num_key_value_heads *
                         (std::size_t)_cfg.llm.head_dim;
  const std::size_t I  = (std::size_t)_cfg.llm.intermediate_size;
  // Exactly the eleven buffers ensure_scratch_ allocates, in the same
  // arithmetic -- a predictor that drifts from the allocator it predicts
  // is worse than none, because the number is used to reserve room.
  return (std::size_t)n * kElt *
         (2 * H + 3 * QD + 2 * KD + 3 * I);
}

std::size_t
U15Backbone::scratch_resident_bytes() const
{
  return scratch_bytes(_scratch_tokens);
}

void
U15Backbone::layer_step_(CommandStream& stream, const MotLayer& L,
                         const PassSlot& s, int li, int n, Expert e,
                         Attn attn, const SharedBuffer& rope_cos,
                         const SharedBuffer& rope_sin,
                         const MetalOps::SteelAttn* steel)
{
  const int H  = _cfg.llm.hidden_size;
  const int NH = _cfg.llm.num_attention_heads;
  const int KV = _cfg.llm.num_key_value_heads;
  const int HD = _cfg.llm.head_dim;
  const int I  = _cfg.llm.intermediate_size;
  const int d_t = _cfg.llm.t_dim();
  const int d_h = _cfg.llm.h_dim();
  const float eps = (float)_cfg.llm.rms_norm_eps;

  const ExpertLayer& w = (e == Expert::Gen) ? L.gen : L.und;
  const SharedBuffer& x = *s.x;
  KvCache& kv = *s.kv;

  // ---- attention ------------------------------------------------
  {
    auto enc = stream.begin_compute();
    _ops->rms_norm(enc, x, w.input_ln, _normed, n, H, eps);
    _ops->linear(enc, _normed, w.q, nullptr, _q, n, H, NH * HD);
    _ops->linear(enc, _normed, w.k, nullptr, _k, n, H, KV * HD);
    _ops->linear(enc, _normed, w.v, nullptr, _v, n, H, KV * HD);

    // The split norm and the three-way rope run TOKEN-MAJOR, on what
    // the GEMM just produced, so no transpose is needed before them.
    _ops->split_qk_norm(enc, _q, w.q_norm, w.q_norm_hw, NH, n, HD, eps);
    _ops->split_qk_norm(enc, _k, w.k_norm, w.k_norm_hw, KV, n, HD, eps);
    _ops->split_rope(enc, _q, rope_cos, rope_sin, NH, n, HD, d_t, d_h);
    _ops->split_rope(enc, _k, rope_cos, rope_sin, KV, n, HD, d_t, d_h);

    // v gets neither a norm nor a rope -- it is a plain projection --
    // but it still needs the same head-major placement as k.
    _ops->transpose_thd(enc, _q, _qh, n, NH, HD);
    _ops->transpose_into_cache(enc, _k, kv.k[(std::size_t)li], n, KV, HD,
                               kv.capacity, s.kv_off);
    _ops->transpose_into_cache(enc, _v, kv.v[(std::size_t)li], n, KV, HD,
                               kv.capacity, s.kv_off);
  }
  {
    auto enc = stream.begin_compute();
    // Queries sit at kv_off within the key range. For a prefill that
    // is 0; for a generation block attending a frozen prefix it is
    // prefix_len, and it only matters in the causal case.
    // SDPA writes HEAD-major [NH][n][HD]; o_proj is a GEMM over token
    // rows, so the output is turned back token-major in between.
    if (steel != nullptr) {
      // SAGE LENDS FROM THE TWO PLANES THE TRANSPOSES JUST EMPTIED.
      // `_q` went into `_qh` and `_v` into the cache a few lines above,
      // and neither is read again until the next layer's projections
      // rewrite them -- the MLP below uses its own. Anything that does
      // not fit is allocated privately, per buffer, so this is a saving
      // rather than a requirement.
      if (_ops->sage_takes(HD)) { _ops->sage_lend(_q, _v); }
      _ops->sdpa_steel(enc, *steel, _qh, kv.k[(std::size_t)li],
                       kv.v[(std::size_t)li], _attn, li);
    } else if (attn == Attn::BlockCausal) {
      _ops->sdpa_block_causal(enc, _qh, kv.k[(std::size_t)li],
                              kv.v[(std::size_t)li], _attn, *s.t_index, NH,
                              KV, n, s.kv_len, HD, kv.capacity, s.kv_off);
    } else {
      _ops->sdpa(enc, _qh, kv.k[(std::size_t)li], kv.v[(std::size_t)li],
                 _attn, NH, KV, n, s.kv_len, HD, kv.capacity,
                 attn == Attn::Causal, s.kv_off);
    }
    _ops->transpose_htd(enc, _attn, _attn_t, n, NH, HD);
    _ops->linear(enc, _attn_t, w.o, nullptr, _proj, n, NH * HD, H);
    _ops->add(enc, x, _proj, n * H);                      // residual
  }

  // ---- MLP -------------------------------------------------------
  {
    auto enc = stream.begin_compute();
    _ops->rms_norm(enc, x, w.post_ln, _normed, n, H, eps);
    _ops->linear(enc, _normed, w.gate, nullptr, _gate, n, H, I);
    _ops->linear(enc, _normed, w.up, nullptr, _up, n, H, I);
    _ops->swiglu(enc, _gate, _up, _ff, n, I);
    _ops->linear(enc, _ff, w.down, nullptr, _proj, n, I, H);
    _ops->add(enc, x, _proj, n * H);                      // residual
  }
}

bool
U15Backbone::forward(CommandStream& stream, const SharedBuffer& x, int n,
                     Expert e, KvCache& kv, int kv_off, int kv_len,
                     const std::vector<int>& pos_t,
                     const std::vector<int>& pos_h,
                     const std::vector<int>& pos_w, Attn attn,
                     std::string* err, const SharedBuffer* t_index)
{
  PassSlot s;
  s.x = &x;
  s.kv = &kv;
  s.kv_off = kv_off;
  s.kv_len = kv_len;
  s.pos_t = &pos_t;
  s.pos_h = &pos_h;
  s.pos_w = &pos_w;
  s.t_index = t_index;
  return forward_many(stream, {s}, n, e, attn, err);
}

bool
U15Backbone::forward_many(CommandStream& stream,
                          const std::vector<PassSlot>& slots, int n,
                          Expert e, Attn attn, std::string* err)
{
  const auto fail = [err](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (slots.empty()) { return fail("no branches to run"); }

  for (const PassSlot& s : slots) {
    if (s.x == nullptr || s.kv == nullptr) { return fail("empty slot"); }
    if (!s.kv->valid()) { return fail("no KV cache"); }
    if (s.kv_off + n > s.kv->capacity) {
      return fail("KV cache too small: need " +
                  std::to_string(s.kv_off + n) + ", capacity " +
                  std::to_string(s.kv->capacity));
    }
    if (s.kv_len < s.kv_off + n) {
      return fail("kv_len is shorter than the tokens just written");
    }
    if (s.pos_t == nullptr || s.pos_h == nullptr || s.pos_w == nullptr ||
        (int)s.pos_t->size() != n || (int)s.pos_h->size() != n ||
        (int)s.pos_w->size() != n) {
      return fail("position arrays do not match the token count");
    }
    if (attn == Attn::BlockCausal &&
        (s.t_index == nullptr || s.t_index->empty())) {
      return fail("Attn::BlockCausal needs the t-index vector");
    }
  }
  if (!ensure_scratch_(n, err)) { return false; }
  // The arena now exists, so the residency policy must stop reserving
  // room for it: a budget that has already subtracted these bytes and
  // is then asked to keep them free as well is asking twice, and the
  // model streams to protect memory sitting in its own hands.
  _w->note_arena_allocated(scratch_resident_bytes());
  // The scratch takes its place in the wired pool BEFORE begin_pass()
  // wires the trunk and before this pass starts admitting layers. See
  // scratch_buffers(): protecting the shed-able half first is the
  // failure this ordering exists to avoid.
  _w->wire_scratch(scratch_buffers());

  const int HD = _cfg.llm.head_dim;
  const int NH = _cfg.llm.num_attention_heads;
  const int KV = _cfg.llm.num_key_value_heads;
  const int d_t = _cfg.llm.t_dim();
  const int d_h = _cfg.llm.h_dim();

  // ---- per-slot preparation, ONCE for the whole stack ---------------
  //
  // The rope tables and the steel plan depend on the branch's positions
  // and key length, not on the layer, so building them here rather than
  // inside the loop is 42x less host work for identical bytes.
  struct Prep {
    MetalOps::RopeTables tab;
    bool                 steel = false;
  };
  std::vector<Prep> prep(slots.size());

  // The bidirectional path is the hot one: 42 layers x every step x
  // every CFG branch. MEASURED at 1024 queries against 1285 keys, GQA
  // 32/8: the scalar sdpa_full_f16 runs at 176 GFLOP/s and the steel
  // flash kernel at 4269 -- 24x, or 5.1 s per forward saved.
  //
  // The two masked paths are NOT switched over, and deliberately: each
  // runs ONCE per generation (the prefill), so at 1024^2 they are ~0.1 s
  // against a 20-step render's ~150 s. Block-causal has no steel
  // instantiation at all -- it would need the mask buffer, which is
  // O(T^2) per layer.
  const bool use_steel =
      (attn == Attn::Bidirectional) && _ops->steel_attn_available(HD);
  if (_steel.size() < slots.size()) { _steel.resize(slots.size()); }
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const PassSlot& s = slots[i];
    prep[i].tab = _ops->build_rope_tables(*s.pos_t, *s.pos_h, *s.pos_w, HD,
                                          d_t, d_h, _cfg.llm.rope_theta,
                                          _cfg.llm.rope_theta_hw);
    if (prep[i].tab.cos.empty()) {
      return fail("could not build the rope tables");
    }
    if (use_steel) {
      prep[i].steel =
          _ops->steel_attn_plan(&_steel[i], NH, KV, n, s.kv_len, HD,
                                s.kv->capacity) &&
          _steel[i].valid();
    }
  }

  // ---- the stack ----------------------------------------------------
  //
  // Layer outermost, branch innermost. On a streamed model that is what
  // makes one step cost one read of the checkpoint; on a preloaded one
  // it changes nothing but the order the commands are queued in.
  _w->begin_pass();
  const int n_layers = _cfg.llm.num_hidden_layers;
  const bool streaming = _w->streaming();
  // JOIN ANY OUTSTANDING READ ON EVERY EXIT. A prefetch may be filling
  // a slot, and returning while a reader thread writes into it is a
  // use-after-free. A scope guard is the only version of this that
  // cannot be forgotten at the next early return.
  struct ReadJoin {
    U15Weights* w;
    ~ReadJoin() { w->join_reads(); }
  } read_join{_w};
  CommandStream::Fence fence;
  for (int li = 0; li < n_layers; ++li) {
    // THE COOPERATIVE STOP, before the layer's weights are asked for.
    // Here rather than between forwards because on a streamed model one
    // forward IS the checkpoint: at 1024 square the stack is minutes,
    // and a Stop noticed only at the end of it is a Stop that appears
    // not to work. The ReadJoin guard above makes this early return
    // safe -- a prefetch may be filling a slot, and leaving while a
    // reader writes into it is a use-after-free.
    if (_stop && _stop()) {
      if (err != nullptr) { *err = kStopped; }
      return false;
    }
    const MotLayer* L = _w->layer(li, err);
    if (L == nullptr) { return false; }

    for (std::size_t i = 0; i < slots.size(); ++i) {
      layer_step_(stream, *L, slots[i], li, n, e, attn, prep[i].tab.cos,
                  prep[i].tab.sin,
                  prep[i].steel ? &_steel[i] : nullptr);
    }

    // Commit per layer rather than building one command buffer for the
    // whole stack: 42 layers x ~15 dispatches x the branch count is
    // large enough that a single buffer delays first work and makes a
    // mid-stack abort impossible. Commits on one stream are ordered, so
    // waiting on the LAST fence waits for all of them.
    fence = stream.commit();
    if (streaming) {
      // BETWEEN THE COMMIT AND THE WAIT the GPU is busy with layer `li`
      // and this thread has nothing to do. That window is where the
      // next layer's read goes, and it is why the slots are worth
      // keeping: the destination already exists, so issuing the read
      // costs no memory and asks no growth question.
      _w->prefetch_after(li);
      // A streamed layer's buffers are OVERWRITTEN (or copied out) by a
      // later acquire, so nothing encoded may still point at them.
      fence.wait();
      _w->end_layer(li);
    }
  }
  fence.wait();

  for (const PassSlot& s : slots) {
    if (s.kv_off == 0) { s.kv->prefix_len = n; }
  }
  return true;
}

}  // namespace u15
