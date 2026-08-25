#include "u15-weights.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/streamed-refill.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace u15 {

// Shorthand for the per-tensor placement the refill needs stated.
using P = vpipe::genai::Placement;

namespace {

// Cached, or read once and dropped.
//
// The distinction is the WeightSet contract, not a preference: a cached
// entry is deduped, parkable and held for the set's life, and a streamed
// one is counted as throughput and owned by whoever asked. Caching a
// streamed layer would put the whole model back in RAM and defeat the
// mechanism, so the two paths are named rather than left to a bool at
// the call sites.
enum class Retain { Cached, Streamed };

// The base expert ships BF16 and the gen expert F32, but that is read
// off each tensor rather than assumed from which expert it belongs to:
// the dtype is per-tensor in the safetensors header, and a checkpoint
// that ships differently must still load. A tensor already in bf16 is
// bound with NO copy at all.
// The two dtype conversions, factored out so the BUILD path and the
// REFILL path cannot diverge.
//
// They must be BIT-IDENTICAL: a refilled layer and a freshly built one
// are the same weights arriving by different routes, and the streaming
// test holds the rendered image to equality. A second copy of "round to
// nearest even" that drifted by a half-ULP would fail that test without
// pointing at itself.
void
f32_to_bf16_(const void* src, void* dst, std::size_t n)
{
  const auto* in = static_cast<const float*>(src);
  auto* o = static_cast<std::uint16_t*>(dst);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t u;
    std::memcpy(&u, &in[i], 4);
    // Round to nearest even, as torch does. Truncating instead is a
    // half-ULP bias that accumulates over 42 layers in one direction.
    const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
    o[i] = (std::uint16_t)((u + r) >> 16);
  }
}

void
f16_to_bf16_(const void* src, void* dst, std::size_t n)
{
  const auto* in = static_cast<const std::uint16_t*>(src);
  auto* o = static_cast<std::uint16_t*>(dst);
  for (std::size_t i = 0; i < n; ++i) {
    // f16 -> f32 -> bf16. Going straight would have to renormalise the
    // exponent bias by hand for no gain; this runs once per tensor.
    const std::uint16_t h = in[i];
    const std::uint32_t sign = (std::uint32_t)(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t man = h & 0x3ffu;
    std::uint32_t f;
    if (exp == 0) {
      if (man == 0) { f = sign; }
      else {
        exp = 127 - 15 + 1;
        while ((man & 0x400u) == 0) { man <<= 1; --exp; }
        man &= 0x3ffu;
        f = sign | (exp << 23) | (man << 13);
      }
    } else if (exp == 31) {
      f = sign | 0x7f800000u | (man << 13);
    } else {
      f = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    const std::uint32_t r = ((f >> 16) & 1u) + 0x7fffu;   // RNE
    o[i] = (std::uint16_t)((f + r) >> 16);
  }
}

// The base expert ships BF16 and the gen expert F32, but that is read
// off each tensor rather than assumed from which expert it belongs to:
// the dtype is per-tensor in the safetensors header, and a checkpoint
// that ships differently must still load. A tensor already in bf16 is
// bound with NO copy at all.
SharedBuffer
bind_bf16_(WeightSet& ws, MetalCompute* mc, const std::string& name,
           const std::string& part, std::size_t* converted,
           Retain retain = Retain::Cached)
{
  if (!ws.has(name)) { return SharedBuffer{}; }
  const auto* ti = ws.src().info(name);
  if (ti == nullptr) { return SharedBuffer{}; }

  if (ti->dtype == "BF16") {
    // Keep it as it sits. This is the accessor for "the model KEEPS the
    // tensor as-is" -- cached, deduped across models, and parkable.
    if (retain == Retain::Streamed) {
      return ws.stream_tensor(name, mc, WeightSet::Residency::Copied);
    }
    return ws.tensor(name, mc, WeightSet::Residency::Copied, part);
  }

  if (ti->dtype != "F32" && ti->dtype != "F16") { return SharedBuffer{}; }

  // The key names the transform AND the source dtype, because that key
  // is all the cache compares.
  const std::string key = "u15/bf16/" + ti->dtype + "/" + name;
  const auto build = [&ws, mc, &name, ti]() {
    // Read UNCACHED: the source is consumed by the conversion and must
    // not survive next to the product.
    const SharedBuffer src =
        ws.read(name, mc, WeightSet::Residency::Copied);
    if (src.empty()) { return SharedBuffer{}; }
    const std::size_t n = src.byte_size() / (ti->dtype == "F16" ? 2 : 4);
    SharedBuffer dst = mc->make_shared_buffer(n * 2);
    if (dst.empty()) { return SharedBuffer{}; }
    if (ti->dtype == "F16") {
      f16_to_bf16_(src.contents(), dst.contents(), n);
    } else {
      f32_to_bf16_(src.contents(), dst.contents(), n);
    }
    return dst;
  };

  // stream_derived() runs the builder EVERY time and caches nothing --
  // reaching for derived() here would silently pin the layer that was
  // just streamed, which is the one mistake this path exists to avoid.
  SharedBuffer out = retain == Retain::Streamed
                         ? ws.stream_derived(build)
                         : ws.derived(key, build, part);

  if (!out.empty() && converted != nullptr) { *converted += out.byte_size(); }
  return out;
}

// Bind and record the miss, so a failed load names the FIRST tensor it
// could not find rather than reporting a generic failure 400 tensors in.
bool
need_(WeightSet& ws, MetalCompute* mc, const std::string& name,
      const std::string& part, SharedBuffer& dst, std::string* miss,
      std::size_t* converted, Retain retain = Retain::Cached)
{
  dst = bind_bf16_(ws, mc, name, part, converted, retain);
  if (dst.empty()) {
    if (miss != nullptr && miss->empty()) { *miss = name; }
    return false;
  }
  return true;
}

// THE PIXEL HEAD AND THE SCALAR EMBEDDERS, ALWAYS AS BF16.
//
// These twelve tensors drive every modulation and every output pixel,
// and the checkpoint has published them BOTH ways: the original release
// stored them F32, and the bf16 re-upload that became `main` a few days
// later stores them BF16. Same weights, different width.
//
// This used to bind whatever was in the file and hand it to a reader
// that assumed F32 -- so on the bf16 revision the modulation MLP took
// its frequency count from byte_size()/4 (half the real one) and read
// bf16 pairs as floats. It did not fail: it rendered a plausible image
// made of wrong pixels, with every test in this repo green, because the
// backbone is bf16 either way and the layer and streaming tests never
// touch fm_modules.
//
// So the dtype is CHECKED rather than assumed, and BF16 is the one form
// callers get. Down-converting F32 is the direction that costs nothing
// -- these are a few hundred kilobytes, and bf16 is what `main` ships
// now -- where widening bf16 to f32 would fabricate precision the
// checkpoint does not have.
SharedBuffer
bind_bf16_scalar_(WeightSet& ws, MetalCompute* mc, const std::string& name,
                  const std::string& part, std::string* why)
{
  if (!ws.has(name)) { return SharedBuffer{}; }
  const auto* ti = ws.src().info(name);
  if (ti == nullptr) { return SharedBuffer{}; }
  if (ti->dtype == "BF16") {
    return ws.tensor(name, mc, WeightSet::Residency::Copied, part);
  }
  if (ti->dtype != "F32") {
    // NAMED, with the dtype that surprised us. A third width here is a
    // checkpoint this build has never seen, and guessing at it is how
    // the bug above happened.
    if (why != nullptr && why->empty()) {
      *why = name + " has dtype " + ti->dtype +
             ", which is neither F32 nor BF16";
    }
    return SharedBuffer{};
  }
  // F32 -> BF16, once, cached. The key names the transform and the
  // destination width, because that is all the cache compares.
  return ws.derived("u15/scalar-bf16/" + name, [&]() {
    const SharedBuffer src =
        ws.read(name, mc, WeightSet::Residency::Copied);
    if (src.empty()) { return SharedBuffer{}; }
    const std::size_t n = src.byte_size() / sizeof(float);
    SharedBuffer dst = mc->make_shared_buffer(n * 2);
    if (dst.empty()) { return SharedBuffer{}; }
    const auto* f = static_cast<const float*>(src.contents());
    auto* d = static_cast<std::uint16_t*>(dst.contents());
    // Round-to-nearest-even, matching the conversion the streamed
    // layers use -- a truncating cast here would be a second, quieter
    // way for these tensors to be slightly wrong.
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u;
      std::memcpy(&u, f + i, 4);
      const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
      d[i] = (std::uint16_t)((u + r) >> 16);
    }
    return dst;
  }, part);
}

bool
need_bf16_scalar_(WeightSet& ws, MetalCompute* mc, const std::string& name,
                  const std::string& part, SharedBuffer& dst,
                  std::string* miss)
{
  std::string why;
  dst = bind_bf16_scalar_(ws, mc, name, part, &why);
  if (dst.empty()) {
    if (miss != nullptr && miss->empty()) {
      *miss = why.empty() ? name : why;
    }
    return false;
  }
  return true;
}

// Bind a MATRIX, quantized or dense.
//
// The pack's shape is what says which: a `.scales` sibling means the
// `.weight` is u32 CODES rather than values. bits and group are then
// DERIVED from the two column counts against the known K --
//
//   group = K / scales_cols        codes_cols * 32 = K * bits
//
// -- rather than read from a config, because model-quantize can write a
// mixed pack and a per-checkpoint answer would be wrong for it. A shape
// that does not close is refused loudly: the tensor IS there, it just
// does not mean what a quantized tensor means, and running it anyway
// produces a checkpoint that loads and generates the wrong thing.
bool
bind_matrix_(WeightSet& ws, MetalCompute* mc, const std::string& name,
             int K, QWeight& out, int* bits, int* group,
             const std::string& part, std::string* miss, std::string* qerr,
             std::size_t* converted, Retain retain = Retain::Cached)
{
  const auto& src = ws.src();
  const auto* si = src.info(name + ".scales");
  const auto* ci = src.info(name + ".weight");
  if (si != nullptr && ci != nullptr && si->shape.size() == 2 &&
      ci->shape.size() == 2 && K > 0) {
    const long scols = si->shape[1];
    const long gcols = ci->shape[1];
    const long g = scols > 0 ? (long)K / scols : 0;
    const long b = (long)gcols * 32 / (long)K;
    if ((g != 32 && g != 64) || (b != 4 && b != 8) ||
        scols * g != (long)K || gcols * 32 != (long)K * b) {
      if (qerr != nullptr && qerr->empty()) {
        *qerr = "'" + name + "' looks quantized but its shapes do not "
                "close: K=" + std::to_string(K) + ", scales cols=" +
                std::to_string(scols) + ", codes cols=" +
                std::to_string(gcols);
      }
      return false;
    }
    if (*bits == 0) {
      *bits = (int)b;
      *group = (int)g;
    } else if (*bits != (int)b || *group != (int)g) {
      if (qerr != nullptr && qerr->empty()) {
        *qerr = "'" + name + "' is packed at w" + std::to_string(b) + "g" +
                std::to_string(g) + " but the checkpoint already used w" +
                std::to_string(*bits) + "g" + std::to_string(*group) +
                "; this build reads one packing per checkpoint";
      }
      return false;
    }
    out.bits = (int)b;
    out.group = (int)g;
    out.quantized = true;
    // The CODES are u32 and pass through untouched; the scales and
    // biases are F16 in the pack and bfloat to the kernel.
    out.codes = retain == Retain::Streamed
                    ? ws.stream_tensor(name + ".weight", mc,
                                       WeightSet::Residency::Copied)
                    : ws.tensor(name + ".weight", mc,
                                WeightSet::Residency::Copied, part);
    out.scales =
        bind_bf16_(ws, mc, name + ".scales", part, converted, retain);
    out.qbias =
        bind_bf16_(ws, mc, name + ".biases", part, converted, retain);
    if (out.empty()) {
      if (miss != nullptr && miss->empty()) { *miss = name + " (quantized)"; }
      return false;
    }
    return true;
  }
  out.quantized = false;
  out.w = bind_bf16_(ws, mc, name + ".weight", part, converted, retain);
  if (out.w.empty()) {
    if (miss != nullptr && miss->empty()) { *miss = name + ".weight"; }
    return false;
  }
  return true;
}

constexpr const char* kTrunkPart  = "";
constexpr const char* kLayersPart = "u15-layers";

}  // namespace

bool
ExpertLayer::complete() const
{
  return !input_ln.empty() && !post_ln.empty() && !q.empty() &&
         !k.empty() && !v.empty() && !o.empty() && !q_norm.empty() &&
         !k_norm.empty() && !q_norm_hw.empty() && !k_norm_hw.empty() &&
         !gate.empty() && !up.empty() && !down.empty();
}

std::unique_ptr<U15Weights>
U15Weights::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                 const U15Config& cfg, const Options& opt, std::string* err)
{
  const auto fail = [err](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<U15Weights>{};
  };
  if (ws == nullptr || mc == nullptr) { return fail("no weight set"); }

  auto out = std::unique_ptr<U15Weights>(new U15Weights());
  out->_ws = ws;                    // held for this object's lifetime
  out->_mc = mc;
  out->_cfg = cfg;
  WeightSet& s = *ws;
  std::string miss;

  // ---- trunk --------------------------------------------------------
  Trunk& t = out->_trunk;
  bool ok = true;
  ok &= need_(s, mc, "language_model.model.embed_tokens.weight",
              kTrunkPart, t.embed_tokens, &miss, &out->_converted);
  ok &= need_(s, mc, "language_model.model.norm.weight", kTrunkPart,
              t.norm, &miss, &out->_converted);
  ok &= need_(s, mc, "language_model.model.norm_mot_gen.weight",
              kTrunkPart, t.norm_gen, &miss, &out->_converted);
  if (opt.with_lm_head) {
    ok &= need_(s, mc, "language_model.lm_head.weight", kTrunkPart,
                t.lm_head, &miss, &out->_converted);
  }

  // Both patch embedders. The generation one lives under fm_modules and
  // the understanding one at the top level, but they are the same module
  // with different weights.
  //
  // WEIGHTS AND BIASES BOTH GO THROUGH need_(). They ship F32 and are
  // consumed by a bf16 GEMM, so both must be converted -- converting
  // only the weight and handing the GEMM raw f32 bias bytes makes it
  // read two f32s as four bf16s, and the embedding comes out NaN. That
  // is what happened the first time this was written, and it surfaced
  // as a NaN in the PIXEL HEAD, four stages downstream.
  const std::string vg = "fm_modules.vision_model_mot_gen.embeddings.";
  const std::string vu = "vision_model.embeddings.";
  ok &= need_(s, mc, vu + "patch_embedding.weight", kTrunkPart,
              t.und_patch_w, &miss, &out->_converted);
  ok &= need_(s, mc, vu + "patch_embedding.bias", kTrunkPart,
                  t.und_patch_b, &miss, &out->_converted);
  ok &= need_(s, mc, vu + "dense_embedding.weight", kTrunkPart,
              t.und_dense_w, &miss, &out->_converted);
  ok &= need_(s, mc, vu + "dense_embedding.bias", kTrunkPart,
                  t.und_dense_b, &miss, &out->_converted);
  ok &= need_(s, mc, vg + "patch_embedding.weight", kTrunkPart,
              t.gen_patch_w, &miss, &out->_converted);
  ok &= need_(s, mc, vg + "patch_embedding.bias", kTrunkPart,
                  t.gen_patch_b, &miss, &out->_converted);
  ok &= need_(s, mc, vg + "dense_embedding.weight", kTrunkPart,
              t.gen_dense_w, &miss, &out->_converted);
  ok &= need_(s, mc, vg + "dense_embedding.bias", kTrunkPart,
                  t.gen_dense_b, &miss, &out->_converted);

  // The pixel head. Bound raw; ImagePath repacks it for im2col and
  // converts from whatever dtype it turns out to be.
  ok &= need_bf16_scalar_(s, mc, "fm_modules.fm_head.conv1.weight", kTrunkPart,
                  t.conv1_w, &miss);
  ok &= need_bf16_scalar_(s, mc, "fm_modules.fm_head.conv1.bias", kTrunkPart,
                  t.conv1_b, &miss);
  ok &= need_bf16_scalar_(s, mc, "fm_modules.fm_head.conv2.weight", kTrunkPart,
                  t.conv2_w, &miss);
  ok &= need_bf16_scalar_(s, mc, "fm_modules.fm_head.conv2.bias", kTrunkPart,
                  t.conv2_b, &miss);

  const std::string te = "fm_modules.timestep_embedder.mlp.";
  const std::string ne = "fm_modules.noise_scale_embedder.mlp.";
  ok &= need_bf16_scalar_(s, mc, te + "0.weight", kTrunkPart, t.tstep_w0, &miss);
  ok &= need_bf16_scalar_(s, mc, te + "0.bias", kTrunkPart, t.tstep_b0, &miss);
  ok &= need_bf16_scalar_(s, mc, te + "2.weight", kTrunkPart, t.tstep_w2, &miss);
  ok &= need_bf16_scalar_(s, mc, te + "2.bias", kTrunkPart, t.tstep_b2, &miss);
  if (cfg.gen.add_noise_scale_embedding) {
    ok &= need_bf16_scalar_(s, mc, ne + "0.weight", kTrunkPart, t.nscale_w0, &miss);
    ok &= need_bf16_scalar_(s, mc, ne + "0.bias", kTrunkPart, t.nscale_b0, &miss);
    ok &= need_bf16_scalar_(s, mc, ne + "2.weight", kTrunkPart, t.nscale_w2, &miss);
    ok &= need_bf16_scalar_(s, mc, ne + "2.bias", kTrunkPart, t.nscale_b2, &miss);
  }

  if (!ok) { return fail("missing trunk tensor: " + miss); }

  // ---- layers -------------------------------------------------------
  int n = cfg.llm.num_hidden_layers;
  if (opt.max_layers >= 0 && opt.max_layers < n) { n = opt.max_layers; }
  out->_layers.resize((std::size_t)n);

  // The checkpoint's widest layer ON DISK. Only a fallback -- the real
  // figure is measured off the first layer actually built (see
  // layer_bytes()) -- but it is the only one available before that, and
  // it is what sizes the pinned prefix.
  {
    const auto names = ws->src().tensor_names();
    out->_disk_layer_bytes = vpipe::genai::widest_block_bytes(
        names,
        [&ws](const std::string& nm) -> std::size_t {
          const auto* ti = ws->src().info(nm);
          return ti != nullptr ? (std::size_t)ti->nbytes : 0;
        },
        {"language_model.model.layers."});
  }

  out->_streaming = opt.stream_layers;
  out->_wire_allowed = opt.wire_resident;
  out->_refill = opt.refill_streamed;
  out->_pinned = 0;
  if (out->_streaming) {
    out->_pinned = opt.pinned_layers < 0 ? 0 : opt.pinned_layers;
    if (out->_pinned > n) { out->_pinned = n; }
  }

  if (out->_streaming) { out->configure_slots_(); }

  // Build every layer, or -- streaming -- only the pinned prefix. The
  // rest are left EMPTY and arrive through layer(), which is also what
  // makes an unbuilt entry unambiguous: `und.q.empty()` is the one test
  // for "this layer is not here".
  const int n_build = out->_streaming ? out->_pinned : n;
  for (int i = 0; i < n_build; ++i) {
    if (!out->load_layer_(i, &out->_layers[(std::size_t)i], false, err)) {
      return {};
    }
  }

  // A streaming model still has to know its packing before the first
  // forward -- the stage refuses a quantized pack on a host with no qmm
  // kernels, and it cannot wait 42 layers to find out. Layer 0's
  // matrices answer it, so read them and drop them.
  if (out->_streaming && out->_pinned == 0 && n > 0) {
    MotLayer probe;
    if (!out->load_layer_(0, &probe, true, err)) { return {}; }
    // Not a pass, so it is not streaming traffic. The counters report
    // what the RUN moves; a load-time read folded into them would make
    // the first beat look like it read one layer more than it did.
    out->_streamed_bytes = 0;
    out->_streamed_layers = 0;
  }

  // What this object caused to be held. Counted from the buffers rather
  // than from the config, so it reports what was ACTUALLY bound -- the
  // number is used for a memory declaration, and an estimate that
  // disagrees with reality is worse than none.
  std::size_t bytes = 0;
  const auto add = [&bytes](const SharedBuffer& b) {
    bytes += b.byte_size();
  };
  add(t.embed_tokens); add(t.norm); add(t.norm_gen); add(t.lm_head);
  add(t.und_patch_w); add(t.und_patch_b);
  add(t.und_dense_w); add(t.und_dense_b);
  add(t.gen_patch_w); add(t.gen_patch_b);
  add(t.gen_dense_w); add(t.gen_dense_b);
  add(t.conv1_w); add(t.conv1_b); add(t.conv2_w); add(t.conv2_b);
  add(t.tstep_w0); add(t.tstep_b0); add(t.tstep_w2); add(t.tstep_b2);
  add(t.nscale_w0); add(t.nscale_b0); add(t.nscale_w2); add(t.nscale_b2);
  for (const MotLayer& L : out->_layers) {
    for (const ExpertLayer* e : {&L.und, &L.gen}) {
      add(e->input_ln); add(e->post_ln);
      bytes += e->q.byte_size() + e->k.byte_size() + e->v.byte_size() +
               e->o.byte_size() + e->gate.byte_size() +
               e->up.byte_size() + e->down.byte_size();
      add(e->q_norm); add(e->k_norm);
      add(e->q_norm_hw); add(e->k_norm_hw);
    }
  }
  out->_bytes = bytes;
  return out;
}


// ---- one layer -------------------------------------------------------

bool
U15Weights::load_layer_(int i, MotLayer* into, bool streamed,
                        std::string* err)
{
  WeightSet& s = *_ws;
  const Retain r = streamed ? Retain::Streamed : Retain::Cached;
  // Streamed reads retain nothing, so there is no part to attribute
  // them to and no conversion counter worth keeping: `_converted`
  // reports what LOAD cost, and a figure that grew on every pass would
  // stop meaning that.
  std::size_t* conv = streamed ? nullptr : &_converted;
  const std::string part = kLayersPart;

  const int H  = _cfg.llm.hidden_size;
  const int I  = _cfg.llm.intermediate_size;
  const int NH = _cfg.llm.num_attention_heads;
  const int HD = _cfg.llm.head_dim;

  const std::string p =
      "language_model.model.layers." + std::to_string(i) + ".";
  std::string miss, qerr;
  bool ok = true;

  // The suffix pair IS the MoT split: the same thirteen tensors, once
  // bare and once with _mot_gen. `lsfx` differs from `sfx` because the
  // MLP and the layernorms suffix the MODULE (mlp_mot_gen.gate_proj)
  // while the attention suffixes the TENSOR (q_proj_mot_gen).
  struct Bind { ExpertLayer* e; const char* sfx; const char* lsfx; };
  const Bind binds[2] = {{&into->und, "", ""},
                         {&into->gen, "_mot_gen", "_mot_gen"}};
  for (const Bind& b : binds) {
    ExpertLayer& e = *b.e;
    const std::string sfx(b.sfx), lsfx(b.lsfx);
    const std::string a = p + "self_attn.";
    ok &= need_(s, _mc, p + "input_layernorm" + lsfx + ".weight", part,
                e.input_ln, &miss, conv, r);
    ok &= need_(s, _mc, p + "post_attention_layernorm" + lsfx + ".weight",
                part, e.post_ln, &miss, conv, r);
    // The four attention MATRICES. K is the INPUT width, which is what
    // the quantized-shape check closes against: hidden for q/k/v, and
    // heads*head_dim for o.
    ok &= bind_matrix_(s, _mc, a + "q_proj" + sfx, H, e.q, &_qbits,
                       &_qgroup, part, &miss, &qerr, conv, r);
    ok &= bind_matrix_(s, _mc, a + "k_proj" + sfx, H, e.k, &_qbits,
                       &_qgroup, part, &miss, &qerr, conv, r);
    ok &= bind_matrix_(s, _mc, a + "v_proj" + sfx, H, e.v, &_qbits,
                       &_qgroup, part, &miss, &qerr, conv, r);
    ok &= bind_matrix_(s, _mc, a + "o_proj" + sfx, NH * HD, e.o, &_qbits,
                       &_qgroup, part, &miss, &qerr, conv, r);
    ok &= need_(s, _mc, a + "q_norm" + sfx + ".weight", part, e.q_norm,
                &miss, conv, r);
    ok &= need_(s, _mc, a + "k_norm" + sfx + ".weight", part, e.k_norm,
                &miss, conv, r);
    ok &= need_(s, _mc, a + "q_norm_hw" + sfx + ".weight", part,
                e.q_norm_hw, &miss, conv, r);
    ok &= need_(s, _mc, a + "k_norm_hw" + sfx + ".weight", part,
                e.k_norm_hw, &miss, conv, r);
    ok &= bind_matrix_(s, _mc, p + "mlp" + lsfx + ".gate_proj", H, e.gate,
                       &_qbits, &_qgroup, part, &miss, &qerr, conv, r);
    ok &= bind_matrix_(s, _mc, p + "mlp" + lsfx + ".up_proj", H, e.up,
                       &_qbits, &_qgroup, part, &miss, &qerr, conv, r);
    ok &= bind_matrix_(s, _mc, p + "mlp" + lsfx + ".down_proj", I, e.down,
                       &_qbits, &_qgroup, part, &miss, &qerr, conv, r);
    if (!ok) {
      // A quantization-shape complaint says far more than "missing X"
      // -- the tensor IS there, it just does not close -- so it wins
      // the message.
      if (err != nullptr) {
        *err = "layer " + std::to_string(i) + ": " +
               (qerr.empty() ? "missing '" + miss + "'" : qerr);
      }
      return false;
    }
  }

  // The measured per-layer figure, taken from the first layer that is
  // ever built. Every layer of this stack is the same shape, so one is
  // enough -- and this is the number admission decisions run on, so it
  // has to be what a layer really costs rather than what the checkpoint
  // says it weighs (the two differ by the whole F32 conversion).
  if (_layer_bytes == 0) { _layer_bytes = layer_resident_bytes_(*into); }
  // NOT counted here. layer() counts every streamed read once, and a
  // rebuild reaches this function THROUGH layer() -- counting in both
  // reported a rebuild-only run at exactly twice its real traffic,
  // which read as the fast path being slower than it is.
  (void)streamed;
  return true;
}

std::size_t
U15Weights::layer_resident_bytes_(const MotLayer& L) const
{
  std::size_t b = 0;
  for (const ExpertLayer* e : {&L.und, &L.gen}) {
    b += e->input_ln.byte_size() + e->post_ln.byte_size();
    b += e->q.byte_size() + e->k.byte_size() + e->v.byte_size() +
         e->o.byte_size() + e->gate.byte_size() + e->up.byte_size() +
         e->down.byte_size();
    b += e->q_norm.byte_size() + e->k_norm.byte_size() +
         e->q_norm_hw.byte_size() + e->k_norm_hw.byte_size();
  }
  return b;
}

std::size_t
U15Weights::resident_bytes() const
{
  // The promoted layers are the model's own buffers, not cache entries
  // in the weight set, so nothing else counts them. _resid.bytes() is
  // exactly what admission booked.
  return _bytes + _resid.bytes();
}

std::size_t
U15Weights::layer_bytes() const
{
  return _layer_bytes > 0 ? _layer_bytes : _disk_layer_bytes;
}

int
U15Weights::resident_layers() const
{
  return _streaming ? _pinned + (int)_promoted.size() : (int)_layers.size();
}

// ---- wiring ----------------------------------------------------------
//
// All-or-nothing per layer, and the return is what the POOL took rather
// than what was asked: a partly wired layer is partly protected, which
// is the state wirable() exists to keep the model out of.

std::size_t
U15Weights::wire_layer_(MotLayer& L, bool on)
{
  std::size_t got = 0;
  bool stop = false;
  const auto one = [&](vpipe::metal_compute::SharedBuffer& b) {
    if (stop) { return; }
    const std::size_t n = _wire.wire_one(_mc, b, on);
    // STOP at the first refusal rather than unwinding: a partly wired
    // layer is partly protected, which is strictly better than none --
    // and giving protection back on the way out means competing for it
    // again on the next layer, against a pool that has just said no.
    if (on && n == 0 && b.byte_size() > 0 && !b.is_wired()) {
      stop = true;
      return;
    }
    got += n;
  };
  for (ExpertLayer* e : {&L.und, &L.gen}) {
    one(e->input_ln); one(e->post_ln);
    for (QWeight* q : {&e->q, &e->k, &e->v, &e->o, &e->gate, &e->up,
                       &e->down}) {
      one(q->w); one(q->codes); one(q->scales); one(q->qbias);
    }
    one(e->q_norm); one(e->k_norm); one(e->q_norm_hw); one(e->k_norm_hw);
  }
  return got;
}

// Give the pool back, because this model has been told to let go.
//
// WIRED AND PARKED ARE OPPOSITES and cannot both be true: mark_inactive()
// refuses a wired buffer outright, so parking a model that is still
// wired reclaims nothing at all. The idle path therefore unwires first
// and parks second, and the next pass re-wires -- wire_trunk_ runs every
// pass and is a no-op per buffer that is already in the state asked for.
//
// The activation arena is deliberately left alone: it is this model's
// working set rather than its weights, it will be used again on the next
// beat, and `park` is a weights policy.
std::size_t
U15Weights::wire_down()
{
  if (_mc == nullptr || !_wire.on()) { return 0; }
  const std::size_t n = wire_trunk_(false);
  _wire.note_unwired(n);
  return n;
}

void
U15Weights::wire_scratch(
    const std::vector<vpipe::metal_compute::SharedBuffer*>& bufs)
{
  if (_mc == nullptr || !_wire.on()) { return; }
  for (vpipe::metal_compute::SharedBuffer* b : bufs) {
    if (b != nullptr) { _wire.wire_one(_mc, *b, true); }
  }
}

void
U15Weights::unwire_scratch(
    const std::vector<vpipe::metal_compute::SharedBuffer*>& bufs)
{
  if (_mc == nullptr || !_wire.on()) { return; }
  for (vpipe::metal_compute::SharedBuffer* b : bufs) {
    if (b != nullptr) { _wire.wire_one(_mc, *b, false); }
  }
}

// The TRUNK: everything the weight set CACHED for this model, which for
// a streaming model is what it holds for the whole run. Read on every
// layer of every pass and never shed, so it has a better claim on the
// pool than any single resident layer does.
//
// Through for_each_weight() rather than by naming the Trunk members,
// which matters for two reasons. It wires the CACHE ENTRY rather than
// this model's alias of it, so a second model over the same checkpoint
// finds it already wired instead of paying for it twice. And it
// includes the PINNED PREFIX, whose tensors are cached under the layers
// part -- a hand-written list of trunk members would silently leave
// them out.
//
// What it does NOT reach is `derived()` entries: for_each_weight yields
// only owned tensors whose source name is still known, and a derived
// one has no retained transform to re-read it with. On this checkpoint
// that is the whole F32->bf16 half -- see resident_pages_ for why that
// is exactly the case the page walk still has to cover.
//
// Re-run every pass rather than latched: the activation arena is
// replaced when it grows, and wire_one() is a no-op for a buffer
// already in the state asked for, so the repeat costs a branch each.
std::size_t
U15Weights::wire_trunk_(bool on)
{
  if (_mc == nullptr || !_wire.on() || _ws == nullptr) { return 0; }
  std::size_t changed = 0;
  _ws->for_each_weight([&](vpipe::metal_compute::SharedBuffer& b) {
    changed += _wire.wire_one(_mc, b, on);
  });
  return changed;
}

// How much of the RESIDENT set is still in RAM.
//
// Every held layer, but SAMPLED inside each buffer: a pin either holds
// or it does not, so every 64th page finds it, and at 16 KB pages that
// is one byte of vector per megabyte examined. The pinned prefix counts
// too -- it is the part whose eviction hurts most, since nothing will
// ever re-admit it.
//
// A WIRED BUFFER CANNOT HAVE LEFT RAM, so asking is spending the walk
// to be told what mlock already guarantees. Skipped PER BUFFER rather
// than per layer, because wire_layer_ stops at the first refusal and
// leaves the rest of that layer unwired -- the remainder is exactly
// what still needs measuring. With everything wired `examined` stays 0
// and the caller reads that as "no evidence" rather than as a shortfall
// (it tests examined > 0 first), which is the correct answer: there is
// nothing this walk could have found.
//
// Worth the branch: the walk costs ~57 ms per 4.3 GB, so a fully wired
// 31.6 GB stack would pay ~420 ms per look for a guaranteed answer.
//
// SO WHY KEEP THE WALK AT ALL. Because "everything is wired" is not a
// state this model reaches. Two things are never in the pool:
//
//   * the pool can be off entirely (no wired_pool_mb, or
//     VPIPE_WIRE_RESIDENT=0), and then nothing here is wired;
//   * `derived()` tensors are outside wire_trunk_'s reach, and on a
//     bf16 pack that is the ENTIRE generation expert -- half the
//     checkpoint -- because it ships F32 and is converted at bind.
//
// So on the pack that most needs streaming, roughly half of every
// pinned layer is unwirable by construction, and this is the only thing
// that can tell whether the box is actually holding it.
void
U15Weights::resident_pages_(std::size_t* examined, std::size_t* incore,
                            std::size_t* paged_out) const
{
  *examined = 0;
  *incore = 0;
  *paged_out = 0;
  const auto walk = [&](const vpipe::metal_compute::SharedBuffer& b) {
    if (b.empty() || b.is_wired()) { return; }
    const auto r = b.page_residency(64);
    if (!r.valid) { return; }
    *examined += r.examined;
    *incore += r.incore;
    *paged_out += r.paged_out;
  };
  const int n = (int)_layers.size();
  for (int i = 0; i < n; ++i) {
    const MotLayer& L = _layers[(std::size_t)i];
    if (L.und.q.empty()) { continue; }
    for (const ExpertLayer* e : {&L.und, &L.gen}) {
      for (const QWeight* q : {&e->q, &e->k, &e->v, &e->o, &e->gate,
                               &e->up, &e->down}) {
        walk(q->w); walk(q->codes); walk(q->scales); walk(q->qbias);
      }
    }
  }
}

// Give back the most recently promoted layer. Recency is the right
// signal HERE and nowhere else in this file: the scan is cyclic, so no
// promoted layer is more useful than another, and taking the newest
// back is what makes a shed undo the admission that caused it.
std::size_t
U15Weights::evict_tail_layer_()
{
  if (_promoted.empty()) { return 0; }
  const int i = _promoted.back();
  _promoted.pop_back();
  MotLayer& L = _layers[(std::size_t)i];
  const std::size_t nb = layer_resident_bytes_(L);
  // The wiring goes back BEFORE the buffers do: dropping a wired buffer
  // unwires it in the kernel but tells the pool nothing, which leaks a
  // layer's worth of budget per eviction.
  _wire.note_unwired(wire_layer_(L, false));
  L = MotLayer{};
  return nb;
}

std::size_t
U15Weights::release_at_idle()
{
  if (!_streaming || _promoted.empty()) { return 0; }
  const std::size_t freed =
      _resid.release(_resid.bytes(),
                     [this]() { return evict_tail_layer_(); });
  // See the header: the ratchet must not read a deliberate release as
  // evidence about what this box will hold.
  _resid.note_landscape_changed();
  return freed;
}

void
U15Weights::set_residency_reserve(std::size_t bytes)
{
  // Calling this at all is what turns growth ON: a model that has not
  // been told what its activations cost keeps exactly the behaviour it
  // had rather than growing against a guess.
  _resid.set_reserve(bytes);
}

void
U15Weights::note_kv_allocated(std::size_t bytes)
{
  _res_kv = bytes;
  _resid.note_reserve_allocated(_res_kv + _res_arena);
}

void
U15Weights::note_arena_allocated(std::size_t bytes)
{
  _res_arena = bytes;
  _resid.note_reserve_allocated(_res_kv + _res_arena);
}

void
U15Weights::set_residency_schedule(int passes)
{
  if (_mc == nullptr) { return; }
  // THE POOL IS OPENED WHETHER OR NOT THIS MODEL STREAMS. Wiring is not
  // a streaming feature: a preloaded model that is wired is protected
  // from the compressor for the whole run, and the pool is where that
  // is negotiated with everybody else. MEASURED in the host's own
  // wired-pool.h on a preloaded 35 GB DiT: 1.21x, with compression
  // falling across the run rather than rising.
  if (_wire_allowed) { _wire.open(_mc); }
  if (!_streaming) { return; }        // the rest is residency policy
  _resid.set_schedule(passes, (int)_layers.size(), layer_bytes(),
                      _wire.on(), _mc->memory_budget());
  if (_mc->session() != nullptr) {
    _mc->session()->log_debug(vpipe::fmt(
        "U15Weights: streaming {} of {} layers at {} MB each; residency "
        "may take {} per pass, wire pool {}",
        (int)_layers.size() - _pinned, (int)_layers.size(),
        layer_bytes() >> 20, _resid.per_forward_cap(),
        _wire.on() ? "on" : "off"));
  }
}


// ---- the reusable slot ----------------------------------------------

void
U15Weights::each_layer_tensor_(
    int i, MotLayer& L,
    const vpipe::genai::BlockSlots<MotLayer>::TensorFn& fn) const
{
  const std::string p =
      "language_model.model.layers." + std::to_string(i) + ".";
  const std::string a = p + "self_attn.";

  // A MATRIX is three tensors when quantized and one when dense, and
  // which it is was decided when the slot was built -- so the layout
  // comes from the QWeight rather than from the checkpoint. A pack that
  // disagrees fails the size check inside the refill and forces a
  // rebuild, which is the correct answer to "these are not the same
  // weights".
  const auto matrix = [&fn](const std::string& base, QWeight& m) {
    if (m.quantized) {
      // The codes are the checkpoint's OWN u32 words: `raw` marks them
      // so a REBUILD copies them rather than reading them as bf16.
      fn(base + ".weight", m.codes, P::kRaw);
      fn(base + ".scales", m.scales, P::kBf16);
      fn(base + ".biases", m.qbias, P::kBf16);
    } else {
      fn(base + ".weight", m.w, P::kBf16);
    }
  };

  struct Bind { ExpertLayer* e; const char* sfx; const char* lsfx; };
  const Bind binds[2] = {{&L.und, "", ""}, {&L.gen, "_mot_gen", "_mot_gen"}};
  for (const Bind& b : binds) {
    ExpertLayer& e = *b.e;
    const std::string sfx(b.sfx), lsfx(b.lsfx);
    fn(p + "input_layernorm" + lsfx + ".weight", e.input_ln, P::kBf16);
    fn(p + "post_attention_layernorm" + lsfx + ".weight", e.post_ln, P::kBf16);
    matrix(a + "q_proj" + sfx, e.q);
    matrix(a + "k_proj" + sfx, e.k);
    matrix(a + "v_proj" + sfx, e.v);
    matrix(a + "o_proj" + sfx, e.o);
    fn(a + "q_norm" + sfx + ".weight", e.q_norm, P::kBf16);
    fn(a + "k_norm" + sfx + ".weight", e.k_norm, P::kBf16);
    fn(a + "q_norm_hw" + sfx + ".weight", e.q_norm_hw, P::kBf16);
    fn(a + "k_norm_hw" + sfx + ".weight", e.k_norm_hw, P::kBf16);
    matrix(p + "mlp" + lsfx + ".gate_proj", e.gate);
    matrix(p + "mlp" + lsfx + ".up_proj", e.up);
    matrix(p + "mlp" + lsfx + ".down_proj", e.down);
  }
}

bool
U15Weights::convert_into_bf16_(const std::string& name,
                               const SharedBuffer& dst)
{
  const auto* ti = _ws->src().info(name);
  if (ti == nullptr || dst.empty()) { return false; }
  // F32 is the only dtype that reaches here: BF16 and U32 are placed
  // raw by the refill and F16 is converted in place by it, because both
  // are the destination's own width. F32 is twice it, which is exactly
  // why there is nowhere to put the bytes without a second buffer.
  if (ti->dtype != "F32") { return false; }
  const std::size_t n = dst.byte_size() / 2;
  if ((std::size_t)ti->nbytes != n * 4) { return false; }

  if (_f32_scratch.byte_size() < (std::size_t)ti->nbytes) {
    _f32_scratch = _mc->make_shared_buffer((std::size_t)ti->nbytes);
    if (_f32_scratch.empty()) { return false; }
  }
  // Through the WEIGHT SET rather than MetalLlamaWeights::pread_into
  // directly, so the bytes are counted as streaming throughput and the
  // manager can still see what this model is moving.
  if (_ws->stream_into(name, _f32_scratch.contents(),
                       (std::size_t)ti->nbytes)) {
    f32_to_bf16_(_f32_scratch.contents(), dst.contents(), n);
    return true;
  }
  // The pread refused -- a GGUF-backed checkpoint, or a short read.
  // Fall back to the mapped copy rather than failing the layer: slower
  // is not the same as wrong.
  const SharedBuffer src =
      _ws->stream_tensor(name, _mc, WeightSet::Residency::Copied);
  if (src.empty() || src.byte_size() != n * 4) { return false; }
  f32_to_bf16_(src.contents(), dst.contents(), n);
  return true;
}

// Allocate `dst` with `src`'s shapes and flags, optionally copying the
// bytes. One function for two uses: a promotion and the second slot
// differ only in whether the contents come along.
bool
U15Weights::clone_layer_(const MotLayer& src, MotLayer* dst,
                         bool copy) const
{
  bool ok = true;
  const auto one = [&](const SharedBuffer& s, SharedBuffer& d) {
    if (!ok || s.empty()) { d = SharedBuffer{}; return; }
    d = _mc->make_shared_buffer(s.byte_size());
    if (d.empty()) { ok = false; return; }
    if (copy) { std::memcpy(d.contents(), s.contents(), s.byte_size()); }
  };
  const auto qw = [&](const QWeight& s, QWeight& d) {
    d.quantized = s.quantized;
    d.bits      = s.bits;
    d.group     = s.group;
    one(s.w, d.w);
    one(s.codes, d.codes);
    one(s.scales, d.scales);
    one(s.qbias, d.qbias);
  };
  const ExpertLayer* se[2] = {&src.und, &src.gen};
  ExpertLayer* de[2] = {&dst->und, &dst->gen};
  for (int i = 0; i < 2; ++i) {
    one(se[i]->input_ln, de[i]->input_ln);
    one(se[i]->post_ln, de[i]->post_ln);
    qw(se[i]->q, de[i]->q); qw(se[i]->k, de[i]->k);
    qw(se[i]->v, de[i]->v); qw(se[i]->o, de[i]->o);
    one(se[i]->q_norm, de[i]->q_norm);
    one(se[i]->k_norm, de[i]->k_norm);
    one(se[i]->q_norm_hw, de[i]->q_norm_hw);
    one(se[i]->k_norm_hw, de[i]->k_norm_hw);
    qw(se[i]->gate, de[i]->gate); qw(se[i]->up, de[i]->up);
    qw(se[i]->down, de[i]->down);
  }
  if (!ok) { *dst = MotLayer{}; }
  return ok;
}

void
U15Weights::configure_slots_()
{
  vpipe::genai::BlockSlots<MotLayer>::Ops ops;
  ops.each = [this](int i, MotLayer& L,
                    const vpipe::genai::BlockSlots<MotLayer>::TensorFn& fn) {
    each_layer_tensor_(i, L, fn);
  };
  // A tensor a raw read could not place AND could not be filled in
  // place: allocate a replacement the same way load_layer_ would, so a
  // repaired tensor is byte-identical to a freshly built one.
  ops.rebuild_one = [this](const std::string& nm, P how) {
    if (how == P::kRaw) {
      return _ws->stream_tensor(nm, _mc, WeightSet::Residency::Copied);
    }
    return bind_bf16_(*_ws, _mc, nm, std::string(), nullptr,
                      Retain::Streamed);
  };
  // F32 is half this checkpoint, so filling in place rather than
  // reallocating is the difference between the fast path covering half
  // the bytes and covering all of them.
  ops.fill_unservable = [this](const std::string& nm,
                               const SharedBuffer& dst) {
    return convert_into_bf16_(nm, dst);
  };
  ops.build = [this](int i, MotLayer& L) {
    std::string err;
    const bool ok = load_layer_(i, &L, /*streamed=*/true, &err);
    if (ok) { ++_rebuilt_layers; }
    return ok;
  };
  ops.clone = [this](const MotLayer& s, MotLayer& d, bool copy) {
    return clone_layer_(s, &d, copy);
  };
  ops.bytes = [this](const MotLayer& L) {
    return layer_resident_bytes_(L);
  };
  ops.empty = [](const MotLayer& L) { return L.und.q.empty(); };
  _slots.set_weight_set(_ws.get());
  _slots.configure(_mc, std::move(ops), "U15Weights",
                   _refill ? "VPIPE_U15_NO_SLOTS" : nullptr);
  // `refill_streamed = false` is the A/B switch: force the fallback by
  // making the kill switch one that is always set.
  if (!_refill) { _slots.disable(); }
}

// ---- the stack-pass protocol -----------------------------------------

void
U15Weights::begin_pass()
{
  if (_mc == nullptr) { return; }
  if (_wire.on()) {
    // The pool may have refused earlier because another process was
    // spiking. Asking again is gated inside retry(); a budget that
    // actually rose is news BlockResidency cannot see for itself.
    if (_wire.retry(_mc, layer_bytes())) { _resid.note_landscape_changed(); }
    // The trunk takes its place in the pool before this pass's layer
    // admissions start asking for room: the layers are the shed-able
    // half, so a pool that runs out should run out there. The SCRATCH
    // went in ahead of both, from the backbone -- see wire_scratch().
    //
    // Not gated on streaming. A PRELOADED model has no layers to admit,
    // and for it for_each_weight() is the whole checkpoint -- which is
    // the case the pool was measured on.
    wire_trunk_(true);
  }
  if (!_streaming) { return; }        // the rest is residency policy

  const auto mb = _mc->memory_budget();
  _resid.begin_forward(mb, [this]() { return evict_tail_layer_(); });

  // Then the measurement that actually finds the limit: are the layers
  // we kept still in RAM? Anything less means a pin has failed and this
  // box holds less than it was asked to, so one layer goes back.
  //
  // TWO gates, both cheap, and both needed:
  //
  //   count() > 0   nothing PROMOTED means nothing this can act on --
  //                 evict_tail_layer_ only sheds promoted layers, never
  //                 the pinned prefix, so a walk here would buy a
  //                 measurement with no available response.
  //   self_compression_grew   the walk costs ~57 ms per 4.3 GB, and a
  //                 healthy run would pay it every pass to be told
  //                 nothing. When none of OUR pages are being
  //                 compressed there is nothing for it to find.
  bool shortfall = false;
  if (_resid.count() > 0 &&
      _resid.self_compression_grew(mb.self_compressed)) {
    std::size_t examined = 0, incore = 0, paged_out = 0;
    resident_pages_(&examined, &incore, &paged_out);
    if (examined > 0 && incore < examined) {
      shortfall = true;
      const std::size_t freed = _resid.note_weight_residency(
          examined, incore, [this]() { return evict_tail_layer_(); });
      if (_mc->session() != nullptr) {
        _mc->session()->log_normal(vpipe::fmt(
            "U15Weights: resident layers are only {}% in RAM ({} of {} "
            "sampled pages paged out, {} MB wired) -- released {} MB, "
            "now {} of {} resident",
            (int)(100.0 * (double)incore / (double)examined), paged_out,
            examined, _wire.wired_bytes() >> 20, freed >> 20,
            resident_layers(), (int)_layers.size()));
      }
    }
  }
  if (!shortfall) { _resid.note_healthy_forward(); }
}

const MotLayer*
U15Weights::layer(int i, std::string* err)
{
  if (i < 0 || i >= (int)_layers.size()) {
    if (err != nullptr) { *err = "layer index out of range"; }
    return nullptr;
  }
  MotLayer& R = _layers[(std::size_t)i];
  if (!R.und.q.empty()) { return &R; }      // pinned or promoted
  if (!_streaming) {
    if (err != nullptr) {
      *err = "layer " + std::to_string(i) + " is not bound";
    }
    return nullptr;
  }
  const MotLayer* L = _slots.acquire(i);
  if (L == nullptr) {
    if (err != nullptr) {
      *err = "could not read layer " + std::to_string(i);
    }
    return nullptr;
  }
  _streamed_bytes += layer_resident_bytes_(*L);
  ++_streamed_layers;
  return L;
}

void
U15Weights::prefetch_after(int after)
{
  if (!_streaming) { return; }
  // The next layer that will actually be STREAMED. Resident ones are
  // skipped -- prefetching one would read bytes the pass already has.
  // Safe to look ahead: promotion only ever adds the layer just
  // finished, never one further down the stack.
  int nxt = -1;
  for (int n = after + 1; n < (int)_layers.size(); ++n) {
    if (_layers[(std::size_t)n].und.q.empty()) { nxt = n; break; }
  }
  _slots.prefetch(nxt);
}

void
U15Weights::join_reads()
{
  _slots.join();
}

void
U15Weights::end_layer(int i)
{
  if (!_streaming || i < 0 || i >= (int)_layers.size()) { return; }
  if (!_layers[(std::size_t)i].und.q.empty()) { return; }   // was resident
  // KEEPING IT COSTS LITTLE EXTRA -- the destination has to be
  // allocated either way, so what promotion adds over the read is one
  // memcpy, and what it retires is re-reading this layer on every
  // remaining pass.
  const std::size_t nb = _slots.last_bytes();
  if (nb == 0) { return; }
  if (_wire.wirable(nb) && _resid.admit(_mc, nb) &&
      _slots.promote_into(_layers[(std::size_t)i])) {
    // Wired LAST, after every write this layer will ever get: mlock
    // pins the pages that exist NOW.
    _wire.note_wired(_mc, wire_layer_(_layers[(std::size_t)i], true), nb);
    _resid.note_admitted(nb);
    _promoted.push_back(i);
    if (_mc != nullptr && _mc->session() != nullptr) {
      _mc->session()->log_debug(vpipe::fmt(
          "U15Weights: layer {} resident ({} of {}, {} MB, {} MB wired)",
          i, resident_layers(), (int)_layers.size(),
          (_resid.bytes() >> 20), _wire.wired_bytes() >> 20));
    }
  }
  // NOT promoted: the slots keep their contents and the next read
  // overwrites them, which is the point.
}

}  // namespace u15
