// Layer 0 of the REAL checkpoint, on the GPU, against the CPU reference.
//
// This is the check that says the whole chain holds: the CPU reference
// is verified against the reference implementation's goldens
// (u15-ref-test), the kernels are verified against the CPU reference at
// toy dimensions (u15-kernels-test), and this runs BOTH at full width
// -- 4096 hidden, 32/8 heads, head_dim 128, intermediate 12288 -- on
// weights read off the shipped 50 GB file.
//
// It runs BOTH experts, because the whole point of a MoT checkpoint is
// that they are different, and a bug that bound one for both would
// otherwise show up only as a slightly wrong image.
//
// The bar is bf16 round-off over ~15 dependent ops. Gated on
// VPIPE_U15_TEST_MODEL_PATH; SKIPS when unset and says so.

#include "u15-backbone.h"
#include "u15-config.h"
#include "u15-metal-ops.h"
#include "u15-ref.h"
#include "u15-weights.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/weight-set.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace {

int g_fail = 0;
int g_ran  = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  ++g_ran;
  if (!ok) { ++g_fail; }
}

float
from_bf16(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t
to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
  return (std::uint16_t)((u + r) >> 16);
}

std::vector<float>
as_float(const SharedBuffer& b)
{
  const std::size_t n = b.byte_size() / 2;
  std::vector<float> v(n);
  const auto* p = static_cast<const std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) { v[i] = from_bf16(p[i]); }
  return v;
}

double
rel_l2(const std::vector<float>& a, const std::vector<float>& b)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

int
main()
{
  const char* p = std::getenv("VPIPE_U15_TEST_MODEL_PATH");
  if (p == nullptr || *p == '\0') {
    std::printf("SKIPPED: VPIPE_U15_TEST_MODEL_PATH unset -- this test "
                "did NOT run.\n");
    return 0;
  }

  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no usable Metal device. NOTHING was checked.\n");
    return 0;
  }
  extern const unsigned char u15_kernels_bf16_metallib[];
  extern const unsigned long u15_kernels_bf16_metallib_len;
  if (!mc.register_metal_library(u15::kMetalLibBf16,
                                 u15_kernels_bf16_metallib,
                                 u15_kernels_bf16_metallib_len)) {
    std::printf("FAILED to register the metallib\n");
    return 1;
  }

  u15::U15Config cfg;
  std::string why;
  if (!u15::parse_config(p, &cfg, &why)) {
    std::printf("FAILED to parse the config: %s\n", why.c_str());
    return 1;
  }
  std::printf("checkpoint: %s\n", p);
  std::printf("hidden %d, heads %d/%d, head_dim %d, inter %d\n\n",
              cfg.llm.hidden_size, cfg.llm.num_attention_heads,
              cfg.llm.num_key_value_heads, cfg.llm.head_dim,
              cfg.llm.intermediate_size);

  auto ws = WeightSet::open(p, nullptr);
  if (ws == nullptr) { std::printf("FAILED to open\n"); return 1; }

  u15::U15Weights::Options opt;
  opt.max_layers = 1;
  std::string err;
  const auto t0 = std::chrono::steady_clock::now();
  auto w = u15::U15Weights::load(ws, &mc, cfg, opt, &err);
  const double load_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  if (w == nullptr) {
    std::printf("FAILED to bind: %s\n", err.c_str());
    return 1;
  }
  std::printf("bound trunk + 1 layer in %.1f s (%.0f MB)\n\n", load_s,
              (double)w->resident_bytes() / (1024.0 * 1024.0));

  u15::MetalOps ops;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED MetalOps init: %s\n", err.c_str());
    return 1;
  }

  // A ONE-layer stack, so the config the backbone validates against
  // matches the weights actually bound.
  u15::U15Config one = cfg;
  one.llm.num_hidden_layers = 1;
  auto bb = u15::U15Backbone::create(&ops, one, w.get(), &err);
  check(bb != nullptr, "backbone built" + (err.empty() ? "" : " (" + err + ")"));
  if (bb == nullptr) { return 1; }

  // Each expert gets the token layout it actually sees in a real run:
  //
  //   und   a PURE TEXT prefix -- t = arange, h = w = 0. Every t is
  //         distinct, so the reference's block-causal mask reduces to
  //         plain causal, which is what makes libvpipe's mask-free
  //         causal kernel usable. Comparing against the real
  //         block_causal_mask here is what VERIFIES that reduction
  //         rather than assuming it.
  //   gen   a 2x3 image block sharing ONE t and varying only in (h, w),
  //         attended bidirectionally.
  const int text_len = 9, tok_h = 2, tok_w = 3;

  const int H = cfg.llm.hidden_size;

  for (int pass = 0; pass < 2; ++pass) {
    const bool gen = (pass == 1);
    const char* tag = gen ? "gen" : "und";
    std::printf("%s expert (%s attention)\n", tag,
                gen ? "bidirectional" : "causal");

    std::vector<int> pt, ph, pw;
    if (gen) {
      for (int i = 0; i < tok_h * tok_w; ++i) {
        pt.push_back(text_len);
        ph.push_back(i / tok_w);
        pw.push_back(i % tok_w);
      }
    } else {
      for (int i = 0; i < text_len; ++i) {
        pt.push_back(i); ph.push_back(0); pw.push_back(0);
      }
    }
    const int n = (int)pt.size();

    std::vector<float> x0((std::size_t)n * H);
    {
      std::mt19937 g(1234);
      std::normal_distribution<float> d(0.0f, 1.0f);
      for (auto& v : x0) { v = d(g); }
    }

    // ---- GPU --------------------------------------------------------
    SharedBuffer xb = mc.make_shared_buffer(x0.size() * 2);
    {
      auto* q = static_cast<std::uint16_t*>(xb.contents());
      for (std::size_t i = 0; i < x0.size(); ++i) { q[i] = to_bf16(x0[i]); }
    }
    auto kv = bb->make_cache(n, &err);
    check(kv.valid(), std::string(tag) + ": KV cache allocated");
    if (!kv.valid()) { return 1; }

    auto stream = mc.make_command_stream();
    const auto g0 = std::chrono::steady_clock::now();
    const bool ok = bb->forward(
        stream, xb, n, gen ? u15::Expert::Gen : u15::Expert::Und, kv, 0, n,
        pt, ph, pw, gen ? u15::Attn::Bidirectional : u15::Attn::Causal,
        &err);
    const double gpu_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - g0).count();
    check(ok, std::string(tag) + ": GPU forward" +
                  (err.empty() ? "" : " (" + err + ")"));
    if (!ok) { return 1; }
    const std::vector<float> got = as_float(xb);

    // Finite before anything else: a NaN makes rel-L2 meaningless and
    // reads as a huge model error rather than as one bad tensor.
    bool finite = true;
    for (float v : got) {
      if (!std::isfinite(v)) { finite = false; break; }
    }
    check(finite, std::string(tag) + ": every output is finite");

    // ---- CPU reference ---------------------------------------------
    const u15::MotLayer& L = w->layers()[0];
    const u15::ExpertLayer& e = gen ? L.gen : L.und;
    const std::vector<float> in_ln = as_float(e.input_ln);
    const std::vector<float> po_ln = as_float(e.post_ln);
    // The CPU reference needs FLOAT weights, so this comparison only
    // means anything against a dense pack. A quantized one is checked
    // end to end instead (u15-generate-test against the bf16 render).
    if (e.q.quantized) {
      std::printf("       SKIPPED: the checkpoint is quantized (w%d g%d); "
                  "the CPU reference compares dense weights\n",
                  e.q.bits, e.q.group);
      continue;
    }
    const std::vector<float> qw = as_float(e.q.w);
    const std::vector<float> kw = as_float(e.k.w);
    const std::vector<float> vw = as_float(e.v.w);
    const std::vector<float> ow = as_float(e.o.w);
    const std::vector<float> qn = as_float(e.q_norm);
    const std::vector<float> kn = as_float(e.k_norm);
    const std::vector<float> qnh = as_float(e.q_norm_hw);
    const std::vector<float> knh = as_float(e.k_norm_hw);
    const std::vector<float> gw2 = as_float(e.gate.w);
    const std::vector<float> uw = as_float(e.up.w);
    const std::vector<float> dw = as_float(e.down.w);

    u15::ref::LayerWeights rw;
    rw.input_layernorm = in_ln.data();
    rw.post_attention_layernorm = po_ln.data();
    rw.q_proj = qw.data(); rw.k_proj = kw.data();
    rw.v_proj = vw.data(); rw.o_proj = ow.data();
    rw.q_norm = qn.data(); rw.k_norm = kn.data();
    rw.q_norm_hw = qnh.data(); rw.k_norm_hw = knh.data();
    rw.gate_proj = gw2.data(); rw.up_proj = uw.data();
    rw.down_proj = dw.data();

    u15::ref::LayerDims d;
    d.hidden = (std::size_t)H;
    d.heads = (std::size_t)cfg.llm.num_attention_heads;
    d.kv_heads = (std::size_t)cfg.llm.num_key_value_heads;
    d.head_dim = (std::size_t)cfg.llm.head_dim;
    d.intermediate = (std::size_t)cfg.llm.intermediate_size;
    d.rms_eps = cfg.llm.rms_norm_eps;
    d.theta_t = cfg.llm.rope_theta;
    d.theta_hw = cfg.llm.rope_theta_hw;

    // The und comparison uses the REFERENCE's block-causal mask, not a
    // plain causal one. With a pure-text prefix the two are identical,
    // and checking against the real rule is what proves that rather
    // than assuming it.
    const std::vector<float> mask = u15::ref::block_causal_mask(pt);
    std::vector<float> ref((std::size_t)n * H);
    const auto c0 = std::chrono::steady_clock::now();
    u15::ref::decoder_layer(x0.data(), rw, d, pt, ph, pw,
                            gen ? nullptr : mask.data(), (std::size_t)n,
                            ref.data());
    const double cpu_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - c0).count();

    const double r = rel_l2(got, ref);
    std::printf("       GPU %.2f s, CPU reference %.1f s, rel-L2 %.3e\n",
                gpu_s, cpu_s, r);
    // ~15 dependent bf16 ops at width 4096 and 12288. The kernel test
    // sees 3e-3 over far fewer ops at width 32; this is the same
    // round-off with more of it.
    check(r < 2e-2, std::string(tag) + ": matches the CPU reference");
  }

  std::printf("\n%d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
