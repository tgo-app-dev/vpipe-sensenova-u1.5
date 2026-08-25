// The patch embedder and the pixel head, on the GPU with REAL weights,
// against the CPU reference.
//
// These two were the last GPU paths without a full-width check. The
// backbone had one (u15-layer0-test) and the plugin's kernels had one
// (u15-kernels-test), but "embed a patch grid" and "decode hidden
// states to pixels" were only ever exercised end to end -- where a
// wrong answer looks like a bad image rather than a failure.
//
// Deliberately tiny (64x64 => a 4x4 patch grid, 2x2 tokens): the CPU
// reference convolves in double precision, and conv1 alone is
// 1024x1024x9 per output pixel.
//
// Gated on VPIPE_U15_TEST_MODEL_PATH.

#include "u15-config.h"
#include "u15-image.h"
#include "u15-metal-ops.h"
#include "u15-ref.h"
#include "u15-weights.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/weight-set.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

extern "C" const unsigned char u15_kernels_bf16_metallib[];
extern "C" const unsigned long u15_kernels_bf16_metallib_len;

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
as_float(const SharedBuffer& b, std::size_t n)
{
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
  if (!mc.register_metal_library(u15::kMetalLibBf16,
                                 u15_kernels_bf16_metallib,
                                 u15_kernels_bf16_metallib_len)) {
    std::printf("FAILED to register the metallib\n");
    return 1;
  }

  u15::U15Config cfg;
  std::string err;
  if (!u15::parse_config(p, &cfg, &err)) {
    std::printf("FAILED config: %s\n", err.c_str());
    return 1;
  }
  auto ws = WeightSet::open(p, nullptr);
  if (ws == nullptr) { std::printf("FAILED to open\n"); return 1; }

  // Only the trunk is needed: neither path touches a layer.
  u15::U15Weights::Options opt;
  opt.max_layers = 0;
  auto w = u15::U15Weights::load(ws, &mc, cfg, opt, &err);
  check(w != nullptr, "trunk bound" + (err.empty() ? "" : " (" + err + ")"));
  if (w == nullptr) { return 1; }

  u15::MetalOps ops;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED MetalOps: %s\n", err.c_str());
    return 1;
  }
  auto ip = u15::ImagePath::create(&ops, cfg, w.get(), &err);
  check(ip != nullptr, "image path" + (err.empty() ? "" : " (" + err + ")"));
  if (ip == nullptr) { return 1; }

  const int P = cfg.gen.patch_size;               // 16
  const int m = cfg.vision.merge_size();          // 2
  const int C = cfg.vision.hidden_size;           // 1024
  const int LLM = cfg.llm.hidden_size;            // 4096
  const int SZ = 64;                              // 64x64 image
  const int gh = SZ / P, gw = SZ / P;             // 4x4 patches
  const int th = SZ / 32, tw = SZ / 32;           // 2x2 tokens

  auto stream = mc.make_command_stream();

  // ---- the patch embedder ------------------------------------------
  {
    std::printf("patch embedder (%dx%d image -> %dx%d patches -> %dx%d "
                "tokens)\n", SZ, SZ, gh, gw, th, tw);
    std::vector<float> img((std::size_t)3 * SZ * SZ);
    std::mt19937 g(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : img) { v = nd(g); }

    std::vector<float> patch((std::size_t)gh * gw * 3 * P * P);
    u15::ImagePath::patchify_for_embed(img.data(), SZ, SZ, P,
                                       patch.data());

    SharedBuffer px = mc.make_shared_buffer(patch.size() * 2);
    {
      auto* q = static_cast<std::uint16_t*>(px.contents());
      for (std::size_t i = 0; i < patch.size(); ++i) {
        q[i] = to_bf16(patch[i]);
      }
    }
    SharedBuffer out = mc.make_shared_buffer(
        (std::size_t)(gh / m) * (gw / m) * LLM * 2);
    check(ip->embed(stream, px, gh, gw, /*gen=*/true, out, &err),
          "embed ran" + (err.empty() ? "" : " (" + err + ")"));
    stream.commit().wait();

    // CPU reference, from the SAME bound weights.
    const auto pw = as_float(w->trunk().gen_patch_w,
                             (std::size_t)C * 3 * P * P);
    const auto pb = as_float(w->trunk().gen_patch_b, (std::size_t)C);
    const auto dw = as_float(w->trunk().gen_dense_w,
                             (std::size_t)LLM * C * m * m);
    const auto db = as_float(w->trunk().gen_dense_b, (std::size_t)LLM);
    u15::ref::VisionWeights vw;
    vw.patch_w = pw.data(); vw.patch_b = pb.data();
    vw.dense_w = dw.data(); vw.dense_b = db.data();

    std::vector<float> ref((std::size_t)(gh / m) * (gw / m) * LLM);
    u15::ref::vision_embed(patch.data(), vw, gh, gw, P, C, LLM, m,
                           cfg.vision.rope_theta_vision, ref.data());

    const auto got = as_float(out, ref.size());
    const double r = rel_l2(got, ref);
    std::printf("       rel-L2 %.3e\n", r);
    check(r < 2e-2, "patch embedder matches the CPU reference");
  }

  // ---- the pixel head ----------------------------------------------
  {
    std::printf("pixel head (%dx%d tokens -> %dx%d pixels)\n", th, tw,
                th * 32, tw * 32);
    std::vector<float> hid((std::size_t)th * tw * LLM);
    std::mt19937 g(6);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : hid) { v = nd(g); }

    SharedBuffer hb = mc.make_shared_buffer(hid.size() * 2);
    {
      auto* q = static_cast<std::uint16_t*>(hb.contents());
      for (std::size_t i = 0; i < hid.size(); ++i) { q[i] = to_bf16(hid[i]); }
    }
    SharedBuffer px = mc.make_shared_buffer(
        (std::size_t)(th * 32) * (tw * 32) * 3 * 2);
    check(ip->decode(stream, hb, th, tw, px, &err),
          "decode ran" + (err.empty() ? "" : " (" + err + ")"));
    stream.commit().wait();

    // The CPU reference is CHANNEL-FIRST, so transpose both ends. That
    // is deliberate: the reference's layout is the one verified against
    // torch, and comparing through the transpose checks the GPU's
    // channel-LAST formulation against it rather than against itself.
    std::vector<float> hid_chw((std::size_t)LLM * th * tw);
    for (int y = 0; y < th; ++y) {
      for (int x = 0; x < tw; ++x) {
        for (int c = 0; c < LLM; ++c) {
          hid_chw[((std::size_t)c * th + y) * tw + x] =
              hid[((std::size_t)y * tw + x) * LLM + c];
        }
      }
    }
    const int c1_out = C;
    const int c1_in = LLM / 4;
    const int c2_in = c1_out / 4;
    const auto c1w = as_float(w->trunk().conv1_w,
                              (std::size_t)c1_out * c1_in * 9);
    const auto c1b = as_float(w->trunk().conv1_b, (std::size_t)c1_out);
    const auto c2w = as_float(w->trunk().conv2_w,
                              (std::size_t)192 * c2_in * 9);
    const auto c2b = as_float(w->trunk().conv2_b, 192);
    u15::ref::HeadWeights hw;
    hw.conv1_w = c1w.data(); hw.conv1_b = c1b.data();
    hw.conv2_w = c2w.data(); hw.conv2_b = c2b.data();

    std::vector<float> ref_chw((std::size_t)3 * (th * 32) * (tw * 32));
    u15::ref::conv_decoder(hid_chw.data(), hw, LLM, th, tw, c1_out,
                           ref_chw.data());

    const int OH = th * 32, OW = tw * 32;
    std::vector<float> ref_hwc(ref_chw.size());
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < OH; ++y) {
        for (int x = 0; x < OW; ++x) {
          ref_hwc[((std::size_t)y * OW + x) * 3 + c] =
              ref_chw[((std::size_t)c * OH + y) * OW + x];
        }
      }
    }

    const auto got = as_float(px, ref_hwc.size());
    const double r = rel_l2(got, ref_hwc);
    std::printf("       rel-L2 %.3e\n", r);
    check(r < 3e-2, "pixel head matches the CPU reference");
  }

  std::printf("\n%d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
