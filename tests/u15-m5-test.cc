// The M5 matrix-core paths, against the simdgroup paths they replace.
//
// Both are a RATE choice, not a numerics one: matmul2d and the NAX
// flash-attention compute the same contraction as the steel tiles
// through hardware matrix units, so the only expected difference is the
// order f16/bf16 rounding happens in. That is exactly why they need a
// test -- a wrong tiling, a params block filled for the other kernel's
// tile, or a grid that leaves a tail undispatched all produce a
// plausible tensor of the right shape.
//
// Runs WITHOUT the checkpoint: the shapes are the model's (hidden 4096,
// 32 q heads over 8 kv heads, head_dim 128, ffn 12288), the data is
// random, and what is under test is the routing rather than the
// weights. On a GPU without matrix cores every arm is the steel path
// and the test says so rather than passing vacuously.
//
//   VPIPE_U15_NO_MMA2=1       force the steel dense GEMM
//   VPIPE_U15_NO_NAX_ATTN=1   force the steel flash-attention
//   VPIPE_U15_MMA_MIN_M=N     the row floor matmul2d is worth

#include "u15-config.h"
#include "u15-metal-ops.h"

extern "C" const unsigned char u15_kernels_bf16_metallib[];
extern "C" const unsigned long u15_kernels_bf16_metallib_len;

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <string>
#include <vector>

using vpipe::metal_compute::ComputeEncoder;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace {

int g_fail = 0, g_ran = 0;

// The two paths differ only in rounding ORDER over the same K-length
// dot products, so at K = 4096-12288 in bf16 this is the accumulation
// spread and nothing else. Stated once so loosening it is a visible
// edit.
constexpr double kBar = 6e-3;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  ++g_ran;
  if (!ok) { ++g_fail; }
}

std::uint16_t to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
  return (std::uint16_t)((u + r) >> 16);
}

float from_bf16(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::vector<float> randn(std::size_t n, unsigned seed, float s = 1.0f)
{
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, s);
  std::vector<float> v(n);
  for (auto& x : v) { x = d(g); }
  return v;
}

SharedBuffer up(MetalCompute& mc, const std::vector<float>& v)
{
  SharedBuffer b = mc.make_shared_buffer(v.size() * 2);
  auto* p = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { p[i] = to_bf16(v[i]); }
  return b;
}

std::vector<float> down(const SharedBuffer& b, std::size_t n)
{
  std::vector<float> v(n);
  const auto* p = static_cast<const std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) { v[i] = from_bf16(p[i]); }
  return v;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

bool all_finite(const std::vector<float>& v)
{
  for (float x : v) { if (!std::isfinite(x)) { return false; } }
  return true;
}

// One timed run of `work`, as wall-seconds over `reps` submissions.
template <class F>
double time_it(MetalCompute& mc, int reps, F&& work)
{
  const auto t0 = std::chrono::steady_clock::now();
  {
    auto st = mc.make_command_stream();
    auto enc = st.begin_compute();
    for (int r = 0; r < reps; ++r) { work(enc); }
    enc.end();
    st.commit().wait();
  }
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - t0).count();
}

// ---- dense projections -------------------------------------------------
//
// The four the backbone runs per layer, at two sequence lengths.
struct Shape { int M, K, N; const char* what; };

void
test_gemm(MetalCompute& mc, const u15::MetalOps& on, const u15::MetalOps& off)
{
  std::printf("dense projections: matmul2d vs steel\n");
  const Shape shapes[] = {
      // Below the default row floor, so BOTH arms are steel here unless
      // VPIPE_U15_MMA_MIN_M lowers it -- which is how the floor itself
      // gets measured rather than assumed.
      {1,    4096,  6144,  "qkv    M=1  "},
      {4,    4096,  6144,  "qkv    M=4  "},
      {8,    4096,  6144,  "qkv    M=8  "},
      {16,   4096,  6144,  "qkv    M=16 "},
      {32,   4096,  6144,  "qkv    M=32 "},
      {64,   4096,  6144,  "qkv    M=64 "},
      {256,  4096,  6144,  "qkv    M=256"},
      {256,  4096,  4096,  "o_proj M=256"},
      {256,  4096, 24576,  "gate|up M=256"},
      {256, 12288,  4096,  "ff_down M=256"},
      {1024, 4096,  6144,  "qkv    M=1024"},
      {1024, 12288, 4096,  "ff_down M=1024"},
  };
  for (const Shape& s : shapes) {
    const auto xv = randn((std::size_t)s.M * s.K, 11, 0.05f);
    const auto wv = randn((std::size_t)s.N * s.K, 22, 0.05f);
    SharedBuffer x = up(mc, xv), w = up(mc, wv);
    SharedBuffer ya = mc.make_shared_buffer((std::size_t)s.M * s.N * 2);
    SharedBuffer yb = mc.make_shared_buffer((std::size_t)s.M * s.N * 2);
    if (x.empty() || w.empty() || ya.empty() || yb.empty()) {
      check(false, std::string(s.what) + ": allocation failed");
      continue;
    }
    auto run = [&](const u15::MetalOps& ops, const SharedBuffer& y) {
      return [&](ComputeEncoder& e) {
        ops.linear(e, x, w, nullptr, y, s.M, s.K, s.N);
      };
    };
    // Warm BOTH before timing EITHER, then interleave the rounds. A
    // first-touch cost or a clock state landing on one arm is how an
    // A/B invents a result -- measured in this tree more than once.
    (void)time_it(mc, 1, run(on, ya));
    (void)time_it(mc, 1, run(off, yb));
    double ta = 1e9, tb = 1e9;
    for (int r = 0; r < 3; ++r) {
      ta = std::min(ta, time_it(mc, 4, run(on, ya)));
      tb = std::min(tb, time_it(mc, 4, run(off, yb)));
    }
    const auto a = down(ya, (std::size_t)s.M * s.N);
    const auto b = down(yb, (std::size_t)s.M * s.N);
    const double e = rel_l2(a, b);
    const double gf = 2.0 * s.M * (double)s.K * s.N * 4.0 / 1e9;
    std::printf("       %s  rel-L2 %.2e   mma %6.2f TFLOP/s  steel %6.2f"
                "   %.2fx\n", s.what, e, gf / ta / 1e3, gf / tb / 1e3,
                tb / ta);
    check(all_finite(a) && e < kBar, std::string(s.what) + " agrees");
  }
}

// ---- flash attention ---------------------------------------------------
void
test_attn(MetalCompute& mc, const u15::MetalOps& on, const u15::MetalOps& off)
{
  std::printf("flash attention: NAX vs steel (h=32/kv=8, d=128)\n");
  const int HQ = 32, HKV = 8, D = 128;
  for (int T : {256, 1024}) {
    const auto qv = randn((std::size_t)HQ * T * D, 33, 0.3f);
    const auto kv = randn((std::size_t)HKV * T * D, 44, 0.3f);
    const auto vv = randn((std::size_t)HKV * T * D, 55, 0.3f);
    SharedBuffer q = up(mc, qv), k = up(mc, kv), v = up(mc, vv);
    SharedBuffer oa = mc.make_shared_buffer((std::size_t)HQ * T * D * 2);
    SharedBuffer ob = mc.make_shared_buffer((std::size_t)HQ * T * D * 2);
    u15::MetalOps::SteelAttn pa, pb;
    const bool oka = on.steel_attn_plan(&pa, HQ, HKV, T, T, D, T);
    const bool okb = off.steel_attn_plan(&pb, HQ, HKV, T, T, D, T);
    if (!oka || !okb || oa.empty() || ob.empty()) {
      check(false, "attention plan failed at T=" + std::to_string(T));
      continue;
    }
    check(pa.bq == (on.attn_is_nax() ? 64 : 32),
          "T=" + std::to_string(T) + ": query tile matches the kernel");
    auto run = [&](const u15::MetalOps& ops,
                   const u15::MetalOps::SteelAttn& p,
                   const SharedBuffer& o) {
      return [&](ComputeEncoder& e) { ops.sdpa_steel(e, p, q, k, v, o); };
    };
    (void)time_it(mc, 1, run(on, pa, oa));
    (void)time_it(mc, 1, run(off, pb, ob));
    double ta = 1e9, tb = 1e9;
    for (int r = 0; r < 3; ++r) {
      ta = std::min(ta, time_it(mc, 4, run(on, pa, oa)));
      tb = std::min(tb, time_it(mc, 4, run(off, pb, ob)));
    }
    const auto a = down(oa, (std::size_t)HQ * T * D);
    const auto b = down(ob, (std::size_t)HQ * T * D);
    const double e = rel_l2(a, b);
    std::printf("       T=%-5d rel-L2 %.2e   nax %7.3f ms  steel %7.3f ms"
                "   %.2fx\n", T, e, ta * 250.0, tb * 250.0, tb / ta);
    check(all_finite(a) && e < kBar,
          "T=" + std::to_string(T) + ": attention agrees");
  }
}

// ---- quantized projections --------------------------------------------
//
// w8g64, the pack the docs tell people to deploy. The steel qmm unpacks
// inside its tile loop; matmul2d cannot, so the M5 path expands once
// and runs dense. What that costs is one pass over N*K whatever M is,
// which is why this has a row floor the dense path does not.
//
// The pack is built here rather than loaded, and the ORACLE for whether
// it is built correctly is the comparison itself: the steel qmm and the
// dequant kernel are two independent readers of the same convention, so
// a packing either of them disagrees with shows up as a rel-L2 blowup
// rather than as two matching wrong answers.
struct Pack {
  SharedBuffer codes, scales, biases;
  std::vector<float> deq;      // what the codes actually mean
};

Pack
quantize_w8(MetalCompute& mc, const std::vector<float>& w, int N, int K,
            int group)
{
  Pack p;
  const int Kg = K / group, Kw = K / 4;
  p.codes  = mc.make_shared_buffer((std::size_t)N * Kw * 4);
  p.scales = mc.make_shared_buffer((std::size_t)N * Kg * 2);
  p.biases = mc.make_shared_buffer((std::size_t)N * Kg * 2);
  if (p.codes.empty() || p.scales.empty() || p.biases.empty()) { return p; }
  auto* cw = static_cast<std::uint32_t*>(p.codes.contents());
  auto* sc = static_cast<std::uint16_t*>(p.scales.contents());
  auto* bi = static_cast<std::uint16_t*>(p.biases.contents());
  p.deq.assign((std::size_t)N * K, 0.0f);
  std::vector<int> q((std::size_t)K);
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < Kg; ++g) {
      const int k0 = g * group;
      float lo = w[(std::size_t)n * K + k0], hi = lo;
      for (int i = 1; i < group; ++i) {
        const float v = w[(std::size_t)n * K + k0 + i];
        lo = std::min(lo, v); hi = std::max(hi, v);
      }
      // Round-tripped through bf16 BEFORE the codes are computed: the
      // kernels read these as bf16, so quantizing against the f32 value
      // would leave a residual that is the fixture's, not the kernel's.
      const float sf = from_bf16(to_bf16((hi - lo) / 255.0f));
      const float bf = from_bf16(to_bf16(lo));
      sc[(std::size_t)n * Kg + g] = to_bf16(sf);
      bi[(std::size_t)n * Kg + g] = to_bf16(bf);
      for (int i = 0; i < group; ++i) {
        const float v = w[(std::size_t)n * K + k0 + i];
        int c = sf > 0.0f ? (int)std::lround((v - bf) / sf) : 0;
        c = std::max(0, std::min(255, c));
        q[(std::size_t)k0 + i] = c;
        p.deq[(std::size_t)n * K + k0 + i] = sf * (float)c + bf;
      }
    }
    for (int wI = 0; wI < Kw; ++wI) {
      const int k0 = wI * 4;
      cw[(std::size_t)n * Kw + wI] =
          (std::uint32_t)q[k0] | ((std::uint32_t)q[k0 + 1] << 8) |
          ((std::uint32_t)q[k0 + 2] << 16) |
          ((std::uint32_t)q[k0 + 3] << 24);
    }
  }
  return p;
}

void
test_gemm_q(MetalCompute& mc, const u15::MetalOps& on,
            const u15::MetalOps& off)
{
  std::printf("quantized projections (w8g64): dequant-once + matmul2d vs "
              "steel qmm\n");
  const int K = 4096, N = 6144, G = 64;
  const auto wv = randn((std::size_t)N * K, 77, 0.05f);
  Pack pk = quantize_w8(mc, wv, N, K, G);
  if (pk.codes.empty()) { check(false, "pack allocation failed"); return; }
  // The pack round-trips: what the codes mean is what the fixture meant
  // them to. A packing error here would otherwise be invisible, since
  // both arms read the same codes.
  // 8 bits over a group of 64 gives a step of (max-min)/255 and a
  // uniform residual, which at this spread is ~7e-3 relative. A bar
  // tighter than the quantization itself would only ever measure the
  // fixture.
  check(rel_l2(pk.deq, wv) < 1.5e-2, "w8g64 pack round-trips");
  u15::QWeight qw;
  // SharedBuffer is move-only; the pack has no further use for them.
  qw.codes = std::move(pk.codes);
  qw.scales = std::move(pk.scales);
  qw.qbias = std::move(pk.biases);
  qw.bits = 8; qw.group = G; qw.quantized = true;

  for (int M : {1, 8, 32, 64, 96, 128, 160, 192, 256, 512, 1024}) {
    const auto xv = randn((std::size_t)M * K, 88, 0.05f);
    SharedBuffer x = up(mc, xv);
    SharedBuffer ya = mc.make_shared_buffer((std::size_t)M * N * 2);
    SharedBuffer yb = mc.make_shared_buffer((std::size_t)M * N * 2);
    if (x.empty() || ya.empty() || yb.empty()) {
      check(false, "alloc failed at M=" + std::to_string(M));
      continue;
    }
    auto run = [&](const u15::MetalOps& ops, const SharedBuffer& y) {
      return [&](ComputeEncoder& e) {
        ops.linear(e, x, qw, nullptr, y, M, K, N);
      };
    };
    (void)time_it(mc, 1, run(on, ya));
    (void)time_it(mc, 1, run(off, yb));
    double ta = 1e9, tb = 1e9;
    for (int r = 0; r < 3; ++r) {
      ta = std::min(ta, time_it(mc, 4, run(on, ya)));
      tb = std::min(tb, time_it(mc, 4, run(off, yb)));
    }
    const auto a = down(ya, (std::size_t)M * N);
    const auto b = down(yb, (std::size_t)M * N);
    const double e = rel_l2(a, b);
    std::printf("       M=%-5d rel-L2 %.2e   mma %7.3f ms  steel %7.3f ms"
                "   %.2fx\n", M, e, ta * 250.0, tb * 250.0, tb / ta);
    check(all_finite(a) && e < kBar,
          "w8g64 M=" + std::to_string(M) + " agrees");
  }
}

}  // namespace

int
main()
{
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no Metal device -- this test did NOT run.\n");
    return 0;
  }
  // The M5 paths ON (the default) and OFF, in one process. init_ reads
  // the environment, so the OFF instance is built after setting the
  // kill switches -- which also means this test exercises the switches
  // themselves.
  // The plugin's own metallib, which a test binary has to register for
  // itself: MetalOps::init() validates every kernel it binds, including
  // the ones that live there.
  if (!mc.register_metal_library(u15::kMetalLibBf16,
                                 u15_kernels_bf16_metallib,
                                 u15_kernels_bf16_metallib_len)) {
    std::printf("FAILED to register the metallib\n");
    return 1;
  }
  u15::MetalOps on;
  std::string why;
  if (!on.init(&mc, &why)) {
    std::printf("SKIPPED: MetalOps::init failed (%s)\n", why.c_str());
    return 0;
  }
  ::setenv("VPIPE_U15_NO_MMA2", "1", 1);
  ::setenv("VPIPE_U15_NO_NAX_ATTN", "1", 1);
  u15::MetalOps off;
  if (!off.init(&mc, &why)) {
    std::printf("SKIPPED: reference MetalOps::init failed (%s)\n",
                why.c_str());
    return 0;
  }
  if (!mc.supports_matrix_cores()) {
    // NOT a pass. Both arms would be the same kernel, and a green tick
    // on that says nothing about the paths this file exists to check.
    std::printf("SKIPPED: no matrix cores on this GPU -- the M5 paths are "
                "not reachable here and this test did NOT run.\n");
    return 0;
  }
  std::printf("matrix cores: yes; mma2=%s mma2_quant=%s nax_attn=%s\n",
              on.dense_is_mma2() ? "on" : "OFF",
              on.quant_is_mma2() ? "on" : "OFF",
              on.attn_is_nax() ? "on" : "OFF");
  if (!on.dense_is_mma2() || !on.attn_is_nax()) {
    check(false, "the M5 paths are available but did not engage");
  }
  test_gemm(mc, on, off);
  test_gemm_q(mc, on, off);
  test_attn(mc, on, off);
  std::printf("%d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
