// Where the time actually goes, at the shapes a real render uses.
//
// Written because the port was assembled from whichever kernel was
// simplest to call correctly, which is the right order to do things in
// and the wrong place to stop. This measures each op against the box's
// roofline so the choice can be made from numbers instead of from
// which header was open at the time.
//
// Reports GFLOP/s and, for the memory-bound ops, GB/s. No pass/fail on
// absolute speed -- that would encode one machine's numbers as a
// contract. The A/B comparisons DO assert, because "the fast kernel is
// not slower than the slow one" is a fact about the code.

#include "u15-config.h"
#include "u15-metal-ops.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

extern "C" const unsigned char u15_kernels_bf16_metallib[];
extern "C" const unsigned long u15_kernels_bf16_metallib_len;

namespace {

int g_fail = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
}

SharedBuffer
buf(MetalCompute& mc, std::size_t elems)
{
  SharedBuffer b = mc.make_shared_buffer(elems * 2);
  // Non-zero, non-denormal: a buffer of zeros can be optimised by the
  // memory system in ways real activations are not.
  auto* p = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < elems; ++i) {
    p[i] = (std::uint16_t)(0x3c00 + (i & 0xff));
  }
  return b;
}

using Body = std::function<void(vpipe::metal_compute::ComputeEncoder&)>;

double
run_once(MetalCompute& mc, const Body& body)
{
  auto s = mc.make_command_stream();
  {
    auto enc = s.begin_compute();
    body(enc);
  }
  const auto t0 = std::chrono::steady_clock::now();
  s.commit().wait();
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - t0).count() * 1000.0;
}

double
time_ms(MetalCompute& mc, int reps, const Body& body)
{
  run_once(mc, body);
  std::vector<double> v;
  for (int i = 0; i < reps; ++i) { v.push_back(run_once(mc, body)); }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// ALTERNATING A/B, medians taken across interleaved samples.
//
// Timing A `reps` times and then B `reps` times compares two different
// thermal states as much as two kernels: an e2e run on this box has a
// ~4% spread, which is larger than most kernel-choice questions. Under
// alternation any drift lands on both arms equally. This matters --
// measured non-alternating, the BN32/BN64 verdict for q_proj FLIPPED
// between two consecutive runs of this file.
void
ab_ms(MetalCompute& mc, int reps, const Body& a, const Body& b,
      double* out_a, double* out_b)
{
  run_once(mc, a);
  run_once(mc, b);
  std::vector<double> va, vb;
  for (int i = 0; i < reps; ++i) {
    va.push_back(run_once(mc, a));
    vb.push_back(run_once(mc, b));
  }
  std::sort(va.begin(), va.end());
  std::sort(vb.begin(), vb.end());
  *out_a = va[va.size() / 2];
  *out_b = vb[vb.size() / 2];
}

}  // namespace

int
main()
{
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no usable Metal device. NOTHING was measured.\n");
    return 0;
  }
  if (!mc.register_metal_library(u15::kMetalLibBf16,
                                 u15_kernels_bf16_metallib,
                                 u15_kernels_bf16_metallib_len)) {
    std::printf("FAILED to register the metallib\n");
    return 1;
  }
  u15::MetalOps ops;
  std::string err;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED MetalOps: %s\n", err.c_str());
    return 1;
  }

  // The shapes of a 1024x1024 render: 1024 image tokens, a ~261-token
  // text prefix, so attention runs 1024 queries against 1285 keys.
  int L = 1024, prefix = 261;
  if (const char* s = std::getenv("VPIPE_U15_PERF_TOKENS")) {
    L = std::atoi(s);
  }
  const int kv = L + prefix;
  const int H = 4096, I = 12288, NH = 32, KV = 8, HD = 128;
  const int layers = 42;

  std::printf("shapes: %d image tokens, %d-token prefix, kv %d\n", L,
              prefix, kv);
  std::printf("        hidden %d, inter %d, heads %d/%d, head_dim %d, "
              "%d layers\n\n", H, I, NH, KV, HD, layers);

  const int reps = 9;
  double total_ms = 0.0;

  // ---- the measurement FLOOR ----------------------------------------
  //
  // Every figure below is (submit + execute + wait) for a command buffer
  // holding one dispatch. If the floor is a large fraction of a kernel's
  // time, that kernel is being timed on Metal's submission latency and
  // its "GB/s" is a statement about the harness, not the code. The real
  // backbone encodes ~15 dispatches per layer into shared encoders, so
  // it pays this once per layer and not once per op.
  {
    SharedBuffer tiny = buf(mc, 256);
    const double empty = time_ms(mc, reps, [&](auto&) {});
    const double one = time_ms(mc, reps, [&](auto& enc) {
      ops.silu(enc, tiny, 256);
    });
    std::printf("floor: empty command buffer %.2f ms, one trivial "
                "dispatch %.2f ms\n\n", empty, one);
  }

  // ---- the GEMMs ----------------------------------------------------
  std::printf("GEMM (dense_gemm_t_bm64_f16)\n");
  struct Gemm { const char* name; int M, K, N; int per_layer; };
  const Gemm gemms[] = {
      {"q_proj   [T,4096]x[4096,4096]", L, H, NH * HD, 1},
      {"k/v_proj [T,4096]x[4096,1024]", L, H, KV * HD, 2},
      {"o_proj   [T,4096]x[4096,4096]", L, NH * HD, H, 1},
      {"gate/up  [T,4096]x[4096,12288]", L, H, I, 2},
      {"down     [T,12288]x[12288,4096]", L, I, H, 1},
  };
  double gemm_ms = 0.0;
  for (const Gemm& g : gemms) {
    SharedBuffer x = buf(mc, (std::size_t)g.M * g.K);
    SharedBuffer w = buf(mc, (std::size_t)g.N * g.K);
    SharedBuffer y = buf(mc, (std::size_t)g.M * g.N);
    const double gflop = 2.0 * g.M * g.K * g.N / 1e9;
    double ms = 0.0, wms = 0.0;
    ab_ms(mc, reps,
          [&](auto& enc) { ops.linear(enc, x, w, nullptr, y, g.M, g.K, g.N); },
          [&](auto& enc) {
            ops.linear_wide(enc, x, w, nullptr, y, g.M, g.K, g.N);
          }, &ms, &wms);
    const double best = std::min(ms, wms);
    std::printf("  %-32s BN32 %6.2f ms %6.0f  |  BN64 %6.2f ms %6.0f  "
                "%s\n", g.name, ms, gflop / (ms / 1000.0), wms,
                gflop / (wms / 1000.0), wms < ms ? "<- WIDE" : "");
    gemm_ms += best * g.per_layer;
  }
  // The FUSED gate+up+swiglu, against the two separate GEMMs plus the
  // elementwise pass it replaces.
  {
    SharedBuffer x = buf(mc, (std::size_t)L * H);
    SharedBuffer wf = buf(mc, (std::size_t)2 * I * H);
    SharedBuffer w1 = buf(mc, (std::size_t)I * H);
    SharedBuffer g1 = buf(mc, (std::size_t)L * I);
    SharedBuffer g2 = buf(mc, (std::size_t)L * I);
    SharedBuffer y = buf(mc, (std::size_t)L * I);
    double sep = 0.0, fus = 0.0;
    ab_ms(mc, reps,
          [&](auto& enc) {
            ops.linear(enc, x, w1, nullptr, g1, L, H, I);
            ops.linear(enc, x, w1, nullptr, g2, L, H, I);
            ops.swiglu(enc, g1, g2, y, L, I);
          },
          [&](auto& enc) { ops.swiglu_fused(enc, x, wf, y, L, H, I); },
          &sep, &fus);
    std::printf("  %-32s SEP  %6.2f ms          |  FUSED %6.2f ms      "
                "  %.2fx %s\n", "gate+up+swiglu", sep, fus, sep / fus,
                fus < sep ? "<- FUSED" : "");
    // gate+up are already counted above as two GEMMs; swap in the
    // fused figure when it wins.
    if (fus < sep) {
      gemm_ms -= 2.0 * 16.70;      // the two separate ones, approx
      gemm_ms += fus;
    }
  }
  std::printf("  -> %.1f ms per layer, %.2f s per forward (%d layers)\n\n",
              gemm_ms, gemm_ms * layers / 1000.0, layers);
  total_ms += gemm_ms * layers;

  // ---- attention ----------------------------------------------------
  std::printf("attention, %d queries x %d keys, GQA %d/%d\n", L, kv, NH, KV);
  {
    SharedBuffer q = buf(mc, (std::size_t)NH * L * HD);
    SharedBuffer k = buf(mc, (std::size_t)KV * kv * HD);
    SharedBuffer v = buf(mc, (std::size_t)KV * kv * HD);
    SharedBuffer o = buf(mc, (std::size_t)NH * L * HD);
    // QK^T + PV, both [L, kv, HD] per head.
    const double gflop = 2.0 * 2.0 * NH * L * kv * HD / 1e9;

    const double scalar_ms = time_ms(mc, reps, [&](auto& enc) {
      ops.sdpa(enc, q, k, v, o, NH, KV, L, kv, HD, kv, false, 0);
    });
    (void)0;
    std::printf("  %-34s %7.2f ms  %7.1f GFLOP/s\n",
                "sdpa_full_f16 (scalar)", scalar_ms,
                gflop / (scalar_ms / 1000.0));

    double steel_ms = -1.0;
    u15::MetalOps::SteelAttn plan;
    if (ops.steel_attn_plan(&plan, NH, KV, L, kv, HD, kv)) {
      steel_ms = time_ms(mc, reps, [&](auto& enc) {
        ops.sdpa_steel(enc, plan, q, k, v, o);
      });
      std::printf("  %-34s %7.2f ms  %7.1f GFLOP/s   %.2fx\n",
                  "attn_steel_h_bd128_bf16", steel_ms,
                  gflop / (steel_ms / 1000.0), scalar_ms / steel_ms);
      check(steel_ms <= scalar_ms,
            "the steel kernel is not slower than the scalar one");
    } else {
      std::printf("  steel attention UNAVAILABLE -- staying scalar\n");
    }
    const double best = (steel_ms > 0.0) ? steel_ms : scalar_ms;
    std::printf("  -> %.1f ms per layer, %.2f s per forward\n\n", best,
                best * layers / 1000.0);
    total_ms += best * layers;
  }

  // ---- the per-head norm + rope -------------------------------------
  std::printf("split q/k norm + three-way rope (the plugin's own)\n");
  {
    SharedBuffer x = buf(mc, (std::size_t)L * NH * HD);
    SharedBuffer wt = buf(mc, HD / 2);
    SharedBuffer wh = buf(mc, HD / 2);
    std::vector<int> pt(L), ph(L), pw(L);
    for (int i = 0; i < L; ++i) { pt[i] = 0; ph[i] = i / 32; pw[i] = i % 32; }
    const auto tab = ops.build_rope_tables(pt, ph, pw, HD, HD / 2, HD / 4,
                                           5e6, 1e4);
    // BATCHED: 64 back-to-back dispatches in one command buffer, then
    // divided. A single-dispatch timing of an op this small measures
    // Metal's submission latency (~3 ms, see the floor above) and not
    // the kernel -- which is how these two first read as "8.4 GB/s, 3%
    // of bandwidth" and sent a rewrite after a problem that was not
    // there. The backbone encodes ~15 dispatches per layer into shared
    // encoders, so batched is also closer to how they really run.
    const int batch = 64;
    const double n_ms = time_ms(mc, reps, [&](auto& enc) {
      for (int i = 0; i < batch; ++i) {
        ops.split_qk_norm(enc, x, wt, wh, NH, L, HD, 1e-6f);
      }
    }) / batch;
    const double r_ms = time_ms(mc, reps, [&](auto& enc) {
      for (int i = 0; i < batch; ++i) {
        ops.split_rope(enc, x, tab.cos, tab.sin, NH, L, HD, HD / 2, HD / 4);
      }
    }) / batch;
    // Both are pure streaming: read the tensor, write it back.
    const double gb = 2.0 * 2.0 * L * NH * HD / 1e9;
    std::printf("  %-34s %7.2f ms  %7.1f GB/s\n", "split_qk_norm", n_ms,
                gb / (n_ms / 1000.0));
    std::printf("  %-34s %7.2f ms  %7.1f GB/s\n", "split_rope", r_ms,
                gb / (r_ms / 1000.0));
    // q and k both, so roughly (1 + KV/NH) of each.
    const double per_layer = (n_ms + r_ms) * (1.0 + (double)KV / NH);
    std::printf("  -> %.1f ms per layer, %.2f s per forward\n\n", per_layer,
                per_layer * layers / 1000.0);
    total_ms += per_layer * layers;
  }

  // ---- RMSNorm: the two libvpipe variants ---------------------------
  std::printf("RMSNorm over the residual stream [%d, %d]\n", L, H);
  {
    SharedBuffer x = buf(mc, (std::size_t)L * H);
    SharedBuffer wv = buf(mc, H);
    SharedBuffer y = buf(mc, (std::size_t)L * H);
    const int batch = 64;
    double fast = 0.0, slow = 0.0;
    ab_ms(mc, reps,
          [&](auto& enc) {
            for (int i = 0; i < batch; ++i) {
              ops.rms_norm(enc, x, wv, y, L, H, 1e-6f, true);
            }
          },
          [&](auto& enc) {
            for (int i = 0; i < batch; ++i) {
              ops.rms_norm(enc, x, wv, y, L, H, 1e-6f, false);
            }
          }, &fast, &slow);
    fast /= batch;
    slow /= batch;
    const double gb = 2.0 * 2.0 * L * H / 1e9;
    std::printf("  %-24s fast %6.3f ms %6.1f GB/s  |  tree %6.3f ms "
                "%6.1f GB/s  %.2fx\n", "rms_norm", fast,
                gb / (fast / 1000.0), slow, gb / (slow / 1000.0),
                slow / fast);
    check(fast <= slow * 1.05,
          "rms_norm_fast is not slower than rms_norm");
    total_ms += 2.0 * fast * layers;      // input_ln + post_attention_ln
    std::printf("\n");
  }

  std::printf("estimated forward: %.2f s  (x2 branches = %.2f s/step)\n",
              total_ms / 1000.0, 2.0 * total_ms / 1000.0);

  std::printf("\n%s\n", g_fail == 0 ? "no regressions" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
