// The plugin's own Metal kernels, on the GPU, against the CPU reference
// -- which is itself verified against the reference implementation's
// goldens (u15-ref-test). So a pass here means the kernel agrees with
// SenseNova's own code, through two independently checked hops.
//
// The bar is bf16 round-off, not f32: the backbone runs bf16 and these
// are its bf16 twin, so ~3 decimal digits is all there is. A tight f32
// bar would be measuring the wrong thing.
//
// Every kernel here is one a WRONG implementation runs cleanly:
//
//   u15_split_qk_norm   two reductions per head with two weights. Doing
//                       one reduction over the whole head, or four over
//                       the quarters, produces the same shapes.
//   u15_split_rope      three sub-ranges, two thetas, three position
//                       vectors. Every wrong variant is shaped right.
//   u15_vision_rope2d   INTERLEAVED pairs, where the backbone uses
//                       rotate_half. Same tensor shape either way.
//   u15_pixel_shuffle   channel-SLOWEST. Channel-fastest gives an image
//                       of the right size made of the wrong pixels.
//   u15_gelu_erf        the erf GELU, not the tanh one libvpipe ships.

#include "u15-config.h"
#include "u15-metal-ops.h"
#include "u15-ref.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

extern "C" const unsigned char u15_kernels_bf16_metallib[];
extern "C" const unsigned long u15_kernels_bf16_metallib_len;

using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace {

int g_fail = 0;
int g_ran  = 0;

// bf16 round-off over a few dependent ops. Stated once, used everywhere,
// so loosening it is a visible edit.
constexpr double kBf16Bar = 4e-3;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  ++g_ran;
  if (!ok) { ++g_fail; }
}

std::uint16_t
to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
  return (std::uint16_t)((u + r) >> 16);
}

float
from_bf16(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

SharedBuffer
upload_bf16(MetalCompute& mc, const std::vector<float>& v)
{
  SharedBuffer b = mc.make_shared_buffer(v.size() * 2);
  auto* p = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { p[i] = to_bf16(v[i]); }
  return b;
}

std::vector<float>
download_bf16(const SharedBuffer& b, std::size_t n)
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

std::vector<float>
randn(std::size_t n, unsigned seed)
{
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) { x = d(g); }
  return v;
}

// -------------------------------------------------------------------
// the split q/k norm + the three-way rope, together
// -------------------------------------------------------------------
void
test_split_head(MetalCompute& mc, const u15::MetalOps& ops)
{
  std::printf("split q/k norm + three-way rope\n");
  const int H = 4, T = 12, D = 32;
  const int d_t = D / 2, d_h = D / 4;
  const float eps = 1e-6f;
  const double theta_t = 5000000.0, theta_hw = 10000.0;

  const std::vector<float> x = randn((std::size_t)T * H * D, 7);
  // Norm weights around 1, not exactly 1: an all-ones weight cannot
  // distinguish "applied" from "skipped".
  std::vector<float> w_t(d_t), w_hw(d_t);
  {
    std::mt19937 g(8);
    std::normal_distribution<float> d(1.0f, 0.2f);
    for (auto& v : w_t) { v = d(g); }
    for (auto& v : w_hw) { v = d(g); }
  }

  // Positions: text then an image block sharing one t.
  const int text_len = 5, tok_w = 7;
  std::vector<int> pt, ph, pw;
  for (int i = 0; i < text_len; ++i) { pt.push_back(i); ph.push_back(0); pw.push_back(0); }
  for (int i = 0; i < T - text_len; ++i) {
    pt.push_back(text_len);
    ph.push_back(i / tok_w);
    pw.push_back(i % tok_w);
  }

  // ---- CPU reference, in the same two steps the kernels take --------
  std::vector<float> ref = x;
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < H; ++h) {
      float* row = ref.data() + ((std::size_t)t * H + h) * D;
      std::vector<float> a(row, row + d_t), b(row + d_t, row + D);
      std::vector<float> na(d_t), nb(d_t);
      u15::ref::rms_norm(a.data(), w_t.data(), eps, 1, (std::size_t)d_t,
                         na.data());
      u15::ref::rms_norm(b.data(), w_hw.data(), eps, 1, (std::size_t)d_t,
                         nb.data());
      std::copy(na.begin(), na.end(), row);
      std::copy(nb.begin(), nb.end(), row + d_t);
    }
  }
  const std::vector<float> inv_t = u15::ref::rope_inv_freq(d_t, theta_t);
  const std::vector<float> inv_h = u15::ref::rope_inv_freq(d_h, theta_hw);
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < H; ++h) {
      float* row = ref.data() + ((std::size_t)t * H + h) * D;
      u15::ref::rope_half_inplace(row, inv_t, d_t, (double)pt[(std::size_t)t]);
      u15::ref::rope_half_inplace(row + d_t, inv_h, d_h,
                                  (double)ph[(std::size_t)t]);
      u15::ref::rope_half_inplace(row + d_t + d_h, inv_h, d_h,
                                  (double)pw[(std::size_t)t]);
    }
  }

  // ---- GPU ---------------------------------------------------------
  SharedBuffer xb = upload_bf16(mc, x);
  SharedBuffer wt = upload_bf16(mc, w_t);
  SharedBuffer wh = upload_bf16(mc, w_hw);
  const auto tab = ops.build_rope_tables(pt, ph, pw, D, d_t, d_h, theta_t,
                                         theta_hw);
  check(!tab.cos.empty(), "rope tables built");

  auto stream = mc.make_command_stream();
  {
    auto enc = stream.begin_compute();
    ops.split_qk_norm(enc, xb, wt, wh, H, T, D, eps);
  }
  {
    auto enc = stream.begin_compute();
    ops.split_rope(enc, xb, tab.cos, tab.sin, H, T, D, d_t, d_h);
  }
  stream.commit().wait();

  const auto got = download_bf16(xb, x.size());
  const double r = rel_l2(got, ref);
  std::printf("       rel-L2 %.3e (bf16 bar %.0e)\n", r, kBf16Bar);
  check(r < kBf16Bar, "matches the CPU reference");

  // Negative control: the same kernels with h and w EXCHANGED must not
  // match. This is the one mistake every shape check in the suite
  // passes.
  {
    SharedBuffer x2 = upload_bf16(mc, x);
    const auto bad =
        ops.build_rope_tables(pt, pw, ph, D, d_t, d_h, theta_t, theta_hw);
    auto s2 = mc.make_command_stream();
    { auto e = s2.begin_compute();
      ops.split_qk_norm(e, x2, wt, wh, H, T, D, eps); }
    { auto e = s2.begin_compute();
      ops.split_rope(e, x2, bad.cos, bad.sin, H, T, D, d_t, d_h); }
    s2.commit().wait();
    const double rr = rel_l2(download_bf16(x2, x.size()), ref);
    check(rr > 1e-2, "swapping h/w gives a DIFFERENT answer (rel-L2 " +
                         std::to_string(rr) + ")");
  }

  // And a single whole-head norm instead of two half norms.
  {
    std::vector<float> one = x;
    std::vector<float> wfull(D);
    for (int i = 0; i < d_t; ++i) {
      wfull[(std::size_t)i] = w_t[(std::size_t)i];
      wfull[(std::size_t)(i + d_t)] = w_hw[(std::size_t)i];
    }
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < H; ++h) {
        float* row = one.data() + ((std::size_t)t * H + h) * D;
        std::vector<float> o(D);
        u15::ref::rms_norm(row, wfull.data(), eps, 1, (std::size_t)D,
                           o.data());
        std::copy(o.begin(), o.end(), row);
      }
    }
    // rope it the same way so only the norm differs
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < H; ++h) {
        float* row = one.data() + ((std::size_t)t * H + h) * D;
        u15::ref::rope_half_inplace(row, inv_t, d_t,
                                    (double)pt[(std::size_t)t]);
        u15::ref::rope_half_inplace(row + d_t, inv_h, d_h,
                                    (double)ph[(std::size_t)t]);
        u15::ref::rope_half_inplace(row + d_t + d_h, inv_h, d_h,
                                    (double)pw[(std::size_t)t]);
      }
    }
    check(rel_l2(one, ref) > 1e-2,
          "ONE whole-head norm differs from TWO half norms");
  }
}

// -------------------------------------------------------------------
void
test_vision_rope(MetalCompute& mc, const u15::MetalOps& ops)
{
  std::printf("vision 2-D rope (interleaved)\n");
  const int gh = 4, gw = 6, C = 16;
  const int N = gh * gw;
  const double theta = 10000.0;

  const std::vector<float> x = randn((std::size_t)N * C, 21);
  std::vector<float> ref = x;
  const std::vector<float> inv = u15::ref::rope_inv_freq(C / 2, theta);
  for (int r = 0; r < gh; ++r) {
    for (int c = 0; c < gw; ++c) {
      float* e = ref.data() + ((std::size_t)r * gw + c) * C;
      u15::ref::rope_interleaved_inplace(e, inv, C / 2, (double)c);
      u15::ref::rope_interleaved_inplace(e + C / 2, inv, C / 2, (double)r);
    }
  }

  SharedBuffer xb = upload_bf16(mc, x);
  SharedBuffer ib = ops.make_inv_freq(C / 2, theta);
  auto stream = mc.make_command_stream();
  { auto enc = stream.begin_compute();
    ops.vision_rope2d(enc, xb, ib, N, C, gw); }
  stream.commit().wait();

  const double r = rel_l2(download_bf16(xb, x.size()), ref);
  std::printf("       rel-L2 %.3e\n", r);
  check(r < kBf16Bar, "matches the CPU reference");

  // Negative control: the backbone's rotate_half convention on the same
  // data must NOT match. Both conventions are shape-compatible.
  {
    std::vector<float> half = x;
    for (int rr = 0; rr < gh; ++rr) {
      for (int c = 0; c < gw; ++c) {
        float* e = half.data() + ((std::size_t)rr * gw + c) * C;
        u15::ref::rope_half_inplace(e, inv, C / 2, (double)c);
        u15::ref::rope_half_inplace(e + C / 2, inv, C / 2, (double)rr);
      }
    }
    check(rel_l2(half, ref) > 1e-2,
          "rotate_half differs from interleaved (the two conventions are "
          "not interchangeable)");
  }
}

// -------------------------------------------------------------------
void
test_pixel_shuffle(MetalCompute& mc, const u15::MetalOps& ops)
{
  std::printf("pixel shuffle (channel-slowest)\n");
  const int C = 16, h = 3, w = 5, r = 2;
  std::vector<float> x((std::size_t)C * h * w);
  for (std::size_t i = 0; i < x.size(); ++i) { x[i] = (float)i; }

  std::vector<float> ref((std::size_t)(C / (r * r)) * (h * r) * (w * r));
  u15::ref::pixel_shuffle(x.data(), C, h, w, r, ref.data());

  SharedBuffer xb = upload_bf16(mc, x);
  SharedBuffer ob = mc.make_shared_buffer(ref.size() * 2);
  auto stream = mc.make_command_stream();
  { auto enc = stream.begin_compute();
    ops.pixel_shuffle(enc, xb, ob, C, h, w, r); }
  stream.commit().wait();

  const auto got = download_bf16(ob, ref.size());
  // Values are small integers, exactly representable in bf16 up to 256,
  // so this is an EXACT comparison -- a permutation bug cannot hide in
  // round-off.
  bool exact = true;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (got[i] != ref[i]) { exact = false; break; }
  }
  check(exact, "exact permutation match against the CPU reference");
}

// -------------------------------------------------------------------
void
test_gelu(MetalCompute& mc, const u15::MetalOps& ops)
{
  std::printf("GELU (exact erf, not tanh)\n");
  std::vector<float> x(256);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = -6.0f + 12.0f * (float)i / (float)(x.size() - 1);
  }
  std::vector<float> ref = x;
  u15::ref::gelu_erf(ref.data(), ref.size());

  SharedBuffer xb = upload_bf16(mc, x);
  auto stream = mc.make_command_stream();
  { auto enc = stream.begin_compute();
    ops.gelu_erf(enc, xb, (int)x.size()); }
  stream.commit().wait();

  const auto got = download_bf16(xb, x.size());
  const double r = rel_l2(got, ref);
  std::printf("       rel-L2 %.3e\n", r);
  check(r < kBf16Bar, "matches the CPU reference (erf form)");

  // How far the tanh approximation actually is. MEASURED, not assumed:
  // max abs 4.7e-4 over [-6, 6], which is BELOW bf16's own resolution
  // near 1.0 (2^-8 = 3.9e-3). So libvpipe's gelu_tanh_ff_f16 would have
  // been numerically fine here and this kernel is not load-bearing --
  // it is exact and free, which is reason enough to keep it, but the
  // claim that the erf form is REQUIRED would be false.
  //
  // Pinned as an upper bound so that if either form ever changes, the
  // number is re-derived rather than inherited.
  {
    std::vector<float> tanh_g(x.size());
    double max_abs = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double v = (double)x[i];
      tanh_g[i] = (float)(0.5 * v * (1.0 + std::tanh(
          0.7978845608028654 * (v + 0.044715 * v * v * v))));
      max_abs = std::max(max_abs, std::fabs((double)tanh_g[i] - ref[i]));
    }
    std::printf("       tanh-GELU vs erf-GELU: rel-L2 %.3e, max abs "
                "%.3e (bf16 ULP near 1.0 is %.1e)\n",
                rel_l2(tanh_g, ref), max_abs, 1.0 / 256.0);
    check(max_abs < 1e-3,
          "the two GELU forms agree to well under a bf16 ULP -- the erf "
          "kernel is exact, not necessary");
  }
}

// -------------------------------------------------------------------
void
test_transpose(MetalCompute& mc, const u15::MetalOps& ops)
{
  std::printf("token-major -> head-major transpose\n");
  const int T = 7, H = 3, D = 8;
  std::vector<float> x((std::size_t)T * H * D);
  for (std::size_t i = 0; i < x.size(); ++i) { x[i] = (float)i; }

  std::vector<float> ref(x.size());
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < H; ++h) {
      for (int d = 0; d < D; ++d) {
        ref[((std::size_t)h * T + t) * D + d] =
            x[((std::size_t)t * H + h) * D + d];
      }
    }
  }

  SharedBuffer xb = upload_bf16(mc, x);
  SharedBuffer ob = mc.make_shared_buffer(x.size() * 2);
  auto stream = mc.make_command_stream();
  { auto enc = stream.begin_compute();
    ops.transpose_thd(enc, xb, ob, T, H, D); }
  stream.commit().wait();

  const auto got = download_bf16(ob, x.size());
  bool exact = true;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (got[i] != ref[i]) { exact = false; break; }
  }
  check(exact, "exact");
}


// -------------------------------------------------------------------
// libvpipe's OWN kernels, as THIS code calls them.
//
// The kernels above are the plugin's and were verified first; these are
// libvpipe's, and what is under test is the CALLING CONVENTION -- buffer
// order, constant indices, dispatch shape, and which way round the
// weight is stored. A wrong convention here runs cleanly and produces a
// numerically wrong model, which is exactly what it did the first time.
// -------------------------------------------------------------------
void
test_linear(MetalCompute& mc, const u15::MetalOps& ops)
{
  std::printf("libvpipe GEMM (weight stored [out][in])\n");
  const int M = 9, K = 64, N = 48;
  const std::vector<float> x = randn((std::size_t)M * K, 31);
  const std::vector<float> w = randn((std::size_t)N * K, 32);

  std::vector<float> ref((std::size_t)M * N);
  u15::ref::linear(x.data(), w.data(), nullptr, (std::size_t)M,
                   (std::size_t)K, (std::size_t)N, ref.data());

  SharedBuffer xb = upload_bf16(mc, x);
  SharedBuffer wb = upload_bf16(mc, w);
  SharedBuffer yb = mc.make_shared_buffer((std::size_t)M * N * 2);
  auto stream = mc.make_command_stream();
  { auto enc = stream.begin_compute();
    ops.linear(enc, xb, wb, nullptr, yb, M, K, N); }
  stream.commit().wait();

  const double r = rel_l2(download_bf16(yb, (std::size_t)M * N), ref);
  std::printf("       rel-L2 %.3e\n", r);
  check(r < kBf16Bar, "y = x @ w^T matches the CPU reference");
}

void
test_rms(MetalCompute& mc, const u15::MetalOps& ops)
{
  std::printf("libvpipe RMSNorm\n");
  const int rows = 7, dim = 64;
  const std::vector<float> x = randn((std::size_t)rows * dim, 41);
  std::vector<float> w(dim);
  { std::mt19937 g(42); std::normal_distribution<float> d(1.0f, 0.2f);
    for (auto& v : w) { v = d(g); } }

  std::vector<float> ref((std::size_t)rows * dim);
  u15::ref::rms_norm(x.data(), w.data(), 1e-6f, (std::size_t)rows,
                     (std::size_t)dim, ref.data());

  SharedBuffer xb = upload_bf16(mc, x);
  SharedBuffer wb = upload_bf16(mc, w);
  SharedBuffer yb = mc.make_shared_buffer((std::size_t)rows * dim * 2);
  auto stream = mc.make_command_stream();
  { auto enc = stream.begin_compute();
    ops.rms_norm(enc, xb, wb, yb, rows, dim, 1e-6f); }
  stream.commit().wait();

  const double r = rel_l2(download_bf16(yb, (std::size_t)rows * dim), ref);
  std::printf("       rel-L2 %.3e\n", r);
  check(r < kBf16Bar, "matches the CPU reference");
}

void
test_swiglu(MetalCompute& mc, const u15::MetalOps& ops)
{
  std::printf("libvpipe SwiGLU\n");
  const int rows = 5, inter = 32;
  const std::vector<float> g = randn((std::size_t)rows * inter, 51);
  const std::vector<float> u = randn((std::size_t)rows * inter, 52);

  std::vector<float> ref = g;
  u15::ref::silu(ref.data(), ref.size());
  for (std::size_t i = 0; i < ref.size(); ++i) { ref[i] *= u[i]; }

  SharedBuffer gb = upload_bf16(mc, g);
  SharedBuffer ub = upload_bf16(mc, u);
  SharedBuffer yb = mc.make_shared_buffer((std::size_t)rows * inter * 2);
  auto stream = mc.make_command_stream();
  { auto enc = stream.begin_compute();
    ops.swiglu(enc, gb, ub, yb, rows, inter); }
  stream.commit().wait();

  const double r = rel_l2(download_bf16(yb, (std::size_t)rows * inter), ref);
  std::printf("       rel-L2 %.3e\n", r);
  check(r < kBf16Bar, "silu(gate)*up matches the CPU reference");
}

void
test_sdpa(MetalCompute& mc, const u15::MetalOps& ops)
{
  std::printf("libvpipe SDPA (GQA, cache-strided)\n");
  const int hq = 4, hkv = 2, tq = 6, d = 32;
  const int cap = 10, tk = 6;    // cache longer than the keys in use
  const std::vector<float> q = randn((std::size_t)hq * tq * d, 61);
  std::vector<float> k((std::size_t)hkv * cap * d, 0.0f);
  std::vector<float> v((std::size_t)hkv * cap * d, 0.0f);
  {
    const auto kk = randn((std::size_t)hkv * tk * d, 62);
    const auto vv = randn((std::size_t)hkv * tk * d, 63);
    for (int h = 0; h < hkv; ++h) {
      for (int t = 0; t < tk; ++t) {
        for (int c = 0; c < d; ++c) {
          k[((std::size_t)h * cap + t) * d + c] =
              kk[((std::size_t)h * tk + t) * d + c];
          v[((std::size_t)h * cap + t) * d + c] =
              vv[((std::size_t)h * tk + t) * d + c];
        }
      }
    }
  }
  // The CPU reference takes tightly-packed k/v, so pack a copy for it.
  std::vector<float> kt((std::size_t)hkv * tk * d);
  std::vector<float> vt((std::size_t)hkv * tk * d);
  for (int h = 0; h < hkv; ++h) {
    for (int t = 0; t < tk; ++t) {
      for (int c = 0; c < d; ++c) {
        kt[((std::size_t)h * tk + t) * d + c] =
            k[((std::size_t)h * cap + t) * d + c];
        vt[((std::size_t)h * tk + t) * d + c] =
            v[((std::size_t)h * cap + t) * d + c];
      }
    }
  }

  SharedBuffer qb = upload_bf16(mc, q);
  SharedBuffer kb = upload_bf16(mc, k);
  SharedBuffer vb = upload_bf16(mc, v);
  SharedBuffer ob = mc.make_shared_buffer((std::size_t)tq * hq * d * 2);

  for (int causal = 0; causal < 2; ++causal) {
    std::vector<float> ref((std::size_t)tq * hq * d);
    std::vector<float> mask;
    if (causal != 0) {
      mask.assign((std::size_t)tq * tk, 0.0f);
      for (int i = 0; i < tq; ++i) {
        for (int j = 0; j < tk; ++j) {
          if (j > i) {
            mask[(std::size_t)i * tk + j] =
                -std::numeric_limits<float>::infinity();
          }
        }
      }
    }
    u15::ref::attention(q.data(), kt.data(), vt.data(),
                        mask.empty() ? nullptr : mask.data(), hq, hkv, tq,
                        tk, d, ref.data());
    // The CPU reference emits TOKEN-major [tq][hq*d]; libvpipe's SDPA
    // writes HEAD-major [hq][tq][d]. Compare in the kernel's layout --
    // the backbone transposes separately, and folding that transpose
    // into this comparison would hide whichever of the two is wrong.
    std::vector<float> ref_h(ref.size());
    for (int t = 0; t < tq; ++t) {
      for (int h = 0; h < hq; ++h) {
        for (int c = 0; c < d; ++c) {
          ref_h[((std::size_t)h * tq + t) * d + c] =
              ref[(std::size_t)t * hq * d + h * d + c];
        }
      }
    }

    auto stream = mc.make_command_stream();
    { auto enc = stream.begin_compute();
      ops.sdpa(enc, qb, kb, vb, ob, hq, hkv, tq, tk, d, cap,
               causal != 0, 0); }
    stream.commit().wait();

    const double r =
        rel_l2(download_bf16(ob, (std::size_t)tq * hq * d), ref_h);
    std::printf("       %s rel-L2 %.3e\n",
                causal != 0 ? "causal" : "full  ", r);
    check(r < kBf16Bar,
          std::string(causal != 0 ? "causal" : "bidirectional") +
              " matches the CPU reference");

  }
}


// -------------------------------------------------------------------
// BLOCK-CAUSAL attention: the kernel the EDIT path needs.
// -------------------------------------------------------------------
void
test_block_causal(MetalCompute& mc, const u15::MetalOps& ops)
{
  std::printf("block-causal SDPA (reference images share one t)\n");
  const int hq = 4, hkv = 2, d = 32;
  // A prefix shaped like a real edit: 3 text tokens, a 6-token image
  // block sharing ONE t, then 2 more text tokens. With a pure-text
  // prefix this rule collapses to plain causal, so a test built only on
  // text would certify a kernel that ignores the first clause entirely.
  const std::vector<int> t_idx = {0, 1, 2, 3, 3, 3, 3, 3, 3, 4, 5};
  const int T = (int)t_idx.size();
  const int cap = T + 4;               // cache longer than the keys used

  const std::vector<float> q = randn((std::size_t)hq * T * d, 71);
  std::vector<float> k((std::size_t)hkv * cap * d, 0.0f);
  std::vector<float> v((std::size_t)hkv * cap * d, 0.0f);
  std::vector<float> kt((std::size_t)hkv * T * d);
  std::vector<float> vt((std::size_t)hkv * T * d);
  {
    const auto kk = randn((std::size_t)hkv * T * d, 72);
    const auto vv = randn((std::size_t)hkv * T * d, 73);
    for (int h = 0; h < hkv; ++h) {
      for (int t = 0; t < T; ++t) {
        for (int c = 0; c < d; ++c) {
          k[((std::size_t)h * cap + t) * d + c] =
              kk[((std::size_t)h * T + t) * d + c];
          v[((std::size_t)h * cap + t) * d + c] =
              vv[((std::size_t)h * T + t) * d + c];
          kt[((std::size_t)h * T + t) * d + c] =
              kk[((std::size_t)h * T + t) * d + c];
          vt[((std::size_t)h * T + t) * d + c] =
              vv[((std::size_t)h * T + t) * d + c];
        }
      }
    }
  }

  const auto mask = u15::ref::block_causal_mask(t_idx);
  std::vector<float> ref((std::size_t)T * hq * d);
  u15::ref::attention(q.data(), kt.data(), vt.data(), mask.data(), hq, hkv,
                      (std::size_t)T, (std::size_t)T, d, ref.data());
  std::vector<float> ref_h(ref.size());
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < hq; ++h) {
      for (int c = 0; c < d; ++c) {
        ref_h[((std::size_t)h * T + t) * d + c] =
            ref[(std::size_t)t * hq * d + h * d + c];
      }
    }
  }

  SharedBuffer qb = upload_bf16(mc, q);
  SharedBuffer kb = upload_bf16(mc, k);
  SharedBuffer vb = upload_bf16(mc, v);
  SharedBuffer ob = mc.make_shared_buffer((std::size_t)T * hq * d * 2);
  SharedBuffer tb = mc.make_shared_buffer((std::size_t)T * sizeof(int));
  {
    auto* p = static_cast<int*>(tb.contents());
    for (int i = 0; i < T; ++i) { p[i] = t_idx[(std::size_t)i]; }
  }

  auto stream = mc.make_command_stream();
  { auto enc = stream.begin_compute();
    ops.sdpa_block_causal(enc, qb, kb, vb, ob, tb, hq, hkv, T, T, d, cap,
                          0); }
  stream.commit().wait();

  const double r = rel_l2(download_bf16(ob, (std::size_t)T * hq * d), ref_h);
  std::printf("       rel-L2 %.3e\n", r);
  check(r < kBf16Bar, "matches the CPU reference's block-causal rule");

  // The control that matters: PLAIN causal must give a DIFFERENT answer
  // on this input. If it did not, the whole kernel would be redundant
  // and a port could keep using libvpipe's causal one.
  {
    std::vector<float> causal_mask((std::size_t)T * T, 0.0f);
    for (int i = 0; i < T; ++i) {
      for (int j = 0; j < T; ++j) {
        if (j > i) {
          causal_mask[(std::size_t)i * T + j] =
              -std::numeric_limits<float>::infinity();
        }
      }
    }
    std::vector<float> c_ref((std::size_t)T * hq * d);
    u15::ref::attention(q.data(), kt.data(), vt.data(), causal_mask.data(),
                        hq, hkv, (std::size_t)T, (std::size_t)T, d,
                        c_ref.data());
    const double dd = rel_l2(c_ref, ref);
    check(dd > 1e-2,
          "plain causal gives a DIFFERENT answer here (rel-L2 " +
              std::to_string(dd) + ") -- the block rule is load-bearing");
  }

  // ...and on a PURE-TEXT prefix the two must AGREE, which is what
  // licenses the text-to-image path to use the mask-free causal kernel.
  {
    std::vector<int> text_t((std::size_t)T);
    for (int i = 0; i < T; ++i) { text_t[(std::size_t)i] = i; }
    const auto bm = u15::ref::block_causal_mask(text_t);
    std::vector<float> a((std::size_t)T * hq * d), b((std::size_t)T * hq * d);
    u15::ref::attention(q.data(), kt.data(), vt.data(), bm.data(), hq, hkv,
                        (std::size_t)T, (std::size_t)T, d, a.data());
    std::vector<float> cm((std::size_t)T * T, 0.0f);
    for (int i = 0; i < T; ++i) {
      for (int j = i + 1; j < T; ++j) {
        cm[(std::size_t)i * T + j] =
            -std::numeric_limits<float>::infinity();
      }
    }
    u15::ref::attention(q.data(), kt.data(), vt.data(), cm.data(), hq, hkv,
                        (std::size_t)T, (std::size_t)T, d, b.data());
    check(rel_l2(a, b) < 1e-9,
          "on a pure-TEXT prefix block-causal IS plain causal (which is "
          "what lets t2i use the mask-free kernel)");
  }
}


// -------------------------------------------------------------------
// STEEL flash attention, at the model's REAL head_dim.
//
// This needs its own test because attn_steel is only instantiated for
// head_dim 64 and 128 -- the toy d=32 the tests above use has no
// instantiation at all, so a check folded into them would report
// "unavailable" forever and pass.
//
// Two things are specific to THIS model and wrong by default:
//   * gqa_factor is the GROUP SIZE (32/8 = 4), not the kv head count. A
//     port with no GQA sets it to 1 and reads past an 8-head cache.
//   * the K/V stride is the cache CAPACITY, not the key length.
// Both are exercised here: heads differ, and the cache is longer than
// the keys in use.
// -------------------------------------------------------------------
void
test_steel_attn(MetalCompute& mc, const u15::MetalOps& ops)
{
  std::printf("STEEL flash attention (head_dim 128, GQA)\n");
  const int hq = 8, hkv = 2, tq = 48, tk = 67, d = 128;
  const int cap = tk + 21;             // cache longer than the keys used

  if (!ops.steel_attn_available(d)) {
    std::printf("       UNAVAILABLE on this host -- NOTHING was checked\n");
    return;
  }

  const std::vector<float> q = randn((std::size_t)hq * tq * d, 81);
  std::vector<float> k((std::size_t)hkv * cap * d, 0.0f);
  std::vector<float> v((std::size_t)hkv * cap * d, 0.0f);
  std::vector<float> kt((std::size_t)hkv * tk * d);
  std::vector<float> vt((std::size_t)hkv * tk * d);
  {
    const auto kk = randn((std::size_t)hkv * tk * d, 82);
    const auto vv = randn((std::size_t)hkv * tk * d, 83);
    for (int h = 0; h < hkv; ++h) {
      for (int t = 0; t < tk; ++t) {
        for (int c = 0; c < d; ++c) {
          k[((std::size_t)h * cap + t) * d + c] =
              kk[((std::size_t)h * tk + t) * d + c];
          v[((std::size_t)h * cap + t) * d + c] =
              vv[((std::size_t)h * tk + t) * d + c];
          kt[((std::size_t)h * tk + t) * d + c] =
              kk[((std::size_t)h * tk + t) * d + c];
          vt[((std::size_t)h * tk + t) * d + c] =
              vv[((std::size_t)h * tk + t) * d + c];
        }
      }
    }
  }

  std::vector<float> ref((std::size_t)tq * hq * d);
  u15::ref::attention(q.data(), kt.data(), vt.data(), nullptr, hq, hkv,
                      (std::size_t)tq, (std::size_t)tk, d, ref.data());
  std::vector<float> ref_h(ref.size());
  for (int t = 0; t < tq; ++t) {
    for (int h = 0; h < hq; ++h) {
      for (int c = 0; c < d; ++c) {
        ref_h[((std::size_t)h * tq + t) * d + c] =
            ref[(std::size_t)t * hq * d + h * d + c];
      }
    }
  }

  SharedBuffer qb = upload_bf16(mc, q);
  SharedBuffer kb = upload_bf16(mc, k);
  SharedBuffer vb = upload_bf16(mc, v);
  SharedBuffer ob = mc.make_shared_buffer((std::size_t)tq * hq * d * 2);

  u15::MetalOps::SteelAttn plan;
  check(ops.steel_attn_plan(&plan, hq, hkv, tq, tk, d, cap),
        "steel plan built (non-square, unaligned tq and tk)");
  if (!plan.valid()) { return; }

  auto stream = mc.make_command_stream();
  { auto enc = stream.begin_compute();
    ops.sdpa_steel(enc, plan, qb, kb, vb, ob); }
  stream.commit().wait();

  const double r = rel_l2(download_bf16(ob, (std::size_t)tq * hq * d),
                          ref_h);
  std::printf("       rel-L2 %.3e\n", r);
  check(r < kBf16Bar, "matches the CPU reference");

  // A wrong gqa_factor is the failure this kernel invites, and it is
  // NOT a crash -- query head h simply reads kv head h. With hkv < hq
  // that also runs off the end of the cache, so the control both proves
  // the field matters and shows what getting it wrong looks like.
  {
    // Built fresh: SteelAttn holds a ComputeFunction and is move-only.
    u15::MetalOps::SteelAttn bad;
    // gqa_factor forced to 1 by claiming hq kv heads.
    if (ops.steel_attn_plan(&bad, hq, hq, tq, tk, d, cap)) {
      SharedBuffer bo =
          mc.make_shared_buffer((std::size_t)tq * hq * d * 2);
      auto s2 = mc.make_command_stream();
      { auto e = s2.begin_compute();
        ops.sdpa_steel(e, bad, qb, kb, vb, bo); }
      s2.commit().wait();
      const double rb =
          rel_l2(download_bf16(bo, (std::size_t)tq * hq * d), ref_h);
      check(rb > 1e-2,
            "gqa_factor is load-bearing: forcing it to 1 gives a "
            "DIFFERENT answer (rel-L2 " + std::to_string(rb) + ")");
    }
  }
}

}  // namespace

int
main()
{
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no usable Metal device. NOTHING was checked.\n");
    return 0;
  }
  // Register the plugin's metallib the way the plugin does at load: a
  // test binary is not a plugin, so nothing has done it for us.
  if (!mc.register_metal_library(u15::kMetalLibBf16,
                                 u15_kernels_bf16_metallib,
                                 u15_kernels_bf16_metallib_len)) {
    std::printf("FAILED to register the metallib\n");
    return 1;
  }

  u15::MetalOps ops;
  std::string err;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED to init MetalOps: %s\n", err.c_str());
    return 1;
  }
  std::printf("MetalOps init OK (every kernel validated)\n\n");

  test_linear(mc, ops);
  test_rms(mc, ops);
  test_swiglu(mc, ops);
  test_sdpa(mc, ops);
  test_steel_attn(mc, ops);
  test_block_causal(mc, ops);
  test_split_head(mc, ops);
  test_vision_rope(mc, ops);
  test_pixel_shuffle(mc, ops);
  test_gelu(mc, ops);
  test_transpose(mc, ops);

  std::printf("\n%d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
