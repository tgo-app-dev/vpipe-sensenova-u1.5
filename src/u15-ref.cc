#include "u15-ref.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace u15 {
namespace ref {

// ---------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------

void
rms_norm(const float* x, const float* w, float eps, std::size_t rows,
         std::size_t dim, float* out)
{
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * dim;
    float*       o  = out + r * dim;
    // Accumulate in double: the reference computes the mean in f32 but
    // over a small dim, and a double accumulator can only be closer to
    // the exact value the golden was produced from.
    double acc = 0.0;
    for (std::size_t c = 0; c < dim; ++c) { acc += (double)xr[c] * xr[c]; }
    const double inv = 1.0 / std::sqrt(acc / (double)dim + (double)eps);
    for (std::size_t c = 0; c < dim; ++c) {
      o[c] = (float)((double)xr[c] * inv * (double)w[c]);
    }
  }
}

void
linear(const float* x, const float* w, const float* b, std::size_t rows,
       std::size_t in_dim, std::size_t out_dim, float* out)
{
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * in_dim;
    float*       o  = out + r * out_dim;
    for (std::size_t oc = 0; oc < out_dim; ++oc) {
      const float* wr = w + oc * in_dim;
      double acc = (b != nullptr) ? (double)b[oc] : 0.0;
      for (std::size_t ic = 0; ic < in_dim; ++ic) {
        acc += (double)xr[ic] * (double)wr[ic];
      }
      o[oc] = (float)acc;
    }
  }
}

void
silu(float* x, std::size_t n)
{
  for (std::size_t i = 0; i < n; ++i) {
    const double v = (double)x[i];
    x[i] = (float)(v / (1.0 + std::exp(-v)));
  }
}

void
gelu_erf(float* x, std::size_t n)
{
  // torch's nn.GELU() default is the EXACT erf form, not the tanh
  // approximation. Both appear in this model's wider family, so the
  // choice is named rather than assumed.
  for (std::size_t i = 0; i < n; ++i) {
    const double v = (double)x[i];
    x[i] = (float)(0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0))));
  }
}

// ---------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------

std::vector<float>
rope_inv_freq(int dim, double theta)
{
  std::vector<float> f((std::size_t)(dim / 2));
  for (int i = 0; i < dim / 2; ++i) {
    f[(std::size_t)i] =
        (float)(1.0 / std::pow(theta, (double)(2 * i) / (double)dim));
  }
  return f;
}

void
rope_half_inplace(float* x, const std::vector<float>& inv_freq, int dim,
                  double position)
{
  const int half = dim / 2;
  for (int j = 0; j < half; ++j) {
    const double ang = position * (double)inv_freq[(std::size_t)j];
    const double c = std::cos(ang), s = std::sin(ang);
    const double a = (double)x[j];
    const double b = (double)x[j + half];
    x[j]        = (float)(a * c - b * s);
    x[j + half] = (float)(b * c + a * s);
  }
}

void
rope_interleaved_inplace(float* x, const std::vector<float>& inv_freq,
                         int dim, double position)
{
  const int pairs = dim / 2;
  for (int i = 0; i < pairs; ++i) {
    const double ang = position * (double)inv_freq[(std::size_t)i];
    const double c = std::cos(ang), s = std::sin(ang);
    const double a = (double)x[2 * i];
    const double b = (double)x[2 * i + 1];
    x[2 * i]     = (float)(a * c - b * s);
    x[2 * i + 1] = (float)(a * s + b * c);
  }
}

// ---------------------------------------------------------------------
// the three-way split head
// ---------------------------------------------------------------------

void
split_rope_project(const float* x, const float* w_proj, const float* w_norm,
                   const float* w_norm_hw, const std::vector<int>& pos_t,
                   const std::vector<int>& pos_h,
                   const std::vector<int>& pos_w, std::size_t tokens,
                   std::size_t hidden, const SplitRopeParams& p, float* out)
{
  const std::size_t hd   = (std::size_t)p.head_dim;
  const std::size_t H    = (std::size_t)p.heads;
  const std::size_t t_d  = hd / 2;          // 64
  const std::size_t hw_d = hd / 2;          // 64, before the h/w split
  const std::size_t q_d  = hd / 4;          // 32 each

  // project: [tokens][H*hd]
  std::vector<float> proj(tokens * H * hd);
  linear(x, w_proj, nullptr, tokens, hidden, H * hd, proj.data());

  // The norms run over the HALVES, per (token, head). Gather each half
  // into a contiguous row so rms_norm sees the shape it expects.
  std::vector<float> th(tokens * H * t_d);
  std::vector<float> hwh(tokens * H * hw_d);
  for (std::size_t s = 0; s < tokens; ++s) {
    for (std::size_t h = 0; h < H; ++h) {
      const float* src = proj.data() + (s * H + h) * hd;
      std::copy(src, src + t_d, th.data() + (s * H + h) * t_d);
      std::copy(src + t_d, src + hd, hwh.data() + (s * H + h) * hw_d);
    }
  }
  std::vector<float> thn(th.size()), hwn(hwh.size());
  rms_norm(th.data(), w_norm, (float)p.rms_eps, tokens * H, t_d,
           thn.data());
  // NOTE: one norm over all 64 of the hw half, BEFORE splitting it into
  // h and w. Normalising the 32-wide quarters separately is a different
  // function and a silent one -- the shapes still line up.
  rms_norm(hwh.data(), w_norm_hw, (float)p.rms_eps, tokens * H, hw_d,
           hwn.data());

  const std::vector<float> inv_t  = rope_inv_freq((int)t_d, p.theta_t);
  const std::vector<float> inv_hw = rope_inv_freq((int)q_d, p.theta_hw);

  // emit head-major [H][tokens][hd] as [t | h | w]
  for (std::size_t h = 0; h < H; ++h) {
    for (std::size_t s = 0; s < tokens; ++s) {
      float* o = out + (h * tokens + s) * hd;
      std::copy(thn.data() + (s * H + h) * t_d,
                thn.data() + (s * H + h) * t_d + t_d, o);
      std::copy(hwn.data() + (s * H + h) * hw_d,
                hwn.data() + (s * H + h) * hw_d + hw_d, o + t_d);

      rope_half_inplace(o, inv_t, (int)t_d, (double)pos_t[s]);
      rope_half_inplace(o + t_d, inv_hw, (int)q_d, (double)pos_h[s]);
      rope_half_inplace(o + t_d + q_d, inv_hw, (int)q_d, (double)pos_w[s]);
    }
  }
}

// ---------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------

std::vector<float>
block_causal_mask(const std::vector<int>& t_index)
{
  const std::size_t L = t_index.size();
  std::vector<float> m(L * L);
  const float ninf = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < L; ++i) {
    for (std::size_t j = 0; j < L; ++j) {
      const bool ok = (t_index[j] == t_index[i]) || (j <= i);
      m[i * L + j] = ok ? 0.0f : ninf;
    }
  }
  return m;
}

void
attention(const float* q, const float* k, const float* v, const float* mask,
          std::size_t hq, std::size_t hkv, std::size_t tq, std::size_t tk,
          std::size_t d, float* out)
{
  const double scale = 1.0 / std::sqrt((double)d);
  const std::size_t group = hq / hkv;
  std::vector<double> row(tk);
  for (std::size_t h = 0; h < hq; ++h) {
    const std::size_t hk = h / group;
    for (std::size_t i = 0; i < tq; ++i) {
      const float* qi = q + (h * tq + i) * d;
      double mx = -std::numeric_limits<double>::infinity();
      for (std::size_t j = 0; j < tk; ++j) {
        const float* kj = k + (hk * tk + j) * d;
        double acc = 0.0;
        for (std::size_t c = 0; c < d; ++c) {
          acc += (double)qi[c] * (double)kj[c];
        }
        acc *= scale;
        if (mask != nullptr) { acc += (double)mask[i * tk + j]; }
        row[j] = acc;
        mx = std::max(mx, acc);
      }
      double sum = 0.0;
      for (std::size_t j = 0; j < tk; ++j) {
        row[j] = std::exp(row[j] - mx);
        sum += row[j];
      }
      const double inv = (sum > 0.0) ? 1.0 / sum : 0.0;
      float* o = out + i * (hq * d) + h * d;
      for (std::size_t c = 0; c < d; ++c) {
        double acc = 0.0;
        for (std::size_t j = 0; j < tk; ++j) {
          acc += row[j] * inv * (double)v[(hk * tk + j) * d + c];
        }
        o[c] = (float)acc;
      }
    }
  }
}

// ---------------------------------------------------------------------
// one MoT decoder layer
// ---------------------------------------------------------------------

void
decoder_layer(const float* x, const LayerWeights& w, const LayerDims& d,
              const std::vector<int>& pos_t, const std::vector<int>& pos_h,
              const std::vector<int>& pos_w, const float* mask,
              std::size_t tokens, float* out)
{
  const std::size_t H = d.heads, KV = d.kv_heads, hd = d.head_dim;
  const std::size_t hidden = d.hidden;

  std::vector<float> normed(tokens * hidden);
  rms_norm(x, w.input_layernorm, (float)d.rms_eps, tokens, hidden,
           normed.data());

  SplitRopeParams p;
  p.head_dim = (int)hd;
  p.theta_t  = d.theta_t;
  p.theta_hw = d.theta_hw;
  p.rms_eps  = d.rms_eps;

  std::vector<float> q(H * tokens * hd), k(KV * tokens * hd);
  p.heads = (int)H;
  split_rope_project(normed.data(), w.q_proj, w.q_norm, w.q_norm_hw, pos_t,
                     pos_h, pos_w, tokens, hidden, p, q.data());
  p.heads = (int)KV;
  split_rope_project(normed.data(), w.k_proj, w.k_norm, w.k_norm_hw, pos_t,
                     pos_h, pos_w, tokens, hidden, p, k.data());

  // v has no norm and no rope, so it is a plain projection -- but it
  // still needs the head-major transpose the other two get for free.
  std::vector<float> vt(tokens * KV * hd);
  linear(normed.data(), w.v_proj, nullptr, tokens, hidden, KV * hd,
         vt.data());
  std::vector<float> v(KV * tokens * hd);
  for (std::size_t s = 0; s < tokens; ++s) {
    for (std::size_t h = 0; h < KV; ++h) {
      std::copy(vt.data() + (s * KV + h) * hd,
                vt.data() + (s * KV + h) * hd + hd,
                v.data() + (h * tokens + s) * hd);
    }
  }

  std::vector<float> att(tokens * H * hd);
  attention(q.data(), k.data(), v.data(), mask, H, KV, tokens, tokens, hd,
            att.data());

  std::vector<float> proj(tokens * hidden);
  linear(att.data(), w.o_proj, nullptr, tokens, H * hd, hidden,
         proj.data());
  for (std::size_t i = 0; i < tokens * hidden; ++i) {
    proj[i] += x[i];                                     // residual
  }

  // ---- MLP: SwiGLU ----
  std::vector<float> n2(tokens * hidden);
  rms_norm(proj.data(), w.post_attention_layernorm, (float)d.rms_eps,
           tokens, hidden, n2.data());

  const std::size_t I = d.intermediate;
  std::vector<float> g(tokens * I), u(tokens * I);
  linear(n2.data(), w.gate_proj, nullptr, tokens, hidden, I, g.data());
  linear(n2.data(), w.up_proj, nullptr, tokens, hidden, I, u.data());
  silu(g.data(), g.size());
  for (std::size_t i = 0; i < g.size(); ++i) { g[i] *= u[i]; }

  std::vector<float> down(tokens * hidden);
  linear(g.data(), w.down_proj, nullptr, tokens, I, hidden, down.data());
  for (std::size_t i = 0; i < tokens * hidden; ++i) {
    out[i] = proj[i] + down[i];
  }
}

// ---------------------------------------------------------------------
// the patch embedder
// ---------------------------------------------------------------------

void
vision_embed(const float* px, const VisionWeights& w, int grid_h,
             int grid_w, int patch_px, int hidden, int llm_hidden,
             int merge, double theta, float* out)
{
  const std::size_t n   = (std::size_t)grid_h * (std::size_t)grid_w;
  const std::size_t pin = (std::size_t)(3 * patch_px * patch_px);

  // patch_embedding is Conv2d(3 -> hidden, k=patch, s=patch) whose kernel
  // equals its input, so it IS a per-patch linear. The weight's memory
  // order [hidden][3][p][p] is exactly [hidden][3*p*p], so it can be used
  // as a linear weight with no repacking -- provided the input patch was
  // packed CHANNEL-FIRST, which is why patchify is called with
  // channel_first=True for this path and not for z.
  std::vector<float> emb(n * (std::size_t)hidden);
  linear(px, w.patch_w, w.patch_b, n, pin, (std::size_t)hidden,
         emb.data());
  gelu_erf(emb.data(), emb.size());

  // 2-D RoPE, INTERLEAVED, in f32: the first half by the column, the
  // second half by the row.
  const int half = hidden / 2;
  const std::vector<float> inv = rope_inv_freq(half, theta);
  for (int r = 0; r < grid_h; ++r) {
    for (int c = 0; c < grid_w; ++c) {
      float* e = emb.data() + ((std::size_t)r * grid_w + c) * hidden;
      rope_interleaved_inplace(e, inv, half, (double)c);          // abs_x
      rope_interleaved_inplace(e + half, inv, half, (double)r);   // abs_y
    }
  }

  // dense_embedding is Conv2d(hidden -> llm, k=merge, s=merge) over the
  // [grid_h, grid_w] map. Its weight is [llm][hidden][merge][merge], so a
  // merged tile has to be gathered CHANNEL-SLOWEST to match.
  const int oh = grid_h / merge, ow = grid_w / merge;
  const std::size_t tile = (std::size_t)hidden * merge * merge;
  std::vector<float> gathered((std::size_t)oh * ow * tile);
  for (int r = 0; r < oh; ++r) {
    for (int c = 0; c < ow; ++c) {
      float* g = gathered.data() + ((std::size_t)r * ow + c) * tile;
      for (int ch = 0; ch < hidden; ++ch) {
        for (int i = 0; i < merge; ++i) {
          for (int j = 0; j < merge; ++j) {
            const std::size_t src =
                ((std::size_t)(r * merge + i) * grid_w + (c * merge + j)) *
                    (std::size_t)hidden + (std::size_t)ch;
            g[((std::size_t)ch * merge + i) * merge + j] = emb[src];
          }
        }
      }
    }
  }
  linear(gathered.data(), w.dense_w, w.dense_b,
         (std::size_t)oh * ow, tile, (std::size_t)llm_hidden, out);
}

// ---------------------------------------------------------------------
// the pixel head
// ---------------------------------------------------------------------

void
pixel_shuffle(const float* x, int ch_in, int h, int w, int r, float* out)
{
  const int co = ch_in / (r * r);
  const int oh = h * r, ow = w * r;
  for (int c = 0; c < co; ++c) {
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < r; ++j) {
        const int cin = c * r * r + i * r + j;
        for (int y = 0; y < h; ++y) {
          for (int x2 = 0; x2 < w; ++x2) {
            out[((std::size_t)c * oh + (y * r + i)) * ow + (x2 * r + j)] =
                x[((std::size_t)cin * h + y) * w + x2];
          }
        }
      }
    }
  }
}

void
conv3x3(const float* x, const float* kernel, const float* b, int cin,
        int cout, int h, int w_, float* out)
{
  for (int oc = 0; oc < cout; ++oc) {
    for (int y = 0; y < h; ++y) {
      for (int x2 = 0; x2 < w_; ++x2) {
        double acc = (b != nullptr) ? (double)b[oc] : 0.0;
        for (int ic = 0; ic < cin; ++ic) {
          for (int ky = 0; ky < 3; ++ky) {
            const int sy = y + ky - 1;
            if (sy < 0 || sy >= h) { continue; }
            for (int kx = 0; kx < 3; ++kx) {
              const int sx = x2 + kx - 1;
              if (sx < 0 || sx >= w_) { continue; }
              const std::size_t wi =
                  (((std::size_t)oc * cin + ic) * 3 + ky) * 3 + kx;
              acc += (double)kernel[wi] *
                     (double)x[((std::size_t)ic * h + sy) * w_ + sx];
            }
          }
        }
        out[((std::size_t)oc * h + y) * w_ + x2] = (float)acc;
      }
    }
  }
}

void
conv_decoder(const float* x, const HeadWeights& w, int in_ch, int th,
             int tw, int hidden_ch, float* out)
{
  // [in_ch][th][tw] -> PS(2) -> [in_ch/4][2th][2tw]
  const int c1 = in_ch / 4;
  std::vector<float> a((std::size_t)c1 * (2 * th) * (2 * tw));
  pixel_shuffle(x, in_ch, th, tw, 2, a.data());

  // conv1: c1 -> hidden_ch
  std::vector<float> b((std::size_t)hidden_ch * (2 * th) * (2 * tw));
  conv3x3(a.data(), w.conv1_w, w.conv1_b, c1, hidden_ch, 2 * th, 2 * tw,
          b.data());
  gelu_erf(b.data(), b.size());

  // PS(2) -> [hidden_ch/4][4th][4tw]
  const int c2 = hidden_ch / 4;
  std::vector<float> c((std::size_t)c2 * (4 * th) * (4 * tw));
  pixel_shuffle(b.data(), hidden_ch, 2 * th, 2 * tw, 2, c.data());

  // conv2: c2 -> 192 (= 3 * 8 * 8). No activation after it.
  std::vector<float> e((std::size_t)192 * (4 * th) * (4 * tw));
  conv3x3(c.data(), w.conv2_w, w.conv2_b, c2, 192, 4 * th, 4 * tw,
          e.data());

  // PS(8) -> [3][32th][32tw]
  pixel_shuffle(e.data(), 192, 4 * th, 4 * tw, 8, out);
}

// ---------------------------------------------------------------------
// patchify
// ---------------------------------------------------------------------

void
patchify_channel_last(const float* img, int c, int h, int w, int p,
                      float* out)
{
  const int gh = h / p, gw = w / p;
  const std::size_t stride = (std::size_t)p * p * c;
  for (int hi = 0; hi < gh; ++hi) {
    for (int wi = 0; wi < gw; ++wi) {
      float* o = out + ((std::size_t)hi * gw + wi) * stride;
      for (int pi = 0; pi < p; ++pi) {
        for (int qi = 0; qi < p; ++qi) {
          for (int ci = 0; ci < c; ++ci) {
            o[((std::size_t)pi * p + qi) * c + ci] =
                img[((std::size_t)ci * h + (hi * p + pi)) * w +
                    (wi * p + qi)];
          }
        }
      }
    }
  }
}

void
patchify_channel_first(const float* img, int c, int h, int w, int p,
                       float* out)
{
  const int gh = h / p, gw = w / p;
  const std::size_t stride = (std::size_t)p * p * c;
  for (int hi = 0; hi < gh; ++hi) {
    for (int wi = 0; wi < gw; ++wi) {
      float* o = out + ((std::size_t)hi * gw + wi) * stride;
      for (int ci = 0; ci < c; ++ci) {
        for (int pi = 0; pi < p; ++pi) {
          for (int qi = 0; qi < p; ++qi) {
            o[((std::size_t)ci * p + pi) * p + qi] =
                img[((std::size_t)ci * h + (hi * p + pi)) * w +
                    (wi * p + qi)];
          }
        }
      }
    }
  }
}

void
patchify_channel_first_hwc(const float* img_hwc, int c, int h, int w, int p,
                           float* out)
{
  const int gh = h / p, gw = w / p;
  const std::size_t stride = (std::size_t)p * p * c;
  for (int hi = 0; hi < gh; ++hi) {
    for (int wi = 0; wi < gw; ++wi) {
      float* o = out + ((std::size_t)hi * gw + wi) * stride;
      for (int ci = 0; ci < c; ++ci) {
        for (int pi = 0; pi < p; ++pi) {
          for (int qi = 0; qi < p; ++qi) {
            o[((std::size_t)ci * p + pi) * p + qi] =
                img_hwc[((std::size_t)(hi * p + pi) * w + (wi * p + qi)) *
                            c + ci];
          }
        }
      }
    }
  }
}

void
unpatchify_channel_last(const float* x, int c, int h, int w, int p,
                        float* img)
{
  const int gh = h / p, gw = w / p;
  const std::size_t stride = (std::size_t)p * p * c;
  for (int hi = 0; hi < gh; ++hi) {
    for (int wi = 0; wi < gw; ++wi) {
      const float* o = x + ((std::size_t)hi * gw + wi) * stride;
      for (int pi = 0; pi < p; ++pi) {
        for (int qi = 0; qi < p; ++qi) {
          for (int ci = 0; ci < c; ++ci) {
            img[((std::size_t)ci * h + (hi * p + pi)) * w + (wi * p + qi)] =
                o[((std::size_t)pi * p + qi) * c + ci];
          }
        }
      }
    }
  }
}


// ---------------------------------------------------------------------
// reference-image preprocessing (the EDIT path)
// ---------------------------------------------------------------------

namespace {

int
round_by_factor_(double v, int f)
{
  return (int)std::llround(v / (double)f) * f;
}

int
floor_by_factor_(double v, int f)
{
  return (int)std::floor(v / (double)f) * f;
}

int
ceil_by_factor_(double v, int f)
{
  return (int)std::ceil(v / (double)f) * f;
}

// PIL's bicubic, a = -0.5. Support 2.0.
double
bicubic_(double x)
{
  constexpr double a = -0.5;
  if (x < 0.0) { x = -x; }
  if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
  if (x < 2.0) { return (((x - 5.0) * x + 8.0) * x - 4.0) * a; }
  return 0.0;
}

unsigned char
clip8_(double v)
{
  const double r = std::floor(v + 0.5);
  if (r <= 0.0) { return 0; }
  if (r >= 255.0) { return 255; }
  return (unsigned char)r;
}

// One resample pass along a single axis, exactly as PIL's
// precompute_coeffs + ImagingResampleHorizontal do it.
//
// The part that is easy to miss: `filterscale` is the DOWNSCALE factor
// clamped at 1, and the filter support is multiplied by it. That is what
// makes PIL antialias when shrinking -- a fixed-support bicubic does
// not, and aliases instead.
void
resample_axis_(const unsigned char* src, int s_len, int other, int stride,
               int step, unsigned char* dst, int d_len, int d_stride,
               int d_step, int channels)
{
  const double scale = (double)s_len / (double)d_len;
  const double filterscale = scale < 1.0 ? 1.0 : scale;
  const double support = 2.0 * filterscale;
  const double ss = 1.0 / filterscale;

  std::vector<double> k;
  for (int xx = 0; xx < d_len; ++xx) {
    const double center = (xx + 0.5) * scale;
    int xmin = (int)(center - support + 0.5);
    if (xmin < 0) { xmin = 0; }
    int xmax = (int)(center + support + 0.5);
    if (xmax > s_len) { xmax = s_len; }
    const int n = xmax - xmin;
    if (n <= 0) { continue; }
    k.assign((std::size_t)n, 0.0);
    double ww = 0.0;
    for (int i = 0; i < n; ++i) {
      const double w = bicubic_(((double)(i + xmin) - center + 0.5) * ss);
      k[(std::size_t)i] = w;
      ww += w;
    }
    if (ww != 0.0) {
      for (int i = 0; i < n; ++i) { k[(std::size_t)i] /= ww; }
    }
    for (int o = 0; o < other; ++o) {
      for (int c = 0; c < channels; ++c) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) {
          acc += k[(std::size_t)i] *
                 (double)src[(std::size_t)o * stride +
                             (std::size_t)(i + xmin) * step + c];
        }
        dst[(std::size_t)o * d_stride + (std::size_t)xx * d_step + c] =
            clip8_(acc);
      }
    }
  }
}

}  // namespace

bool
smart_resize(int height, int width, int factor, long min_pixels,
             long max_pixels, int* out_h, int* out_w)
{
  if (height <= 0 || width <= 0 || factor <= 0) { return false; }
  const double mx = (double)std::max(height, width);
  const double mn = (double)std::min(height, width);
  if (mx / mn > 200.0) { return false; }

  int h_bar = std::max(factor, round_by_factor_((double)height, factor));
  int w_bar = std::max(factor, round_by_factor_((double)width, factor));
  const double area = (double)height * (double)width;
  if ((long)h_bar * w_bar > max_pixels) {
    const double beta = std::sqrt(area / (double)max_pixels);
    h_bar = std::max(factor, floor_by_factor_((double)height / beta, factor));
    w_bar = std::max(factor, floor_by_factor_((double)width / beta, factor));
  } else if ((long)h_bar * w_bar < min_pixels) {
    const double beta = std::sqrt((double)min_pixels / area);
    h_bar = ceil_by_factor_((double)height * beta, factor);
    w_bar = ceil_by_factor_((double)width * beta, factor);
  }
  *out_h = h_bar;
  *out_w = w_bar;
  return true;
}

void
resize_bicubic_u8(const unsigned char* src, int sw, int sh,
                  unsigned char* dst, int dw, int dh, int channels)
{
  if (sw == dw && sh == dh) {
    std::copy(src, src + (std::size_t)sw * sh * channels, dst);
    return;
  }
  // Horizontal first, into an 8-bit intermediate -- as PIL does. Staying
  // in float across both passes would drift from it by more than the
  // rounding this reproduces.
  std::vector<unsigned char> tmp((std::size_t)dw * sh * channels);
  resample_axis_(src, sw, sh, (std::size_t)sw * channels, channels,
                 tmp.data(), dw, (std::size_t)dw * channels, channels,
                 channels);
  // Vertical: the "axis" is now rows, so the strides swap roles.
  for (int x = 0; x < dw; ++x) {
    resample_axis_(tmp.data() + (std::size_t)x * channels, sh, 1,
                   0, (std::size_t)dw * channels,
                   dst + (std::size_t)x * channels, dh, 0,
                   (std::size_t)dw * channels, channels);
  }
}

void
reference_patches(const unsigned char* rgb, int w, int h, int patch,
                  float* out)
{
  const int gh = h / patch, gw = w / patch;
  const std::size_t stride = (std::size_t)3 * patch * patch;
  for (int hi = 0; hi < gh; ++hi) {
    for (int wi = 0; wi < gw; ++wi) {
      float* o = out + ((std::size_t)hi * gw + wi) * stride;
      for (int ci = 0; ci < 3; ++ci) {
        for (int pi = 0; pi < patch; ++pi) {
          for (int qi = 0; qi < patch; ++qi) {
            const std::size_t si =
                ((std::size_t)(hi * patch + pi) * w + (wi * patch + qi)) * 3 +
                (std::size_t)ci;
            const double v = (double)rgb[si] / 255.0;
            o[((std::size_t)ci * patch + pi) * patch + qi] =
                (float)((v - kImagenetMean[ci]) / kImagenetStd[ci]);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------
// sampling
// ---------------------------------------------------------------------

std::vector<float>
time_schedule(int num_steps, double shift)
{
  std::vector<float> t((std::size_t)num_steps + 1);
  for (int i = 0; i <= num_steps; ++i) {
    const double lin = (double)i / (double)num_steps;
    double sigma = 1.0 - lin;
    sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma);
    t[(std::size_t)i] = (float)(1.0 - sigma);
  }
  return t;
}

std::vector<float>
timestep_sinusoid(double t, int dim, double max_period)
{
  const int half = dim / 2;
  std::vector<float> e((std::size_t)dim, 0.0f);
  for (int i = 0; i < half; ++i) {
    const double freq =
        std::exp(-std::log(max_period) * (double)i / (double)half);
    const double a = t * freq;
    // COSINE FIRST -- the reference concatenates [cos, sin], which is the
    // flip_sin_to_cos=True convention. Getting this backwards is a
    // half-period phase error that still trains-looking output.
    e[(std::size_t)i]        = (float)std::cos(a);
    e[(std::size_t)(i + half)] = (float)std::sin(a);
  }
  return e;
}

double
optimized_scale(const float* pos, const float* neg, std::size_t n)
{
  double dot = 0.0, nrm = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    dot += (double)pos[i] * (double)neg[i];
    nrm += (double)neg[i] * (double)neg[i];
  }
  return dot / (nrm + 1e-8);
}

}  // namespace ref
}  // namespace u15
