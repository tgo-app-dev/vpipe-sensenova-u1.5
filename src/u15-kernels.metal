// The Metal kernels SenseNova-U1.5 needs that libvpipe does not have.
//
// DELIBERATELY SHORT. libvpipe's embedded libraries are reachable from a
// plugin by name, so the GEMMs (dense_gemm_bf16), RMSNorm
// (rms_norm_bf16), the SwiGLU MLP, the residual add, im2col for the 3x3
// convolutions and flash attention are all REUSED. What is here is only
// what is genuinely this model's own.
//
// VPIPE_ELT is set at compile time (bfloat for the backbone, half and
// float for the twins) exactly as the in-tree kernels do it.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT bfloat
#endif

// ---------------------------------------------------------------------
// The split q/k RMSNorm.
//
// head_dim is 128 but all four norm weights are 64 wide: the head is cut
// in half, the t half normalised by q_norm and the hw half by
// q_norm_hw, and only THEN is the hw half split into h and w. So this is
// two independent reductions inside one head, with two different
// weights -- which is why libvpipe's rms_norm_f16 (one reduction over a
// whole row) cannot do it.
//
// Normalising the 32-wide quarters separately instead would produce the
// same SHAPES and a different model. There is no way to catch that
// downstream, which is the reason this kernel exists rather than a
// clever reuse.
//
// Runs TOKEN-MAJOR, on the [T][H*D] a GEMM produces, so no transpose is
// needed before it.
//
// One SIMDGROUP per (token, head), reducing with simd_sum.
//
// The first version gave each (token, head) ONE thread walking 128
// elements serially. It measured 8.4 GB/s against this box's ~273 --
// 3% of bandwidth -- because a single lane cannot coalesce a 256-byte
// row and 32 of every 32 lanes sat idle. This reads the row across the
// simdgroup, so the loads coalesce and both reductions are one
// simd_sum each.
//
//   x   [T][H*D]  in/out
//   w_t [D/2]     the t-half weight
//   w_hw[D/2]     the hw-half weight
// grid: one simdgroup per row of (T*H); threadgroup 256 = 8 simdgroups.
kernel void u15_split_qk_norm(
    device VPIPE_ELT*       x    [[buffer(0)]],
    const device VPIPE_ELT* w_t  [[buffer(1)]],
    const device VPIPE_ELT* w_hw [[buffer(2)]],
    constant int&           H    [[buffer(3)]],
    constant int&           T    [[buffer(4)]],
    constant int&           D    [[buffer(5)]],
    constant float&         eps  [[buffer(6)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
  const uint row = tgid * nsg + sg;          // (t * H + h)
  if (row >= (uint)(T * H)) { return; }

  const int half_d = D / 2;
  device VPIPE_ELT* r = x + (uint)row * D;

  // Two reductions, one per half, in f32.
  float acc_t = 0.0f, acc_hw = 0.0f;
  for (int i = (int)lane; i < half_d; i += 32) {
    const float a = (float)r[i];
    const float b = (float)r[i + half_d];
    acc_t += a * a;
    acc_hw += b * b;
  }
  acc_t = simd_sum(acc_t);
  acc_hw = simd_sum(acc_hw);
  const float inv_t = rsqrt(acc_t / (float)half_d + eps);
  const float inv_hw = rsqrt(acc_hw / (float)half_d + eps);

  for (int i = (int)lane; i < half_d; i += 32) {
    r[i] = (VPIPE_ELT)((float)r[i] * inv_t * (float)w_t[i]);
    r[i + half_d] =
        (VPIPE_ELT)((float)r[i + half_d] * inv_hw * (float)w_hw[i]);
  }
}

// ---------------------------------------------------------------------
// The three-way split RoPE.
//
// One head's 128 dims carry THREE independent rotary embeddings:
//
//   [0, 64)    the sequence axis t, theta 5e6, rotate_half over 32 pairs
//   [64, 96)   the row axis h,      theta 1e4, rotate_half over 16 pairs
//   [96, 128)  the column axis w,   theta 1e4, rotate_half over 16 pairs
//
// libvpipe's mrope_partial_f16 is the closest existing kernel and does
// not fit: it rotates ONE sub-range anchored at offset 0, and cannot
// address the two later ranges. rope_partial_f16 has the same limit.
//
// The cos/sin tables arrive PACKED to the head's own layout -- [T][D],
// each of the three ranges already doubled in the cat(freqs, freqs) way
// its own rotate_half expects -- so the host owns the ladder arithmetic
// and this kernel owns only the rotation. Tables are f32 because the
// angles reach theta-scale and rounding them to bf16 before taking cos
// is a real phase error.
//
//   x    [T][H*D]  in/out, token-major
//   cosf [T][D]    f32, packed per the layout above
//   sinf [T][D]
//   d_t  the width of the t range (64); d_h the width of h AND of w (32)
//
// FLAT 1-D dispatch, pair index fastest. A 3-D grid of (D/2, T, H) with
// a 64-thread group launches one tiny threadgroup per (token, head) --
// ~1 KB of work each -- and spent its time on launch rather than on
// memory. Flat, consecutive lanes read consecutive pairs of one row.
kernel void u15_split_rope(
    device VPIPE_ELT*       x    [[buffer(0)]],
    const device float*     cosf [[buffer(1)]],
    const device float*     sinf [[buffer(2)]],
    constant int&           H    [[buffer(3)]],
    constant int&           T    [[buffer(4)]],
    constant int&           D    [[buffer(5)]],
    constant int&           d_t  [[buffer(6)]],
    constant int&           d_h  [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
  const int pairs = D / 2;
  if ((int)gid >= pairs * T * H) { return; }
  const int p = (int)(gid % (uint)pairs);
  const uint rest = gid / (uint)pairs;
  const int h = (int)(rest % (uint)H);
  const int t = (int)(rest / (uint)H);

  // Which of the three ranges this pair belongs to. The ranges have
  // different widths, so the pair index is walked rather than divided.
  const int pt = d_t / 2;      // pairs in the t range
  const int ph = d_h / 2;      // pairs in each of h and w
  int off, half_r, lp;
  if (p < pt) {
    off = 0;            half_r = pt; lp = p;
  } else if (p < pt + ph) {
    off = d_t;          half_r = ph; lp = p - pt;
  } else {
    off = d_t + d_h;    half_r = ph; lp = p - pt - ph;
  }

  const uint base = ((uint)t * H + h) * D + off;
  const uint ti   = (uint)t * D + off + lp;
  const float c = cosf[ti];
  const float s = sinf[ti];
  const float x1 = (float)x[base + lp];
  const float x2 = (float)x[base + lp + half_r];
  x[base + lp]          = (VPIPE_ELT)(x1 * c - x2 * s);
  x[base + lp + half_r] = (VPIPE_ELT)(x2 * c + x1 * s);
}

// ---------------------------------------------------------------------
// BLOCK-CAUSAL attention.
//
// The reference's prefill rule is
//
//     attend(i, j)  iff  t[j] == t[i]  or  j <= i
//
// -- causal ACROSS t-blocks, fully bidirectional WITHIN one. For a
// pure-TEXT prefix every t is distinct, the first clause fires only on
// the diagonal, and the rule collapses to plain causal; that is why the
// text-to-image path can use libvpipe's mask-free sdpa_causal_f16.
//
// It stops collapsing the moment a REFERENCE IMAGE is in the prefix,
// because all of an image's <IMG_CONTEXT> tokens share ONE t. Using the
// causal kernel there would make each reference-image token attend only
// to the earlier half of its own image -- which is not an error, just a
// worse reference.
//
// The rule is evaluated from the t-index VECTOR rather than a
// materialised [T][T] mask: the mask is O(T^2) (5.8 MB at T=1200, per
// layer, per branch) and carries no information the vector does not.
//
// Layout matches libvpipe's SDPA kernels exactly -- head-major q/k/v/out
// and a kv_stride in TOKENS -- so this is a drop-in for them.
//   0:q[Hq,n_q,D] 1:k 2:v 3:out 4:scale 5:T_kv 6:D 7:Hq 8:Hkv 9:n_q
//   10:q_offset 11:kv_stride 12:t_index[T_kv]
// grid (32, Hq, n_q); threadgroup (32,1,1).
kernel void u15_sdpa_block_causal(
    const device VPIPE_ELT* q         [[buffer(0)]],
    const device VPIPE_ELT* k         [[buffer(1)]],
    const device VPIPE_ELT* v         [[buffer(2)]],
    device VPIPE_ELT*       out       [[buffer(3)]],
    constant float&         scale     [[buffer(4)]],
    constant int&           T_kv      [[buffer(5)]],
    constant int&           D         [[buffer(6)]],
    constant int&           Hq        [[buffer(7)]],
    constant int&           Hkv       [[buffer(8)]],
    constant int&           n_q       [[buffer(9)]],
    constant int&           q_offset  [[buffer(10)]],
    constant int&           kv_stride [[buffer(11)]],
    const device int*       t_index   [[buffer(12)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int h  = (int)tid.y;
  const int qi = (int)tid.z;
  const int kv = h / (Hq / Hkv);
  const int q_pos = q_offset + qi;
  const int t_q = t_index[q_pos];

  const int per = (D + 31) / 32;
  const device VPIPE_ELT* qh  = q + ((uint)h * n_q + qi) * D;
  const device VPIPE_ELT* kkv = k + (uint)kv * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kv * kv_stride * D;

  float qreg[8];
  float acc[8];
  for (int p = 0; p < per; ++p) {
    const int idx = lane * per + p;
    qreg[p] = idx < D ? float(qh[idx]) * scale : 0.0f;
    acc[p] = 0.0f;
  }

  float m = -INFINITY, l = 0.0f;
  for (int j = 0; j < T_kv; ++j) {
    // The block-causal rule, per key.
    if (!(t_index[j] == t_q || j <= q_pos)) { continue; }
    float dot = 0.0f;
    for (int p = 0; p < per; ++p) {
      const int idx = lane * per + p;
      if (idx < D) { dot += qreg[p] * float(kkv[(uint)j * D + idx]); }
    }
    dot = simd_sum(dot);
    const float m_new = max(m, dot);
    const float corr = exp(m - m_new);
    const float pj = exp(dot - m_new);
    l = l * corr + pj;
    for (int p = 0; p < per; ++p) {
      const int idx = lane * per + p;
      if (idx < D) {
        acc[p] = acc[p] * corr + pj * float(vkv[(uint)j * D + idx]);
      }
    }
    m = m_new;
  }

  const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int p = 0; p < per; ++p) {
    const int idx = lane * per + p;
    if (idx < D) {
      out[((uint)h * n_q + qi) * D + idx] = (VPIPE_ELT)(acc[p] * inv_l);
    }
  }
}

// ---------------------------------------------------------------------
// The vision embedder's 2-D RoPE -- INTERLEAVED, not rotate_half.
//
// The same forward pass uses both conventions: the backbone rotates
// (i, i+half) pairs, the patch embedder rotates (2i, 2i+1) pairs. Using
// one for the other runs cleanly and scrambles the patch positions.
//
// The first half of the embedding is rotated by the patch COLUMN, the
// second half by its ROW. The reference computes this in float32 and
// casts back, so the angles here are f32 throughout.
//
//   x   [N][C]  in/out, N = grid_h*grid_w row-major
//   inv [C/4]   the shared ladder: C/2 dims per half, so C/4 pairs
// grid (C/4, N, 1).
kernel void u15_vision_rope2d(
    device VPIPE_ELT*   x       [[buffer(0)]],
    const device float* inv     [[buffer(1)]],
    constant int&       N       [[buffer(2)]],
    constant int&       C       [[buffer(3)]],
    constant int&       grid_w  [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int i = (int)gid.x;
  const int n = (int)gid.y;
  const int half_c = C / 2;
  if (n >= N || i >= half_c / 2) { return; }

  const int col = n % grid_w;     // abs_x
  const int row = n / grid_w;     // abs_y
  device VPIPE_ELT* p = x + (uint)n * C;
  const float f = inv[i];

  // first half <- column
  {
    const float a = (float)p[2 * i];
    const float b = (float)p[2 * i + 1];
    const float c = cos((float)col * f), s = sin((float)col * f);
    p[2 * i]     = (VPIPE_ELT)(a * c - b * s);
    p[2 * i + 1] = (VPIPE_ELT)(a * s + b * c);
  }
  // second half <- row
  {
    const int o = half_c + 2 * i;
    const float a = (float)p[o];
    const float b = (float)p[o + 1];
    const float c = cos((float)row * f), s = sin((float)row * f);
    p[o]     = (VPIPE_ELT)(a * c - b * s);
    p[o + 1] = (VPIPE_ELT)(a * s + b * c);
  }
}

// ---------------------------------------------------------------------
// GELU, the EXACT erf form.
//
// libvpipe ships gelu_tanh_ff_f16, which is the tanh APPROXIMATION. The
// reference uses torch's nn.GELU() default in both places this model
// needs it (the patch embedder and the pixel head), and that default is
// erf.
//
// MEASURED, so it is not overstated: the two forms differ by at most
// 4.7e-4 absolute over [-6, 6], which is BELOW bf16's own resolution
// near 1.0 (2^-8 = 3.9e-3). The tanh kernel would have been numerically
// fine. This one is kept because it is exact and costs nothing, not
// because it is required -- see u15-kernels-test, which pins that
// number rather than asserting a difference that is not there.
//
// MSL has no erf(), so it is spelled out: Abramowitz & Stegun 7.1.26,
// whose max absolute error is 1.5e-7. That is two orders below f32's own
// resolution here and four below bf16's, so the approximation is not
// what limits accuracy -- switching to the tanh GELU would be.
static inline float u15_erf_(float x)
{
  const float s = sign(x);
  const float a = fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * a);
  const float p = t * (0.254829592f +
                  t * (-0.284496736f +
                  t * (1.421413741f +
                  t * (-1.453152027f +
                  t * 1.061405429f))));
  return s * (1.0f - p * exp(-a * a));
}

kernel void u15_gelu_erf(
    device VPIPE_ELT* x [[buffer(0)]],
    constant int&     n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  if ((int)gid >= n) { return; }
  const float v = (float)x[gid];
  x[gid] = (VPIPE_ELT)(0.5f * v *
                       (1.0f + u15_erf_(v * 0.70710678118654752440f)));
}

// ---------------------------------------------------------------------
// PixelShuffle(r), channel-SLOWEST.
//
// Input channel c*r^2 + i*r + j lands at output channel c, offset (i,j).
// The pixel head applies this three times (2, 2, then 8), and it is the
// only place the head's geometry can go wrong silently: a channel-
// FASTEST reading produces an image of the right size made of the wrong
// pixels.
//
//   in  [C_in][H][W]
//   out [C_in/(r*r)][H*r][W*r]
// grid (W, H, C_in).
kernel void u15_pixel_shuffle(
    const device VPIPE_ELT* in   [[buffer(0)]],
    device VPIPE_ELT*       out  [[buffer(1)]],
    constant int&           C_in [[buffer(2)]],
    constant int&           Hin  [[buffer(3)]],
    constant int&           Win  [[buffer(4)]],
    constant int&           r    [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int x = (int)gid.x;
  const int y = (int)gid.y;
  const int ci = (int)gid.z;
  if (x >= Win || y >= Hin || ci >= C_in) { return; }

  const int rr = r * r;
  const int c  = ci / rr;
  const int rem = ci % rr;
  const int i = rem / r;
  const int j = rem % r;

  const int OW = Win * r;
  out[((uint)c * (Hin * r) + (y * r + i)) * OW + (x * r + j)] =
      in[((uint)ci * Hin + y) * Win + x];
}

// PixelShuffle(r) on a CHANNEL-LAST map -- the layout the pixel head
// actually runs in.
//
// The backbone's output is [tokens][hidden] with tokens in row-major
// (h, w) order, which IS [th][tw][4096] channel-last already; and
// libvpipe's im2col_hwc_3x3 wants channel-last too. So the whole head
// stays HWC and never transposes. The semantics are identical to the
// channel-first kernel above -- input channel c*r^2 + i*r + j lands at
// output channel c, offset (i, j) -- and the test pins that by
// comparing the two through a transpose.
//
//   in  [H][W][C_in]
//   out [H*r][W*r][C_in/(r*r)]
// grid (C_in, W, H).
kernel void u15_pixel_shuffle_hwc(
    const device VPIPE_ELT* in   [[buffer(0)]],
    device VPIPE_ELT*       out  [[buffer(1)]],
    constant int&           C_in [[buffer(2)]],
    constant int&           Hin  [[buffer(3)]],
    constant int&           Win  [[buffer(4)]],
    constant int&           r    [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int ci = (int)gid.x;
  const int x  = (int)gid.y;
  const int y  = (int)gid.z;
  if (ci >= C_in || x >= Win || y >= Hin) { return; }

  const int rr = r * r;
  const int c  = ci / rr;
  const int rem = ci % rr;
  const int i = rem / r;
  const int j = rem % r;
  const int C_out = C_in / rr;
  const int OW = Win * r;

  out[((uint)(y * r + i) * OW + (x * r + j)) * C_out + c] =
      in[((uint)y * Win + x) * C_in + ci];
}

// Gather each m x m tile of a channel-last map into one row, ordered
// (channel, i, j) -- CHANNEL-SLOWEST.
//
// This is the patch embedder's `dense_embedding`, a Conv2d(C -> llm,
// k=m, s=m). Kernel equals stride, so it is a per-tile LINEAR, and its
// checkpoint weight [llm][C][m][m] is contiguous as [llm][C*m*m] with
// exactly this ordering. Gathering channel-FASTEST instead would feed
// the same GEMM a correctly-shaped row with its elements permuted.
//
//   in  [gh][gw][C]
//   out [gh/m][gw/m][C*m*m]
// grid (C, gw/m, gh/m).
kernel void u15_merge_tiles(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&           gh  [[buffer(2)]],
    constant int&           gw  [[buffer(3)]],
    constant int&           C   [[buffer(4)]],
    constant int&           m   [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  const int ox = (int)gid.y;
  const int oy = (int)gid.z;
  const int ow = gw / m, oh = gh / m;
  if (c >= C || ox >= ow || oy >= oh) { return; }

  const int tile = C * m * m;
  device VPIPE_ELT* o = out + ((uint)oy * ow + ox) * tile;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      o[((uint)c * m + i) * m + j] =
          in[((uint)(oy * m + i) * gw + (ox * m + j)) * C + c];
    }
  }
}

// The final pixel conversion: channel-last float in the model's [-1, 1]
// space -> PLANAR u8 RGB, which is what a TensorBeat carrying
// `rgb-frames` holds and what save-image consumes.
//
// The reference's denormalisation is (x * std + mean).clamp(0,1) with
// mean = std = 0.5, i.e. the model works in [-1, 1]. Skipping it gives
// a washed-out or clipped image rather than an error.
//
//   in  [H][W][3]  float-ish, model space
//   out [3][H][W]  u8
// grid (W, H, 3).
kernel void u15_to_u8_planar(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device uchar*           out [[buffer(1)]],
    constant int&           H   [[buffer(2)]],
    constant int&           W   [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int x = (int)gid.x;
  const int y = (int)gid.y;
  const int c = (int)gid.z;
  if (x >= W || y >= H || c >= 3) { return; }
  const float v = (float)in[((uint)y * W + x) * 3 + c];
  const float d = clamp(v * 0.5f + 0.5f, 0.0f, 1.0f);
  out[((uint)c * H + y) * W + x] = (uchar)(d * 255.0f + 0.5f);
}

// ---------------------------------------------------------------------
// Small utilities. libvpipe has residual_add_f16, but it works on the
// f16/bf16 twin only and these two also serve the f32 twin where the
// scalar embedders run.
// ---------------------------------------------------------------------

//   dst += src   (elementwise, same length)
kernel void u15_add(
    device VPIPE_ELT*       dst [[buffer(0)]],
    const device VPIPE_ELT* src [[buffer(1)]],
    constant int&           n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if ((int)gid >= n) { return; }
  dst[gid] = (VPIPE_ELT)((float)dst[gid] + (float)src[gid]);
}

// Add one ROW to every row of a [rows][dim] matrix. The timestep and
// noise-scale embeddings are one row broadcast over every image token.
kernel void u15_add_row(
    device VPIPE_ELT*       dst  [[buffer(0)]],
    const device VPIPE_ELT* row  [[buffer(1)]],
    constant int&           rows [[buffer(2)]],
    constant int&           dim  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  const int r = (int)gid.y;
  if (c >= dim || r >= rows) { return; }
  const uint i = (uint)r * dim + c;
  dst[i] = (VPIPE_ELT)((float)dst[i] + (float)row[c]);
}

// SiLU in place, for the two scalar embedders' MLPs.
kernel void u15_silu(
    device VPIPE_ELT* x [[buffer(0)]],
    constant int&     n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  if ((int)gid >= n) { return; }
  const float v = (float)x[gid];
  x[gid] = (VPIPE_ELT)(v / (1.0f + exp(-v)));
}

// Transpose [T][H*D] -> a KV CACHE slice [H][cap][D] at `offset`.
//
// The generation path attends over [prefix || current] with ONE
// contiguous K/V per layer, exactly as the reference's flash_k_cache
// does: the prefix is written once at prefill and the current block is
// overwritten in place every step. So the destination stride is the
// cache CAPACITY, not the number of tokens being written -- which is
// why the plain transpose above cannot serve.
//
// Getting `cap` wrong writes a correctly-shaped cache whose rows belong
// to the wrong head, and attention then reads a blend of heads.
kernel void u15_transpose_into_cache(
    const device VPIPE_ELT* in     [[buffer(0)]],
    device VPIPE_ELT*       cache  [[buffer(1)]],
    constant int&           T      [[buffer(2)]],
    constant int&           H      [[buffer(3)]],
    constant int&           D      [[buffer(4)]],
    constant int&           cap    [[buffer(5)]],
    constant int&           offset [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  if (d >= D || t >= T || h >= H) { return; }
  cache[((uint)h * cap + (offset + t)) * D + d] =
      in[((uint)t * H + h) * D + d];
}

// Transpose [H][T][D] -> [T][H*D]: the INVERSE of the above.
//
// libvpipe's SDPA kernels write their output HEAD-major -- `out[(h*n_q +
// qi)*D + idx]` -- while o_proj is a GEMM over token rows. Feeding the
// attention output straight to o_proj therefore reads a transposed
// matrix of exactly the right size, which is why it produced finite,
// plausible, completely wrong numbers rather than an error.
kernel void u15_transpose_htd(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&           T   [[buffer(2)]],
    constant int&           H   [[buffer(3)]],
    constant int&           D   [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  if (d >= D || t >= T || h >= H) { return; }
  out[((uint)t * H + h) * D + d] = in[((uint)h * T + t) * D + d];
}

// Transpose [T][H*D] -> [H][T][D]. libvpipe's transpose_abd_f16 does
// this, and is used where it applies; this exists for the f32 twin and
// for the k/v paths whose head count differs from q's.
kernel void u15_transpose_thd(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&           T   [[buffer(2)]],
    constant int&           H   [[buffer(3)]],
    constant int&           D   [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  if (d >= D || t >= T || h >= H) { return; }
  out[((uint)h * T + t) * D + d] = in[((uint)t * H + h) * D + d];
}
