#include "u15-metal-ops.h"

#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/mma-tile.h"

#include "u15-config.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using vpipe::metal_compute::ComputeEncoder;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace u15 {

namespace {

// C++ mirror of mlx::steel::AttnParams, which lives in the vendored
// steel headers and is not on the plugin's include path. Field order and
// types are the contract; nothing here may be reordered.
struct SteelAttnParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
  std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

void
dispatch_1d_(ComputeEncoder& enc, std::size_t n)
{
  const unsigned g = (unsigned)((n + 255) / 256 * 256);
  enc.dispatch({g, 1, 1}, {256, 1, 1});
}

}  // namespace

bool
MetalOps::init(MetalCompute* mc, std::string* err)
{
  const auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (mc == nullptr || !mc->valid()) {
    return fail("no usable Metal device");
  }
  _mc = mc;

  // libvpipe's own libraries, by name. The entry points keep an `_f16`
  // suffix in BOTH dtype twins -- the LIBRARY name selects bf16 -- which
  // reads wrong and is correct.
  _lib_gemm = mc->load_library("dense_gemm_bf16");
  _lib_elt  = mc->load_library("llm_elementwise_bf16");
  _lib_sdpa = mc->load_library("sdpa_bf16");
  _lib_rms  = mc->load_library("rms_norm_bf16");
  _lib_u15  = mc->load_library(kMetalLibBf16);
  // OPTIONAL, and deliberately not in the need[] table below: a host
  // that does not ship it still runs correctly on the scalar path, and
  // failing init() here would turn a missing OPTIONAL kernel into "the
  // plugin does not load".
  _lib_attn = mc->load_library("attn_steel");
  _steel_ok = _lib_attn.valid();

  // ---- M5 matrix cores -------------------------------------------------
  //
  // Gated on supports_matrix_cores() rather than on a device-name test:
  // what decides is whether matmul2d is present, and the host already
  // answers that. Both libraries are OPTIONAL for the same reason the
  // steel one is -- a host that does not ship them still runs, on the
  // simdgroup tiles this had before.
  //
  // Kill switches, because a rate change nobody can turn off cannot be
  // measured. Naming them after the in-tree ones (VPIPE_*_NO_MMA2 /
  // _NO_NAX_ATTN) so an A/B across host and plugin reads the same.
  const bool mcore = mc->supports_matrix_cores();
  if (mcore && std::getenv("VPIPE_U15_NO_NAX_ATTN") == nullptr) {
    _lib_attn_nax = mc->load_library("attn_steel_nax");
    // VALIDATED, not merely present. An unvalidated ComputeFunction is a
    // silent no-op, and steel_attn_plan() commits to the NAX tiling on
    // the strength of this flag before it ever looks a function up --
    // so "the library loaded" is not a strong enough thing to know.
    // The probe constants are the all-aligned case; the specialisation
    // a real plan asks for differs only in 200/201.
    if (_lib_attn_nax.valid()) {
      vpipe::metal_compute::FunctionConstants probe;
      probe.set_bool(200, true).set_bool(201, true)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      _use_nax = _lib_attn_nax
                     .function("attn_steel_nax_h_bd128_bf16", probe)
                     .valid();
    }
  }
  if (mcore && std::getenv("VPIPE_U15_NO_MMA2") == nullptr) {
    _lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
    if (_lib_dense_mma.valid()) {
      _fn_dense_mma = _lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
      _fn_dense_mma_wide =
          _lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
      // BOTH or neither: the tile rule picks between them per shape, and
      // a half-bound pair would silently send every deep-K GEMM back to
      // steel while reporting matmul2d as on.
      _use_mma2 = _fn_dense_mma.valid() && _fn_dense_mma_wide.valid();
    }
  }
  if (const char* e = std::getenv("VPIPE_U15_MMA_Q_MIN_M")) {
    const int v = std::atoi(e);
    if (v > 0) { _mma_q_min_m = v; }
  }
  if (const char* e = std::getenv("VPIPE_U15_MMA_MIN_M")) {
    const int v = std::atoi(e);
    if (v > 0) { _mma_min_m = v; }
  }

  // The affine qmm kernels, for a QUANTIZED checkpoint. OPTIONAL and
  // deliberately not in the need[] table below: a host that does not
  // ship them still runs a dense checkpoint perfectly well, and failing
  // init() here would turn a missing optional kernel into "the plugin
  // does not load". The loader checks quant_available() when it actually
  // meets a quantized weight, so the error names the cause where it
  // matters.
  _lib_qmm = mc->load_library("affine_qmm_steel_bf16");
  if (_lib_qmm.valid()) {
    _quant_ok = true;
    const int bits[2] = {4, 8};
    const int grp[2] = {32, 64};
    for (int b = 0; b < 2; ++b) {
      for (int g = 0; g < 2; ++g) {
        const std::string base = "affine_qmm_steel_w" +
                                 std::to_string(bits[b]) + "g" +
                                 std::to_string(grp[g]);
        _fn_qmm[b][g][0] = _lib_qmm.function(base);
        // The WIDE tile exists only at group 64 -- there is no
        // affine_qmm_steel_w4g32_bm64 -- so it is optional and falls
        // back to the narrow one. REQUIRING it turns quantization off
        // wholesale, and quietly.
        _fn_qmm[b][g][1] = _lib_qmm.function(base + "_bm64");
        if (!_fn_qmm[b][g][0].valid()) { _quant_ok = false; }
      }
    }
  }
  // The dequant-once companion: expanding a quantized weight into bf16
  // and running the SAME matmul2d. Separate flag from _use_mma2 because
  // it needs the dequant kernels too, and a host that ships matmul2d
  // without them should still get the dense win.
  if (_use_mma2 && _quant_ok) {
    _lib_dequant = mc->load_library("affine_dequant_bf16");
    if (_lib_dequant.valid()) {
      const int bits[2] = {4, 8};
      const int grp[2]  = {32, 64};
      bool all = true;
      for (int b = 0; b < 2; ++b) {
        for (int g = 0; g < 2; ++g) {
          _fn_dequant[b][g] = _lib_dequant.function(
              "affine_dequant_w" + std::to_string(bits[b]) + "g" +
              std::to_string(grp[g]));
          if (!_fn_dequant[b][g].valid()) { all = false; }
        }
      }
      _use_mma2_q = all;
    }
  }
  if (!_lib_gemm.valid()) { return fail("no dense_gemm_bf16 library"); }
  if (!_lib_elt.valid())  { return fail("no llm_elementwise_bf16 library"); }
  if (!_lib_sdpa.valid()) { return fail("no sdpa_bf16 library"); }
  if (!_lib_rms.valid())  { return fail("no rms_norm_bf16 library"); }
  if (!_lib_u15.valid()) {
    return fail(std::string("no '") + kMetalLibBf16 +
                "' library -- the plugin's own kernels were not registered");
  }

  _fn_gemm   = _lib_gemm.function("dense_gemm_t_bm64_f16");
  _fn_gemm_wide = _lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  _fn_ff_swiglu = _lib_gemm.function("dense_gemm_swiglu_bm64_f16");
  // rms_norm_FAST: a simd_sum reduction instead of a threadgroup
  // tree. Same buffer contract, so it is a drop-in.
  _fn_rms    = _lib_rms.function("rms_norm_fast_f16");
  _fn_rms_slow = _lib_rms.function("rms_norm_f16");
  _fn_swiglu = _lib_elt.function("swiglu_f16");
  _fn_im2col = _lib_elt.function("im2col_hwc_3x3_f16");
  _fn_sdpa   = _lib_sdpa.function("sdpa_full_f16");
  _fn_sdpa_causal = _lib_sdpa.function("sdpa_causal_f16");
  _fn_sdpa_block = _lib_u15.function("u15_sdpa_block_causal");

  _fn_split_norm = _lib_u15.function("u15_split_qk_norm");
  _fn_split_rope = _lib_u15.function("u15_split_rope");
  _fn_vis_rope   = _lib_u15.function("u15_vision_rope2d");
  _fn_gelu       = _lib_u15.function("u15_gelu_erf");
  _fn_silu       = _lib_u15.function("u15_silu");
  _fn_add        = _lib_u15.function("u15_add");
  _fn_add_row    = _lib_u15.function("u15_add_row");
  _fn_ps         = _lib_u15.function("u15_pixel_shuffle");
  _fn_ps_hwc     = _lib_u15.function("u15_pixel_shuffle_hwc");
  _fn_merge      = _lib_u15.function("u15_merge_tiles");
  _fn_u8         = _lib_u15.function("u15_to_u8_planar");
  _fn_bias_add   = _lib_elt.function("bias_add_rows_f16");
  _fn_transpose  = _lib_u15.function("u15_transpose_thd");
  _fn_untranspose = _lib_u15.function("u15_transpose_htd");
  _fn_cache      = _lib_u15.function("u15_transpose_into_cache");

  // EVERY function is validated. An unvalidated ComputeFunction
  // dispatches as a silent no-op, so skipping this turns a missing
  // kernel into a numerically wrong model -- which is far harder to
  // diagnose than a refusal at load.
  struct Need { const vpipe::metal_compute::ComputeFunction* fn;
                const char* name; };
  const Need need[] = {
      {&_fn_gemm, "dense_gemm_t_bm64_f16"},
      {&_fn_gemm_wide, "dense_gemm_t_bm64bn64_f16"},
      {&_fn_ff_swiglu, "dense_gemm_swiglu_bm64_f16"},
      {&_fn_rms, "rms_norm_fast_f16"},
      {&_fn_rms_slow, "rms_norm_f16"},
      {&_fn_swiglu, "swiglu_f16"},
      {&_fn_im2col, "im2col_hwc_3x3_f16"},
      {&_fn_sdpa, "sdpa_full_f16"},
      {&_fn_sdpa_causal, "sdpa_causal_f16"},
      {&_fn_sdpa_block, "u15_sdpa_block_causal"},
      {&_fn_split_norm, "u15_split_qk_norm"},
      {&_fn_split_rope, "u15_split_rope"},
      {&_fn_vis_rope, "u15_vision_rope2d"},
      {&_fn_gelu, "u15_gelu_erf"},
      {&_fn_silu, "u15_silu"},
      {&_fn_add, "u15_add"},
      {&_fn_add_row, "u15_add_row"},
      {&_fn_ps, "u15_pixel_shuffle"},
      {&_fn_ps_hwc, "u15_pixel_shuffle_hwc"},
      {&_fn_merge, "u15_merge_tiles"},
      {&_fn_u8, "u15_to_u8_planar"},
      {&_fn_bias_add, "bias_add_rows_f16"},
      {&_fn_transpose, "u15_transpose_thd"},
      {&_fn_untranspose, "u15_transpose_htd"},
      {&_fn_cache, "u15_transpose_into_cache"},
  };
  for (const Need& n : need) {
    if (!n.fn->valid()) {
      return fail(std::string("kernel not found or failed to validate: ") +
                  n.name);
    }
  }
  return true;
}

// matmul2d, when the box has matrix cores and the shape is worth the
// 128-row tile.
//
// TILE BY K, not by N. The host's shared rule (mma_use_wide_tile) is
// K-only and says so with a measurement: an N term looked right on a
// non-interleaved probe and cost 7% end-to-end once the arms were
// interleaved. This model lands either side of it -- K=4096 for qkv/o
// and the gate|up pair, K=12288 for ff-down -- both tiles are live.
//
// NO BIAS SLOT. matmul2d writes the product and nothing else, exactly
// as the quantized path above finds, so a bias is a second pass through
// the same bias_add() that path already uses.
MetalOps::MetalOps()  = default;
MetalOps::~MetalOps() = default;

bool
MetalOps::enable_i8_gemm(bool want)
{
  _i8.reset();
  if (_mc == nullptr) { return false; }
  // bf16, matching this model's element type end to end -- it loads the
  // `_bf16` twins of the dense-GEMM and dequant kernels, and `_w_deq` is
  // sized two bytes per element. Handing bf16 buffers to the f16 kernels
  // would reinterpret the bits rather than fail.
  auto ctx = std::make_unique<vpipe::genai::I8GemmContext>(_mc, want,
                                                           /*bf16=*/true);
  if (ctx->enabled()) { _i8 = std::move(ctx); }
  return _i8 != nullptr;
}

void
MetalOps::release_i8_scratch()
{
  if (_i8) { _i8->release_scratch(); }
}

bool
MetalOps::linear_mma_(ComputeEncoder& enc, const SharedBuffer& x,
                      const SharedBuffer& w, const SharedBuffer* bias,
                      const SharedBuffer& y, int M, int K, int N) const
{
  // N < 16 is a projector, not a GEMM: the tile is nearly all padding
  // and steel wins outright.
  if (!_use_mma2 || M < _mma_min_m || N < 16) { return false; }
  // ACCELERATED MODE, before the bf16 tiles. Placed HERE rather than in
  // each caller because this function is the single point both GEMM
  // paths reach: a dense weight arrives directly, and a quantized one
  // arrives as the `_w_deq` expansion linear_mma_q_ just wrote -- which
  // is the "dense, or a dequant scratch" input the mode is defined on.
  //
  // Declines silently on a shape it cannot help (M under the ~1k
  // crossover, K not a whole number of 512-groups), and the tiles below
  // then run exactly as before. `enc` is untouched on a decline.
  if (_i8 && _i8->gemm(enc, x, 0, w, y, 0, M, N, K)) {
    // The int8 kernel has no bias slot, so the same add the tiles use
    // runs after it -- and it must, or a biased projection silently
    // loses its bias only in accelerated mode.
    if (bias != nullptr) { bias_add(enc, y, *bias, M, N); }
    return true;
  }
  const bool wide = vpipe::genai::mma_use_wide_tile(N, K);
  const int RN = wide ? 256 : 128;   // N-region per threadgroup
  enc.set_function(wide ? _fn_dense_mma_wide : _fn_dense_mma);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  // Slot 2 is the bias the kernel does not use. Bound to `w` because an
  // unbound argument is undefined behaviour rather than a null the
  // kernel can test -- the same reason the steel path binds it.
  enc.set_buffer(2, w);
  enc.set_buffer(3, y);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  if (bias != nullptr) { bias_add(enc, y, *bias, M, N); }
  return true;
}

void
MetalOps::linear(ComputeEncoder& enc, const SharedBuffer& x,
                 const SharedBuffer& w, const SharedBuffer* bias,
                 const SharedBuffer& y, int M, int K, int N) const
{
  if (linear_mma_(enc, x, w, bias, y, M, K, N)) { return; }
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  // The bias slot must be bound even when unused: an unbound buffer
  // argument is undefined behaviour, not a null the kernel can test.
  enc.set_buffer(2, bias != nullptr ? *bias : w);
  enc.set_buffer(3, y);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias != nullptr ? 1 : 0);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

// DEQUANT ONCE, then the dense matmul2d -- the same trade the in-tree
// DiTs take on M5.
//
// The steel qmm unpacks its codes inside the tile loop, so every
// threadgroup that touches a column pays for that column's unpacking
// again. matmul2d cannot unpack at all, so the weight is expanded into
// bf16 first and the matmul reads it dense. That swaps repeated
// in-register unpacking for one pass over N*K -- which is why this has
// a REAL row floor where the dense path has none: the expansion costs
// the same whether M is 1 or 1024, and only the matmul it feeds scales
// with M.
bool
MetalOps::linear_mma_q_(ComputeEncoder& enc, const SharedBuffer& x,
                        const QWeight& w, const SharedBuffer* bias,
                        const SharedBuffer& y, int M, int K, int N) const
{
  if (!_use_mma2_q || !w.quantized || M < _mma_q_min_m || N < 16) {
    return false;
  }
  if (K % w.group != 0) { return false; }   // the kernels assume it divides
  const int bi = w.bits == 8 ? 1 : 0;
  const int gi = w.group == 64 ? 1 : 0;
  const vpipe::metal_compute::ComputeFunction& dq =
      _fn_dequant[bi][gi];
  if (!dq.valid()) { return false; }
  const std::size_t need = (std::size_t)N * K * 2;
  if (_w_deq.empty() || _w_deq.byte_size() < need) {
    _w_deq = _mc->make_shared_buffer(need);
    // A refusal here is not a failure: the steel path is still correct
    // and still there. Falling back is the whole point of returning a
    // bool rather than reporting an error.
    if (_w_deq.empty()) { return false; }
  }
  // One thread per packed u32 word: w8 packs 4 codes per word, w4 packs
  // 8, so the K extent is K/4 or K/8 words.
  enc.set_function(dq);
  enc.set_buffer(0, w.codes);
  enc.set_buffer(1, w.scales);
  enc.set_buffer(2, w.qbias);
  enc.set_buffer(3, _w_deq);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  const unsigned words = (unsigned)(w.bits == 8 ? (K / 4) : (K / 8));
  enc.dispatch({words, (unsigned)N, 1}, {64, 1, 1});
  // From here it is the dense path, and deliberately the SAME code: a
  // second copy of the tile rule is a second thing to keep in step.
  const bool ok = linear_mma_(enc, x, _w_deq, bias, y, M, K, N);
  // linear_mma_ can only decline on the shape, which was already
  // checked above -- but if it ever does, the expansion is wasted work
  // and the caller must still get an answer, so say so rather than
  // returning true on a dispatch that never happened.
  return ok;
}

void
MetalOps::linear(ComputeEncoder& enc, const SharedBuffer& x,
                 const QWeight& w, const SharedBuffer* bias,
                 const SharedBuffer& y, int M, int K, int N) const
{
  if (!w.quantized) {
    linear(enc, x, w.w, bias, y, M, K, N);
    return;
  }
  if (linear_mma_q_(enc, x, w, bias, y, M, K, N)) { return; }
  const int bi = w.bits == 8 ? 1 : 0;
  const int gi = w.group == 64 ? 1 : 0;
  // The wide tile only pays once there are enough rows to fill it, and
  // it does not exist at every (bits, group).
  const bool bm64 = M >= 64 && _fn_qmm[bi][gi][1].valid();
  enc.set_function(_fn_qmm[bi][gi][bm64 ? 1 : 0]);
  enc.set_buffer(0, w.codes);
  enc.set_buffer(1, w.scales);
  enc.set_buffer(2, w.qbias);
  enc.set_buffer(3, x);
  enc.set_buffer(4, y);
  enc.set_constant(5, K);
  enc.set_constant(6, N);
  enc.set_constant(7, M);
  const int bm = bm64 ? 64 : 32;
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});

  // buffer(2) above is the quantization ZERO-POINT, not this linear's
  // bias -- the kernel has no slot for one -- so a bias is a second pass.
  if (bias != nullptr) { bias_add(enc, y, *bias, M, N); }
}

void
MetalOps::bias_add(ComputeEncoder& enc, const SharedBuffer& y,
                   const SharedBuffer& bias, int M, int N) const
{
  const std::size_t total = (std::size_t)M * N;
  enc.set_function(_fn_bias_add);
  enc.set_buffer(0, y);
  enc.set_buffer(1, bias);
  enc.set_constant(2, N);
  enc.set_constant(3, (int)total);
  dispatch_1d_(enc, total);
}

void
MetalOps::linear_wide(ComputeEncoder& enc, const SharedBuffer& x,
                      const SharedBuffer& w, const SharedBuffer* bias,
                      const SharedBuffer& y, int M, int K, int N) const
{
  // Same routing as linear(): the caller's narrow/wide choice is about
  // the STEEL tiles, and matmul2d picks its own by K.
  if (linear_mma_(enc, x, w, bias, y, M, K, N)) { return; }
  enc.set_function(_fn_gemm_wide);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, bias != nullptr ? *bias : w);
  enc.set_buffer(3, y);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias != nullptr ? 1 : 0);
  // BN = 64 here, so the N grid steps by 64 rather than 32.
  enc.dispatch({(unsigned)(((N + 63) / 64) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

void
MetalOps::swiglu_fused(ComputeEncoder& enc, const SharedBuffer& x,
                       const SharedBuffer& w_fused, const SharedBuffer& y,
                       int M, int K, int inter) const
{
  enc.set_function(_fn_ff_swiglu);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w_fused);
  enc.set_buffer(2, y);
  enc.set_constant(3, K);
  // N is the FUSED width (2*inter). The kernel emits N/2 columns; a
  // caller that passes the output width instead dispatches half the
  // tiles and computes half the model, at a rate that looks like a
  // speed-up.
  const int nf = 2 * inter;
  enc.set_constant(4, nf);
  enc.set_constant(5, M);
  enc.set_constant(6, 0);          // out_stride 0 = packed (N/2)
  enc.set_constant(7, 0);          // out_off
  enc.dispatch({(unsigned)(((nf + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

void
MetalOps::rms_norm(ComputeEncoder& enc, const SharedBuffer& x,
                   const SharedBuffer& w, const SharedBuffer& y, int rows,
                   int dim, float eps, bool fast) const
{
  enc.set_function(fast ? _fn_rms : _fn_rms_slow);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, y);
  enc.set_constant(3, dim);
  enc.set_constant(4, eps);
  enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
}

void
MetalOps::split_qk_norm(ComputeEncoder& enc, const SharedBuffer& x,
                        const SharedBuffer& w_t, const SharedBuffer& w_hw,
                        int heads, int tokens, int head_dim,
                        float eps) const
{
  enc.set_function(_fn_split_norm);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w_t);
  enc.set_buffer(2, w_hw);
  enc.set_constant(3, heads);
  enc.set_constant(4, tokens);
  enc.set_constant(5, head_dim);
  enc.set_constant(6, eps);
  // One simdgroup per (token, head); 8 simdgroups per threadgroup.
  const unsigned rows = (unsigned)tokens * (unsigned)heads;
  const unsigned groups = (rows + 7) / 8;
  enc.dispatch({groups * 256, 1, 1}, {256, 1, 1});
}

void
MetalOps::split_rope(ComputeEncoder& enc, const SharedBuffer& x,
                     const SharedBuffer& cos, const SharedBuffer& sin,
                     int heads, int tokens, int head_dim, int d_t,
                     int d_h) const
{
  enc.set_function(_fn_split_rope);
  enc.set_buffer(0, x);
  enc.set_buffer(1, cos);
  enc.set_buffer(2, sin);
  enc.set_constant(3, heads);
  enc.set_constant(4, tokens);
  enc.set_constant(5, head_dim);
  enc.set_constant(6, d_t);
  enc.set_constant(7, d_h);
  const std::size_t n =
      (std::size_t)(head_dim / 2) * tokens * heads;
  dispatch_1d_(enc, n);
}

MetalOps::RopeTables
MetalOps::build_rope_tables(const std::vector<int>& pos_t,
                            const std::vector<int>& pos_h,
                            const std::vector<int>& pos_w, int head_dim,
                            int d_t, int d_h, double theta_t,
                            double theta_hw) const
{
  RopeTables out;
  const std::size_t T = pos_t.size();
  const std::size_t n = T * (std::size_t)head_dim;
  out.cos = _mc->make_shared_buffer(n * sizeof(float));
  out.sin = _mc->make_shared_buffer(n * sizeof(float));
  if (out.cos.empty() || out.sin.empty()) { return RopeTables{}; }

  auto* c = static_cast<float*>(out.cos.contents());
  auto* s = static_cast<float*>(out.sin.contents());

  // Each of the three ranges is filled with its OWN ladder, doubled in
  // the cat(freqs, freqs) way rotate_half expects, so the kernel reads
  // cos[t][off + i] for the pair (off+i, off+i+half) and needs no
  // arithmetic of its own.
  //
  // The angle is formed in double and narrowed once. The t ladder's
  // theta is 5e6, so its lowest frequency is ~1 and its highest is
  // ~theta^-1; forming these in f32 from the start costs real phase at
  // long positions.
  struct Range { int off, width; double theta; const std::vector<int>* p; };
  const Range ranges[3] = {
      {0, d_t, theta_t, &pos_t},
      {d_t, d_h, theta_hw, &pos_h},
      {d_t + d_h, d_h, theta_hw, &pos_w},
  };

  for (const Range& r : ranges) {
    const int half = r.width / 2;
    std::vector<double> inv((std::size_t)half);
    for (int i = 0; i < half; ++i) {
      inv[(std::size_t)i] =
          1.0 / std::pow(r.theta, (double)(2 * i) / (double)r.width);
    }
    for (std::size_t t = 0; t < T; ++t) {
      const double pos = (double)(*r.p)[t];
      for (int i = 0; i < half; ++i) {
        const double a = pos * inv[(std::size_t)i];
        const float cv = (float)std::cos(a);
        const float sv = (float)std::sin(a);
        const std::size_t base = t * (std::size_t)head_dim + r.off;
        c[base + i] = cv;
        s[base + i] = sv;
        // The doubled half. The kernel only reads the first half of each
        // range, but filling both keeps the table meaning what its shape
        // says and costs nothing.
        c[base + i + half] = cv;
        s[base + i + half] = sv;
      }
    }
  }
  return out;
}

SharedBuffer
MetalOps::make_inv_freq(int dim, double theta) const
{
  const int half = dim / 2;
  SharedBuffer b = _mc->make_shared_buffer((std::size_t)half * sizeof(float));
  if (b.empty()) { return b; }
  auto* f = static_cast<float*>(b.contents());
  for (int i = 0; i < half; ++i) {
    f[i] = (float)(1.0 / std::pow(theta, (double)(2 * i) / (double)dim));
  }
  return b;
}

void
MetalOps::vision_rope2d(ComputeEncoder& enc, const SharedBuffer& x,
                        const SharedBuffer& inv_freq, int n, int chan,
                        int grid_w) const
{
  enc.set_function(_fn_vis_rope);
  enc.set_buffer(0, x);
  enc.set_buffer(1, inv_freq);
  enc.set_constant(2, n);
  enc.set_constant(3, chan);
  enc.set_constant(4, grid_w);
  enc.dispatch({(unsigned)(chan / 4), (unsigned)n, 1}, {32, 1, 1});
}

void
MetalOps::swiglu(ComputeEncoder& enc, const SharedBuffer& gate,
                 const SharedBuffer& up, const SharedBuffer& y, int rows,
                 int inter) const
{
  const int n = rows * inter;
  enc.set_function(_fn_swiglu);
  enc.set_buffer(0, gate);
  enc.set_buffer(1, up);
  enc.set_buffer(2, y);
  enc.set_constant(3, n);
  dispatch_1d_(enc, (std::size_t)n);
}

void
MetalOps::sdpa(ComputeEncoder& enc, const SharedBuffer& q,
               const SharedBuffer& k, const SharedBuffer& v,
               const SharedBuffer& out, int hq, int hkv, int tq, int tk,
               int d, int cap, bool causal, int q_offset) const
{
  const float scale = 1.0f / std::sqrt((float)d);
  enc.set_function(causal ? _fn_sdpa_causal : _fn_sdpa);
  enc.set_buffer(0, q);
  enc.set_buffer(1, k);
  enc.set_buffer(2, v);
  enc.set_buffer(3, out);
  enc.set_constant(4, scale);
  enc.set_constant(5, tk);
  enc.set_constant(6, d);
  enc.set_constant(7, hq);
  enc.set_constant(8, hkv);
  enc.set_constant(9, tq);
  // The K/V stride is the cache CAPACITY in TOKENS -- the kernel
  // multiplies by D itself (`k + kv * kv_stride * D`). Passing elements
  // here indexes past the end of head 0 and reads another head's keys.
  const int kv_stride = cap;
  if (causal) {
    enc.set_constant(10, q_offset);
    enc.set_constant(11, kv_stride);
  } else {
    enc.set_constant(10, kv_stride);
  }
  enc.dispatch({32, (unsigned)hq, (unsigned)tq}, {32, 1, 1});
}

bool
MetalOps::steel_attn_available(int head_dim) const noexcept
{
  // Only 64 and 128 are instantiated in attn_steel.metal. This model's
  // head_dim is 128.
  return _steel_ok && (head_dim == 64 || head_dim == 128);
}

bool
MetalOps::steel_attn_plan(SteelAttn* p, int heads_q, int heads_kv, int tq,
                          int tkv, int head_dim, int kv_stride_tokens) const
{
  if (p == nullptr || !steel_attn_available(head_dim)) { return false; }
  if (heads_q <= 0 || heads_kv <= 0 || tq <= 0 || tkv <= 0) { return false; }
  if (heads_q % heads_kv != 0) { return false; }

  // NAX tiles 64x32 where steel tiles 32x16. These two numbers drive the
  // params block below AND the launch grid in sdpa_steel(), so they have
  // to be picked here, once, rather than assumed anywhere downstream --
  // filling NQ/NK from the steel tiling and then launching the NAX
  // kernel is the shape of bug that reads as a wrong picture, not a
  // crash.
  const bool nax = _use_nax && _lib_attn_nax.valid();
  const int bq = nax ? 64 : 32, bk = nax ? 32 : 16;
  if (p->params.empty()) {
    p->params = _mc->make_shared_buffer(sizeof(SteelAttnParams));
    if (p->params.empty()) { return false; }
  }
  auto* s = static_cast<SteelAttnParams*>(p->params.contents());
  s->B = 1;
  s->H = heads_q;
  s->D = head_dim;
  s->qL = tq;
  s->kL = tkv;
  // The kernel computes kv_head_idx = query_head / gqa_factor, so this
  // is the GROUP SIZE (32/8 = 4 here), not the kv head count. Leaving it
  // at 1 -- which a port with no GQA can do -- would give query head h
  // the keys of kv head h, i.e. read past the end of an 8-head cache.
  s->gqa_factor = heads_q / heads_kv;
  s->scale = (float)(1.0 / std::sqrt((double)head_dim));
  s->NQ = (tq + bq - 1) / bq;
  s->NK = (tkv + bk - 1) / bk;
  s->NQ_aligned = tq / bq;
  s->NK_aligned = tkv / bk;
  s->qL_rem = tq - s->NQ_aligned * bq;
  s->kL_rem = tkv - s->NK_aligned * bk;
  s->qL_off = 0;

  // Q and O hold tq rows per head; K and V hold kv_stride_tokens, which
  // is the cache CAPACITY rather than the key length in use. Filling all
  // four from one length is the bug a port with a tightly-packed cache
  // never sees.
  s->Q_strides[0] = (std::int64_t)heads_q * tq * head_dim;
  s->Q_strides[1] = (std::int64_t)tq * head_dim;
  s->Q_strides[2] = head_dim;
  s->K_strides[0] =
      (std::int64_t)heads_kv * kv_stride_tokens * head_dim;
  s->K_strides[1] = (std::int64_t)kv_stride_tokens * head_dim;
  s->K_strides[2] = head_dim;
  for (int i = 0; i < 3; ++i) {
    s->V_strides[i] = s->K_strides[i];
    s->O_strides[i] = s->Q_strides[i];
  }

  // 200 says the last QUERY tile is full, 201 the last KEY tile -- so
  // 201 comes from the KEY length. A square-only caller can fill both
  // from one number and never notice.
  vpipe::metal_compute::FunctionConstants fc;
  fc.set_bool(200, (tq % bq) == 0)
      .set_bool(201, (tkv % bk) == 0)
      .set_bool(300, false)        // has_mask
      .set_bool(301, false)        // do_causal
      .set_bool(302, false);       // has_sinks
  // `nax` is settled at init, so there is no fallback to take here: a
  // NAX library that did not validate never set _use_nax, and the
  // tiling above was chosen from the same flag. Deciding it here
  // instead would mean either re-planning with the other tiling or
  // launching one kernel with the other's params block.
  p->fn = nax
      ? _lib_attn_nax.function(head_dim == 128 ? "attn_steel_nax_h_bd128_bf16"
                                               : "attn_steel_nax_h_bd64_bf16",
                               fc)
      : _lib_attn.function(head_dim == 128 ? "attn_steel_h_bd128_bf16"
                                           : "attn_steel_h_bd64_bf16", fc);
  if (!p->fn.valid()) { return false; }

  // AND THE INT8 TWIN, whenever this GPU could ever run it. See the note
  // on SteelAttn::fn_i8 for why the condition is the hardware and not
  // the setting.
  if (nax && vpipe::genai::MetalSageAttention::available(_mc)) {
    vpipe::metal_compute::FunctionConstants fi = fc;
    fi.set_bool(vpipe::genai::sage::kQkInt8Constant, true);
    p->fn_i8 = _lib_attn_nax.function(
        head_dim == 128 ? "attn_steel_nax_h_bd128_bf16"
                        : "attn_steel_nax_h_bd64_bf16", fi);
  } else {
    p->fn_i8 = vpipe::metal_compute::ComputeFunction{};
  }

  p->heads = heads_q;
  p->tq = tq;
  p->tkv = tkv;
  p->head_dim = head_dim;
  p->bq = bq;
  p->heads_kv = heads_kv;
  p->bk = bk;
  p->kv_stride = kv_stride_tokens;
  return true;
}

bool
MetalOps::set_sage(const vpipe::genai::sage::Config& cfg, std::string* err)
{
  _sage_cfg = cfg;
  if (const char* e = std::getenv("VPIPE_SAGE_ATTN")) {
    _sage_cfg.enabled = (*e != '0');
  }
  _sage.reset();
  if (!_sage_cfg.enabled || _mc == nullptr) { return true; }
  bool fatal = false;
  _sage = vpipe::genai::MetalSageAttention::load_for_model(
      _mc, /*bf16=*/true, _sage_cfg, "sensenova-u1.5", &fatal);
  if (!_sage) {
    // OFF rather than half on, so sage_takes() cannot say yes to a tier
    // with no driver behind it. The caller still holds what it asked
    // for, which is what a log line about the decline needs.
    _sage_cfg.enabled = false;
    if (fatal) {
      if (err != nullptr) {
        *err = "the int8 attention kernels would not build";
      }
      return false;
    }
  }
  return true;
}

bool
MetalOps::sage_takes(int head_dim) const noexcept
{
  return (bool)_sage && _sage_cfg.enabled && _use_nax &&
         _lib_attn_nax.valid() && steel_attn_available(head_dim);
}

void
MetalOps::sage_lend(const SharedBuffer& a, const SharedBuffer& b) const
{
  if (_sage) { _sage->set_arena(a, b); }
}

std::size_t
MetalOps::sage_resident_bytes() const noexcept
{
  return _sage ? _sage->resident_bytes() : 0;
}

void
MetalOps::sdpa_steel(ComputeEncoder& enc, const SteelAttn& p,
                     const SharedBuffer& q, const SharedBuffer& k,
                     const SharedBuffer& v, const SharedBuffer& out,
                     int sage_layer) const
{
  // THE INT8 PROLOGUE, into this encoder and immediately before the
  // dispatch that reads what it wrote. The encoder is serial, so that
  // ordering is the whole synchronisation between them.
  //
  // The tile check is not paranoia: the scales are one per the ATTENTION
  // kernel's own tiles, so a prologue sized against different numbers
  // than the kernel indexes with would read past its scale arrays. These
  // are the same numbers by construction -- Sage only runs on the NAX
  // plan, whose tiles are 64/32 -- and asking is what keeps that true if
  // either side ever moves.
  bool i8 = false;
  if (sage_layer >= 0 && p.fn_i8.valid() && (bool)_sage &&
      sage_layer >= _sage_cfg.dense_layers &&
      p.bq == vpipe::genai::MetalSageAttention::nax_query_block() &&
      p.bk == vpipe::genai::MetalSageAttention::nax_key_block()) {
    using Operand = vpipe::genai::MetalSageAttention::Operand;
    // THE SAME THREE NUMBERS THE PARAMS BLOCK CARRIES, and the K head
    // stride is the cache CAPACITY rather than the key length -- the
    // quantizer walks the bytes the kernel walks, so deriving that from
    // `tkv` would quantize the wrong heads and do it silently.
    const Operand qo{&q, 0, p.head_dim, p.tq * p.head_dim};
    const Operand ko{&k, 0, p.head_dim, p.kv_stride * p.head_dim};
    std::string serr;
    // K is quantized once per KEY head, not once per query head: the
    // kernel indexes it the same way it indexes the f16 K, through the
    // GQA group size.
    i8 = _sage->prepare(enc, qo, ko, p.heads, p.heads_kv, p.tq, p.tkv,
                        p.head_dim, p.bq, p.bk, _sage_cfg, &serr);
  }
  enc.set_function(i8 ? p.fn_i8 : p.fn);
  enc.set_buffer(0, q);
  enc.set_buffer(1, k);
  enc.set_buffer(2, v);
  enc.set_buffer(3, out);
  enc.set_buffer(4, p.params);
  // Slots 5/6 (mask, sinks) are guarded by function constants 300/302
  // and are not declared in this specialisation, so binding them would
  // be binding arguments the pipeline does not have.
  if (i8) { _sage->bind(enc); }        // 15..18, and only on the twin
  enc.dispatch({32 * (unsigned)((p.tq + p.bq - 1) / p.bq),
                4 * (unsigned)p.heads, 1}, {32, 4, 1});
}

void
MetalOps::sdpa_block_causal(ComputeEncoder& enc, const SharedBuffer& q,
                            const SharedBuffer& k, const SharedBuffer& v,
                            const SharedBuffer& out,
                            const SharedBuffer& t_index, int hq, int hkv,
                            int tq, int tk, int d, int cap,
                            int q_offset) const
{
  const float scale = 1.0f / std::sqrt((float)d);
  enc.set_function(_fn_sdpa_block);
  enc.set_buffer(0, q);
  enc.set_buffer(1, k);
  enc.set_buffer(2, v);
  enc.set_buffer(3, out);
  enc.set_constant(4, scale);
  enc.set_constant(5, tk);
  enc.set_constant(6, d);
  enc.set_constant(7, hq);
  enc.set_constant(8, hkv);
  enc.set_constant(9, tq);
  enc.set_constant(10, q_offset);
  enc.set_constant(11, cap);
  enc.set_buffer(12, t_index);
  enc.dispatch({32, (unsigned)hq, (unsigned)tq}, {32, 1, 1});
}

void
MetalOps::transpose_thd(ComputeEncoder& enc, const SharedBuffer& in,
                        const SharedBuffer& out, int T, int H, int D) const
{
  enc.set_function(_fn_transpose);
  enc.set_buffer(0, in);
  enc.set_buffer(1, out);
  enc.set_constant(2, T);
  enc.set_constant(3, H);
  enc.set_constant(4, D);
  enc.dispatch({(unsigned)D, (unsigned)T, (unsigned)H}, {(unsigned)
                std::min(D, 64), 1, 1});
}

void
MetalOps::transpose_htd(ComputeEncoder& enc, const SharedBuffer& in,
                        const SharedBuffer& out, int T, int H, int D) const
{
  enc.set_function(_fn_untranspose);
  enc.set_buffer(0, in);
  enc.set_buffer(1, out);
  enc.set_constant(2, T);
  enc.set_constant(3, H);
  enc.set_constant(4, D);
  enc.dispatch({(unsigned)D, (unsigned)T, (unsigned)H},
               {(unsigned)std::min(D, 64), 1, 1});
}

void
MetalOps::transpose_into_cache(ComputeEncoder& enc, const SharedBuffer& in,
                               const SharedBuffer& cache, int T, int H,
                               int D, int cap, int offset) const
{
  enc.set_function(_fn_cache);
  enc.set_buffer(0, in);
  enc.set_buffer(1, cache);
  enc.set_constant(2, T);
  enc.set_constant(3, H);
  enc.set_constant(4, D);
  enc.set_constant(5, cap);
  enc.set_constant(6, offset);
  enc.dispatch({(unsigned)D, (unsigned)T, (unsigned)H},
               {(unsigned)std::min(D, 64), 1, 1});
}

void
MetalOps::gelu_erf(ComputeEncoder& enc, const SharedBuffer& x, int n) const
{
  enc.set_function(_fn_gelu);
  enc.set_buffer(0, x);
  enc.set_constant(1, n);
  dispatch_1d_(enc, (std::size_t)n);
}

void
MetalOps::silu(ComputeEncoder& enc, const SharedBuffer& x, int n) const
{
  enc.set_function(_fn_silu);
  enc.set_buffer(0, x);
  enc.set_constant(1, n);
  dispatch_1d_(enc, (std::size_t)n);
}

void
MetalOps::add(ComputeEncoder& enc, const SharedBuffer& dst,
              const SharedBuffer& src, int n) const
{
  enc.set_function(_fn_add);
  enc.set_buffer(0, dst);
  enc.set_buffer(1, src);
  enc.set_constant(2, n);
  dispatch_1d_(enc, (std::size_t)n);
}

void
MetalOps::add_row(ComputeEncoder& enc, const SharedBuffer& dst,
                  const SharedBuffer& row, int rows, int dim) const
{
  enc.set_function(_fn_add_row);
  enc.set_buffer(0, dst);
  enc.set_buffer(1, row);
  enc.set_constant(2, rows);
  enc.set_constant(3, dim);
  enc.dispatch({(unsigned)dim, (unsigned)rows, 1}, {64, 1, 1});
}

void
MetalOps::pixel_shuffle(ComputeEncoder& enc, const SharedBuffer& in,
                        const SharedBuffer& out, int c_in, int h, int w,
                        int r) const
{
  enc.set_function(_fn_ps);
  enc.set_buffer(0, in);
  enc.set_buffer(1, out);
  enc.set_constant(2, c_in);
  enc.set_constant(3, h);
  enc.set_constant(4, w);
  enc.set_constant(5, r);
  enc.dispatch({(unsigned)w, (unsigned)h, (unsigned)c_in}, {32, 1, 1});
}

void
MetalOps::pixel_shuffle_hwc(ComputeEncoder& enc, const SharedBuffer& in,
                            const SharedBuffer& out, int c_in, int h,
                            int w, int r) const
{
  enc.set_function(_fn_ps_hwc);
  enc.set_buffer(0, in);
  enc.set_buffer(1, out);
  enc.set_constant(2, c_in);
  enc.set_constant(3, h);
  enc.set_constant(4, w);
  enc.set_constant(5, r);
  enc.dispatch({(unsigned)c_in, (unsigned)w, (unsigned)h}, {64, 1, 1});
}

void
MetalOps::merge_tiles(ComputeEncoder& enc, const SharedBuffer& in,
                      const SharedBuffer& out, int gh, int gw, int c,
                      int m) const
{
  enc.set_function(_fn_merge);
  enc.set_buffer(0, in);
  enc.set_buffer(1, out);
  enc.set_constant(2, gh);
  enc.set_constant(3, gw);
  enc.set_constant(4, c);
  enc.set_constant(5, m);
  enc.dispatch({(unsigned)c, (unsigned)(gw / m), (unsigned)(gh / m)},
               {64, 1, 1});
}

void
MetalOps::to_u8_planar(ComputeEncoder& enc, const SharedBuffer& in,
                       const SharedBuffer& out, int h, int w) const
{
  enc.set_function(_fn_u8);
  enc.set_buffer(0, in);
  enc.set_buffer(1, out);
  enc.set_constant(2, h);
  enc.set_constant(3, w);
  enc.dispatch({(unsigned)w, (unsigned)h, 3}, {32, 1, 1});
}

void
MetalOps::conv3x3_hwc(ComputeEncoder& enc, const SharedBuffer& in,
                      const SharedBuffer& col, const SharedBuffer& w,
                      const SharedBuffer* bias, const SharedBuffer& out,
                      int h, int w_, int cin, int cout) const
{
  // im2col first: [h][w][cin] -> [h*w][9*cin], zero-padded at the border.
  enc.set_function(_fn_im2col);
  enc.set_buffer(0, in);
  enc.set_buffer(1, col);
  enc.set_constant(2, h);
  enc.set_constant(3, w_);
  enc.set_constant(4, cin);
  // The documented 2-D grid {9*C, rows}: it is what keeps the kernel's
  // index arithmetic 32-bit. The 1-D form emulates 64-bit div/mod in
  // software and runs at a fraction of the bandwidth.
  enc.dispatch({(unsigned)(9 * cin), (unsigned)(h * w_), 1}, {32, 1, 1});

  // then the GEMM. The weight must already be packed [cout][9*cin] with
  // the same (ky, kx, cin) order im2col emits -- done once at load, not
  // per call.
  linear(enc, col, w, bias, out, h * w_, 9 * cin, cout);
}

}  // namespace u15
