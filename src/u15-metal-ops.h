#ifndef VPIPE_U15_METAL_OPS_H
#define VPIPE_U15_METAL_OPS_H

// The op vocabulary the U1.5 forward is written in.
//
// Most of it is libvpipe's, reached by name from the plugin: the GEMM,
// RMSNorm, SwiGLU, the residual add, im2col for the 3x3 convolutions and
// flash attention. Only the operations that are genuinely this model's
// own come from u15-kernels.metal -- see that file for why each one
// could not be a reuse.
//
// EVERY ComputeFunction here is validated at init(). An unvalidated one
// dispatches as a silent no-op, which reads downstream as a numerically
// wrong model rather than as a missing kernel.

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/metal-sage-attention.h"

#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
class I8GemmContext;    // fwd (generative-models/shared/i8-gemm.h)
}
}

namespace u15 {

// A linear's weight, dense or affine-quantized.
//
// bits and group are per TENSOR, not per checkpoint: model-quantize can
// write a mixed pack, and a loader that reads one tensor's group and
// applies it to the rest produces a checkpoint that loads and generates
// the wrong thing.
struct QWeight {
  vpipe::metal_compute::SharedBuffer w;                  // dense
  vpipe::metal_compute::SharedBuffer codes, scales, qbias;
  int  bits = 0;                 // 4 or 8
  int  group = 0;                // 32 or 64
  bool quantized = false;

  bool empty() const noexcept
  {
    return quantized ? (codes.empty() || scales.empty() || qbias.empty())
                     : w.empty();
  }
  std::size_t byte_size() const noexcept
  {
    return quantized ? (codes.byte_size() + scales.byte_size() +
                        qbias.byte_size())
                     : w.byte_size();
  }
};

class MetalOps {
 public:
  MetalOps();
  // Out of line, because `_i8` is a unique_ptr to a type this header
  // only forward-declares: the implicit destructor would need it
  // complete in every TU that destroys a MetalOps.
  ~MetalOps();
  MetalOps(const MetalOps&)            = delete;
  MetalOps& operator=(const MetalOps&) = delete;

  bool init(vpipe::metal_compute::MetalCompute* mc, std::string* err);

  // ACCELERATED MODE (LOSSY, opt-in): dynamic-int8 GEMMs for the big
  // projections in place of the bf16 matmul2d tiles. The activation and
  // the weight are quantized to i8 on the fly with per-512-group scales
  // and the product runs on the matrix units' int8 pipe.
  //
  // This model qualifies on both counts the mode needs. Its projections
  // already reach matmul2d with a DENSE bf16 weight -- either the
  // weight itself, or the `_w_deq` expansion a quantized one is
  // dequant-once'd into -- which is exactly the "dense, or a dequant
  // scratch" input I8GemmContext takes. And its shapes clear the gate:
  // K in {4096, 12288} are whole 512-groups, so nothing is padded, and
  // N in {4096, 6144, 24576} is far above the floor. Only M decides,
  // against a crossover of ~1k rows.
  //
  // Call AFTER init(): the context needs the MetalCompute init() stored,
  // and it loads its own kernels, so a host without them leaves the mode
  // off however it is asked. `want=false` is a no-op. VPIPE_I8_GEMM
  // overrides either way, which is how an A/B is run.
  // Returns whether the mode is ON afterwards, which is not the same
  // as `want`: env can turn it on, and a host without the kernels
  // leaves it off. The caller logs it -- MetalOps has no session.
  bool enable_i8_gemm(bool want);

  // Drop the mode's grow-only requant scratches. For a generator that
  // keeps its models between beats: the scratches re-grow on the next
  // qualifying GEMM, and holding them idle crowds a downstream VAE
  // decode on a memory-bounded box.
  void release_i8_scratch();

  vpipe::metal_compute::MetalCompute* mc() const { return _mc; }

  // y[M][N] = x[M][K] @ w[N][K]^T (+ bias[N]). The weight is stored
  // [out][in], which is what torch gives and what the _t GEMM wants.
  void linear(vpipe::metal_compute::ComputeEncoder& enc,
              const vpipe::metal_compute::SharedBuffer& x,
              const vpipe::metal_compute::SharedBuffer& w,
              const vpipe::metal_compute::SharedBuffer* bias,
              const vpipe::metal_compute::SharedBuffer& y, int M, int K,
              int N) const;

  // Same as linear(), but with the WIDE (BN=64) tile. Which of the two
  // wins is shape-dependent, so the caller picks per call site from
  // measurement rather than a rule.
  void linear_wide(vpipe::metal_compute::ComputeEncoder& enc,
                   const vpipe::metal_compute::SharedBuffer& x,
                   const vpipe::metal_compute::SharedBuffer& w,
                   const vpipe::metal_compute::SharedBuffer* bias,
                   const vpipe::metal_compute::SharedBuffer& y, int M,
                   int K, int N) const;

  // Fused gate+up+SwiGLU: ONE GEMM emitting silu(gate)*up directly.
  // Replaces two GEMMs and an elementwise pass.
  //
  // THE WEIGHT MUST BE ROW-INTERLEAVED, not concatenated:
  //   row 2i   = gate_proj row i
  //   row 2i+1 = up_proj   row i
  // The kernel reads the pair out of one accumulator fragment (even
  // column = gate, odd = up), so a concatenated [gate | up] weight
  // produces a correctly-shaped result computed from the wrong pairs.
  // See fuse_swiglu_weight().
  void swiglu_fused(vpipe::metal_compute::ComputeEncoder& enc,
                    const vpipe::metal_compute::SharedBuffer& x,
                    const vpipe::metal_compute::SharedBuffer& w_fused,
                    const vpipe::metal_compute::SharedBuffer& y, int M,
                    int K, int inter) const;

  // The same, taking a weight that may be quantized. Falls back to the
  // dense path when it is not, so one call site serves both pack kinds.
  void linear(vpipe::metal_compute::ComputeEncoder& enc,
              const vpipe::metal_compute::SharedBuffer& x, const QWeight& w,
              const vpipe::metal_compute::SharedBuffer* bias,
              const vpipe::metal_compute::SharedBuffer& y, int M, int K,
              int N) const;

  // True when the host ships the affine qmm kernels. A dense checkpoint
  // runs perfectly well without them, so this is asked at the point a
  // quantized weight is actually met rather than at init.
  bool quant_available() const noexcept { return _quant_ok; }

  // y += bias, broadcast over rows. The qmm kernel has NO bias slot --
  // its buffer(2) is the quantization ZERO-POINT, not the linear's bias
  // -- so a quantized linear with a bias needs this second pass.
  void bias_add(vpipe::metal_compute::ComputeEncoder& enc,
                const vpipe::metal_compute::SharedBuffer& y,
                const vpipe::metal_compute::SharedBuffer& bias, int M,
                int N) const;

  // Whole-row RMSNorm with gain -- the layer norms. `fast` selects the
  // simd_sum reduction over the threadgroup-tree one; both are
  // libvpipe's and share a buffer contract, so this exists to be
  // measured rather than assumed.
  void rms_norm(vpipe::metal_compute::ComputeEncoder& enc,
                const vpipe::metal_compute::SharedBuffer& x,
                const vpipe::metal_compute::SharedBuffer& w,
                const vpipe::metal_compute::SharedBuffer& y, int rows,
                int dim, float eps, bool fast = true) const;

  // The split q/k norm: two reductions per head, two weights.
  void split_qk_norm(vpipe::metal_compute::ComputeEncoder& enc,
                     const vpipe::metal_compute::SharedBuffer& x,
                     const vpipe::metal_compute::SharedBuffer& w_t,
                     const vpipe::metal_compute::SharedBuffer& w_hw,
                     int heads, int tokens, int head_dim, float eps) const;

  // The three-way split rope. `cos`/`sin` are f32 [T][D], packed to the
  // head layout by build_rope_tables().
  void split_rope(vpipe::metal_compute::ComputeEncoder& enc,
                  const vpipe::metal_compute::SharedBuffer& x,
                  const vpipe::metal_compute::SharedBuffer& cos,
                  const vpipe::metal_compute::SharedBuffer& sin, int heads,
                  int tokens, int head_dim, int d_t, int d_h) const;

  // Build those tables on the host. The ladder arithmetic lives here
  // rather than in the kernel so the angles are formed in double and
  // narrowed once -- the t ladder reaches theta 5e6, where forming the
  // angle in bf16 is a visible phase error.
  //
  // Returns a [tokens][head_dim] f32 pair.
  struct RopeTables {
    vpipe::metal_compute::SharedBuffer cos, sin;
  };
  RopeTables build_rope_tables(const std::vector<int>& pos_t,
                               const std::vector<int>& pos_h,
                               const std::vector<int>& pos_w, int head_dim,
                               int d_t, int d_h, double theta_t,
                               double theta_hw) const;

  // The patch embedder's interleaved 2-D rope.
  void vision_rope2d(vpipe::metal_compute::ComputeEncoder& enc,
                     const vpipe::metal_compute::SharedBuffer& x,
                     const vpipe::metal_compute::SharedBuffer& inv_freq,
                     int n, int chan, int grid_w) const;

  vpipe::metal_compute::SharedBuffer make_inv_freq(int dim,
                                                   double theta) const;

  // SwiGLU: y = silu(gate) * up, both [rows][inter].
  void swiglu(vpipe::metal_compute::ComputeEncoder& enc,
              const vpipe::metal_compute::SharedBuffer& gate,
              const vpipe::metal_compute::SharedBuffer& up,
              const vpipe::metal_compute::SharedBuffer& y, int rows,
              int inter) const;

  // Grouped-query attention. q [hq][tq][d], k/v [hkv][cap][d] (the KV
  // cache, so the stride is the CAPACITY), out [tq][hq*d].
  //
  // `causal` selects the und prefill's horizon. For a pure-text prefix
  // the reference's block-causal mask REDUCES to plain causal, because
  // every text token has a distinct t and the `t[j] == t[i]` clause then
  // only fires on the diagonal. That equivalence is what lets this use
  // libvpipe's mask-free kernels; it stops holding as soon as a
  // reference image puts several tokens at one t, which is why the edit
  // path needs a masked kernel and is not wired here.
  //
  // `q_offset` places the queries within the key range (causal only).
  void sdpa(vpipe::metal_compute::ComputeEncoder& enc,
            const vpipe::metal_compute::SharedBuffer& q,
            const vpipe::metal_compute::SharedBuffer& k,
            const vpipe::metal_compute::SharedBuffer& v,
            const vpipe::metal_compute::SharedBuffer& out, int hq, int hkv,
            int tq, int tk, int d, int cap, bool causal,
            int q_offset = 0) const;

  // ---- STEEL flash attention ---------------------------------------
  //
  // The MMA/threadgroup-tiled flash kernel every image DiT in the tree
  // uses (attn_steel_h_bd128_bf16). The scalar sdpa_full_f16 above
  // materialises one query at a time across 32 lanes; this tiles both
  // axes and keeps the running softmax in registers.
  //
  // Only head_dim 64 and 128 are instantiated, so callers ask rather
  // than assume -- binding a kernel compiled for a different D would
  // read past every row.
  struct SteelAttn {
    vpipe::metal_compute::ComputeFunction fn;
    // The SAME kernel with constant 306 true: SageAttention's int8 QK
    // twin. Built beside the f16 one, not instead of it, because
    // `sage_dense_layers` leaves a prefix of the stack on the f16
    // kernel and one pass therefore dispatches both.
    //
    // Built from what the GPU CAN do rather than from what the config
    // asked. A plan is cached per pass shape and outlives a beat, where
    // the setting is per beat -- so a twin conditioned on the setting
    // would be missing for every later beat that turned the tier on
    // after one that had it off, and that image would render dense with
    // the log saying otherwise.
    vpipe::metal_compute::ComputeFunction fn_i8;
    vpipe::metal_compute::SharedBuffer    params;
    int heads = 0, tq = 0, tkv = 0, head_dim = 0, bq = 0;
    // What Sage needs and the f16 dispatch does not have to remember:
    // the KEY head count (this model is GQA) and the cache CAPACITY,
    // which is the K head stride and is NOT tkv. The quantizer walks the
    // same bytes the kernel does, so it needs the same three numbers.
    int heads_kv = 0, bk = 0, kv_stride = 0;
    bool valid() const { return fn.valid() && !params.empty(); }
  };

  bool steel_attn_available(int head_dim) const noexcept;

  // Which path the M5 gates actually settled on. For the A/B test and
  // for a log line that has to be able to say "off": a rate change
  // nobody can observe is one nobody can measure.
  bool dense_is_mma2() const noexcept { return _use_mma2; }
  bool attn_is_nax() const noexcept { return _use_nax; }
  bool quant_is_mma2() const noexcept { return _use_mma2_q; }

  // `kv_stride_tokens` is the K/V cache CAPACITY, not the key length:
  // the cache is sized for prefix + generation block and only partly
  // filled, so the per-head stride is the capacity.
  bool steel_attn_plan(SteelAttn* p, int heads_q, int heads_kv, int tq,
                       int tkv, int head_dim, int kv_stride_tokens) const;

  // `sage_layer` is the stack index when the caller will accept
  // SageAttention here and -1 when it will not. When the tier is live
  // and the layer is past `sage_dense_layers`, the int8 prologue is
  // encoded HERE -- into the same encoder, immediately before the
  // dispatch that reads what it wrote, which is the whole of the
  // ordering between them.
  void sdpa_steel(vpipe::metal_compute::ComputeEncoder& enc,
                  const SteelAttn& p,
                  const vpipe::metal_compute::SharedBuffer& q,
                  const vpipe::metal_compute::SharedBuffer& k,
                  const vpipe::metal_compute::SharedBuffer& v,
                  const vpipe::metal_compute::SharedBuffer& out,
                  int sage_layer = -1) const;

  // ---- SageAttention -------------------------------------------------
  //
  // The QK^T product of the flash attention in INT8, with one scale per
  // attention block and the key side quantized as K - mean(K) over
  // tokens. That smoothing is exact rather than approximate: a
  // per-channel shift moves every score in a row by the same amount and
  // softmax does not see it. P*V stays in bf16.
  //
  // IT REACHES THE BIDIRECTIONAL PASS AND NOTHING ELSE, which is not a
  // property of the method but of this port: the causal and
  // block-causal branches do not run on the steel kernel at all, and
  // the int8 twin is that kernel's function constant.
  //
  // MATRIX CORES ARE THE WHOLE CONDITION -- the int8 fragment MMA has
  // no ALU fallback -- so this declines on a box without them and the
  // model runs bf16. `err` is set only when the tier was asked for and
  // its kernels would not build, which is a refusal rather than a
  // decline.
  bool set_sage(const vpipe::genai::sage::Config& cfg, std::string* err);
  bool sage_takes(int head_dim) const noexcept;
  void sage_lend(const vpipe::metal_compute::SharedBuffer& a,
                 const vpipe::metal_compute::SharedBuffer& b) const;
  std::size_t sage_resident_bytes() const noexcept;
  const vpipe::genai::sage::Config& sage_config() const noexcept
  {
    return _sage_cfg;
  }

  // BLOCK-CAUSAL attention: attend iff t[j] == t[i] or j <= i. Needed
  // once a reference image is in the prefix, where several tokens share
  // one t and the rule stops collapsing to plain causal.
  //
  // `t_index` is an i32 buffer of T_kv entries.
  void sdpa_block_causal(vpipe::metal_compute::ComputeEncoder& enc,
                         const vpipe::metal_compute::SharedBuffer& q,
                         const vpipe::metal_compute::SharedBuffer& k,
                         const vpipe::metal_compute::SharedBuffer& v,
                         const vpipe::metal_compute::SharedBuffer& out,
                         const vpipe::metal_compute::SharedBuffer& t_index,
                         int hq, int hkv, int tq, int tk, int d, int cap,
                         int q_offset) const;

  void transpose_thd(vpipe::metal_compute::ComputeEncoder& enc,
                     const vpipe::metal_compute::SharedBuffer& in,
                     const vpipe::metal_compute::SharedBuffer& out, int T,
                     int H, int D) const;

  // [H][T][D] -> [T][H*D]. The SDPA kernels write head-major; o_proj is
  // a GEMM over token rows, so this sits between them.
  void transpose_htd(vpipe::metal_compute::ComputeEncoder& enc,
                     const vpipe::metal_compute::SharedBuffer& in,
                     const vpipe::metal_compute::SharedBuffer& out, int T,
                     int H, int D) const;

  // [T][H*D] -> a KV cache slice [H][cap][D] starting at `offset`.
  void transpose_into_cache(vpipe::metal_compute::ComputeEncoder& enc,
                            const vpipe::metal_compute::SharedBuffer& in,
                            const vpipe::metal_compute::SharedBuffer& cache,
                            int T, int H, int D, int cap, int offset) const;

  void gelu_erf(vpipe::metal_compute::ComputeEncoder& enc,
                const vpipe::metal_compute::SharedBuffer& x, int n) const;
  void silu(vpipe::metal_compute::ComputeEncoder& enc,
            const vpipe::metal_compute::SharedBuffer& x, int n) const;
  void add(vpipe::metal_compute::ComputeEncoder& enc,
           const vpipe::metal_compute::SharedBuffer& dst,
           const vpipe::metal_compute::SharedBuffer& src, int n) const;
  void add_row(vpipe::metal_compute::ComputeEncoder& enc,
               const vpipe::metal_compute::SharedBuffer& dst,
               const vpipe::metal_compute::SharedBuffer& row, int rows,
               int dim) const;

  // PixelShuffle on a CHANNEL-LAST map -- what the pixel head runs in.
  void pixel_shuffle_hwc(vpipe::metal_compute::ComputeEncoder& enc,
                         const vpipe::metal_compute::SharedBuffer& in,
                         const vpipe::metal_compute::SharedBuffer& out,
                         int c_in, int h, int w, int r) const;

  // The patch embedder's m x m tile gather, channel-SLOWEST.
  void merge_tiles(vpipe::metal_compute::ComputeEncoder& enc,
                   const vpipe::metal_compute::SharedBuffer& in,
                   const vpipe::metal_compute::SharedBuffer& out, int gh,
                   int gw, int c, int m) const;

  // [H][W][3] model-space -> [3][H][W] u8, denormalised and clamped.
  void to_u8_planar(vpipe::metal_compute::ComputeEncoder& enc,
                    const vpipe::metal_compute::SharedBuffer& in,
                    const vpipe::metal_compute::SharedBuffer& out, int h,
                    int w) const;

  void pixel_shuffle(vpipe::metal_compute::ComputeEncoder& enc,
                     const vpipe::metal_compute::SharedBuffer& in,
                     const vpipe::metal_compute::SharedBuffer& out, int c_in,
                     int h, int w, int r) const;

  // 3x3 stride-1 pad-1 convolution over a CHANNEL-LAST [h][w][cin] map,
  // as im2col + GEMM. Channel-last because that is the layout libvpipe's
  // im2col_hwc_3x3 expects, and it is also what makes the GEMM's K the
  // contiguous axis.
  void conv3x3_hwc(vpipe::metal_compute::ComputeEncoder& enc,
                   const vpipe::metal_compute::SharedBuffer& in,
                   const vpipe::metal_compute::SharedBuffer& col,
                   const vpipe::metal_compute::SharedBuffer& w,
                   const vpipe::metal_compute::SharedBuffer* bias,
                   const vpipe::metal_compute::SharedBuffer& out, int h,
                   int w_, int cin, int cout) const;

 private:
  vpipe::metal_compute::MetalCompute* _mc = nullptr;

  vpipe::metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_sdpa;
  vpipe::metal_compute::ComputeLibrary _lib_rms, _lib_u15;
  vpipe::metal_compute::ComputeLibrary _lib_attn;
  bool _steel_ok = false;

  // ---- M5 matrix cores (Apple10+) -------------------------------------
  //
  // Both are the SAME arithmetic through hardware matrix units rather
  // than simdgroup ops, so they are a rate choice and not a numerics
  // one -- which is why each has a kill switch and neither changes what
  // the model computes beyond f16/bf16 rounding order.
  //
  //   _lib_dense_mma  matmul2d for the dense projections. The backbone's
  //                   GEMMs are M=tokens by N in {4096, 6144, 24576} at
  //                   K in {4096, 12288}: squarely the regime the 128-row
  //                   tile was built for.
  //   _lib_attn_nax   the NAX twin of the steel flash-attention this
  //                   already uses, at the same bd128_bf16 head width.
  //                   It tiles 64x32 where steel tiles 32x16, so the
  //                   params block and the launch grid change with it --
  //                   see steel_attn_plan().
  //
  // Absent on a pre-M5 GPU, and absent from a host that does not ship
  // them; both stay OPTIONAL for the same reason the steel library does.
  vpipe::metal_compute::ComputeLibrary _lib_attn_nax;
  vpipe::metal_compute::ComputeLibrary _lib_dense_mma;
  vpipe::metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_wide;
  bool _use_nax  = false;
  bool _use_mma2 = false;
  // A row floor for matmul2d, kept as a knob and set to 1 because the
  // crossover it was meant to guard AGAINST DOES NOT EXIST on this
  // model's shapes. MEASURED on the M5 at K=4096, N=6144, interleaved
  // arms, min of three rounds -- matmul2d over steel:
  //
  //   M     1     4     8    16    32    64   256   1024
  //   x  1.47  1.62  1.47  1.78  1.77  1.74  3.30   3.75
  //
  // The 128-row tile is nearly all padding at M=1 and still wins, so a
  // floor of 64 (the first guess here) was giving away 1.5-1.8x on
  // every short sequence. Left overridable (VPIPE_U15_MMA_MIN_M) so the
  // next GPU can be probed the same way rather than trusted to agree.
  int  _mma_min_m = 1;

  // The DEQUANT-ONCE half, for a quantized weight. Expands codes into
  // `_w_deq` and runs the same matmul2d, which is worth it exactly when
  // the expansion amortizes -- see the definition for the measured
  // floor. False when it does not qualify.
  bool linear_mma_q_(vpipe::metal_compute::ComputeEncoder& enc,
                     const vpipe::metal_compute::SharedBuffer& x,
                     const QWeight& w,
                     const vpipe::metal_compute::SharedBuffer* bias,
                     const vpipe::metal_compute::SharedBuffer& y,
                     int M, int K, int N) const;

  // One expansion buffer, reused across a pass. MUTABLE because the
  // linear() overloads are const and this is a scratch, not state the
  // caller can observe: what comes out of a linear() does not depend on
  // whether a previous one grew it. Reuse across the GEMMs of one
  // encoder is safe on Metal's WAR hazard tracking -- each
  // dequant/matmul pair is ordered before the next pair writes it --
  // and that serialisation is the price of not holding an expansion per
  // weight.
  mutable vpipe::metal_compute::SharedBuffer _w_deq;

  // Null unless accelerated mode is on AND its kernels loaded. MUTABLE
  // for the same reason `_w_deq` is: the linear() overloads are const
  // and this is a scratch-owning encoder helper, not state a caller can
  // observe in the result.
  mutable std::unique_ptr<vpipe::genai::I8GemmContext> _i8;

  // SageAttention's driver and the settings it was built from. mutable
  // for the reason `_i8` is: every dispatch helper here is const and
  // this one allocates on first sight of a geometry.
  mutable std::unique_ptr<vpipe::genai::MetalSageAttention> _sage;
  vpipe::genai::sage::Config _sage_cfg;
  vpipe::metal_compute::ComputeLibrary _lib_dequant;
  // [bits 4|8][group 32|64]
  vpipe::metal_compute::ComputeFunction _fn_dequant[2][2];
  bool _use_mma2_q = false;
  // Rows below which expanding the whole weight costs more than the
  // matmul saves. Unlike the dense floor this one is REAL and it is the
  // reason the two paths cannot share a threshold: the expansion is
  // O(N*K) whatever M is, so it has to be amortized before it pays.
  //
  // MEASURED on the M5, w8g64 at K=4096 N=6144, interleaved arms, min
  // of three -- dequant-once+matmul2d over the steel qmm:
  //
  //   M      1     8    32    64    96   128   192   256   512  1024
  //   x   0.53  0.56  0.53  0.93  1.66  1.64  1.85  2.40  2.94  3.46
  //
  // Flat at ~0.55x while the expansion dominates, then a step. 96 is
  // the first row count that wins; 64 still loses 7%, so the floor sits
  // above it rather than at the sign change. Overridable with
  // VPIPE_U15_MMA_Q_MIN_M, because this crossover is a property of the
  // weight's N*K against the GPU's rate and will move on another box.
  int  _mma_q_min_m = 96;

  // The matmul2d half of linear()/linear_wide(). False when the shape or
  // the box does not qualify, and the caller then keeps its steel path.
  bool linear_mma_(vpipe::metal_compute::ComputeEncoder& enc,
                   const vpipe::metal_compute::SharedBuffer& x,
                   const vpipe::metal_compute::SharedBuffer& w,
                   const vpipe::metal_compute::SharedBuffer* bias,
                   const vpipe::metal_compute::SharedBuffer& y,
                   int M, int K, int N) const;

  vpipe::metal_compute::ComputeFunction _fn_gemm, _fn_gemm_wide;
  vpipe::metal_compute::ComputeFunction _fn_rms, _fn_rms_slow;
  vpipe::metal_compute::ComputeFunction _fn_swiglu;
  vpipe::metal_compute::ComputeFunction _fn_ff_swiglu;
  vpipe::metal_compute::ComputeFunction _fn_sdpa, _fn_sdpa_causal;
  vpipe::metal_compute::ComputeFunction _fn_sdpa_block;
  vpipe::metal_compute::ComputeFunction _fn_im2col;
  vpipe::metal_compute::ComputeFunction _fn_split_norm, _fn_split_rope;
  vpipe::metal_compute::ComputeFunction _fn_vis_rope, _fn_gelu, _fn_silu;
  vpipe::metal_compute::ComputeFunction _fn_add, _fn_add_row, _fn_ps;
  vpipe::metal_compute::ComputeFunction _fn_ps_hwc, _fn_merge, _fn_u8;
  vpipe::metal_compute::ComputeFunction _fn_transpose, _fn_cache;
  vpipe::metal_compute::ComputeFunction _fn_bias_add;
  // [bits 4|8][group 32|64][narrow|wide tile]
  vpipe::metal_compute::ComputeLibrary  _lib_qmm;
  vpipe::metal_compute::ComputeFunction _fn_qmm[2][2][2];
  bool _quant_ok = false;
  vpipe::metal_compute::ComputeFunction _fn_untranspose;
};

}  // namespace u15

#endif  // VPIPE_U15_METAL_OPS_H
