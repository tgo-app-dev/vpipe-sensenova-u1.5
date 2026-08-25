#ifndef VPIPE_U15_REF_H
#define VPIPE_U15_REF_H

// A CPU reference for the pieces of SenseNova-U1.5 that no existing
// vpipe path already covers. It exists to settle every ordering and
// indexing question with NO kernels involved -- so that when the Metal
// path disagrees with the goldens, a transcription bug and a kernel bug
// cannot be debugged at the same time.
//
// Layout conventions, chosen to match what the reference produces so a
// golden can be compared elementwise without a transpose in between:
//
//   activations   [tokens][dim]                 token-major
//   linear weight [out][in]                     as torch stores it
//   q / k / v     [head][token][head_dim]       head-major, POST-transpose
//
// Everything is float32. This is a reference, not a fast path.

#include <cstddef>
#include <vector>

namespace u15 {
namespace ref {

// ---------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------

// y[r][c] = x[r][c] * rsqrt(mean(x[r]^2) + eps) * w[c], per row of `dim`.
void rms_norm(const float* x, const float* w, float eps, std::size_t rows,
              std::size_t dim, float* out);

// y = x @ W^T + b   with W [out_dim][in_dim], b optional (may be null).
void linear(const float* x, const float* w, const float* b,
            std::size_t rows, std::size_t in_dim, std::size_t out_dim,
            float* out);

void silu(float* x, std::size_t n);
void gelu_erf(float* x, std::size_t n);      // the exact erf GELU

// ---------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------

// inv_freq[i] = 1 / theta^(2i / dim), i < dim/2.
std::vector<float> rope_inv_freq(int dim, double theta);

// The BACKBONE convention: rotate_half. cos/sin are the doubled table
// (emb = cat(freqs, freqs)), so both have `dim` entries per position.
//   out[j]       = x[j]*cos[j]       - x[j+dim/2]*sin[j]
//   out[j+dim/2] = x[j+dim/2]*cos[j] + x[j]*sin[j]
void rope_half_inplace(float* x, const std::vector<float>& inv_freq,
                       int dim, double position);

// The VISION convention: INTERLEAVED pairs. Same model, different
// convention, in the same forward pass -- see ARCHITECTURE.md section 5.
//   out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
//   out[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
void rope_interleaved_inplace(float* x, const std::vector<float>& inv_freq,
                              int dim, double position);

// ---------------------------------------------------------------------
// the three-way split head
// ---------------------------------------------------------------------

// One expert's projection + norm + split RoPE, for q or k.
//
//   x        [tokens][hidden]
//   w_proj   [heads*head_dim][hidden]
//   w_norm   [head_dim/2]        applied to the t half
//   w_norm_hw[head_dim/2]        applied to the hw half, BEFORE the h/w
//                                split -- it spans all 64, not 32
//   pos_t/h/w  one position per token
//   out      [heads][tokens][head_dim], laid out [t | h | w]
struct SplitRopeParams {
  int    heads      = 0;
  int    head_dim   = 0;
  double theta_t    = 5000000.0;
  double theta_hw   = 10000.0;
  double rms_eps    = 1e-6;
};

void split_rope_project(const float* x, const float* w_proj,
                        const float* w_norm, const float* w_norm_hw,
                        const std::vector<int>& pos_t,
                        const std::vector<int>& pos_h,
                        const std::vector<int>& pos_w,
                        std::size_t tokens, std::size_t hidden,
                        const SplitRopeParams& p, float* out);

// ---------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------

// Grouped-query attention over head-major q/k/v.
//   q [hq][tokens_q][d], k/v [hkv][tokens_k][d]
//   mask, when non-null, is ADDITIVE and [tokens_q][tokens_k]
//   out [tokens_q][hq*d]   token-major, ready for o_proj
void attention(const float* q, const float* k, const float* v,
               const float* mask, std::size_t hq, std::size_t hkv,
               std::size_t tq, std::size_t tk, std::size_t d, float* out);

// The reference's block-causal rule:
//   attend(i,j) iff t[j] == t[i] or j <= i
// Causal ACROSS t-blocks, bidirectional WITHIN one. Additive 0 / -inf.
std::vector<float> block_causal_mask(const std::vector<int>& t_index);

// ---------------------------------------------------------------------
// one MoT decoder layer
// ---------------------------------------------------------------------

// Both experts share this structure and differ only in which weights
// are bound, which is exactly why one struct serves both.
struct LayerWeights {
  const float* input_layernorm          = nullptr;  // [hidden]
  const float* post_attention_layernorm = nullptr;  // [hidden]
  const float* q_proj = nullptr;   // [heads*hd][hidden]
  const float* k_proj = nullptr;   // [kv_heads*hd][hidden]
  const float* v_proj = nullptr;   // [kv_heads*hd][hidden]
  const float* o_proj = nullptr;   // [hidden][heads*hd]
  const float* q_norm = nullptr;   // [hd/2]
  const float* k_norm = nullptr;   // [hd/2]
  const float* q_norm_hw = nullptr;
  const float* k_norm_hw = nullptr;
  const float* gate_proj = nullptr;  // [inter][hidden]
  const float* up_proj   = nullptr;  // [inter][hidden]
  const float* down_proj = nullptr;  // [hidden][inter]
};

struct LayerDims {
  std::size_t hidden      = 0;
  std::size_t heads       = 0;
  std::size_t kv_heads    = 0;
  std::size_t head_dim    = 0;
  std::size_t intermediate = 0;
  double      rms_eps     = 1e-6;
  double      theta_t     = 5000000.0;
  double      theta_hw    = 10000.0;
};

// x [tokens][hidden] -> out [tokens][hidden].
// `mask` may be null (the generation path runs unmasked/bidirectional).
void decoder_layer(const float* x, const LayerWeights& w,
                   const LayerDims& d, const std::vector<int>& pos_t,
                   const std::vector<int>& pos_h,
                   const std::vector<int>& pos_w, const float* mask,
                   std::size_t tokens, float* out);

// ---------------------------------------------------------------------
// the patch embedder
// ---------------------------------------------------------------------

struct VisionWeights {
  const float* patch_w = nullptr;  // [hidden][3*p*p]
  const float* patch_b = nullptr;  // [hidden]
  const float* dense_w = nullptr;  // [llm_hidden][hidden*merge*merge]
  const float* dense_b = nullptr;  // [llm_hidden]
};

// px [grid_h*grid_w][3*p*p] -> out [(grid_h/2)*(grid_w/2)][llm_hidden]
void vision_embed(const float* px, const VisionWeights& w, int grid_h,
                  int grid_w, int patch_px, int hidden, int llm_hidden,
                  int merge, double theta, float* out);

// ---------------------------------------------------------------------
// the pixel head
// ---------------------------------------------------------------------

struct HeadWeights {
  const float* conv1_w = nullptr;  // [c1_out][c1_in][3][3]
  const float* conv1_b = nullptr;
  const float* conv2_w = nullptr;  // [c2_out][c2_in][3][3]
  const float* conv2_b = nullptr;
};

// x [in_ch][th][tw] -> out [3][th*32][tw*32].
//   PixelShuffle(2) -> conv1 3x3 -> GELU -> PixelShuffle(2) -> conv2 3x3
//   -> PixelShuffle(8)
void conv_decoder(const float* x, const HeadWeights& w, int in_ch, int th,
                  int tw, int hidden_ch, float* out);

// PixelShuffle(r): [c*r*r][h][w] -> [c][h*r][w*r], input channel
// c*r^2 + i*r + j landing at output channel c offset (i, j) --
// channel-SLOWEST, which is the half of this that is easy to invert.
void pixel_shuffle(const float* x, int ch_in, int h, int w, int r,
                   float* out);

// 3x3 convolution, stride 1, zero pad 1. [cin][h][w] -> [cout][h][w].
void conv3x3(const float* x, const float* kernel, const float* b, int cin,
             int cout, int h, int width, float* out);

// ---------------------------------------------------------------------
// patchify -- TWO different packings, from one image, in one step
// ---------------------------------------------------------------------

// channel_last (the reference's default): 'nchpwq->nhwpqc'.
// Used for z, the thing being denoised.
void patchify_channel_last(const float* img, int c, int h, int w, int p,
                           float* out);

// channel_first=True: 'nchpwq->nhwcpq'. Used for the embedder input.
void patchify_channel_first(const float* img, int c, int h, int w, int p,
                            float* out);

// The same packing, but reading a CHANNEL-LAST [h][w][c] image.
//
// The reference keeps its working image planar [3][H][W]; this port
// keeps it interleaved [H][W][3], because that is what the pixel head
// emits and what the u8 conversion consumes. Feeding an interleaved
// image to patchify_channel_first reads the first third of the buffer
// as the red plane -- which for the pure NOISE of step 0 is
// statistically identical and therefore invisible, and from step 1
// destroys every bit of structure the denoise has built. That is
// exactly how it presented: a rich, prompt-sensitive first step
// followed by immediate collapse to a flat field.
void patchify_channel_first_hwc(const float* img_hwc, int c, int h, int w,
                                int p, float* out);

// The inverse of patchify_channel_last.
void unpatchify_channel_last(const float* x, int c, int h, int w, int p,
                             float* img);

// ---------------------------------------------------------------------
// reference-image preprocessing (the EDIT path)
// ---------------------------------------------------------------------
//
// A reference image goes through a pipeline the generated image does
// NOT: a Qwen-VL style smart_resize to a multiple of 32, PIL's default
// BICUBIC resample, and IMAGENET normalisation. That last one is the
// trap -- the generated image uses mean = std = 0.5, and a reference
// normalised the same way is a picture the model reads differently.
inline constexpr double kImagenetMean[3] = {0.485, 0.456, 0.406};
inline constexpr double kImagenetStd[3]  = {0.229, 0.224, 0.225};

// Rescale so both dimensions are divisible by `factor` and the total
// pixel count lands in [min_pixels, max_pixels]. Returns false when the
// aspect ratio exceeds 200:1, which the reference refuses outright.
bool smart_resize(int height, int width, int factor, long min_pixels,
                  long max_pixels, int* out_h, int* out_w);

// PIL-compatible resample of a channel-last u8 image.
//
// PIL's default for an RGB image is BICUBIC (a = -0.5) with the filter
// support SCALED by the downscale factor -- i.e. antialiased on the way
// down, which a naive bicubic is not. It runs as two passes with an
// 8-bit intermediate, and this reproduces that rather than staying in
// float, so the two agree to within rounding.
void resize_bicubic_u8(const unsigned char* src, int sw, int sh,
                       unsigned char* dst, int dw, int dh, int channels);

// u8 channel-last image -> the patch tensor the und embedder consumes:
// [grid_h*grid_w][3*p*p], ImageNet-normalised, packed CHANNEL-FIRST
// within each patch (the same packing the generation path uses for its
// embedder input, and the opposite of the one it uses for z).
void reference_patches(const unsigned char* rgb, int w, int h, int patch,
                       float* out);

// ---------------------------------------------------------------------
// sampling
// ---------------------------------------------------------------------

// sigma = 1-t; sigma = shift*sigma/(1+(shift-1)*sigma); t = 1-sigma.
std::vector<float> time_schedule(int num_steps, double shift);

// The sinusoid the two scalar embedders share: half = dim/2 frequencies,
// then cat([cos, sin]) -- COSINE FIRST.
std::vector<float> timestep_sinusoid(double t, int dim,
                                     double max_period = 10000.0);

// alpha = <pos, neg> / (||neg||^2 + 1e-8), computed in double because
// the reference forces f32 there regardless of the surrounding autocast.
double optimized_scale(const float* pos, const float* neg, std::size_t n);

}  // namespace ref
}  // namespace u15

#endif  // VPIPE_U15_REF_H
