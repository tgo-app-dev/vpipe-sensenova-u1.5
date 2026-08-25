#ifndef VPIPE_U15_IMAGE_H
#define VPIPE_U15_IMAGE_H

// The image half of the model: the patch embedder in, the pixel head
// out, and the two scalar embedders that ride along with the timestep.
//
// EVERYTHING HERE IS CHANNEL-LAST. That is not a preference: the
// backbone's output is [tokens][hidden] with tokens in row-major (h, w)
// order, which IS a [th][tw][hidden] channel-last map already, and
// libvpipe's im2col_hwc_3x3 wants channel-last too. So the pixel head
// never transposes, and the patch embedder's per-patch linear reads the
// checkpoint's conv weight as-is.

#include "u15-config.h"
#include "u15-metal-ops.h"
#include "u15-weights.h"

#include "apple-silicon/metal-compute/command-stream.h"

#include <memory>
#include <string>
#include <vector>

namespace u15 {

class ImagePath {
 public:
  static std::unique_ptr<ImagePath> create(MetalOps* ops,
                                           const U15Config& cfg,
                                           const U15Weights* w,
                                           std::string* err);

  // ---- in ---------------------------------------------------------
  //
  // Patchify an image into what the embedder consumes.
  //
  //   img [3][H][W] float, MODEL space ([-1, 1])
  //   out [grid_h*grid_w][3*p*p], packed CHANNEL-FIRST within a patch
  //
  // Channel-first here and channel-LAST for the denoised z: the
  // reference computes both from the same image in the same step, with
  // different packings, and using either for both is silent.
  static void patchify_for_embed(const float* img, int h, int w,
                                 int patch_px, float* out);

  // Run the patch embedder. `gen` selects the generation twin
  // (fm_modules.vision_model_mot_gen) over the understanding one.
  //
  //   px  [grid_h*grid_w][3*p*p] bf16
  //   out [(grid_h/2)*(grid_w/2)][hidden] bf16
  bool embed(vpipe::metal_compute::CommandStream& stream,
             const vpipe::metal_compute::SharedBuffer& px, int grid_h,
             int grid_w, bool gen,
             const vpipe::metal_compute::SharedBuffer& out,
             std::string* err);

  // The timestep + noise-scale embedding, as ONE row of `hidden`. The
  // reference adds them together and broadcasts over every image token,
  // so it is computed once per step rather than per token.
  //
  // Runs on the HOST in f32: it is two 4096x4096 matvecs against
  // weights the checkpoint keeps in f32, ~34 MFLOP against the stack's
  // ~40 TFLOP, and rounding the value that drives every token's
  // conditioning to bf16 would be a real loss for no measurable gain.
  // Same reasoning as the LTX adaLN chains.
  std::vector<float> timestep_row(double t, double noise_scale_ratio) const;

  // ---- out --------------------------------------------------------
  //
  // The pixel head. `hidden` is the backbone's output for the image
  // tokens, [th*tw][hidden] bf16, read as a [th][tw][hidden] map.
  //
  //   out_px [th*32][tw*32][3] bf16, MODEL space
  bool decode(vpipe::metal_compute::CommandStream& stream,
              const vpipe::metal_compute::SharedBuffer& hidden, int th,
              int tw, const vpipe::metal_compute::SharedBuffer& out_px,
              std::string* err);

  // [H][W][3] model-space -> [3][H][W] u8, denormalised and clamped.
  void to_u8_planar(vpipe::metal_compute::CommandStream& stream,
                    const vpipe::metal_compute::SharedBuffer& px, int h,
                    int w, const vpipe::metal_compute::SharedBuffer& u8);

 private:
  ImagePath() = default;
  bool build_conv_weights_(std::string* err);

  MetalOps*         _ops = nullptr;
  const U15Weights* _w   = nullptr;
  U15Config         _cfg;

  // The vision rope ladder, built once.
  vpipe::metal_compute::SharedBuffer _vis_inv;

  // conv1/conv2 repacked [cout][ky][kx][cin] to match im2col's column
  // order (ky*3+kx major, channel minor) -- the checkpoint stores
  // [cout][cin][ky][kx], which is a different thing of the same size.
  vpipe::metal_compute::SharedBuffer _conv1_w, _conv2_w;
  vpipe::metal_compute::SharedBuffer _conv1_b, _conv2_b;

  // scratch, grown on demand
  int _tokens = 0, _pix = 0;
  vpipe::metal_compute::SharedBuffer _emb, _merged;
  vpipe::metal_compute::SharedBuffer _a, _b, _c, _col;
};

}  // namespace u15

#endif  // VPIPE_U15_IMAGE_H
